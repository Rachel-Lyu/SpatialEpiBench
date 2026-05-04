import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from copy import deepcopy
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from .utils import *
from .metrics import crps_ensemble, get_loss, wis_from_quantiles

US_POP_2019 = {
    "AL": 4903185, "AK": 731545,  "AZ": 7278717, "AR": 3017804, "CA": 39512223,
    "CO": 5758736, "CT": 3565287, "DE": 973764,  "DC": 705749,  "FL": 21477737,
    "GA": 10617423,"HI": 1415872, "ID": 1787065, "IL": 12671821,"IN": 6732219,
    "IA": 3155070, "KS": 2913314, "KY": 4467673, "LA": 4648794, "ME": 1344212,
    "MD": 6045680, "MA": 6892503, "MI": 9986857, "MN": 5639632, "MS": 2976149,
    "MO": 6137428, "MT": 1068778, "NE": 1934408, "NV": 3080156, "NH": 1359711,
    "NJ": 8882190, "NM": 2096829, "NY": 19453561,"NC": 10488084,"ND": 762062,
    "OH": 11689100,"OK": 3956971, "OR": 4217737, "PA": 12801989,"RI": 1059361,
    "SC": 5148714, "SD": 884659,  "TN": 6829174, "TX": 28995881,"UT": 3205958,
    "VT": 623989,  "VA": 8535519, "WA": 7614893, "WV": 1792147, "WI": 5822434,
    "WY": 578759,  "PR": 3193694, "NYC": 8804190
}


def resolve_population_2019(
    M,
    population=None,   # None | scalar | tensor/list length M
    regions=None,      # None | "CA" | list[str] length M
    device=None,
    dtype=None,
):
    """
    Returns N_vec: [M] tensor (population per node).

    Priority:
      1) explicit `population` (scalar or length-M)
      2) `regions` lookup in US_POP_2019 (scalar or length-M)
      3) fallback: ones (treat as normalized population)
    """
    device = device or "cpu"
    dtype = dtype or torch.float32

    # 1) explicit population provided
    if population is not None:
        if torch.is_tensor(population):
            pop = population.to(device=device, dtype=dtype)
            if pop.numel() == 1:
                return pop.view(1).expand(M)
            if pop.numel() == M:
                return pop.view(M)
            raise ValueError(f"population tensor must have numel 1 or {M}, got {pop.numel()}")
        # python number
        if isinstance(population, (int, float)):
            return torch.full((M,), float(population), device=device, dtype=dtype)
        # list/tuple/np array
        pop = torch.as_tensor(population, device=device, dtype=dtype).flatten()
        if pop.numel() == 1:
            return pop.expand(M)
        if pop.numel() == M:
            return pop
        raise ValueError(f"population must be scalar or length {M}, got {pop.numel()}")

    # 2) regions lookup
    if regions is not None:
        if isinstance(regions, str):
            if regions not in US_POP_2019:
                raise KeyError(f"Unknown region code: {regions}")
            return torch.full((M,), float(US_POP_2019[regions]), device=device, dtype=dtype)

        # list of region codes per node
        if isinstance(regions, (list, tuple)):
            if len(regions) != M:
                raise ValueError(f"regions must have length {M}, got {len(regions)}")
            vals = []
            for r in regions:
                if r not in US_POP_2019:
                    raise KeyError(f"Unknown region code: {r}")
                vals.append(float(US_POP_2019[r]))
            return torch.tensor(vals, device=device, dtype=dtype)

        raise ValueError("regions must be a string or list/tuple of strings")

    # 3) fallback
    return torch.ones((M,), device=device, dtype=dtype)

def _inv_tanh_0_1(y, eps=1e-6, device=None, dtype=torch.float32):
    """
    y in (0,1) -> x in R such that (tanh(x)+1)/2 = y
    """
    y = torch.as_tensor(y, device=device, dtype=dtype)
    y = torch.clamp(y, eps, 1 - eps)
    return torch.atanh(2.0 * y - 1.0)

class SIRm_tanh(nn.Module):
    """
    SIR with beta,gamma in (0,1) via tanh reparam.
    State: (S, I, R) in counts (or any consistent unit).
    """
    def __init__(self, population, parameter, dtype=torch.float32):
        super().__init__()
        self.register_buffer("N", torch.as_tensor(population, dtype=dtype))
        self.init_params(parameter, dtype=dtype)

    def init_params(self, params, dtype=torch.float32):
        device = self.N.device
        self.logbeta  = Parameter(_inv_tanh_0_1(params["beta"],  device=device, dtype=dtype), requires_grad=True)
        self.loggamma = Parameter(_inv_tanh_0_1(params["gamma"], device=device, dtype=dtype), requires_grad=True)

    def get_scaled_params(self, convert_cpu=False):
        beta  = (torch.tanh(self.logbeta) + 1.0) * 0.5
        gamma = (torch.tanh(self.loggamma) + 1.0) * 0.5
        out = {"beta": beta, "gamma": gamma}
        if convert_cpu:
            for k, v in out.items():
                out[k] = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu()
        return out

    def ODE(self, state, t=None):
        """
        state: [..., 3] last dim = (S,I,R)
        returns dstate with same shape
        """
        p = self.get_scaled_params()
        beta, gamma = p["beta"], p["gamma"]

        S = state[..., 0]
        I = state[..., 1]
        R = state[..., 2]

        N = self.N
        while N.dim() < S.dim():
            N = N.unsqueeze(0)

        new_inf_rate = beta * S * I / (N + 1e-12)   # flow S->I per unit time
        dS = -new_inf_rate
        dI = new_inf_rate - gamma * I
        dR = gamma * I
        return torch.stack([dS, dI, dR], dim=-1)

    def incidence(self, state):
        """
        Returns the incidence flow (new infections per unit time): beta*S*I/N
        state: [...,3]
        """
        p = self.get_scaled_params()
        beta = p["beta"]
        S = state[..., 0]
        I = state[..., 1]

        N = self.N
        while N.dim() < S.dim():
            N = N.unsqueeze(0)

        return beta * S * I / (N + 1e-12)

class SIRIncidenceRollout(nn.Module):
    """
    Produces a horizon-length incidence series from a latent SIR rollout.

    Inputs:
      feature: [B, W, M, F]
      Uses feature[:, -1, :, target_idx] as the latest observed incidence (cases/step).

    Output:
      cases_pred: [B, H, M]  (new cases per step)
    """
    def __init__(
        self,
        sir: SIRm_tanh,
        target_idx=0,
        dt=1.0,
        learn_r0_frac=False,
        r0_init=0.0,
        enforce_mass=True,
        obs="incidence",              # "incidence" or "ili_percent"
        outpatient_ratio=None,        # needed if obs == "ili_percent"
    ):
        super().__init__()
        self.sir = sir
        self.target_idx = int(target_idx)
        self.dt = float(dt)
        self.enforce_mass = bool(enforce_mass)
        self.obs = str(obs)
        self.outpatient_ratio = outpatient_ratio

        if learn_r0_frac:
            self.log_r0_frac = Parameter(_inv_tanh_0_1(r0_init, device=self.sir.N.device, dtype=self.sir.N.dtype))
        else:
            self.log_r0_frac = None

    def _infer_H(self, pred, target):
        x = pred if pred is not None else target
        if x is None or x.dim() != 3:
            raise ValueError("Need pred or target with shape [B,H,M] or [B,M,H] to infer H.")
        return x.shape[1] if x.shape[1] != x.shape[2] else x.shape[-1]

    def forward(self, feature, pred=None, target=None):
        if feature is None or feature.dim() != 4:
            raise ValueError("feature must be [B,W,M,F]")
        B, W, M, Fdim = feature.shape
        H = self._infer_H(pred, target)

        device = feature.device
        dtype = feature.dtype

        # --- Latest observed incidence (cases/step)
        C0 = feature[:, -1, :, self.target_idx].to(device=device, dtype=dtype)  # [B,M]
        C0 = torch.clamp(C0, min=0.0)

        # --- N as [B,M]
        N = self.sir.N.to(device=device, dtype=dtype)
        if N.numel() == 1:
            N_bm = N.view(1, 1).expand(B, M)
        elif N.dim() == 1 and N.numel() == M:
            N_bm = N.view(1, M).expand(B, M)
        else:
            N_bm = N.expand(B, M)

        # --- Infer initial I0 from incidence: C0 ≈ beta*S0*I0/N, with S0≈N => I0≈C0/beta
        beta = self.sir.get_scaled_params()["beta"].to(device=device, dtype=dtype)
        beta_safe = torch.clamp(beta, min=1e-4)

        I0 = torch.clamp(C0 / beta_safe, min=0.0, max=N_bm)

        # --- Optional R0 fraction
        if self.log_r0_frac is None:
            R0 = torch.zeros_like(I0)
        else:
            r0_frac = (torch.tanh(self.log_r0_frac) + 1.0) * 0.5  # scalar in (0,1)
            R0 = r0_frac * torch.clamp(N_bm - I0, min=0.0)

        S0 = torch.clamp(N_bm - I0 - R0, min=0.0)

        state = torch.stack([S0, I0, R0], dim=-1)  # [B,M,3]

        cases = []
        for _ in range(H):
            # incidence per unit time
            inc = self.sir.incidence(state)  # [B,M]
            # convert to "cases per step" (discrete) via dt
            c_step = self.dt * inc

            # map to observed if needed (EINN-style ILI% proxy)
            if self.obs == "ili_percent":
                if self.outpatient_ratio is None:
                    raise ValueError("outpatient_ratio must be provided for obs='ili_percent'")
                OR = float(self.outpatient_ratio)
                # EINN connects ILI% to incidence scaled by outpatient ratio (see paper).
                # Here we follow their idea: ILI% ≈ (beta*S*I/N) / (N*OR)
                c_step = inc / (N_bm * OR + 1e-12)  # fraction (not counts)
            elif self.obs != "incidence":
                raise ValueError(f"Unknown obs type: {self.obs}")

            cases.append(c_step)

            # Euler update
            dstate = self.sir.ODE(state)
            state = torch.clamp(state + self.dt * dstate, min=0.0)

            if self.enforce_mass:
                total = state.sum(dim=-1, keepdim=True)  # [B,M,1]
                state = state * (N_bm.unsqueeze(-1) / (total + 1e-12))

        return torch.stack(cases, dim=1)  # [B,H,M]

def compute_epi_ngm_forecast(
    adj_prob,          # [B,H,M,M]
    beta,              # [B,H,M]
    gamma,             # [B,H,M]
    x_last,            # [B,M]
    adj_static=None,   # [M,M] or [B,M,M]
    clamp_max=1.0,
    eps=1e-6
):
    assert adj_prob.dim() == 4, "adj_prob must be [B,H,M,M]"
    assert beta.dim() == 3 and gamma.dim() == 3, "beta/gamma must be [B,H,M]"
    assert x_last.dim() == 2, "x_last must be [B,M]"

    B, H, M, _ = adj_prob.shape
    device = adj_prob.device
    dtype = adj_prob.dtype

    # ---- optional mask
    if adj_static is not None:
        if adj_static.dim() == 2:
            mask = (adj_static > 0).to(dtype=dtype, device=device).view(1, 1, M, M)
        elif adj_static.dim() == 3:
            mask = (adj_static > 0).to(dtype=dtype, device=device).view(B, 1, M, M)
        else:
            raise ValueError("adj_static must be [M,M] or [B,M,M]")
        adj_epi = adj_prob * mask
    else:
        adj_epi = adj_prob

    diag_vals = torch.diagonal(adj_epi, dim1=-2, dim2=-1)  # [B,H,M]
    D = torch.diag_embed(diag_vals)                        # [B,H,M,M]

    col_sum = adj_epi.sum(dim=-2)                          # [B,H,M]
    W = torch.diag_embed(col_sum) - D                      # [B,H,M,M]

    A = (adj_epi.transpose(-1, -2) - D) - W                # [B,H,M,M]

    BetaDiag = torch.diag_embed(beta)                      # [B,H,M,M]
    GammaDiag = torch.diag_embed(gamma)                    # [B,H,M,M]

    tmp = (GammaDiag - A).clamp(max=clamp_max)
    I = torch.eye(M, device=device, dtype=dtype).view(1, 1, M, M)
    tmp = tmp + eps * I

    tmp_inv = torch.linalg.inv(tmp)                        # [B,H,M,M]
    ngm = BetaDiag @ tmp_inv                               # [B,H,M,M]

    x = x_last.to(device=device, dtype=dtype).view(B, 1, 1, M).expand(B, H, 1, M)
    y_epi = torch.matmul(x, ngm.transpose(-1, -2)).squeeze(-2)  # [B,H,M]
    return y_epi

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64, 64), out_dim=1, act=nn.Tanh, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def build_adj_prob(dynamic_graph, graph, B, H, M, device, dtype):
    """
    Returns adj_prob [B,H,M,M] (softmax along last dim).
    Prefers dynamic_graph if provided; otherwise uses static graph.
    """
    if dynamic_graph is not None:
        dg = dynamic_graph.to(device=device, dtype=dtype)
        if dg.dim() == 4:
            adj_prob = F.softmax(dg, dim=-1)
            if adj_prob.shape[1] == 1 and H > 1:
                adj_prob = adj_prob.expand(B, H, M, M)
            elif adj_prob.shape[1] != H:
                if adj_prob.shape[1] > H:
                    adj_prob = adj_prob[:, :H]
                else:
                    last = adj_prob[:, -1:].expand(B, H - adj_prob.shape[1], M, M)
                    adj_prob = torch.cat([adj_prob, last], dim=1)
            return adj_prob
        if dg.dim() == 3:
            return F.softmax(dg, dim=-1).view(B, 1, M, M).expand(B, H, M, M)

    if graph is None:
        raise ValueError("Need dynamic_graph or graph to build adj_prob.")

    g = graph.to(device=device, dtype=dtype)
    if g.dim() == 2:
        return F.softmax(g, dim=-1).view(1, 1, M, M).expand(B, H, M, M)
    if g.dim() == 3:
        return F.softmax(g, dim=-1).view(B, 1, M, M).expand(B, H, M, M)

    raise ValueError("graph must be [M,M] or [B,M,M]")

class EinnModule(nn.Module):
    """
    Supports epi_mode in {"sir_incidence", "sir_percent", "ngm"}.

    Targets:
      - raw daily cases: use epi_mode="sir_incidence"
      - per-capita (percent/per-100k): use epi_mode="sir_percent" and set percent_scale accordingly
      - ILI proxy: use epi_mode="sir_percent" with outpatient_ratio (optional) and percent_scale to match target
    """
    def __init__(
        self,
        num_nodes: int,
        horizon: int,
        in_features: int,
        epi_mode: str = "sir_incidence",     # "sir_incidence" | "sir_percent" | "ngm"
        dt: float = 1.0,
        target_idx: int = 0,
        population=None,
        regions=None,
        percent_scale: float = 100.0,        # 100 for %, 1e5 for per-100k, 1 for fraction
        outpatient_ratio: float = None,      # for ILI-like scaling if you want incidence/(N*OR)
        use_context: bool = True,
        context_hidden: int = 32,
        node_emb_dim: int = 8,
        state_hidden=(64, 64),
        param_hidden=(64, 64),
        ode_weight: float = 1.0,
        data_weight: float = 1.0,
        constraint_weight: float = 0.0,      # optional small penalty
        eps: float = 1e-6,
    ):
        super().__init__()
        self.M = int(num_nodes)
        self.H = int(horizon)
        self.F = int(in_features)
        self.epi_mode = str(epi_mode)
        self.dt = float(dt)
        self.target_idx = int(target_idx)
        self.population_spec = population
        self.regions_spec = regions
        self.percent_scale = float(percent_scale)
        self.outpatient_ratio = outpatient_ratio
        self.use_context = bool(use_context)
        self.ode_weight = float(ode_weight)
        self.data_weight = float(data_weight)
        self.constraint_weight = float(constraint_weight)
        self.eps = float(eps)

        self.node_emb = nn.Embedding(self.M, node_emb_dim)

        if self.use_context:
            self.ctx_gru = nn.GRU(self.F, context_hidden, batch_first=True)
            ctx_dim = context_hidden
        else:
            ctx_dim = 0

        # state net: outputs logits for (S,I,R) fractions -> softmax -> *N
        state_in = 1 + node_emb_dim + ctx_dim
        self.state_net = MLP(state_in, hidden=state_hidden, out_dim=3, act=nn.Tanh)

        # param net: outputs raw -> sigmoid -> (beta,gamma) in (0,1)
        param_in = 1 + node_emb_dim + ctx_dim
        self.param_net = MLP(param_in, hidden=param_hidden, out_dim=2, act=nn.Tanh)

    def _expand_N(self, B, device, dtype):
        N_m = resolve_population_2019(
            M=self.M,
            population=self.population_spec,
            regions=self.regions_spec,
            device=device,
            dtype=dtype,
        )  # [M]
        return N_m.view(1, self.M).expand(B, self.M)  # [B,M]

    def _encode_context(self, x):
        # x: [B,W,M,F]
        if not self.use_context:
            return None
        B, W, M, Fdim = x.shape
        x_flat = x.transpose(2, 1).contiguous().flatten(0, 1)  # [B*M, W, F]
        out, _ = self.ctx_gru(x_flat)
        ctx = out[:, -1, :]                                    # [B*M, C]
        return ctx.view(B, M, -1)                              # [B,M,C]

    def _time_grid(self, B, device, dtype):
        # discrete collocation points aligned with horizon
        t = (torch.arange(self.H, device=device, dtype=dtype) * self.dt)
        t = t.view(1, self.H, 1, 1).expand(B, self.H, self.M, 1).clone()
        t.requires_grad_(True)
        return t  # [B,H,M,1], requires_grad

    def _node_features(self, B, device, dtype):
        node_ids = torch.arange(self.M, device=device, dtype=torch.long)
        emb = self.node_emb(node_ids).to(dtype=dtype)     # [M,E]
        emb = emb.view(1, 1, self.M, -1).expand(B, self.H, self.M, -1)  # [B,H,M,E]
        return emb

    def _pack_inputs(self, t, node_emb, ctx):
        # t: [B,H,M,1], node_emb: [B,H,M,E], ctx: [B,M,C] -> broadcast to [B,H,M,C]
        if ctx is None:
            return torch.cat([t, node_emb], dim=-1)
        ctx_h = ctx.view(ctx.shape[0], 1, ctx.shape[1], ctx.shape[2]).expand(-1, self.H, -1, -1)
        return torch.cat([t, node_emb, ctx_h], dim=-1)

    def _states_and_params(self, x):
        """
        Returns:
          S,I,R: [B,H,M]
          beta,gamma: [B,H,M]
          t: [B,H,M,1] (requires_grad)
          N_bm: [B,M]
        """
        B, W, M, Fdim = x.shape
        device, dtype = x.device, x.dtype
        N_bm = self._expand_N(B, device, dtype)

        ctx = self._encode_context(x)  # [B,M,C] or None
        t = self._time_grid(B, device, dtype)
        emb = self._node_features(B, device, dtype)
        inp = self._pack_inputs(t, emb, ctx)              # [B,H,M,D]

        flat = inp.view(B * self.H * self.M, -1)
        state_logits = self.state_net(flat).view(B, self.H, self.M, 3)
        state_frac = F.softmax(state_logits, dim=-1)      # fractions sum to 1

        N_hm = N_bm.view(B, 1, self.M, 1).expand(B, self.H, self.M, 1)
        SIR = state_frac * N_hm                           # counts
        S = SIR[..., 0]
        I = SIR[..., 1]
        R = SIR[..., 2]

        param_raw = self.param_net(flat).view(B, self.H, self.M, 2)
        params = torch.sigmoid(param_raw)                 # (0,1)
        beta = params[..., 0]
        gamma = params[..., 1]

        return S, I, R, beta, gamma, t, N_bm

    def _mix_I(self, adj_prob, I):
        # adj_prob: [B,H,M,M], I: [B,H,M] -> I_mix: [B,H,M]
        B, H, M, _ = adj_prob.shape
        Ih = I.view(B * H, M, 1)
        Ah = adj_prob.view(B * H, M, M)
        I_mix = torch.bmm(Ah, Ih).view(B, H, M)
        return I_mix

    def _sir_rhs(self, S, I, R, beta, gamma, I_mix, N_bm):
        # All inputs [B,H,M] except N_bm [B,M]
        N = N_bm.unsqueeze(1)  # [B,1,M]
        inf = beta * S * I_mix / (N + self.eps)          # per unit time
        dS = -inf
        dI = inf - gamma * I
        dR = gamma * I
        return dS, dI, dR, inf

    def _d_dt(self, y, t):
        # y: [B,H,M], t: [B,H,M,1] -> dy/dt: [B,H,M]
        grad = torch.autograd.grad(
            outputs=y,
            inputs=t,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grad.squeeze(-1)

    def _obs_from_incidence(self, inc, N_bm):
        # inc is per unit time infection flow (not multiplied by dt), shape [B,H,M]
        cases = self.dt * inc
        if self.epi_mode == "sir_incidence":
            y = cases
        elif self.epi_mode == "sir_percent":
            denom = N_bm.unsqueeze(1) + self.eps
            y = (cases / denom) * self.percent_scale
            if self.outpatient_ratio is not None:
                y = y / (float(self.outpatient_ratio) + self.eps)
        else:
            raise ValueError(f"_obs_from_incidence called with epi_mode={self.epi_mode}")
        return y

    def forward(self, x, graph=None, dynamic_graph=None):
        """
        Returns:
          y_einn: [B,H,M]
          losses: dict {"ode":..., "data":..., "constraint":..., "total":...}
        """
        B, W, M, Fdim = x.shape
        device, dtype = x.device, x.dtype
        assert M == self.M, "num_nodes mismatch"
        assert self.H > 0, "invalid horizon"

        if self.epi_mode == "ngm":
            # NGM mode: no ODE residual; use operator propagation + data loss only.
            ctx = self._encode_context(x)
            t = self._time_grid(B, device, dtype)
            emb = self._node_features(B, device, dtype)
            inp = self._pack_inputs(t, emb, ctx)
            flat = inp.view(B * self.H * self.M, -1)
            params = torch.sigmoid(self.param_net(flat)).view(B, self.H, self.M, 2)
            beta = params[..., 0]
            gamma = params[..., 1]

            adj_prob = build_adj_prob(dynamic_graph, graph, B, self.H, self.M, device, dtype)
            x_last = x[:, -1, :, self.target_idx]
            y_einn = compute_epi_ngm_forecast(adj_prob=adj_prob, beta=beta, gamma=gamma, x_last=x_last, adj_static=graph)

            losses = {
                "ode": torch.zeros((), device=device, dtype=dtype),
                "data": torch.zeros((), device=device, dtype=dtype),
                "constraint": torch.zeros((), device=device, dtype=dtype),
                "total": torch.zeros((), device=device, dtype=dtype),
            }
            return y_einn, losses

        # SIR modes
        S, I, R, beta, gamma, t, N_bm = self._states_and_params(x)
        adj_prob = build_adj_prob(dynamic_graph, graph, B, self.H, self.M, device, dtype)
        I_mix = self._mix_I(adj_prob, I)

        dS_rhs, dI_rhs, dR_rhs, inf = self._sir_rhs(S, I, R, beta, gamma, I_mix, N_bm)

        dS_dt = self._d_dt(S, t)
        dI_dt = self._d_dt(I, t)
        dR_dt = self._d_dt(R, t)

        # ODE residual loss
        ode_res = (dS_dt - dS_rhs).pow(2) + (dI_dt - dI_rhs).pow(2) + (dR_dt - dR_rhs).pow(2)
        L_ode = ode_res.mean()

        # Observation/data output (incidence -> target space)
        y_einn = self._obs_from_incidence(inf, N_bm)

        losses = {
            "ode": L_ode,
            "data": torch.zeros((), device=device, dtype=dtype),
            "constraint": torch.zeros((), device=device, dtype=dtype),
            "total": torch.zeros((), device=device, dtype=dtype),
        }
        return y_einn, losses

    def losses(self, x, y, graph=None, dynamic_graph=None):
        """
        Returns:
          L_ode, L_data, y_einn
        """
        y_einn, losses = self.forward(x, graph=graph, dynamic_graph=dynamic_graph)

        # data loss computed here (needs y)
        if y_einn.shape != y.shape and y_einn.dim() == 3 and y_einn.transpose(1, 2).shape == y.shape:
            y_einn_aligned = y_einn.transpose(1, 2)
        else:
            y_einn_aligned = y_einn

        L_data = F.mse_loss(y_einn_aligned, y)

        L_ode = losses["ode"]
        total = self.ode_weight * L_ode + self.data_weight * L_data

        return L_ode, L_data, y_einn_aligned

class _FutureTI(nn.Module):
    def __init__(self, tid_sizes, emb_dim=4, hidden=(16,), node_specific=True, num_nodes=None):
        super().__init__()
        self.keys = list(tid_sizes.keys()) if tid_sizes else []
        self.node_specific = node_specific
        self.num_nodes = num_nodes
        self.embs = nn.ModuleDict({k: nn.Linear(K, emb_dim, bias=False) for k, K in tid_sizes.items()})
        self.total_dim = emb_dim * len(self.keys)

        def mlp(din, dout):
            layers, d = [], din
            for h in hidden: layers += [nn.Linear(d, h), nn.ReLU()]; d = h
            layers.append(nn.Linear(d, dout)); return nn.Sequential(*layers)

        self.head = mlp(self.total_dim, num_nodes if node_specific else 1)
        if node_specific: assert num_nodes is not None, "num_nodes is required when node_specific=True"

    def forward(self, states_future):
        if states_future is None or states_future.numel() == 0 or not self.keys:
            return None
        B, H, C = states_future.shape
        outs = []
        for ch, k in enumerate(self.keys):
            K = self.embs[k].in_features
            idx = states_future[..., ch].long()
            oh = F.one_hot(idx.clamp_min(0).clamp_max(K-1), K).float()
            outs.append(self.embs[k](oh))          # [B,H,emb]
        z = torch.cat(outs, dim=-1)                # [B,H,D]
        return self.head(z)                        # [B,H,N] or [B,H,1]

class _EpiHybridHead(nn.Module):
    """
    Outputs:
      - sir_incidence: [B,H,M] new cases per step (dt * beta * S * (Adj@I) / N)
      - sir_percent:   [B,H,M] percent of pop per step (sir_incidence / N * percent_scale)
      - ngm:           [B,H,M] NGM forecast

    Assumption: target series is NEW CASES (incidence) per step.
    """
    def __init__(
        self,
        in_features,
        horizon,
        population=None,        # optional
        regions=None,           # optional ("CA" or list[str] len M)
        hidden = 32,
        mlp_hidden = 8,
        target_idx = 0,
        dt = 1.0,
        percent_scale = 100.0,  # 100.0 => percent; 1.0 => fraction
        clamp_max = 1.0,
        eps = 1e-6,
        enforce_mass = True,
        learn_r0_frac = False,
        r0_init = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.horizon = horizon
        self.hidden = hidden
        self.mlp_hidden = mlp_hidden
        self.target_idx = int(target_idx)
        self.dt = float(dt)
        self.percent_scale = float(percent_scale)
        self.clamp_max = float(clamp_max)
        self.eps = float(eps)
        self.enforce_mass = bool(enforce_mass)

        self.population_spec = population
        self.regions_spec = regions

        self.gru_beta = nn.GRU(in_features, hidden, batch_first=True)
        self.gru_gamma = nn.GRU(in_features, hidden, batch_first=True)

        self.pred_beta = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, horizon),
            nn.Sigmoid(),
        )
        self.pred_gamma = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, horizon),
            nn.Sigmoid(),
        )

        if learn_r0_frac:
            # scalar -> sigmoid in (0,1)
            self.log_r0_frac = nn.Parameter(torch.tensor(float(r0_init)))
        else:
            self.log_r0_frac = None

    def _mask_adj(self, adj_prob, adj_static):
        if adj_static is None:
            return adj_prob
        B, H, M, _ = adj_prob.shape
        device, dtype = adj_prob.device, adj_prob.dtype
        if adj_static.dim() == 2:
            mask = (adj_static > 0).to(dtype=dtype, device=device).view(1, 1, M, M)
        elif adj_static.dim() == 3:
            mask = (adj_static > 0).to(dtype=dtype, device=device).view(B, 1, M, M)
        else:
            raise ValueError("adj_static must be [M,M] or [B,M,M]")
        return adj_prob * mask

    def _predict_beta_gamma(self, x):
        B, W, M, Fdim = x.shape
        x_flat = x.transpose(2, 1).contiguous().flatten(0, 1)  # [B*M, W, F]

        out_b, _ = self.gru_beta(x_flat)
        out_g, _ = self.gru_gamma(x_flat)

        last_b = out_b[:, -1, :]
        last_g = out_g[:, -1, :]

        beta = self.pred_beta(last_b).view(B, M, self.horizon).transpose(1, 2)   # [B,H,M]
        gamma = self.pred_gamma(last_g).view(B, M, self.horizon).transpose(1, 2) # [B,H,M]
        return beta, gamma

    def _expand_N(self, B, M, device, dtype):
        N_m = resolve_population_2019(
            M=M,
            population=self.population_spec,
            regions=self.regions_spec,
            device=device,
            dtype=dtype,
        )  # [M]
        return N_m.view(1, M).expand(B, M)  # [B,M]

    def _sir_incidence_rollout(self, x, adj_prob, beta, gamma):
        """
        Network-mixed latent SIR rollout producing NEW CASES per step.

        Definitions:
        I_eff(h) = Adj[h] @ I(h)   (network infectious pressure)
        new_inf_rate(h) = beta[h] * S(h) * I_eff(h) / N
        new_cases(h) = dt * new_inf_rate(h)

        Args:
        x:        [B,W,M,F]   input window, where x[:, -1, :, target_idx] is last observed NEW CASES
        adj_prob: [B,H,M,M]   (soft) adjacency weights per horizon step
        beta:     [B,H,M]     per-node transmission factor in (0,1)
        gamma:    [B,H,M]     per-node recovery factor in (0,1)

        Returns:
        cases: [B,H,M] new cases per step
        N_bm:  [B,M]   population per node
        """
        B, W, M, _ = x.shape
        device, dtype = x.device, x.dtype

        # population per node (broadcasted to [B,M])
        N_bm = self._expand_N(B, M, device, dtype)  # [B,M]

        # last observed incidence (new cases per step)
        C0 = x[:, -1, :, self.target_idx].to(device=device, dtype=dtype)
        C0 = torch.clamp(C0, min=0.0)

        # --- initialize latent compartments from incidence
        # C0 ≈ dt * beta0 * S0 * I0 / N, assume S0≈N -> I0 ≈ C0 / (dt*beta0)
        beta0 = torch.clamp(beta[:, 0, :], min=1e-4)  # [B,M], avoid divide-by-zero
        I0 = C0 / (self.dt * beta0 + 1e-12)          # [B,M]
        I0 = I0.clamp(min=0.0)
        I0 = torch.minimum(I0, N_bm)                 # tensor-safe upper bound

        # optional R0 as fraction of remaining mass
        if self.log_r0_frac is None:
            R0 = torch.zeros_like(I0)
        else:
            r0 = torch.sigmoid(self.log_r0_frac)     # scalar in (0,1)
            R0 = r0 * torch.clamp(N_bm - I0, min=0.0)

        S0 = torch.clamp(N_bm - I0 - R0, min=0.0)

        # current state
        S, I, R = S0, I0, R0

        cases = []
        for h in range(self.horizon):
            # --- network mixing: I_eff = Adj @ I
            adj_h = adj_prob[:, h, :, :]  # [B,M,M]
            I_eff = torch.bmm(adj_h, I.unsqueeze(-1)).squeeze(-1)  # [B,M]
            I_eff = I_eff.clamp(min=0.0)

            # --- infection flow
            beta_h = beta[:, h, :].clamp(min=0.0)
            gamma_h = gamma[:, h, :].clamp(min=0.0)

            new_inf_rate = beta_h * S * I_eff / (N_bm + 1e-12)     # per unit time
            new_inf_rate = new_inf_rate.clamp(min=0.0)

            new_cases = self.dt * new_inf_rate                     # per step (e.g., per day)
            cases.append(new_cases)

            # --- Euler update
            dS = -new_inf_rate
            dI = new_inf_rate - gamma_h * I
            dR = gamma_h * I

            S = (S + self.dt * dS).clamp(min=0.0)
            I = (I + self.dt * dI).clamp(min=0.0)
            R = (R + self.dt * dR).clamp(min=0.0)

            # enforce conservation S+I+R = N (helps prevent drift)
            if self.enforce_mass:
                total = S + I + R
                scale = N_bm / (total + 1e-12)
                S = S * scale
                I = I * scale
                R = R * scale

        cases = torch.stack(cases, dim=1)  # [B,H,M]
        return cases, N_bm

    def forward(self, x, adj_prob, adj_static=None, mode="sir_incidence"):
        """
        mode:
          - "sir_incidence": [B,H,M] new cases per step
          - "sir_percent":   [B,H,M] percent of population per step
          - "ngm":           [B,H,M]
        """
        assert x.dim() == 4, "x must be [B,W,M,F]"
        assert adj_prob.dim() == 4, "adj_prob must be [B,H,M,M]"
        assert adj_prob.shape[1] == self.horizon, "adj_prob horizon mismatch"

        adj_prob_masked = self._mask_adj(adj_prob, adj_static)
        beta, gamma = self._predict_beta_gamma(x)

        if mode in ("sir_incidence", "sir_percent"):
            sir_inc, N_bm = self._sir_incidence_rollout(x, adj_prob_masked, beta, gamma)  # [B,H,M], [B,M]
            sir_pct = (sir_inc / (N_bm.unsqueeze(1) + 1e-12)) * self.percent_scale        # [B,H,M]

        if mode == "sir_incidence":
            return sir_inc
        if mode == "sir_percent":
            return sir_pct
        if mode == "ngm":
            x_last = x[:, -1, :, self.target_idx]  # last observed series (assumed incidence)
            return compute_epi_ngm_forecast(
                adj_prob=adj_prob_masked,
                beta=beta,
                gamma=gamma,
                x_last=x_last,
                adj_static=adj_static,
                clamp_max=self.clamp_max,
                eps=self.eps,
            )

        raise ValueError(f"Unknown mode: {mode}")

class _EpiRegLoss(nn.Module):
    def __init__(self, scale=0.5, loss='mse'):
        super().__init__()
        self.scale = float(scale)
        self.loss = str(loss)

    def forward(self, epi_out, target):
        if self.loss == 'mse':
            reg = F.mse_loss(epi_out, target)
        elif self.loss == 'l1':
            reg = F.l1_loss(epi_out, target)
        elif self.loss == 'smooth_l1':
            reg = F.smooth_l1_loss(epi_out, target)
        else:
            raise ValueError(f"Unknown epi_reg_loss: {self.loss}")
        return self.scale * reg

class BaseModel(nn.Module):
    def __init__(self, device = 'cpu', 
                 use_future_ti=False, tid_sizes=None, emb_dim=4, ti_hidden=(16,), node_specific=True, num_nodes=None):
        super(BaseModel, self).__init__()
        self.device = device
        self.future_ti = _FutureTI(tid_sizes, emb_dim, ti_hidden, node_specific, num_nodes).to(device) \
                         if (use_future_ti and tid_sizes) else None

    @staticmethod
    def _conformal_qhat_abs(abs_resid, alpha):
        """
        abs_resid: (N_calib, ...) nonnegative absolute residuals
        Returns qhat(...) with the split-conformal finite-sample correction:
            k = ceil((N+1)*(1-alpha)) - 1  (0-indexed)
            qhat = k-th order statistic along calibration dimension.
        This corresponds to the 'higher' quantile needed for coverage.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        N = abs_resid.shape[0]
        # k in {0,...,N-1}
        k = int(math.ceil((N + 1) * (1.0 - alpha))) - 1
        k = max(0, min(N - 1, k))
        # sort along calibration dimension
        sorted_vals, _ = torch.sort(abs_resid, dim=0)
        return sorted_vals[k]

    @staticmethod
    def _filter_samples_by_iqr(sample_scores, iqr_mult=1.5):
        """
        sample_scores: (N,) robust score per calibration sample (bigger = more outlier-ish)
        returns boolean mask of shape (N,) for samples within IQR fence.
        """
        q1 = torch.quantile(sample_scores, 0.25)
        q3 = torch.quantile(sample_scores, 0.75)
        iqr = q3 - q1
        if iqr == 0:
            lower, upper = q1, q3
        else:
            lower = q1 - iqr_mult * iqr
            upper = q3 + iqr_mult * iqr
        return (sample_scores >= lower) & (sample_scores <= upper)

    def _fit_conformal(
        self,
        feature,
        target,
        states=None,
        graph=None,
        dynamic_graph=None,
        exclude_zeros=True,
        iqr_mult=1.5,
    ):
        """
        Calibrate conformal residuals on a held-out calibration set.

        Stores:
            self._calib_resid          (N, ...)  residuals: (y - pred)
            self._calib_abs_resid      (N, ...)  abs residuals: |y - pred|
            self._calib_resid_filtered
            self._calib_abs_resid_filtered

        Filtering is done on a robust *residual-based* sample score (not target magnitude):
            score_i = mean_j |r_{ij}| over valid elements j
        """
        self.eval()
        with torch.no_grad():
            pred = self.predict(
                feature=feature,
                graph=graph,
                states=states,
                dynamic_graph=dynamic_graph,
            ).to(self.device)
            y = target.to(self.device)
            pred = pred.reshape_as(y)
            # residuals: y - pred (note sign; abs-residual is sign-invariant)
            resid = (y - pred)
            # valid elements mask
            mask_elem = torch.isfinite(y)
            if exclude_zeros:
                mask_elem &= (y != 0)
            # If nothing valid, just use everything
            if mask_elem.sum() == 0:
                resid_valid = resid
                mask_elem = torch.ones_like(y, dtype=torch.bool, device=y.device)
            else:
                # Keep resid but ignore invalid elements for scoring/filtering
                resid_valid = resid.clone()
                resid_valid[~mask_elem] = 0.0
            abs_resid = resid_valid.abs()
            # store unfiltered (still respecting invalid element handling)
            self._calib_resid = resid_valid.detach()
            self._calib_abs_resid = abs_resid.detach()
            # ---- filtered version: remove calibration samples with unusually large residual score
            # score per sample = mean abs residual over valid elements
            # shape: (N,)
            N = resid_valid.shape[0]
            flat_abs = abs_resid.view(N, -1)
            flat_mask = mask_elem.view(N, -1)
            denom = flat_mask.sum(dim=1).clamp(min=1)
            sample_score = (flat_abs.sum(dim=1) / denom)  # mean |resid| per sample
            keep = self._filter_samples_by_iqr(sample_score, iqr_mult=iqr_mult)
            if keep.sum() == 0:
                # fallback: no filtering
                self._calib_resid_filtered = self._calib_resid
                self._calib_abs_resid_filtered = self._calib_abs_resid
            else:
                self._calib_resid_filtered = resid_valid[keep].detach()
                self._calib_abs_resid_filtered = abs_resid[keep].detach()

    def predict_conformal_intervals(
        self,
        feature,
        alphas,
        graph=None,
        states=None,
        dynamic_graph=None,
        filtered=False,
    ):
        """
        Returns dict alpha -> (lower, upper) tensors matching predict() shape.
        Uses split conformal: [pred - qhat_alpha, pred + qhat_alpha].
        Requires: _fit_conformal called beforehand.
        """
        base = self.predict(feature, graph=graph, states=states, dynamic_graph=dynamic_graph).to(self.device)
        abs_resid = self._calib_abs_resid_filtered if filtered else self._calib_abs_resid
        if abs_resid is None:
            raise RuntimeError("Conformal not fitted. Call _fit_conformal(...) on a calibration set first.")
        out = {}
        for alpha in alphas:
            qhat = self._conformal_qhat_abs(abs_resid.to(self.device), float(alpha))
            # broadcast qhat to base.shape if needed (qhat should already be (...))
            while qhat.dim() < base.dim():
                qhat = qhat.unsqueeze(0)
            lower = base - qhat
            upper = base + qhat
            out[float(alpha)] = (lower.detach().cpu(), upper.detach().cpu())
        return out

    def predict_quantiles_conformal_for_wis(
        self,
        feature,
        alphas,
        graph=None,
        states=None,
        dynamic_graph=None,
        filtered=False,
    ):
        """
        Returns q tensor shaped (1 + 2K, ...) in the same layout your wis_from_quantiles expects:
            q[0] = median (here: base point forecast)
            q[1+2k] = lower for alpha_k
            q[2+2k] = upper for alpha_k
        Note: This produces *central* conformal intervals around the point forecast.
        """
        base = self.predict(feature, graph=graph, states=states, dynamic_graph=dynamic_graph).to(self.device)
        abs_resid = self._calib_abs_resid_filtered if filtered else self._calib_abs_resid
        if abs_resid is None:
            raise RuntimeError("Conformal not fitted. Call _fit_conformal(...) on a calibration set first.")
        alphas = [float(a) for a in alphas]
        q_list = [base]  # treat point forecast as "median" for WIS layout
        for alpha in alphas:
            qhat = self._conformal_qhat_abs(abs_resid.to(self.device), alpha)
            while qhat.dim() < base.dim():
                qhat = qhat.unsqueeze(0)
            q_list.append(base - qhat)  # "lower"
            q_list.append(base + qhat)  # "upper"
        q = torch.stack(q_list, dim=0)
        return q.detach().cpu()

    def predict_samples(self, feature, graph=None, states=None, dynamic_graph=None, n_samples=100, filtered=False):
        """
        Residual-bootstrap samples (for CRPS-like ensemble metrics, no strict coverage guarantee):
            sample = base + r*, where r* is a residual drawn from the calibration residuals.
        For distribution-free coverage, use predict_conformal_intervals / predict_quantiles_conformal_for_wis instead.
        """
        base = self.predict(feature, graph=graph, states=states, dynamic_graph=dynamic_graph).to(self.device)
        resid_bank = self._calib_resid_filtered if filtered else self._calib_resid
        if resid_bank is None:
            # fallback: identical copies
            return base.unsqueeze(0).detach().cpu()
        resid_bank = resid_bank.to(self.device)
        N = resid_bank.shape[0]
        # Assume base is (B, ...) with B=batch size. If no batch dim, treat B=1.
        if base.dim() == resid_bank.dim() - 1:
            # base is missing leading sample dimension, add batch dim = 1
            base_b = base.unsqueeze(0)
        else:
            base_b = base
        B = base_b.shape[0]
        # draw residual indices for each (sample, batch)
        idx = torch.randint(0, N, size=(n_samples, B), device=self.device)
        sampled_resid = resid_bank[idx]  # (S, B, ...)
        samples = base_b.unsqueeze(0) + sampled_resid  # (S, B, ...)
        # if we added a fake batch dim, remove it
        if base.dim() == resid_bank.dim() - 1:
            samples = samples[:, 0, ...]  # (S, ...)
        return samples.detach().cpu()

    def predict_quantiles(
        self,
        feature,
        quantiles,
        graph=None,
        states=None,
        dynamic_graph=None,
        n_samples=100,
        filtered=False,
    ):
        """
        Quantiles from residual-bootstrap samples.
        Produces quantiles of a sampled ensemble, not conformal intervals.
        It gives you an approximate predictive distribution (empirical), but no finite-sample coverage guarantee.
        If what you need is WIS-style central intervals with guarantees, prefer:
            predict_quantiles_conformal_for_wis(feature, alphas, ...)
        """
        samples = self.predict_samples(
            feature, graph=graph, states=states, dynamic_graph=dynamic_graph,
            n_samples=n_samples, filtered=filtered
        ).to(self.device)
        q_tensor = torch.tensor(quantiles, device=self.device, dtype=samples.dtype)
        q = torch.quantile(samples, q=q_tensor, dim=0)
        return q.detach().cpu()

    def compute_crps_wis(
        self,
        feature,
        target,
        quantile_levels,
        alphas,
        graph=None,
        states=None,
        dynamic_graph=None,
        n_samples=100,
    ):
        """
        Compute CRPS and WIS for both unfiltered and filtered residual-bootstrap predictions.

        Notes:
            - Assumes _fit_conformal(...) has been called to populate residual banks.
            - If filtered residuals are unavailable, filtered metrics fall back to unfiltered.

        Returns:
            dict with keys: crps, crps_filtered, wis, wis_filtered (torch scalars).
        """
        target = target.to(self.device)

        samples = self.predict_samples(
            feature,
            graph=graph,
            states=states,
            dynamic_graph=dynamic_graph,
            n_samples=n_samples,
            filtered=False,
        )
        samples = samples.reshape(samples.shape[0], *target.shape)
        crps = crps_ensemble(samples, target)

        samples_filtered = self.predict_samples(
            feature,
            graph=graph,
            states=states,
            dynamic_graph=dynamic_graph,
            n_samples=n_samples,
            filtered=True,
        )
        samples_filtered = samples_filtered.reshape(samples_filtered.shape[0], *target.shape)
        crps_filtered = crps_ensemble(samples_filtered, target)

        q = self.predict_quantiles(
            feature,
            quantile_levels,
            graph=graph,
            states=states,
            dynamic_graph=dynamic_graph,
            n_samples=n_samples,
            filtered=False,
        )
        q = q.reshape(q.shape[0], *target.shape)
        wis = wis_from_quantiles(q, target, alphas=alphas)

        q_filtered = self.predict_quantiles(
            feature,
            quantile_levels,
            graph=graph,
            states=states,
            dynamic_graph=dynamic_graph,
            n_samples=n_samples,
            filtered=True,
        )
        q_filtered = q_filtered.reshape(q_filtered.shape[0], *target.shape)
        wis_filtered = wis_from_quantiles(q_filtered, target, alphas=alphas)

        return {
            "crps": crps,
            "crps_filtered": crps_filtered,
            "wis": wis,
            "wis_filtered": wis_filtered,
        }

    def fit(self, 
            train_input, 
            train_target, 
            train_states=None, 
            train_graph=None, 
            train_dynamic_graph=None,
            val_input=None, 
            val_target=None,
            val_states=None, 
            val_graph= None, 
            val_dynamic_graph=None,
            loss='mse', 
            use_epi_reg=False,
            epi_reg_loss='mse',
            epi_hidden=8,
            epi_mode="sir_incidence",   # "sir_incidence" | "sir_percent" | "ngm"
            epi_percent_scale=100.0,    # percent output scaling for sir_percent
            epi_population=None,        # optional scalar / [M] list/tensor
            epi_regions=None,           # optional "CA" or list[str] length M
            epochs=1000, 
            batch_size=10,
            lr=1e-3, 
            initialize=True, 
            patience=100, 
            **kwargs):
        if initialize:
            self.initialize()
        self._setup_epi_reg_from_data(
            train_input, train_target,
            use_epi_reg=use_epi_reg,
            epi_reg_loss=epi_reg_loss,
            epi_hidden=epi_hidden,
            epi_mode=epi_mode,
            epi_percent_scale=epi_percent_scale,
            epi_regions=epi_regions,
            epi_population=epi_population,
            target_idx=0,
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        loss_fn = get_loss(loss)

        training_losses = []
        validation_losses = []
        early_stopping = patience
        best_val = float('inf')
        best_weights = deepcopy(self.state_dict())
        for epoch in tqdm(range(epochs)):
            # train one epoch
            # import ipdb; ipdb.set_trace()
            loss = self.train_epoch(optimizer=optimizer, 
                                    loss_fn=loss_fn, 
                                    feature=train_input, 
                                    states=train_states, 
                                    graph=train_graph, 
                                    dynamic_graph=train_dynamic_graph, 
                                    target=train_target, 
                                    batch_size=batch_size, 
                                    device=self.device)
            training_losses.append(loss)
            # validate
            if val_input is not None and val_input.numel():
                val_loss, output = self.evaluate(loss_fn=loss_fn, 
                                                feature=val_input, 
                                                graph=val_graph, 
                                                dynamic_graph=val_dynamic_graph,
                                                target=val_target, 
                                                states=val_states, 
                                                device=self.device)
                validation_losses.append(val_loss)
                if val_loss is not None and best_val > val_loss:
                    best_val = val_loss
                    self.best_output = output
                    best_weights = deepcopy(self.state_dict())
                    patience = early_stopping
                else:
                    patience -= 1

                if epoch > early_stopping and patience <= 0:
                    print("Early stopping at epoch: ", epoch)
                    break

                if epoch%10 == 0:
                    print(f"######### epoch:{epoch}")
                    print("Training loss: {}".format(training_losses[-1]))
                    print("Validation loss: {}".format(validation_losses[-1]))
            else:
                validation_losses.append(None)
                best_weights = deepcopy(self.state_dict())
                print(f"######### epoch:{epoch}")
                print("Training loss: {}".format(training_losses[-1]))
                print("Validation loss: {}".format(validation_losses[-1]))
            

        print("\n")
        print("Final Training loss: {}".format(training_losses[-1]))
        print("Final Validation loss: {}".format(validation_losses[-1]))

        # plt.figure()
        # plt.plot(training_losses, label="train")
        # plt.plot(validation_losses, label="val")
        # plt.legend()
        # plt.savefig("st_loss.png")
        # plt.close()
        
        self.load_state_dict(best_weights)

    def _apply_future_ti(self, y, states):
        # states: [B,H,C], y: [B,N,H] or [B,H,N]
        if self.future_ti is None or states is None or states.numel() == 0:
            return y
        delta = self.future_ti(states)  # [B,H,N] or [B,H,1]
        if delta is None:
            return y
        # Align shapes to [B,H,N] for addition
        B, Hs, _ = states.shape
        if y.dim() != 3:
            return y
        y_is_BNH = (y.shape[2] == Hs)      # True if [B,N,H]
        y_hn = y.transpose(1, 2) if y_is_BNH else y  # -> [B,H,N]
        if delta.size(-1) == 1:
            delta = delta.expand(-1, -1, y_hn.size(-1))
        y_hn = y_hn + delta
        return y_hn.transpose(1, 2) if y_is_BNH else y_hn

    def _setup_epi_reg_from_data(
        self,
        train_input,
        train_target,
        use_epi_reg=False,
        epi_reg_loss="mse",
        epi_hidden=8,
        epi_mode="sir_incidence",   # "sir_incidence" | "sir_percent" | "ngm"
        epi_percent_scale=100.0,    # percent output scaling for sir_percent
        epi_population=None,        # optional scalar / [M] list/tensor
        epi_regions=None,           # optional "CA" or list[str] length M
        epi_dt=1.0,                 # step size
        target_idx=0,               # which feature is the observed cases series
    ):
        if not use_epi_reg:
            self.epi_head = None
            self.epi_reg = None
            self.epi_mode = None
            return

        _, _, M, Fdim = train_input.shape

        if train_target.dim() != 3:
            raise ValueError("train_target must be 3D")
        if train_target.shape[1] == M and train_target.shape[2] != M:
            H = train_target.shape[2]  # [B,M,H]
        else:
            H = train_target.shape[1]  # [B,H,M]

        self.epi_mode = epi_mode

        self.epi_head = _EpiHybridHead(
            in_features=Fdim,
            horizon=H,
            population=epi_population,
            regions=epi_regions,
            hidden=epi_hidden,
            mlp_hidden=epi_hidden,
            target_idx=target_idx,
            dt=epi_dt,
            percent_scale=epi_percent_scale,
            clamp_max=1.0,
            eps=1e-6,
            enforce_mass=True,
        ).to(self.device)

        scale = float(use_epi_reg) if isinstance(use_epi_reg, (int, float)) else 1.0
        self.epi_reg = _EpiRegLoss(scale=scale, loss=epi_reg_loss).to(self.device)

    def _apply_epi_reg_loss(self, base_loss, feature, graph, dynamic_graph, target):
        if self.epi_head is None or self.epi_reg is None:
            return base_loss

        B, W, M, Fdim = feature.shape
        device = feature.device

        # static adjacency for masking (optional)
        adj_static = graph.to(device) if graph is not None else None

        # build adj_prob [B,H,M,M]
        H = self.epi_head.horizon

        if dynamic_graph is not None:
            dg = dynamic_graph.to(device)
            if dg.dim() == 4:
                # assume [B,H,M,M] (or [B,1,M,M] -> expand)
                adj_prob = F.softmax(dg, dim=-1)
                if adj_prob.shape[1] == 1 and H > 1:
                    adj_prob = adj_prob.expand(B, H, M, M)
                elif adj_prob.shape[1] != H:
                    # best effort: truncate or pad by repeating last
                    if adj_prob.shape[1] > H:
                        adj_prob = adj_prob[:, :H]
                    else:
                        last = adj_prob[:, -1:].expand(B, H - adj_prob.shape[1], M, M)
                        adj_prob = torch.cat([adj_prob, last], dim=1)
            elif dg.dim() == 3:
                # [B,M,M] -> expand to [B,H,M,M]
                adj_prob = F.softmax(dg, dim=-1).view(B, 1, M, M).expand(B, H, M, M)
            else:
                adj_prob = None
        else:
            adj_prob = None

        if adj_prob is None:
            # fallback to static graph
            if adj_static is None:
                return base_loss  # cannot compute epi reg without any graph
            adj_prob = F.softmax(adj_static, dim=-1).view(1, 1, M, M).expand(B, H, M, M)

        # compute epi output in chosen mode
        epi_out = self.epi_head(feature, adj_prob, adj_static=adj_static, mode=self.epi_mode)

        # shape align: allow target [B,M,H]
        if epi_out.shape != target.shape and epi_out.dim() == 3 and epi_out.transpose(1, 2).shape == target.shape:
            epi_out = epi_out.transpose(1, 2)

        return base_loss + self.epi_reg(epi_out, target)

    def train_epoch(self, optimizer, loss_fn, feature, states=None, graph=None, dynamic_graph=None, target=None, batch_size=1, device='cpu'):
        """
        Trains one epoch with the given data.
        :param feature: Training features of shape (num_samples, num_nodes,
        num_timesteps_train, num_features).
        :param target: Training targets of shape (num_samples, num_nodes,
        num_timesteps_predict).
        :param batch_size: Batch size to use during training.
        :return: Average loss for this epoch.
        """
        permutation = torch.randperm(feature.shape[0])

        epoch_training_losses = []
        for i in range(0, feature.shape[0], batch_size):
            self.train()
            optimizer.zero_grad()
            
            indices = permutation[i:i + batch_size]
            X_batch, y_batch = feature[indices], target[indices]

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            X_batch = torch.nan_to_num(X_batch, nan=0.0, posinf=1e4, neginf=-1e4)
            y_batch = torch.nan_to_num(y_batch, nan=0.0, posinf=1e4, neginf=-1e4)
            
            if states is not None:
                X_states = states[indices]
                X_states = X_states.to(device)
            else:
                X_states = None
            
            if dynamic_graph is not None:
                batch_graph = dynamic_graph[indices]
                batch_graph = batch_graph.to(device)
            else:
                batch_graph = None
            
            if graph is not None:
                graph = graph.to(device)
            out = self.forward(X_batch, graph, X_states, batch_graph)
            out = self._apply_future_ti(out, X_states)
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
            loss = loss_fn(out, y_batch)
            # import ipdb; ipdb.set_trace()
            loss = self._apply_epi_reg_loss(loss, X_batch, graph, batch_graph, y_batch)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_training_losses.append(loss.detach().cpu().numpy())
            if len(epoch_training_losses) == 0:
                return float('nan')
        return sum(epoch_training_losses)/len(epoch_training_losses)
    
    def evaluate(self, loss_fn, feature, graph = None, dynamic_graph=None, target = None, states = None, device = 'cpu'):
        with torch.no_grad():
            self.eval()
            feature = feature.to(device=device)
            target = target.to(device=device)

            if graph is not None:
                graph = graph.to(device)

            if dynamic_graph is not None:
                dynamic_graph = dynamic_graph.to(device)

            if states is not None:
                states = states.to(device)

            feature = torch.nan_to_num(feature, nan=0.0, posinf=1e4, neginf=-1e4)
            target = torch.nan_to_num(target, nan=0.0, posinf=1e4, neginf=-1e4)
            out = self.forward(feature, graph, states, dynamic_graph)
            out = self._apply_future_ti(out, states)
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
            val_loss = loss_fn(out, target)
            val_loss = self._apply_epi_reg_loss(val_loss, feature, graph, dynamic_graph, target)
            val_loss = val_loss.detach().cpu().item()
            
            return val_loss, out

    def predict(self, feature, graph=None, states=None, dynamic_graph=None):
        """
        Returns
        -------
        torch.FloatTensor
        """
        with torch.no_grad():
            self.eval()
            if graph is not None:
                graph = graph.to(self.device)

            if dynamic_graph is not None:
                dynamic_graph = dynamic_graph.to(self.device)
            
            if states is not None:
                states = states.to(self.device)
            
            if feature is not None:
                feature = feature.to(self.device)
            # import ipdb; ipdb.set_trace()
            result = self.forward(feature, graph, states, dynamic_graph)
            result = self._apply_future_ti(result, states)
        return result.detach().cpu()

class BaseTemporalModel(nn.Module):
    def __init__(self, device = 'cpu'):
        super(BaseTemporalModel, self).__init__()
        self.device = device

    def fit(self, 
            train_input, 
            train_target, 
            train_states=None, 
            train_graph=None, 
            train_dynamic_graph=None,
            val_input=None, 
            val_target=None,
            val_states=None, 
            val_graph= None, 
            val_dynamic_graph=None,
            loss='mse', 
            epochs=1000, 
            batch_size=10,
            lr=1e-3, 
            initialize=True, 
            verbose=False, 
            patience=100, 
            **kwargs):
        if initialize:
            self.initialize()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = get_loss(loss)

        training_losses = []
        validation_losses = []
        early_stopping = patience
        best_val = float('inf')
        for epoch in tqdm(range(epochs)):
            
            loss = self.train_epoch(optimizer = optimizer, loss_fn = loss_fn, feature = train_input,  target = train_target, batch_size = batch_size, device = self.device)
            training_losses.append(loss)
            if val_input is not None and val_input.numel():
                val_loss, output = self.evaluate(loss_fn = loss_fn, feature = val_input,  target = val_target, device = self.device)
                validation_losses.append(val_loss)

                if best_val > val_loss:
                    best_val = val_loss
                    self.output = output
                    best_weights = deepcopy(self.state_dict())
                    patience = early_stopping
                else:
                    patience -= 1

                if epoch > early_stopping and patience <= 0:
                    break

                if verbose and epoch%10 == 0:
                    print(f"######### epoch:{epoch}")
                    print("Training loss: {}".format(training_losses[-1]))
                    print("Validation loss: {}".format(validation_losses[-1]))
            else:
                validation_losses.append(None)
                best_weights = deepcopy(self.state_dict())
                if verbose and epoch%10 == 0:
                    print(f"######### epoch:{epoch}")
                    print("Training loss: {}".format(training_losses[-1]))
                    print("Validation loss: {}".format(validation_losses[-1]))

        print("\n")
        print("Final Training loss: {}".format(training_losses[-1]))
        print("Final Validation loss: {}".format(validation_losses[-1]))

        self.load_state_dict(best_weights)

        
    def train_epoch(self, optimizer, loss_fn, feature, target = None, batch_size = 1, device = 'cpu'):
        """
        Trains one epoch with the given data.
        :param feature: Training features of shape (num_samples, num_nodes,
        num_timesteps_train, num_features).
        :param target: Training targets of shape (num_samples, num_nodes,
        num_timesteps_predict).
        :param batch_size: Batch size to use during training.
        :return: Average loss for this epoch.
        """
        permutation = torch.randperm(feature.shape[0])

        epoch_training_losses = []
        for i in range(0, feature.shape[0], batch_size):
            self.train()
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            X_batch, y_batch = feature[indices], target[indices]
            X_batch = X_batch.to(device=device)
            y_batch = y_batch.to(device=device)
            X_batch = torch.nan_to_num(X_batch, nan=0.0, posinf=1e4, neginf=-1e4)
            y_batch = torch.nan_to_num(y_batch, nan=0.0, posinf=1e4, neginf=-1e4)
            out = self.forward(X_batch)
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
            loss = loss_fn(out.reshape(y_batch.shape), y_batch)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_training_losses.append(loss.detach().cpu().numpy())
            if len(epoch_training_losses) == 0:
                return float('nan')
        return sum(epoch_training_losses)/len(epoch_training_losses)
    
    def evaluate(self, loss_fn, feature, target = None, device = 'cpu'):
        with torch.no_grad():
            self.eval()
            feature = feature.to(device=device)
            target = target.to(device=device)

            feature = torch.nan_to_num(feature, nan=0.0, posinf=1e4, neginf=-1e4)
            target = torch.nan_to_num(target, nan=0.0, posinf=1e4, neginf=-1e4)
            out = self.forward(feature)
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
            val_loss = loss_fn(out.reshape(target.shape), target)
            val_loss = val_loss.detach().cpu().numpy().item()
            
            return val_loss, out

    def predict(self, feature, graph=None, states=None, dynamic_graph=None):
        """
        Returns
        -------
        torch.FloatTensor
        """
        self.eval()
        result = self.forward(feature.to(self.device))

        return result.detach().cpu()