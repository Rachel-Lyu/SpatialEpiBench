import torch
import torch.nn as nn
from .base import BaseModel

class ARIMA(BaseModel):
    """
    Node-wise ARIMA(p, d, q) using differencing + iterated conditional least squares (CSS).

    Shapes
    ------
    train_input / X_batch: (B, T_in, N, F) with F == 1
    output:               (B, T_out, N)

    Notes
    -----
    - This is NOT full maximum-likelihood ARIMA; MA(q) is fit via iterated CSS.
    - Multi-step forecasts set future innovations to 0 (standard point-forecast convention).
    """

    def __init__(
        self,
        num_features,
        num_timesteps_input,
        num_timesteps_output,
        p=1,
        d=0,
        q=0,
        include_intercept=True,
        css_iters=5,
        ridge=1e-6,
        device="cpu",
    ):
        super().__init__(device=device)

        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output

        self.p = int(p)
        self.d = int(d)
        self.q = int(q)
        self.include_intercept = bool(include_intercept)
        self.css_iters = int(css_iters)
        self.ridge = float(ridge)

        # Fitted per node
        self.phi = None     # (N, p)
        self.theta = None   # (N, q)
        self.c = None       # (N,) or None
        self.num_nodes = None

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _difference_1d(x_1d: torch.Tensor, d: int):
        """
        x_1d: (T,)
        Returns:
          x_d:   (T-d,)
          tails: list of last values of x_k at end for k=0..d-1, each a scalar tensor
        """
        if d == 0:
            return x_1d, []
        tails = []
        cur = x_1d
        for k in range(d):
            tails.append(cur[-1].clone())   # last value of x_k
            cur = cur[1:] - cur[:-1]
        return cur, tails

    @staticmethod
    def _undifference_forecast(pred_d: torch.Tensor, tails: list):
        """
        pred_d: (H,) forecasts in d-times differenced space
        tails:  [last(x0), last(x1), ..., last(x_{d-1})]
        Returns:
          pred_level: (H,) in original level space
        """
        out = pred_d
        # Rebuild x_{d-1}, x_{d-2}, ..., x0
        for k in range(len(tails) - 1, -1, -1):
            out = tails[k] + torch.cumsum(out, dim=0)
        return out

    def _build_regression(self, x_d: torch.Tensor, e: torch.Tensor):
        """
        x_d: (B, T) differenced series
        e:   (B, T) residuals (same length)
        Returns X2, Y2 for regression y_t on [AR lags, MA lags, intercept]
        """
        B, T = x_d.shape
        start = max(self.p, self.q)
        if T <= start:
            raise ValueError(f"Need T > max(p,q). Got T={T}, p={self.p}, q={self.q}")

        Y = x_d[:, start:]  # (B, T-start)

        feats = []
        # AR lags: columns [y_{t-1}, ..., y_{t-p}]
        for k in range(1, self.p + 1):
            feats.append(x_d[:, start - k : T - k].unsqueeze(-1))  # (B, T-start, 1)

        # MA lags: columns [e_{t-1}, ..., e_{t-q}]
        for k in range(1, self.q + 1):
            feats.append(e[:, start - k : T - k].unsqueeze(-1))    # (B, T-start, 1)

        if self.include_intercept:
            feats.append(torch.ones(B, T - start, 1, device=x_d.device))

        if len(feats) == 0:
            # Degenerate: no AR, no MA, no intercept -> can't fit anything meaningful
            X = torch.zeros(B, T - start, 1, device=x_d.device)
        else:
            X = torch.cat(feats, dim=-1)  # (B, T-start, D)

        X2 = X.reshape(-1, X.shape[-1])
        Y2 = Y.reshape(-1, 1)
        return X2, Y2, start

    def _solve_lstsq(self, X2: torch.Tensor, Y2: torch.Tensor):
        """
        Solve ridge-regularized least squares.
        Returns beta: (D, 1)
        """
        if self.ridge and self.ridge > 0:
            D = X2.shape[1]
            eye = torch.eye(D, device=X2.device)
            X_aug = torch.cat([X2, (self.ridge ** 0.5) * eye], dim=0)
            Y_aug = torch.cat([Y2, torch.zeros(D, 1, device=X2.device)], dim=0)
            beta = torch.linalg.lstsq(X_aug, Y_aug).solution
        else:
            beta = torch.linalg.lstsq(X2, Y2).solution
        return beta

    def _compute_residuals(self, x_d: torch.Tensor, phi: torch.Tensor, theta: torch.Tensor, c: torch.Tensor):
        """
        Sequentially compute residuals e_t = x_d[t] - (c + AR + MA) using past residuals.
        x_d:  (B, T)
        phi:  (p,)
        theta:(q,)
        c:    scalar tensor or None
        Returns e: (B, T)
        """
        B, T = x_d.shape
        e = torch.zeros(B, T, device=x_d.device)
        start = max(self.p, self.q)

        for t in range(start, T):
            y_hat = 0.0
            if self.include_intercept:
                y_hat = y_hat + c

            if self.p > 0:
                # ar part: sum_{k=1..p} phi_k * y_{t-k}
                ar = 0.0
                for k in range(1, self.p + 1):
                    ar = ar + phi[k - 1] * x_d[:, t - k]
                y_hat = y_hat + ar

            if self.q > 0:
                ma = 0.0
                for k in range(1, self.q + 1):
                    ma = ma + theta[k - 1] * e[:, t - k]
                y_hat = y_hat + ma

            e[:, t] = x_d[:, t] - y_hat

        return e

    # -------------------------
    # Fit (deterministic-ish)
    # -------------------------
    def fit(
        self,
        train_input,
        train_target=None,
        train_states=None,
        train_graph=None,
        train_dynamic_graph=None,
        val_input=None,
        val_target=None,
        val_states=None,
        val_graph=None,
        val_dynamic_graph=None,
        loss="mse",
        epochs=1,
        batch_size=10,
        lr=1e-3,
        initialize=True,
        verbose=False,
        patience=100,
        **kwargs
    ):
        if initialize:
            self.initialize()

        X = train_input.to(self.device)
        if X.dim() != 4:
            raise ValueError(f"Expected train_input (B, T_in, N, F), got {X.shape}")

        B, T_in, N, F = X.shape
        if F != self.num_features:
            raise ValueError(f"num_features mismatch: model={self.num_features}, data={F}")
        if T_in != self.num_timesteps_input:
            raise ValueError(f"T_in mismatch: model={self.num_timesteps_input}, data={T_in}")
        if self.num_features != 1:
            raise NotImplementedError("This implementation assumes num_features == 1.")

        if T_in - self.d <= max(self.p, self.q):
            raise ValueError(
                f"Lookback too short after differencing. "
                f"T_in={T_in}, d={self.d} => {T_in-self.d} <= max(p,q)={max(self.p,self.q)}"
            )

        self.num_nodes = N
        phi = torch.zeros(N, self.p, device=self.device) if self.p > 0 else torch.zeros(N, 0, device=self.device)
        theta = torch.zeros(N, self.q, device=self.device) if self.q > 0 else torch.zeros(N, 0, device=self.device)
        c = torch.zeros(N, device=self.device) if self.include_intercept else None

        for n in range(N):
            # x: (B, T_in)
            x = X[:, :, n, 0]

            # Difference each sample independently to avoid leaking across windows
            x_d_list = []
            for b in range(B):
                xd, _tails = self._difference_1d(x[b], self.d)
                x_d_list.append(xd)
            x_d = torch.stack(x_d_list, dim=0)  # (B, T_in - d)

            # Iterated CSS:
            # Start with residuals = 0, then alternate:
            #   beta = argmin ||y - X(beta)||^2
            #   recompute residuals sequentially using beta
            e = torch.zeros_like(x_d)

            # Initialize params
            phi_n = torch.zeros(self.p, device=self.device) if self.p > 0 else torch.zeros(0, device=self.device)
            theta_n = torch.zeros(self.q, device=self.device) if self.q > 0 else torch.zeros(0, device=self.device)
            c_n = torch.tensor(0.0, device=self.device)

            iters = max(1, self.css_iters) if self.q > 0 else 1
            for _ in range(iters):
                X2, Y2, start = self._build_regression(x_d, e)
                beta = self._solve_lstsq(X2, Y2).squeeze(-1)  # (D,)

                idx = 0
                if self.p > 0:
                    phi_n = beta[idx : idx + self.p]
                    idx += self.p
                if self.q > 0:
                    theta_n = beta[idx : idx + self.q]
                    idx += self.q
                if self.include_intercept:
                    c_n = beta[idx]

                e = self._compute_residuals(x_d, phi_n, theta_n, c_n)

            if self.p > 0:
                phi[n, :] = phi_n
            if self.q > 0:
                theta[n, :] = theta_n
            if self.include_intercept:
                c[n] = c_n

        self.phi = phi
        self.theta = theta
        self.c = c

        if verbose:
            print(f"ARIMA fitted per node with CSS (p={self.p}, d={self.d}, q={self.q}).")
        return self

    # -------------------------
    # Forward (forecast)
    # -------------------------
    def forward(self, X_batch, graph=None, X_states=None, batch_graph=None):
        if self.phi is None or self.theta is None or (self.include_intercept and self.c is None):
            raise RuntimeError("ARIMA model not fitted. Call .fit(...) first.")

        X_batch = X_batch.to(self.device)
        B, T_in, N, F = X_batch.shape

        if N != self.num_nodes:
            raise ValueError(f"Num nodes mismatch: data={N}, fitted={self.num_nodes}")
        if F != self.num_features or self.num_features != 1:
            raise ValueError("This implementation assumes num_features == 1.")
        if T_in != self.num_timesteps_input:
            raise ValueError(f"T_in mismatch: model={self.num_timesteps_input}, data={T_in}")

        H = self.num_timesteps_output
        out = torch.zeros(B, H, N, device=self.device)

        start = max(self.p, self.q)

        for n in range(N):
            phi_n = self.phi[n]  # (p,)
            theta_n = self.theta[n]  # (q,)
            c_n = self.c[n] if self.include_intercept else torch.tensor(0.0, device=self.device)

            for b in range(B):
                series = X_batch[b, :, n, 0]  # (T_in,)

                # Difference (and collect tails for undifferencing)
                x_d, tails = self._difference_1d(series, self.d)  # (T_in-d,), tails len d
                T = x_d.numel()

                # Estimate in-sample residual history for MA lags (needed for 1-step forecast)
                # (Deterministic: compute residuals sequentially on the history window)
                e_hist = torch.zeros(T, device=self.device)
                if T > start:
                    # sequential residuals
                    for t in range(start, T):
                        y_hat = 0.0
                        if self.include_intercept:
                            y_hat = y_hat + c_n
                        if self.p > 0:
                            for k in range(1, self.p + 1):
                                y_hat = y_hat + phi_n[k - 1] * x_d[t - k]
                        if self.q > 0:
                            for k in range(1, self.q + 1):
                                y_hat = y_hat + theta_n[k - 1] * e_hist[t - k]
                        e_hist[t] = x_d[t] - y_hat

                # Forecast in differenced space, setting future innovations to 0
                hist = x_d.clone()
                e_future = e_hist.clone()
                preds_d = []

                for _ in range(H):
                    # AR context (lag1..lagp)
                    if self.p > 0:
                        if hist.numel() < self.p:
                            y_ctx = torch.zeros(self.p, device=self.device)
                            m = hist.numel()
                            y_ctx[:m] = hist.flip(0)  # fill lag1, lag2...
                        else:
                            y_ctx = hist[-self.p:].flip(0)
                        ar_part = (phi_n * y_ctx).sum()
                    else:
                        ar_part = torch.tensor(0.0, device=self.device)

                    # MA context (lag1..lagq)
                    if self.q > 0:
                        if e_future.numel() < self.q:
                            e_ctx = torch.zeros(self.q, device=self.device)
                            m = e_future.numel()
                            e_ctx[:m] = e_future.flip(0)
                        else:
                            e_ctx = e_future[-self.q:].flip(0)
                        ma_part = (theta_n * e_ctx).sum()
                    else:
                        ma_part = torch.tensor(0.0, device=self.device)

                    y_next_d = (c_n if self.include_intercept else 0.0) + ar_part + ma_part
                    preds_d.append(y_next_d)

                    # append predicted value and assumed future residual 0
                    hist = torch.cat([hist, y_next_d.view(1)], dim=0)
                    e_future = torch.cat([e_future, torch.zeros(1, device=self.device)], dim=0)

                preds_d = torch.stack(preds_d)  # (H,)

                # Undifference back to levels
                if self.d > 0:
                    preds_level = self._undifference_forecast(preds_d, tails)
                else:
                    preds_level = preds_d

                out[b, :, n] = preds_level

        return out  # (B, H, N)

    # -------------------------
    # Init / reset
    # -------------------------
    def reset_parameters(self):
        pass

    def initialize(self):
        self.phi = None
        self.theta = None
        self.c = None
        self.num_nodes = None
