import torch
import torch.nn.functional as F
import torch.nn as nn

from .base import BaseModel

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(
            F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(
                2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        # N, cheb_k, dim_in, dim_out
        weights = torch.einsum(
            'nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports,
                           x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g,
                               weights) + bias  # b, N, dim_out
        return x_gconv

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2 *
                           dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim,
                             dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, num_timesteps_input
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, nlayers=1):
        super(AVWDCRNN, self).__init__()
        assert nlayers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.num_timesteps_input = dim_in
        self.nlayers = nlayers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(
            AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, nlayers):
            self.dcrnn_cells.append(
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (nlayers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.num_timesteps_input
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.nlayers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](
                    current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (nlayers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.nlayers):
            init_states.append(
                self.dcrnn_cells[i].init_hidden_state(batch_size))
        # (nlayers, B, N, hidden_dim)
        return torch.stack(init_states, dim=0)


class AGCRN(BaseModel):
    """
    Paper: Adaptive Graph Convolutional Recurrent Network for Trafﬁc Forecasting
    Official Code: https://github.com/LeiBAI/AGCRN
    Link: https://arxiv.org/abs/2007.02842
    Venue: NeurIPS 2020
    Task: Spatial-Temporal Forecasting
    """

    def __init__(self, num_timesteps_input, num_timesteps_output, adj_m = None, num_nodes = None, num_features = 1,rnn_units = 64, nlayers = 2, embed_dim = 10, cheb_k = 2, device="cpu", use_future_ti=False, tid_sizes=None, emb_dim=4, ti_hidden=(16,), node_specific=True, **kwargs):
        if num_nodes is None and adj_m is not None:
            num_nodes = adj_m.shape[0]
        super().__init__(tid_sizes=tid_sizes,
                         device=device,
                         use_future_ti=use_future_ti,
                         emb_dim=emb_dim,
                         ti_hidden=ti_hidden,
                         node_specific=node_specific,
                         num_nodes=num_nodes)
        num_nodes = adj_m.shape[0]
        self.device = device
        self.num_node = num_nodes
        self.num_timesteps_input = num_timesteps_input
        self.hidden_dim = rnn_units
        self.num_features = num_features
        self.num_timesteps_output = num_timesteps_output
        self.nlayers = nlayers

        self.default_graph = adj_m
        self.node_embeddings = nn.Parameter(torch.randn(
            self.num_node, embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(num_nodes, num_timesteps_input, rnn_units, cheb_k,
                                embed_dim, nlayers)

        # predictor
        self.end_conv = nn.Conv2d(
            1, num_timesteps_output * self.num_features, kernel_size=(1, self.hidden_dim), bias=True)

        self.init_param()

    def init_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X_batch, graph, X_states, batch_graph):
        history_data = X_batch.permute(0, 3, 2, 1).contiguous()
        
        init_state = self.encoder.init_hidden(history_data.shape[0])
        output, _ = self.encoder(
            history_data, init_state, self.node_embeddings)  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv((output))  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.num_timesteps_output,
                                            self.num_features, self.num_node)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C
        if self.num_features == 1:
            output = output.squeeze(-1)

        return output

    def initialize(self):
        pass
