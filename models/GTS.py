import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel

class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter(
                '{}_weight_{}'.format(self._type, str(shape)), nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter(
                '{}_biases_{}'.format(self._type, str(length)), biases)

        return self._biases_dict[length]


class DCGRUCell(torch.nn.Module):
    def __init__(self, num_units, max_diffusion_step, num_nodes, nonlinearity='tanh', filter_type="laplacian", use_gc_for_ru=True):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'gconv')

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape)
        return L

    def _calculate_random_walk_matrix(self, adj_mx):

        # tf.Print(adj_mx, [adj_mx], message="This is adj: ")

        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(adj_mx.device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(
            d_inv.shape).to(d_inv.device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, inputs, hx, adj):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        adj_mx = self._calculate_random_walk_matrix(adj).t()
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(
            fn(inputs, adj_mx, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(
            tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, adj_mx, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(
            x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            x1 = torch.mm(adj_mx, x0)
            x = self._concat(x, x1)

            for k in range(2, self._max_diffusion_step + 1):
                x2 = 2 * torch.mm(adj_mx, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1
        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(
            x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(
            x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights(
            (input_size * num_matrices, output_size)).to(x.device)
        # (batch_size * self._num_nodes, output_size)
        x = torch.matmul(x, weights)

        biases = self._gconv_params.get_biases(
            output_size, bias_start).to(x.device)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def sample_gumbel(shape, eps=1e-20, device=None):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps, device=logits.device)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(logits.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

class Seq2SeqAttrs:
    def __init__(self, rnn_units, max_diffusion_step=2, cl_decay_steps=1000, filter_type="laplacian", num_nodes=1, num_rnn_layers=1):
        #self.adj_mx = adj_mx
        self.max_diffusion_step = int(max_diffusion_step)
        self.cl_decay_steps = int(cl_decay_steps)
        self.filter_type = str(filter_type)
        self.num_nodes = int(num_nodes)
        self.num_rnn_layers = int(num_rnn_layers)
        self.rnn_units = int(rnn_units)
        self.hidden_state_size = self.num_nodes * self.rnn_units

class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, rnn_units, seq_len, input_dim=1, max_diffusion_step=2, cl_decay_steps=1000, filter_type="laplacian", num_nodes=1, num_rnn_layers=1):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(
            self,
            rnn_units=rnn_units,
            max_diffusion_step=max_diffusion_step,
            cl_decay_steps=cl_decay_steps,
            filter_type=filter_type,
            num_nodes=num_nodes,
            num_rnn_layers=num_rnn_layers,
        )
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes, filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size) optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size) hidden_state # shape (num_layers, batch_size, self.hidden_state_size) (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size)).to(inputs.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(
        self,
        rnn_units,
        horizon=1,
        output_dim=1,
        max_diffusion_step=2,
        cl_decay_steps=1000,
        filter_type="laplacian",
        num_nodes=1,
        num_rnn_layers=1,
    ):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(
            self,
            rnn_units=rnn_units,
            max_diffusion_step=max_diffusion_step,
            cl_decay_steps=cl_decay_steps,
            filter_type=filter_type,
            num_nodes=num_nodes,
            num_rnn_layers=num_rnn_layers,
        )
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes, filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj, hidden_state=None):
        """
        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size) optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim) hidden_state # shape (num_layers, batch_size, self.hidden_state_size) (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class GTS(BaseModel, Seq2SeqAttrs):
    """
    Paper: Discrete Graph Structure Learning for Forecasting Multiple Time Series.
    Link: https://arxiv.org/abs/2101.06861
    Official Code: https://github.com/chaoshangcs/GTS
    Venue: ICLR 2021
    Task: Spatial-Temporal Forecasting
    Note: 
        Kindly note that the results of GTS may have some gaps with the original paper, 
        because it calculates the evaluation metrics in a slightly different manner. 
        Some details can be found in the appendix in the original paper and 
            similar issues in its official repository: https://github.com/chaoshangcs/GTS/issues
    """

    def __init__(
        self,
        num_timesteps_input,
        num_timesteps_output,
        adj_m = None,
        num_nodes = None,
        rnn_units=64,
        max_diffusion_step=2,
        cl_decay_steps=1000,
        filter_type="laplacian",
        num_rnn_layers=1,
        input_dim=1,
        output_dim=1,
        use_curriculum_learning=True,
        node_feats=None,
        temp=0.5,
        k=5,
        device="cpu",
        use_future_ti=False, 
        tid_sizes=None, 
        emb_dim=4, 
        ti_hidden=(16,), 
        node_specific=True, 
        **kwargs
    ):
        if num_nodes is None and adj_m is not None:
            num_nodes = adj_m.shape[0]
        seq_len = num_timesteps_input
        horizon = num_timesteps_output
        super().__init__(tid_sizes=tid_sizes,
                         device=device,
                         use_future_ti=use_future_ti,
                         emb_dim=emb_dim,
                         ti_hidden=ti_hidden,
                         node_specific=node_specific,
                         num_nodes=num_nodes)
        Seq2SeqAttrs.__init__(
            self,
            rnn_units=rnn_units,
            max_diffusion_step=max_diffusion_step,
            cl_decay_steps=cl_decay_steps,
            filter_type=filter_type,
            num_nodes=num_nodes,
            num_rnn_layers=num_rnn_layers,
        )
        self.encoder_model = EncoderModel(
            rnn_units=rnn_units,
            seq_len=seq_len,
            input_dim=input_dim,
            max_diffusion_step=max_diffusion_step,
            cl_decay_steps=cl_decay_steps,
            filter_type=filter_type,
            num_nodes=num_nodes,
            num_rnn_layers=num_rnn_layers,
        )
        self.decoder_model = DecoderModel(
            rnn_units=rnn_units,
            horizon=horizon,
            output_dim=output_dim,
            max_diffusion_step=max_diffusion_step,
            cl_decay_steps=cl_decay_steps,
            filter_type=filter_type,
            num_nodes=num_nodes,
            num_rnn_layers=num_rnn_layers,
        )
        self.cl_decay_steps = int(cl_decay_steps)
        self.use_curriculum_learning = bool(use_curriculum_learning)
        self.embedding_dim = 100
        self.conv1 = torch.nn.Conv1d(1, 8, 3, stride=1)  # .to(device)
        self.conv2 = torch.nn.Conv1d(8, 16, 3, stride=1)  # .to(device)
        self.hidden_drop = torch.nn.Dropout(0.2)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
            return labels_onehot
        # Generate off-diagonal interaction graph
        off_diag = np.ones([self.num_nodes, self.num_nodes])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec)
        self.rel_send = torch.FloatTensor(rel_send)

        self.node_feats = node_feats
        self.temp = temp
        if node_feats is None:
            self.node_feats = adj_m
            self.prior_adj = torch.Tensor(np.array(adj_m, dtype=np.float32))
        else:
            from sklearn.neighbors import kneighbors_graph
            g = kneighbors_graph(self.node_feats.T, k, metric='cosine')
            g = np.array(g.todense(), dtype=np.float32)
            self.prior_adj = torch.Tensor(g)
        def out_len_1d(L, k, s=1, p=0, d=1):
            return (L + 2*p - d*(k-1) - 1)//s + 1
        L1 = out_len_1d(self.node_feats.shape[0], 3, 1, 0, 1)   # conv1
        L2 = out_len_1d(L1, 3, 1, 0, 1)  # conv2
        self.fc = torch.nn.Linear(16 * L2, self.embedding_dim)

    def encoder(self, inputs, adj):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """

        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], adj, encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """

        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim)).to(encoder_hidden_state.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, X_batch, graph, X_states, batch_graph):
    
        # reshape data
        batch_size, length, num_nodes, channels = X_batch.shape
        X_batch = X_batch.reshape(batch_size, length, num_nodes * channels)      # [B, L, N*C]
        X_batch = X_batch.transpose(0, 1)         # [L, B, N*C]
    
        # GTS
        inputs = X_batch

        x = self.node_feats.transpose(1, 0).view(self.num_nodes, 1, -1).to(X_batch.device)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        # x = self.hidden_drop(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = x.view(self.num_nodes, -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.bn3(x)

        receivers = torch.matmul(self.rel_rec.to(x.device), x)
        senders = torch.matmul(self.rel_send.to(x.device), x)
        x = torch.cat([senders, receivers], dim=1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        adj = gumbel_softmax(x, temperature=self.temp, hard=True)
        adj = adj[:, 0].clone().reshape(self.num_nodes, -1)
        mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(adj.device)
        adj.masked_fill_(mask, 0)

        encoder_hidden_state = self.encoder(inputs, adj)
        outputs = self.decoder(encoder_hidden_state, adj)
        prediction = outputs.transpose(1, 0)
        return prediction

    def initialize(self):
        pass