import numpy as np
import math
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch.autograd import Variable
import torchdiffeq
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian, dense_to_sparse
# from fastdtw import fastdtw
from .base import BaseModel

# controldiffeq
class VectorFieldGDE_dev(nn.Module):
    def __init__(self, dX_dt, func_f, func_g):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func_f: As cdeint.
            func_g: As cdeint.
        """
        super(VectorFieldGDE_dev, self).__init__()
        if not isinstance(func_f, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        if not isinstance(func_g, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func_f = func_f
        self.func_g = func_g

    def __call__(self, t, hz):
        # control_gradient is of shape (..., input_channels)
        
        control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)

        h = hz[0] # h: torch.Size([64, 207, 32])
        z = hz[1:] # z: torch.Size([64, 207, 32])
        
        
        vector_field_f = self.func_f(h) # vector_field_f: torch.Size([64, 207, 32, 2])
        vector_field_g_list= self.func_g(hz) # vector_field_g: torch.Size([64, 207, 32, 2])  spatial
        
        # vector_field_fg = torch.mul(vector_field_g, vector_field_f) # vector_field_fg: torch.Size([64, 207, 32, 2])
        vector_field_zg = torch.matmul(vector_field_g_list[0], vector_field_f)
        
        vector_field_sg = torch.matmul(vector_field_g_list[1], vector_field_f)
        vector_field_ig = torch.matmul(vector_field_g_list[2], vector_field_f)
        vector_field_rg = torch.matmul(vector_field_g_list[3], vector_field_f)
        
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        dh = (vector_field_f @ control_gradient.unsqueeze(-1)).squeeze(-1)
        out_z = (vector_field_zg @ control_gradient.unsqueeze(-1)).squeeze(-1)
        out_s = (vector_field_sg @ control_gradient.unsqueeze(-1)).squeeze(-1)
        out_i = (vector_field_ig @ control_gradient.unsqueeze(-1)).squeeze(-1)
        out_r = (vector_field_rg @ control_gradient.unsqueeze(-1)).squeeze(-1)
        
        # dh: torch.Size([64, 207, 32])
        # out: torch.Size([64, 207, 32])
        return tuple([dh, out_z , out_s, out_i, out_r])

def cdeint_gde_dev(dX_dt, h0, z0,s0,i0,r0,adjs,func_f, func_g, t, adjoint=True, **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        dX_dt: The control. This should be a callable. It will be evaluated with a scalar tensor with values
            approximately in [t[0], t[-1]]. (In practice variable step size solvers will often go a little bit outside
            this range as well.) Then dX_dt should return a tensor of shape (..., input_channels), where input_channels
            is some number of channels and the '...' is some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        func: Should be an instance of `torch.nn.Module`. Describes the vector field f(z). Will be called with a tensor
            z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `dX_dt` arguments as above. The '...' corresponds to some number of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq. Note that empirically, the solvers
            that seem to work best are dopri5, euler, midpoint, rk4. Avoid all three Adams methods.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(z_s)dX_s, where t_i = t[i]. This
        will be a tensor of shape (len(t), ..., hidden_channels).
    """
    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
                         "dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))

    if control_gradient.requires_grad and adjoint:
        raise ValueError("Gradients do not backpropagate through the control with adjoint=True. (This is a limitation "
                         "of the underlying torchdiffeq library.)")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorFieldGDE_dev(dX_dt=dX_dt, func_f=func_f, func_g=func_g)
    init0 = (h0,z0,s0,i0,r0)
    out = odeint(func=vector_field, y0=init0, t=t, **kwargs)
    return out

def cheap_stack(tensors, dim):
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    else:
        return torch.stack(tensors, dim=dim)

def tridiagonal_solve(b, A_upper, A_diagonal, A_lower):
    """Solves a tridiagonal system Ax = b.

    The arguments A_upper, A_digonal, A_lower correspond to the three diagonals of A. Letting U = A_upper, D=A_digonal
    and L = A_lower, and assuming for simplicity that there are no batch dimensions, then the matrix A is assumed to be
    of size (k, k), with entries:

    D[0] U[0]
    L[0] D[1] U[1]
         L[1] D[2] U[2]                     0
              L[2] D[3] U[3]
                  .    .    .
                       .      .      .
                           .        .        .
                        L[k - 3] D[k - 2] U[k - 2]
           0                     L[k - 2] D[k - 1] U[k - 1]
                                          L[k - 1]   D[k]

    Arguments:
        b: A tensor of shape (..., k), where '...' is zero or more batch dimensions
        A_upper: A tensor of shape (..., k - 1).
        A_diagonal: A tensor of shape (..., k).
        A_lower: A tensor of shape (..., k - 1).

    Returns:
        A tensor of shape (..., k), corresponding to the x solving Ax = b

    Warning:
        This implementation isn't super fast. You probably want to cache the result, if possible.
    """

    # This implementation is very much written for clarity rather than speed.

    A_upper, _ = torch.broadcast_tensors(A_upper, b[..., :-1])
    A_lower, _ = torch.broadcast_tensors(A_lower, b[..., :-1])
    A_diagonal, b = torch.broadcast_tensors(A_diagonal, b)

    channels = b.size(-1)

    new_b = np.empty(channels, dtype=object)
    new_A_diagonal = np.empty(channels, dtype=object)
    outs = np.empty(channels, dtype=object)

    new_b[0] = b[..., 0]
    new_A_diagonal[0] = A_diagonal[..., 0]
    for i in range(1, channels):
        w = A_lower[..., i - 1] / new_A_diagonal[i - 1]
        new_A_diagonal[i] = A_diagonal[..., i] - w * A_upper[..., i - 1]
        new_b[i] = b[..., i] - w * new_b[i - 1]

    outs[channels - 1] = new_b[channels - 1] / new_A_diagonal[channels - 1]
    for i in range(channels - 2, -1, -1):
        outs[i] = (new_b[i] - A_upper[..., i] * outs[i + 1]) / new_A_diagonal[i]

    return torch.stack(outs.tolist(), dim=-1)

def _natural_cubic_spline_coeffs_without_missing_values(times, path):
    # path should be a tensor of shape (..., length)
    # Will return the b, two_c, three_d coefficients of the derivative of the cubic spline interpolating the path.

    length = path.size(-1)

    if length < 2:
        # In practice this should always already be caught in __init__.
        raise ValueError("Must have a time dimension of size at least 2.")
    elif length == 2:
        a = path[..., :1]
        b = (path[..., 1:] - path[..., :1]) / (times[..., 1:] - times[..., :1])
        two_c = torch.zeros(*path.shape[:-1], 1, dtype=path.dtype, device=path.device)
        three_d = torch.zeros(*path.shape[:-1], 1, dtype=path.dtype, device=path.device)
    else:
        # Set up some intermediate values
        time_diffs = times[1:] - times[:-1]
        time_diffs_reciprocal = time_diffs.reciprocal()
        time_diffs_reciprocal_squared = time_diffs_reciprocal ** 2
        three_path_diffs = 3 * (path[..., 1:] - path[..., :-1])
        six_path_diffs = 2 * three_path_diffs
        path_diffs_scaled = three_path_diffs * time_diffs_reciprocal_squared

        # Solve a tridiagonal linear system to find the derivatives at the knots
        system_diagonal = torch.empty(length, dtype=path.dtype, device=path.device)
        system_diagonal[:-1] = time_diffs_reciprocal
        system_diagonal[-1] = 0
        system_diagonal[1:] += time_diffs_reciprocal
        system_diagonal *= 2
        system_rhs = torch.empty_like(path)
        system_rhs[..., :-1] = path_diffs_scaled
        system_rhs[..., -1] = 0
        system_rhs[..., 1:] += path_diffs_scaled
        knot_derivatives = tridiagonal_solve(system_rhs, time_diffs_reciprocal, system_diagonal,
                                                  time_diffs_reciprocal)

        # Do some algebra to find the coefficients of the spline
        a = path[..., :-1]
        b = knot_derivatives[..., :-1]
        two_c = (six_path_diffs * time_diffs_reciprocal
                 - 4 * knot_derivatives[..., :-1]
                 - 2 * knot_derivatives[..., 1:]) * time_diffs_reciprocal
        three_d = (-six_path_diffs * time_diffs_reciprocal
                   + 3 * (knot_derivatives[..., :-1]
                          + knot_derivatives[..., 1:])) * time_diffs_reciprocal_squared

    return a, b, two_c, three_d

def _natural_cubic_spline_coeffs_with_missing_values(t, path):
    if len(path.shape) == 1:
        # We have to break everything down to individual scalar paths because of the possibility of missing values
        # being different in different channels
        return _natural_cubic_spline_coeffs_with_missing_values_scalar(t, path)
    else:
        a_pieces = []
        b_pieces = []
        two_c_pieces = []
        three_d_pieces = []
        for p in path.unbind(dim=0):  # TODO: parallelise over this
            a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(t, p)
            a_pieces.append(a)
            b_pieces.append(b)
            two_c_pieces.append(two_c)
            three_d_pieces.append(three_d)
        return (cheap_stack(a_pieces, dim=0),
                cheap_stack(b_pieces, dim=0),
                cheap_stack(two_c_pieces, dim=0),
                cheap_stack(three_d_pieces, dim=0))

def _natural_cubic_spline_coeffs_with_missing_values_scalar(times, path):
    # times and path both have shape (length,)

    # How to deal with missing values at the start or end of the time series? We're creating some splines, so one
    # option is just to extend the first piece backwards, and the final piece forwards. But polynomials tend to
    # behave badly when extended beyond the interval they were constructed on, so the results can easily end up
    # being awful.
    # Instead we impute an observation at the very start equal to the first actual observation made, and impute an
    # observation at the very end equal to the last actual observation made, and then procede with splines as
    # normal.

    not_nan = ~torch.isnan(path)
    path_no_nan = path.masked_select(not_nan)

    if path_no_nan.size(0) == 0:
        # Every entry is a NaN, so we take a constant path with derivative zero, so return zero coefficients.
        # Note that we may assume that path.size(0) >= 2 by the checks in __init__ so "path.size(0) - 1" is a valid
        # thing to do.
        return (torch.zeros(path.size(0) - 1, dtype=path.dtype, device=path.device),
                torch.zeros(path.size(0) - 1, dtype=path.dtype, device=path.device),
                torch.zeros(path.size(0) - 1, dtype=path.dtype, device=path.device),
                torch.zeros(path.size(0) - 1, dtype=path.dtype, device=path.device))
    # else we have at least one non-NaN entry, in which case we're going to impute at least one more entry (as
    # the path is of length at least 2 so the start and the end aren't the same), so we will then have at least two
    # non-Nan entries. In particular we can call _compute_coeffs safely later.

    need_new_not_nan = False
    if torch.isnan(path[0]):
        if not need_new_not_nan:
            path = path.clone()
            need_new_not_nan = True
        path[0] = path_no_nan[0]
    if torch.isnan(path[-1]):
        if not need_new_not_nan:
            path = path.clone()
            need_new_not_nan = True
        path[-1] = path_no_nan[-1]
    if need_new_not_nan:
        not_nan = ~torch.isnan(path)
        path_no_nan = path.masked_select(not_nan)
    times_no_nan = times.masked_select(not_nan)

    # Find the coefficients on the pieces we do understand
    # These all have shape (len - 1,)
    (a_pieces_no_nan,
     b_pieces_no_nan,
     two_c_pieces_no_nan,
     three_d_pieces_no_nan) = _natural_cubic_spline_coeffs_without_missing_values(times_no_nan, path_no_nan)

    # Now we're going to normalise them to give coefficients on every interval
    a_pieces = []
    b_pieces = []
    two_c_pieces = []
    three_d_pieces = []

    iter_times_no_nan = iter(times_no_nan)
    iter_coeffs_no_nan = iter(zip(a_pieces_no_nan, b_pieces_no_nan, two_c_pieces_no_nan, three_d_pieces_no_nan))
    next_time_no_nan = next(iter_times_no_nan)
    for time in times[:-1]:
        # will always trigger on the first iteration because of how we've imputed missing values at the start and
        # end of the time series.
        if time >= next_time_no_nan:
            prev_time_no_nan = next_time_no_nan
            next_time_no_nan = next(iter_times_no_nan)
            next_a_no_nan, next_b_no_nan, next_two_c_no_nan, next_three_d_no_nan = next(iter_coeffs_no_nan)
        offset = prev_time_no_nan - time
        a_inner = (0.5 * next_two_c_no_nan - next_three_d_no_nan * offset / 3) * offset
        a_pieces.append(next_a_no_nan + (a_inner - next_b_no_nan) * offset)
        b_pieces.append(next_b_no_nan + (next_three_d_no_nan * offset - next_two_c_no_nan) * offset)
        two_c_pieces.append(next_two_c_no_nan - 2 * next_three_d_no_nan * offset)
        three_d_pieces.append(next_three_d_no_nan)

    return (cheap_stack(a_pieces, dim=0),
            cheap_stack(b_pieces, dim=0),
            cheap_stack(two_c_pieces, dim=0),
            cheap_stack(three_d_pieces, dim=0))


# The mathematics of this are adapted from  http://mathworld.wolfram.com/CubicSpline.html, although they only treat the
# case of each piece being parameterised by [0, 1]. (We instead take the length of each piece to be the difference in
# time stamps.)
def natural_cubic_spline_coeffs(t, X):
    """Calculates the coefficients of the natural cubic spline approximation to the batch of controls given.

    Arguments:
        t: One dimensional tensor of times. Must be monotonically increasing.
        X: tensor of values, of shape (..., L, C), where ... is some number of batch dimensions, L is some length
            that must be the same as the length of t, and C is some number of channels. This is interpreted as a
            (batch of) paths taking values in a C-dimensional real vector space, with L observations. Missing values
            are supported, and should be represented as NaNs.

    In particular, the support for missing values allows for batching together elements that are observed at
    different times; just set them to have missing values at each other's observation times.

    Warning:
        Calling this function can be pretty slow. Make sure to cache the result, and don't reinstantiate it on every
        forward pass, if at all possible.

    Returns:
        Four tensors, which should in turn be passed to `controldiffeq.NaturalCubicSpline`.

        Why do we do it like this? Because typically you want to use PyTorch tensors at various interfaces, for example
        when loading a batch from a DataLoader. If we wrapped all of this up into just the
        `controldiffeq.NaturalCubicSpline` class then that sort of thing wouldn't be possible.

        As such the suggested use is to:
        (a) Load your data.
        (b) Preprocess it with this function.
        (c) Save the result.
        (d) Treat the result as your dataset as far as PyTorch's `torch.utils.data.Dataset` and
            `torch.utils.data.DataLoader` classes are concerned.
        (e) Call NaturalCubicSpline as the first part of your model.

        See also the accompanying example.py.
    """
    if not t.is_floating_point():
        raise ValueError("t and X must both be floating point/")
    if not X.is_floating_point():
        raise ValueError("t and X must both be floating point/")
    if len(t.shape) != 1:
        raise ValueError("t must be one dimensional.")
    prev_t_i = -math.inf
    for t_i in t:
        if t_i <= prev_t_i:
            raise ValueError("t must be monotonically increasing.")

    if len(X.shape) < 2:
        raise ValueError("X must have at least two dimensions, corresponding to time and channels.")

    if X.size(-2) != t.size(0):
        raise ValueError("The time dimension of X must equal the length of t.")

    if t.size(0) < 2:
        raise ValueError("Must have a time dimension of size at least 2.")

    if torch.isnan(X).any():
        # Transpose because channels are a batch dimension for the purpose of finding interpolating polynomials.
        # b, two_c, three_d have shape (..., channels, length - 1)
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(t, X.transpose(-1, -2))
    else:
        # Can do things more quickly in this case.
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_without_missing_values(t, X.transpose(-1, -2))

    # These all have shape (..., length - 1, channels)
    a = a.transpose(-1, -2)
    b = b.transpose(-1, -2)
    two_c = two_c.transpose(-1, -2)
    three_d = three_d.transpose(-1, -2)
    return a, b, two_c, three_d

class NaturalCubicSpline:
    """Calculates the natural cubic spline approximation to the batch of controls given. Also calculates its derivative.

    Example:
        times = torch.linspace(0, 1, 7)
        # (2, 1) are batch dimensions. 7 is the time dimension (of the same length as t). 3 is the channel dimension.
        X = torch.rand(2, 1, 7, 3)
        coeffs = natural_cubic_spline_coeffs(times, X)
        # ...at this point you can save the coeffs, put them through PyTorch's Datasets and DataLoaders, etc...
        spline = NaturalCubicSpline(times, coeffs)
        t = torch.tensor(0.4)
        # will be a tensor of shape (2, 1, 3), corresponding to batch and channel dimensions
        out = spline.derivative(t)
    """

    def __init__(self, times, coeffs, **kwargs):
        """
        Arguments:
            times: As was passed as an argument to natural_cubic_spline_coeffs.
            coeffs: As returned by natural_cubic_spline_coeffs.
        """
        super(NaturalCubicSpline, self).__init__(**kwargs)

        a, b, two_c, three_d = coeffs

        self._times = times
        self._a = a
        self._b = b
        # as we're typically computing derivatives, we store the multiples of these coefficients that are more useful
        self._two_c = two_c
        self._three_d = three_d

    def _interpret_t(self, t):
        maxlen = self._b.size(-2) - 1
        index = (t > self._times).sum() - 1
        index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
        # will never access the last element of self._times; this is correct behaviour
        fractional_part = t - self._times[index]
        return fractional_part, index

    def evaluate(self, t):
        """Evaluates the natural cubic spline interpolation at a point t, which should be a scalar tensor."""
        fractional_part, index = self._interpret_t(t)
        inner = 0.5 * self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part / 3
        inner = self._b[..., index, :] + inner * fractional_part
        return self._a[..., index, :] + inner * fractional_part

    def derivative(self, t):
        """Evaluates the derivative of the natural cubic spline at a point t, which should be a scalar tensor."""
        fractional_part, index = self._interpret_t(t)
        inner = self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part
        deriv = self._b[..., index, :] + inner * fractional_part
        return deriv

# ChebnetII_pro
def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

class ChebnetII_prop(MessagePassing):
    def __init__(self, K, Init=False, bias=True, **kwargs):
        super(ChebnetII_prop, self).__init__(aggr='add', **kwargs)
        
        self.K = K
        self.temp = Parameter(torch.Tensor(self.K+1))
        self.Init=Init
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

        if self.Init:
            for j in range(self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                self.temp.data[j] = x_j**2
        
    def forward(self, x, edge_index,edge_weight=None):
        coe_tmp=F.relu(self.temp)
        coe=coe_tmp.clone()
        
        for i in range(self.K+1):
            coe[i]=coe_tmp[0]*cheby(i,math.cos((self.K+0.5)*math.pi/(self.K+1)))
            for j in range(1,self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
            coe[i]=2*coe[i]/(self.K+1)


        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))

        #L_tilde=L-I
        edge_index_tilde, norm_tilde= add_self_loops(edge_index1,norm1,fill_value=-1.0,num_nodes=x.size(self.node_dim))

        Tx_0=x
        Tx_1=self.propagate(edge_index_tilde,x=x,norm=norm_tilde,size=None)

        out=coe[0]/2*Tx_0+coe[1]*Tx_1

        for i in range(2,self.K+1):
            Tx_2=self.propagate(edge_index_tilde,x=Tx_1,norm=norm_tilde,size=None)
            Tx_2=2*Tx_2-Tx_0
            out=out+coe[i]*Tx_2
            Tx_0,Tx_1 = Tx_1, Tx_2
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

# EARTH
def graph_norm_ours(A, batch=False, self_loop=True, symmetric=True):
	# A = A + I    A: (bs, num_nodes, num_nodes
    # Degree
    d = A.sum(-1) # (bs, num_nodes) #[1000, m+1]
    if symmetric:
		# D = D^-1/2
        d = torch.pow(d, -0.5)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A).bmm(D)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A).mm(D)
    else:
		# D=D^-1
        d = torch.pow(d,-1)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A)
        else:
            D =torch.diag(d)
            norm_A = D.mm(A)

    return norm_A

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 or len(sparse_mx.col)==0:
        print(sparse_mx.row,sparse_mx.col)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # return torch.sparse.FloatTensor(indices, values, shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32, device=values.device)

def normalize_adj2(adj):
    """Symmetrically normalize adjacency matrix."""
    # print(adj.shape)
    # adj += sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    eps = 1e-12
    d_inv_sqrt = np.power((rowsum + eps), -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

class FinalTanh_f(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        gain = 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()
        
        # z = self.dropout(z)

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        
        # z: torch.Size([64, 207, 32])
        # self.linear_out(z): torch.Size([64, 207, 64])
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)    
        z = torch.tanh(z)
        # z = self.dropout2(z)
        return z

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SIRLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        gain = 0.1
        self.in_features = in_features
        self.out_features = out_features
        # 我们需要三组权重来分别处理 S, I, R 的更新
        self.weight_s = Parameter(torch.Tensor(in_features *2, out_features))
        self.weight_i = Parameter(torch.Tensor(in_features * 2, out_features))
        self.weight_r = Parameter(torch.Tensor(out_features, out_features))
        nn.init.xavier_uniform_(self.weight_s, gain=gain)
        nn.init.xavier_uniform_(self.weight_i, gain=gain)
        nn.init.xavier_uniform_(self.weight_r, gain=gain)
        self.dropout = 0.4

        if bias:
            self.bias_s = Parameter(torch.Tensor(out_features))
            self.bias_i = Parameter(torch.Tensor(out_features))
            self.bias_r = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias_s.size(0))
            self.bias_s.data.uniform_(-stdv, stdv)
            self.bias_i.data.uniform_(-stdv, stdv)
            self.bias_r.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias_s', None)
            self.register_parameter('bias_i', None)
            self.register_parameter('bias_r', None)

    def forward(self, s, i, r, adj):
        eps = 1.0
        with torch.no_grad(): 
            for w in (self.weight_s, self.weight_i, self.weight_r):
                w.clamp_(-eps, eps)
        # 加入噪声项
        noise_s = torch.randn_like(s) * 0.01
        noise_i = torch.randn_like(i) * 0.01
        
        if self.training:
            support_s = torch.matmul(torch.cat([s, adj @ i], dim=-1), self.weight_i) 
            support_i = torch.matmul(torch.cat([s, adj @ i], dim=-1), self.weight_i) - torch.matmul(i, self.weight_r) 
            support_r = torch.matmul(i, self.weight_r)
        else:
            support_s = torch.matmul(torch.cat([s, adj @ i], dim=-1), self.weight_i) 
            support_i = torch.matmul(torch.cat([s, adj @ i], dim=-1), self.weight_i) - torch.matmul(i, self.weight_r) 
            support_r = torch.matmul(i, self.weight_r)

        for t in (support_s, support_i, support_r):
            t.clamp_(-1e3, 1e3)
        
        s =  - F.dropout(support_s, self.dropout, training=self.training)


        i = F.dropout(support_i, self.dropout, training=self.training)


        r = F.dropout(support_r, self.dropout, training=self.training)
        with torch.no_grad():
            for t in (s, i, r):
                t.nan_to_num_(nan=0.0, posinf=1e3, neginf=-1e3)
        return s, i, r

class SIRGCN(MessagePassing):
    def __init__(self, embedding_size):
        super().__init__(aggr='add')
        self.embedding_size = embedding_size
        #self.toS = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.toI = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.toR = nn.Linear(self.embedding_size, self.embedding_size)


    def forward(self, s, i, r, edge_index, edge_weight):
        # x has shape [N, 3, embedding_size], where the second dim represents S, I, R
        return self.propagate(edge_index, edge_weight=edge_weight, s=s, x=i, r=r)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.reshape(-1,1)

    def update(self, neighbor_i, s, x, r):
        s1 =  - self.toI(torch.cat([s, neighbor_i], dim=1))   # i.shape = 边数x节点数xembedding
        i1 = + self.toI(torch.cat([s, neighbor_i], dim=1)) - self.toR(x)
        r1 = self.toR(x)
        return s1, i1, r1

class ChebNetII(nn.Module):
    def __init__(self, num_features, hidden , K  = 6):
        super(ChebNetII, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, hidden)
        self.prop1 = ChebnetII_prop(K)

        self.dprate = 0.5
        self.dropout = 0.5
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, X, edge_index):
        x = X
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        
        return x

class EdgeWeight(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EdgeWeight, self).__init__()
        self.f_e = nn.Sequential(
            nn.Linear(input_dim , hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
        self.f_self = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
        self.n_hidden = hidden_dim
        
        half_hid = int(self.n_hidden/2)
        self.V = Parameter(torch.Tensor(half_hid))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.b1 = Parameter(torch.Tensor(half_hid))
        self.W2 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.act = F.elu
        
    def forward(self, z_i, z_j):
        # Concatenate z_i and z_j and pass through f_e
        z_ij = torch.cat([z_i @ self.W1.t(), z_j @ self.W2.t()], dim=-1)
        
        edge_weight = self.f_e(z_ij)
        
        return edge_weight.squeeze(-1)

class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, tanhalpha=1):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding):
        # embedding [batchsize, hidden_dim]
        nodevec1 = self.linear1(embedding)
        nodevec2 = self.linear2(embedding)
        nodevec1 = self.alpha * nodevec1
        nodevec2 = self.alpha * nodevec2
        nodevec1 = torch.tanh(nodevec1)
        nodevec2 = torch.tanh(nodevec2)
        
        adj = torch.bmm(nodevec1, nodevec2.permute(0, 2, 1))-torch.bmm(nodevec2, nodevec1.permute(0, 2, 1))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj

class VectorField_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type, adj):
        super(VectorField_g, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        
        #FIXME:
        # self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4
        self.linear_out_z = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4
        self.linear_out_s = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4
        self.linear_out_i = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4
        self.linear_out_r = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4

        self.m = num_nodes
        self.n_hidden = hidden_channels
        
        half_hid = int(self.n_hidden/2)
        self.V = Parameter(torch.Tensor(half_hid))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.b1 = Parameter(torch.Tensor(half_hid))
        self.W2 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.act = F.elu
        # self.Wb = Parameter(torch.Tensor(self.m,self.m))
        # self.wb = Parameter(torch.Tensor(1))
        self.Wb  = Parameter(torch.empty(self.m, self.m))
        self.wb  = Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.Wb, gain=0.1)
        # self.Wb1 = Parameter(torch.Tensor(self.m,self.m))
        # self.wb1 = Parameter(torch.Tensor(1))
        self.Wb1  = Parameter(torch.empty(self.m, self.m))
        self.wb1  = Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.Wb1, gain=0.1)
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))

        self.conv1 = GraphConvLayer(self.n_hidden, self.n_hidden) # self.k
        self.conv2 = GraphConvLayer(self.n_hidden, self.n_hidden)
        
        self.dropout = 0.5
        
        self.SIR_GCN = SIRGCN(self.n_hidden)
        self.sir = SIRLayer(self.n_hidden, self.n_hidden)
        
        self.EdgeWeightMLP = EdgeWeight(self.n_hidden, self.n_hidden, 1)
        
        self.ChebnetII = ChebNetII(self.n_hidden, self.n_hidden)
        
        self.weight = Parameter(torch.Tensor(self.n_hidden, self.n_hidden))
        nn.init.xavier_uniform_(self.weight)
        
        self.bias = Parameter(torch.Tensor(self.n_hidden))
        stdv = 1. / math.sqrt(self.bias.size(0))
        self.bias.data.uniform_(-stdv, stdv)
        
        self.graph_gen = GraphLearner(self.n_hidden)
        
        self.position = 16
        
        self.WQ = nn.Linear(self.n_hidden, self.n_hidden // 2)
        self.WK = nn.Linear(self.n_hidden, self.n_hidden // 2)
        
        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.dropout1 = nn.Dropout(0.2)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z_list):
        t = z_list[0]
        
        z = z_list[1]
        
        b = z.size(0)
        
        a_mx = F.normalize(self.graph_gen(z), p=2, dim=1, eps=1e-12, out=None)
        
        adjs = self.adj.repeat(b,1)
        
        adjs = adjs.view(b, self.m, self.m)
        
        c = torch.sigmoid(a_mx @ self.Wb + self.wb)
        
        a_mx = adjs * c + a_mx * (1-c)
        
        Adj_soft = F.softmax(a_mx, dim=2)
        
        z = self.agc(z, Adj_soft, self.fused_adj)
        
        s, i, r = z_list[2:]
        
        s, i , r = self.sir(s, i, r, Adj_soft)

        z = self.linear_out_z(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        # z = F.dropout(z)
        
        s = self.linear_out_s(s).view(*s.shape[:-1], self.hidden_channels, self.hidden_channels)
        s = s.tanh()
        # s = F.dropout(s)
        
        i = self.linear_out_i(i).view(*i.shape[:-1], self.hidden_channels, self.hidden_channels)
        i = i.tanh()
        # i = F.dropout(i)
        
        r = self.linear_out_r(r).view(*r.shape[:-1], self.hidden_channels, self.hidden_channels)
        r = r.tanh()
        # r = F.dropout(r)
        
        return [z, s, i , r] #torch.Size([64, 307, 64, 1])

    def agc(self, z, adj, origin_adj, power=1):
        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """
        
        z = self.linear_in(z)
        
        z = z.relu()
        
        global_h = torch.matmul(z, self.weight)
        
        for i in range(power):
            global_h = origin_adj @ global_h
            
        x = z + global_h

        x = F.dropout(x, self.dropout, training=self.training)
        
        z = x
        return z

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch_size, num_nodes = query.size(0), query.size(1)

        # Project inputs to the multi-head space
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, num_nodes, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(2, 3)
        key = key.view(batch_size, num_nodes, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(2, 3)
        value = value.view(batch_size, num_nodes, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(2, 3)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted average
        context = torch.matmul(attn_weights, value)
        
        # Reshape back to [batch_size, num_nodes, seq_len, embed_dim]
        context = context.transpose(2, 3).contiguous().view(batch_size, num_nodes, -1, self.embed_dim)
        
        # Project the output back to the original embedding dimension
        output = self.out_proj(context)
        
        return output, attn_weights

class MultiViewFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout = 0.5):
        super(MultiViewFusion, self).__init__()
        self.cross_attention = CrossAttention(embed_dim, num_heads, dropout)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, semantic_feature, multi_view_features):
        # Apply cross-attention
        fusion_output, attn_weights = self.cross_attention(semantic_feature, multi_view_features, multi_view_features)
        
        # Apply residual connection, normalization, and dropout
        fusion_output = self.fc(fusion_output) + semantic_feature
        fusion_output = self.dropout(fusion_output)
        
        return fusion_output, attn_weights

class EARTH(BaseModel):
    def __init__(self, num_timesteps_input, num_timesteps_output, dtw_matrix, adj_m=None, num_nodes=None, num_features=1, dropout=0.2, n_hidden=16, device="cpu", use_future_ti=False, tid_sizes=None, emb_dim=4, ti_hidden=(16,), node_specific=True, **kwargs): 
        if num_nodes is None and adj_m is not None:
            num_nodes = adj_m.shape[0]
        super().__init__(tid_sizes=tid_sizes,
                         device=device,
                         use_future_ti=use_future_ti,
                         emb_dim=emb_dim,
                         ti_hidden=ti_hidden,
                         node_specific=node_specific,
                         num_nodes=num_nodes)
        self.x_h = num_features # 1
        self.m = num_nodes
        self.w = num_timesteps_input
        self.h = num_timesteps_output
        self.adj = adj_m
        #adj转换为edge_index
        self.o_adj = self.adj
        rowsum = 1. / torch.sqrt(self.adj.sum(dim=0))
        self.adj = rowsum[:, np.newaxis] * self.adj * rowsum[np.newaxis, :]
        self.adj = Variable(self.adj)
        self.dtw_matrix = dtw_matrix
        self.device = device
        if self.device == 'cpu':
            self.sem_mask = torch.ones(self.m, self.m)
        else:
            self.sem_mask = torch.ones(self.m, self.m).cuda()
            self.adj = self.adj.cuda()
            self.o_adj = self.o_adj.cuda()
        
        sem_mask = self.dtw_matrix.argsort(axis=1)[:, :(self.m - 5)]
        for i in range(self.sem_mask.shape[0]):
            self.sem_mask[i][sem_mask[i]] = 0
        
        edge_index, _ = dense_to_sparse(self.o_adj)
        
        if self.device == 'cpu':
            self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(self.o_adj.cpu().numpy())).to_dense()
            self.edge_index = edge_index
        else:
            self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(self.o_adj.cpu().numpy())).to_dense().cuda()
            self.edge_index = edge_index.cuda()
        self.dropout = dropout
        self.n_hidden = n_hidden
        
        self.d_model = n_hidden
        
        self.d_state = 16
        
        self.Wb = Parameter(torch.Tensor(self.m,self.m))
        self.wb = Parameter(torch.Tensor(1))
        #667
        
        self.residual_window = 4
        if (self.residual_window > 0):
            self.residual = nn.Linear(self.residual_window, 1);
        
        self.init_type = "fc"
        self.atol = 1e-9
        self.rtol = 1e-7
        self.solver = "rk4"
        self.input_dim = 2
        
        self.vector_field_f = FinalTanh_f(input_channels=self.input_dim , hidden_channels=n_hidden,
                                        hidden_hidden_channels=n_hidden,
                                        num_hidden_layers=1)
        self.vector_field_g = VectorField_g(input_channels=self.input_dim , hidden_channels=n_hidden,
                                        hidden_hidden_channels=n_hidden,
                                        num_hidden_layers=2, num_nodes=self.m, cheb_k=3, embed_dim=n_hidden,
                                        g_type="agc", adj = self.o_adj)
        
        self.vector_field_g.orig_adj = self.o_adj
        
        self.vector_field_g.norm_adj = graph_norm_ours(self.o_adj,batch=False)

        self.vector_field_g.adj = self.adj
        self.vector_field_g.m = self.m
        
        if self.init_type == 'fc':
            self.initial_h = torch.nn.Linear(self.input_dim, n_hidden)
            self.initial_z = torch.nn.Linear(self.input_dim, n_hidden)
            self.initial_s = torch.nn.Linear(self.input_dim, n_hidden)
            self.initial_i = torch.nn.Linear(self.input_dim, n_hidden)
            self.initial_r = torch.nn.Linear(self.input_dim, n_hidden)

        self.final_project = nn.Sequential(
            nn.Linear(self.d_model * 2 , self.h)
        )
        
        self.dtw = self.sem_mask
        edge_index, _ = dense_to_sparse(self.dtw)
        if self.device == 'cpu':
            self.dtw_edge =  edge_index
        else:
            self.dtw_edge =  edge_index.cuda()
        
        self.vector_field_g.dtw_edge = self.dtw_edge
        
        self.vector_field_g.dtw = self.sem_mask

        # 将A_1和A_2转换为布尔类型
        A_1_bool = self.o_adj.bool()
        A_2_bool = self.sem_mask.bool()

        # 计算 A_2 中存在而 A_1 中不存在的连接
        difference = A_2_bool & ~A_1_bool
        # 将difference转换回浮点类型
        difference = difference.float()

        # 更新 A_1
        A_new = self.o_adj + difference
        
        self.vector_field_g.fused_orig_adj = A_new
        if self.device == 'cpu':
            self.vector_field_g.fused_adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(A_new.cpu().numpy())).to_dense()
            self.vector_field_g.dtw_norm = sparse_mx_to_torch_sparse_tensor(normalize_adj2(self.dtw.fill_diagonal_(1).cpu().numpy())).to_dense()
        else:
            self.vector_field_g.fused_adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(A_new.cpu().numpy())).to_dense().cuda()
            self.vector_field_g.dtw_norm = sparse_mx_to_torch_sparse_tensor(normalize_adj2(self.dtw.fill_diagonal_(1).cpu().numpy())).to_dense().cuda()
        
        self.vector_field_g.edge = self.edge_index
        
        self.multi_view_fusion = MultiViewFusion(n_hidden, 4)
        
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, X, adj, states=None, dynamic_adj=None):
        '''
        Args:  x: (batch, time_step, m) -- batch number, window, location number 
            feat: [batch, window, dim, m]
        Returns: (batch, m)
        '''
        X = X[:, :, :, 0]
        times = torch.linspace(0, X.size(1)-1, X.size(1))
        if self.device != 'cpu':
            times = times.cuda()
        augmented_X_tra = [times.unsqueeze(0).unsqueeze(0).repeat(X.size(0), X.size(2), 1).unsqueeze(-1).transpose(1,2)]
        augmented_X_tra.append(torch.Tensor(X[..., :]).unsqueeze(-1))
        X = torch.cat(augmented_X_tra, dim=3)
        coeffs = natural_cubic_spline_coeffs(times, X.transpose(1,2))
        if self.device != 'cpu':
            coeffs = tuple(c.cuda() for c in coeffs)
        spline = NaturalCubicSpline(times, coeffs)
        
        if self.init_type == 'fc':
            h0 = self.initial_h(spline.evaluate(times[0]))
            z0 = self.initial_z(spline.evaluate(times[0]))
            s0 = self.initial_s(spline.evaluate(times[0]))
            i0 = self.initial_i(spline.evaluate(times[0]))
            r0 = self.initial_r(spline.evaluate(times[0]))       
        elif self.init_type == 'conv':
            h0 = self.start_conv_h(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()
            z0 = self.start_conv_z(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()

        adjs  = adj.repeat(X.shape[0],1)
        adjs = adjs.view(X.shape[0], self.m, self.m)
        
        output_list = cdeint_gde_dev(dX_dt=spline.derivative, #dh_dt
                                   h0= h0,
                                   z0= z0,
                                   s0 = s0,
                                   i0 = i0,
                                   r0 = r0,
                                   adjs = adjs,
                                   func_f=self.vector_field_f,
                                   func_g=self.vector_field_g,
                                   t=times,
                                   method=self.solver,
                                   atol=self.atol,
                                   rtol=self.rtol)

        temporal_out = [i[-1] for i in output_list]
        
        t_t = temporal_out[0]
        
        z_t = temporal_out[1].unsqueeze(-2)
        
        multi_view_features = torch.stack(temporal_out[2:-1], dim=-2)
        
        fused_feature, attn_weights = self.multi_view_fusion(z_t, multi_view_features)
        
        fused_feature = fused_feature.squeeze(-2)
        
        x_out = self.final_project(torch.cat([fused_feature,t_t],dim=-1))
        
        outputs = x_out.permute(0, 2, 1).contiguous()
        
        self.ratio = 0.5
        
        if (self.residual_window > 0):
            z = X[:, -self.residual_window:, :, 0]
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window)
            z = self.residual(z)
            z = z.view(-1,self.m)
            z = z.unsqueeze(1).expand(-1, self.h, -1)
            outputs = outputs * self.ratio + z

        return outputs
    
    def initialize(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()