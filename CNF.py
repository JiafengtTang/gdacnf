# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python [conda env:.conda-torch-env]
#     language: python
#     name: conda-env-.conda-torch-env-py
# ---

import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import Parameter
from torch.utils.data import DataLoader
from torchdiffeq import odeint_adjoint as odeint
# my module
import util


# +
class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))) + self._hyper_bias(t.view(1, 1))


class MovingBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-4, decay=0.1, bn_lag=0., affine=True):
        super(MovingBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.decay = decay
        self.bn_lag = bn_lag
        self.register_buffer('step', torch.zeros(1))
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    @property
    def shape(self):
        return [1, -1]

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.zero_()
            self.bias.data.zero_()

    def forward(self, x, logpx=None, lec=None, integration_times=None, reverse=False):
        # lec and integration_times args are not need
        res = self._reverse(x, logpx) if reverse else self._forward(x, logpx)
        return res + tuple([lec])

    def _forward(self, x, logpx=None):
        c = x.size(1)
        used_mean = self.running_mean.clone().detach()
        used_var = self.running_var.clone().detach()

        if self.training:
            # compute batch statistics
            x_t = x.transpose(0, 1).contiguous().view(c, -1)
            batch_mean = torch.mean(x_t, dim=1)
            batch_var = torch.var(x_t, dim=1)

            # moving average
            if self.bn_lag > 0:
                used_mean = batch_mean - (1 - self.bn_lag) * (batch_mean - used_mean.detach())
                used_mean /= (1. - self.bn_lag**(self.step[0] + 1))
                used_var = batch_var - (1 - self.bn_lag) * (batch_var - used_var.detach())
                used_var /= (1. - self.bn_lag**(self.step[0] + 1))

            # update running estimates
            self.running_mean -= self.decay * (self.running_mean - batch_mean.data)
            self.running_var -= self.decay * (self.running_var - batch_var.data)
            self.step += 1

        # perform normalization
        used_mean = used_mean.view(*self.shape).expand_as(x)
        used_var = used_var.view(*self.shape).expand_as(x)

        y = (x - used_mean) * torch.exp(-0.5 * torch.log(used_var + self.eps))

        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(x)
            bias = self.bias.view(*self.shape).expand_as(x)
            y = y * torch.exp(weight) + bias

        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(x, used_var).view(x.size(0), -1).sum(1, keepdim=True)

    def _reverse(self, y, logpy=None):
        used_mean = self.running_mean
        used_var = self.running_var

        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(y)
            bias = self.bias.view(*self.shape).expand_as(y)
            y = (y - bias) * torch.exp(-weight)

        used_mean = used_mean.view(*self.shape).expand_as(y)
        used_var = used_var.view(*self.shape).expand_as(y)
        x = y * torch.exp(0.5 * torch.log(used_var + self.eps)) + used_mean

        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x, used_var).view(x.size(0), -1).sum(1, keepdim=True)

    def _logdetgrad(self, x, used_var):
        logdetgrad = -0.5 * torch.log(used_var + self.eps)
        if self.affine:
            weight = self.weight.view(*self.shape).expand(*x.size())
            logdetgrad += weight
        return logdetgrad

    def __repr__(self):
        return (
            '{name}({num_features}, eps={eps}, decay={decay}, bn_lag={bn_lag},'
            ' affine={affine})'.format(name=self.__class__.__name__, **self.__dict__)
        )


def stable_var(x, mean=None, dim=1):
    if mean is None:
        mean = x.mean(dim, keepdim=True)
    mean = mean.view(-1, 1)
    res = torch.pow(x - mean, 2)
    max_sqr = torch.max(res, dim, keepdim=True)[0]
    var = torch.mean(res / max_sqr, 1, keepdim=True) * max_sqr
    var = var.view(-1)
    # change nan to zero
    var[var != var] = 0
    return var


class ODEnet(nn.Module):
    """
    @param
    input_dims: int, dims of input
    hidden_dims: int, dims of hidden layer
    num_hidden_layers: int num of hidden layers
    """

    def __init__(self, input_dims, hidden_dims=64, num_hidden_layers=3):
        super(ODEnet, self).__init__()
        # build layers and add them
        layers, activation_fns = list(), list()
        dims = [(input_dims, hidden_dims)] + [(hidden_dims, hidden_dims)] * num_hidden_layers + [(hidden_dims, input_dims)]
        for d in dims:
            layers.append(ConcatSquashLinear(dim_in=d[0], dim_out=d[1]))
            activation_fns.append(nn.Softplus())
        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y):
        dx = y
        for i, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if i < len(self.layers) - 1:
                dx = self.activation_fns[i](dx)
        return dx


class ODEfunc(nn.Module):

    def __init__(self, diffeq, divergence_fn):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.register_buffer("_num_evals", torch.tensor(0.))
        if divergence_fn == "brute_force":
            self.divergence_fn = self.divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = self.divergence_approx

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

#     def divergence_bf(self, dx, y, training):
#         sum_diag = 0.
#         for i in range(y.shape[1]):
#             sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=training)[0].contiguous()[:, i].contiguous()
#         return sum_diag.contiguous()

    def divergence_bf(self, dx, y, training=True, **unused_kwargs):
        sum_diag = 0.
        for i in range(y.shape[1]):
            sum_diag += torch.autograd.grad(dx[:, i].sum(), y, retain_graph=True, create_graph=training)[0].contiguous()[:, i].contiguous()
        return sum_diag.contiguous()

    def divergence_approx(self, f, y, e=None, training=True):
        e_dzdx = torch.autograd.grad(f, y, e, retain_graph=True, create_graph=training)[0]
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
        return approx_tr_dzdx

    def sample_rademacher_like(self, y):
        return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1

    def sample_gaussian_like(self, y):
        return torch.randn_like(y)

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]
        #t = torch.tensor(t).type_as(y)
        self._num_evals += 2
        batchsize = y.shape[0]

        if self._e is None:
            self._e = self.sample_rademacher_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            v = self.diffeq(t, y, *states[2:])
            # Hack for 2D data to use brute force divergence computation.
            # if not self.training and y.view(y.shape[0], -1).shape[1] == 2:
            # divergence = self.divergence_bf(v, y, training=self.training).view(batchsize, 1)
            divergence = self.divergence_fn(v, y, e=self._e, training=self.training).view(batchsize, 1)
        return tuple([v, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])

#             for s_ in states[2:]:
#                 s_.requires_grad_(True)
#             dy = self.diffeq(t, y, *states[2:])
#             divergence = self.divergence_bf(dy, y).view(batchsize, 1)
#         return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])


class CNF(nn.Module):
    def __init__(self, odefunc):
        super(CNF, self).__init__()
        self.odefunc = odefunc
        # parameters for TPR are fixed
        self.poly_num_sample = 2
        self.poly_order = 1
        self.poly_coef = 5

    def _flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def num_evals(self):
        return self.odefunc._num_evals.item()

    def sample_unique(self):
        rnd = torch.cat([torch.tensor([0.0]), torch.rand(self.poly_num_sample).sort()[0], torch.tensor([1.0])])
        while torch.any(rnd[:-1] == rnd[1:]):
            rnd = torch.cat([torch.tensor([0.0]), torch.rand(self.poly_num_sample).sort()[0], torch.tensor([1.0])])
        return rnd

    def poly_reg_error(self, t, z):
        """
        Error = ((I-G(G^T G)^{-1}G^T)) @ y)^2
        """
        T = [torch.ones_like(t)]
        for i in range(self.poly_order):
            T.append(T[-1] * (t-t[i]))
        G = torch.stack(T, -1)
        U = torch.svd(G)[0]
        mat = torch.eye(t.shape[0]).to(t) - U[:,:self.poly_order+1]@U.t()[:self.poly_order+1]
        return torch.sum(torch.einsum("ij,j...->i...", mat, z)**2)/t.shape[0]/z.shape[1]

    def forward(self, z, logpz=None, lec=None, integration_times=None, reverse=False):
        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        # modified for TPR
        #return_last = integration_times is None
        lec = None if not self.training else lec

        if integration_times is None:
            integration_times = torch.tensor([0.0, 1.0])

        if lec is not None:  # modified for TPR
            end_time = integration_times[-1]
            integration_times = self.sample_unique() * end_time

        if reverse:
            integration_times = self._flip(integration_times, 0)

        self.odefunc.before_odeint()
        integration_times = integration_times.to(z)
        state_t = odeint(self.odefunc, (z, _logpz), integration_times, atol=1e-5, rtol=1e-5, method='dopri5')

        z_t = state_t[0]  # shape -> [len(integration_times), z.shape[0], z.shape[1]]
        if lec is not None:
            state_t = tuple(s[-1] for s in state_t)
            lec = lec + self.poly_reg_error(integration_times, z_t)
            return tuple(state_t) + (lec,)
        else:
            state_t = tuple(s[1] for s in state_t)
            z_t, logpz_t = state_t[:2]
            return (z_t, logpz_t, lec)


#         if len(integration_times) == 2:
#             state_t = tuple(s[1] for s in state_t)
#         z_t, logpz_t = state_t[:2]

#         if logpz is not None:
#             return z_t, logpz_t
#         else:
#             return z_t


class SequentialFlow(nn.Module):
    """ A generalized nn.Sequential CNF """

    def __init__(self, layersList, input_dims, fn):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)
        self.input_dims = input_dims
        self.fn = fn

    def save_state(self):
        """ Save model state """
        fn = 'state_{}.tar'.format(self.fn)
        torch.save(self.state_dict(), fn)

    def load_state(self, fn=None):
        """ Load model state """
        if fn is None:
            fn = 'state_{}.tar'.format(self.fn)
        #map_location = None if torch.cuda.is_available() else torch.device('cpu')
        map_location = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.load_state_dict(torch.load(fn, map_location=map_location))

    # integration_times added
    def forward(self, x, logpx=None, lec=None, integration_times=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        else:
            # modified for TPR
            for i in inds:
                x, logpx, lec = self.chain[i](x, logpx, lec=lec, integration_times=integration_times, reverse=reverse)
            return x, logpx, lec


def build_cnf(input_dims: int, hidden_dims: int, num_hidden_layers: int, 
              num_blocks: int, fn: str, divergence_fn: str="brute_force"):
    """
    build cnf model

    @param
    input_dims: int, dims of input
    hidden_dims: int, dims of hidden layer
    num_hidden_layers: int num of hidden layers
    num_blocks: int, num of cnf blocks
    fn: str, file name of loss history and model state
    divergence_fn: str, "brute_force" or "approximate"
    """
    flow = []
    for i in range(num_blocks):
        diffeq = ODEnet(input_dims=input_dims, hidden_dims=hidden_dims, num_hidden_layers=num_hidden_layers)
        odefunc = ODEfunc(diffeq, divergence_fn)
        cnf = CNF(odefunc)
        flow.append(cnf)
    # add BatchNorm layer
    bn_layers = [MovingBatchNorm1d(input_dims) for _ in range(num_blocks)]
    bn_flow = [MovingBatchNorm1d(input_dims)]
    for a, b in zip(flow, bn_layers):
        bn_flow.append(a)
        bn_flow.append(b)
    flow = bn_flow
    model = SequentialFlow(flow, input_dims, fn)
    return model


def standard_normal_logprob(z):
    """ 2d standard normal, sum over the second dimension """
    return (-np.log(2 * np.pi) - 0.5 * z.pow(2)).sum(1, keepdim=True)


class RunningAverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self, momentum=0.93):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def count_nfe(model):

    class AccNumEvals(object):
        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, ODEfunc):
                self.num_evals += module._num_evals.item()

    accumulator = AccNumEvals()
    model.apply(accumulator)
    return accumulator.num_evals


def train_cnf(model, x_all, epochs, TPR=True):
    # preprocess
    poly_coef = 5.0
    batch_size = max(map(len, x_all))  # 500
    model = util.torch_to(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_loaders = []
    for x in x_all:
        dataset = util.preprocess_input(x)
        train_loaders.append(DataLoader(dataset.tensors[0], batch_size=batch_size, shuffle=True))
    loss_history = np.full(shape=(len(x_all), epochs), fill_value=np.nan)
    min_loss = np.full(shape=len(x_all), fill_value=1e5)
    nfe = RunningAverageMeter()  # logging NFE forward only
    # start training
    for epoch in tqdm(range(epochs), disable=False):
        for j, train_loader in enumerate(train_loaders):
            integration_times = torch.tensor([0.0, j+1])
            epoch_loss = 0
            for x_sample in train_loader:
                x_sample = util.torch_to(x_sample)
                optimizer.zero_grad()
                # Transform the original samples to base distribution samples.
                zero = torch.zeros(x_sample.shape[0], 1).to(x_sample)
                lec = torch.tensor([0.0]).to(x_sample)
                z_t0, delta_logp, lec = model(x_sample, logpx=zero, lec=lec,
                                              integration_times=integration_times,
                                              reverse=False, inds=None)
                # Calculate a loss
                logp_t0 = standard_normal_logprob(z_t0)
                logp_t = logp_t0 - delta_logp
                loss = -torch.mean(logp_t)
                if TPR:
                    loss = loss + poly_coef * lec
                nfe.update(count_nfe(model))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() / x_sample.shape[0]
            loss_history[j, epoch] = epoch_loss
        pd.to_pickle(loss_history, 'lh_{}.pkl'.format(model.fn))
        pd.to_pickle(nfe, 'nfe_{}.pkl'.format(model.fn))
        if np.all(min_loss > loss_history[:, epoch]):
            min_loss = loss_history[:, epoch].copy()
            model.save_state()
    return model, loss_history


def sampling_from_cnf(model: nn.Module, n_samples: int, t_end: float):
    torch.manual_seed(1234)
    input_dims = model.input_dims
    if isinstance(input_dims, tuple):
        size = (n_samples,) + input_dims
    else:
        size = (n_samples, input_dims)
    z_samples = torch.randn(size)
    logp_samples = standard_normal_logprob(z_samples.reshape(n_samples, -1))
    z_samples, logp_samples, model = util.torch_to(z_samples, logp_samples, model)
    model = model.eval()
    with torch.no_grad():
        integration_times = util.torch_to(torch.tensor([0.0, t_end]))
        z_traj, _, _ = model(z_samples, logpx=logp_samples, lec=None,
                             integration_times=integration_times, reverse=True)
        z_traj = z_traj.cpu().numpy()
    return z_traj


def get_generate_gradual_samples(model: nn.Module, x_all: list, alpha: float):
    num_domain = len(x_all)
    n_samples = x_all[0].shape[0]
    T = np.arange(start=1, stop=num_domain+1, step=1)
    T_gene = 1 + alpha * np.arange(start=1, stop=(num_domain-1)/alpha, step=1, dtype=int)
    T_gene = np.unique(np.hstack([T, T_gene]))
    x_all_generated = []
    # print(f'T_gene \n {T_gene}')
    for t in tqdm(T_gene):
        if np.isclose([t], T).any():
            idx = round(t) - 1
            x_all_generated.append(x_all[idx].copy())
        else:
            x_all_generated.append(sampling_from_cnf(model, n_samples, t))
    return x_all_generated


# def compute_each_time_loss(model, x_all):
#     model = util.torch_to(model)
#     dataset = util.preprocess_input(*x_all)
#     all_loss = []
#     for j, data in enumerate(dataset.tensors):
#         data = util.torch_to(data)
#         integration_times = torch.tensor([0.0, j+1])
#         zero = torch.zeros(data.shape[0], 1).to(data)
#         z_t0, delta_logp = model(data, logpx=zero, integration_times=integration_times, reverse=False, inds=None)
#         # Calculate a loss
#         logp_t0 = standard_normal_logprob(z_t0)
#         logp_t = logp_t0 - delta_logp
#         loss = -torch.mean(logp_t)
#         all_loss.append(loss.item())
#     return all_loss

