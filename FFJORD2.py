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
from util import torch_to, preprocess_input


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

    def forward(self, x, logpx=None, integration_times=None, reverse=False):
        # integration_times arg is not need
        if reverse:
            return self._reverse(x, logpx)
        else:
            return self._forward(x, logpx)

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

    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq

    def divergence_bf(self, dx, y):
        sum_diag = 0.
        for i in range(y.shape[1]):
            sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
        return sum_diag.contiguous()

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]
        #t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]
        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            divergence = self.divergence_bf(dy, y).view(batchsize, 1)
        return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])


class CNF(nn.Module):
    def __init__(self, odefunc):
        super(CNF, self).__init__()
        self.odefunc = odefunc

    def _flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def forward(self, z, logpz=None, integration_times=None, reverse=False):

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, 1.0]).to(z)
        if reverse:
            integration_times = self._flip(integration_times, 0)

        state_t = odeint(self.odefunc, (z, _logpz), integration_times.to(z), atol=1e-5, rtol=1e-5, method='dopri5')

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)
        z_t, logpz_t = state_t[:2]

        if logpz is not None:
            return z_t, logpz_t
        else:
            return z_t


class SequentialFlow(nn.Module):
    """ A generalized nn.Sequential CNF"""

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
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        self.load_state_dict(torch.load(fn, map_location=map_location))

    # integration_times added
    def forward(self, x, logpx=None, integration_times=None, reverse=False, inds=None):
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
            for i in inds:
                x, logpx = self.chain[i](x, logpx, integration_times=integration_times, reverse=reverse)
            return x, logpx


def build_cnf(input_dims: int, hidden_dims: int, num_hidden_layers: int, num_blocks: int, fn: str):
    """
    build cnf model

    @param
    input_dims: int, dims of input
    hidden_dims: int, dims of hidden layer
    num_hidden_layers: int num of hidden layers
    num_blocks: int, num of cnf blocks
    fn: str, file name of loss history and model state
    """
    flow = []
    for i in range(num_blocks):
        diffeq = ODEnet(input_dims=input_dims, hidden_dims=hidden_dims, num_hidden_layers=num_hidden_layers)
        odefunc = ODEfunc(diffeq)
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
    """ 2d standard normal, sum over the second dimension. """
    return (-np.log(2 * np.pi) - 0.5 * z.pow(2)).sum(1, keepdim=True)


def train_cnf(model, x_all, epochs):
    # preprocess
    model = torch_to(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    batch_size = max(map(len, x_all))
    dataset = preprocess_input(*x_all)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_history = np.full(shape=(len(x_all), epochs), fill_value=np.nan)
    min_loss = np.full(shape=len(x_all), fill_value=1e5)
    # start training
    for epoch in tqdm(range(epochs)):
        x_samples = next(iter(train_loader))
        x_samples = torch_to(*x_samples)  # len(x_samples) == len(x_all)
        x_samples = [x_samples] if not isinstance(x_samples, list) else x_samples  # for len(x_all) == 1
        for j, x_sample in enumerate(x_samples):  # x_sample.shape = batch_size * ndim
            optimizer.zero_grad()
            # Transform the original samples to base distribution samples.
            integration_times = torch.tensor([0.0, j+1])
            zero = torch.zeros(x_sample.shape[0], 1).to(x_sample)
            z_t0, delta_logp = model(x_sample, logpx=zero, integration_times=integration_times, reverse=False, inds=None)
            # Calculate a loss
            logp_t0 = standard_normal_logprob(z_t0)
            logp_t = logp_t0 - delta_logp
            loss = -torch.mean(logp_t)
            loss.backward()
            optimizer.step()
            loss_history[j, epoch] = loss.item() / batch_size
        pd.to_pickle(loss_history, 'lh_{}.pkl'.format(model.fn))
        if np.all(min_loss > loss_history[:, epoch]):
            min_loss = loss_history[:, epoch].copy()
            model.save_state()
    return model, loss_history


def sampling_from_cnf(model: nn.Module, n_samples: int, t_end: float):
    torch.manual_seed(1234)
    z_samples = torch.randn(n_samples, model.input_dims)
    logp_samples = standard_normal_logprob(z_samples)
    z_samples, logp_samples, model = torch_to(z_samples, logp_samples, model)
    with torch.no_grad():
        integration_times = torch_to(torch.tensor([0.0, t_end]))
        z_traj, _ = model(z_samples, logp_samples, integration_times, reverse=True)
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


def compute_each_time_loss(model, x_all):
    model = torch_to(model)
    dataset = preprocess_input(*x_all)
    all_loss = []
    for j, data in enumerate(dataset.tensors):
        data = torch_to(data)
        integration_times = torch.tensor([0.0, j+1])
        zero = torch.zeros(data.shape[0], 1).to(data)
        z_t0, delta_logp = model(data, logpx=zero, integration_times=integration_times, reverse=False, inds=None)
        # Calculate a loss
        logp_t0 = standard_normal_logprob(z_t0)
        logp_t = logp_t0 - delta_logp
        loss = -torch.mean(logp_t)
        all_loss.append(loss.item())
    return all_loss


# -

if __name__ == '__main__':
    import datasets2
    x_all, y_all = datasets2.make_gradual_data(steps=3)
    x_eval, y_eval = x_all.pop(), y_all.pop()
    model = build_cnf(input_dims=2, hidden_dims=64, num_hidden_layers=3, num_blocks=3, fn='rot-moon')
    model, lh = train_cnf(model, x_all, 5000)


