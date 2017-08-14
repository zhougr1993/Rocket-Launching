from nested_dict import nested_dict
from collections import OrderedDict
from functools import partial
import torch
import torch.cuda.comm as comm
from torch.autograd import Variable
from torch.nn.init import kaiming_normal
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
import torch.nn.functional as F


def distillation(y, teacher_scores, labels, T, alpha):
    return F.kl_div(F.log_softmax(y / T), F.softmax(teacher_scores / T)) * (T * T * 2. * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


def rocket_distillation(y, teacher_scores, labels, T, alpha):
    return F.kl_div(F.log_softmax(y / T), F.softmax(teacher_scores / T)) * (T * T * 2. * alpha)


def normalize(input, p=2, dim=1, eps=1e-12):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation
        dim (int): the dimension to reduce
        eps (float): small value to avoid division by zero
    """
    return input / torch.norm(input, p, dim).clamp(min=eps).expand_as(input)


def at(x):
    return normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k, v in params.items()}
    else:
        return getattr(params.cuda(), dtype)()


def conv_params(ni, no, k=1):
    return cast(kaiming_normal(torch.Tensor(no, ni, k, k)))


def linear_params(ni, no):
    return cast({'weight': kaiming_normal(torch.Tensor(no, ni)), 'bias': torch.zeros(no)})


def bnparams(n):
    return cast({'weight': torch.rand(n), 'bias': torch.zeros(n)})


def bnstats(n):
    return cast({'running_mean': torch.zeros(n), 'running_var': torch.ones(n)})


def data_parallel(f, input, params, stats, mode, device_ids, output_device=None):
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, stats, mode)

    def replicate(param_dict, g):
        replicas = [{} for d in device_ids]
        for k, v in param_dict.iteritems():
            for i, u in enumerate(g(v)):
                replicas[i][k] = u
        return replicas

    params_replicas = replicate(params, lambda x: Broadcast(device_ids)(x))
    stats_replicas = replicate(stats, lambda x: comm.broadcast(x, device_ids))

    replicas = [partial(f, params=p, stats=s, mode=mode)
                for p, s in zip(params_replicas, stats_replicas)]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten_params(params):
    return OrderedDict(('.'.join(k), Variable(v, requires_grad=True))
                       for k, v in nested_dict(params).iteritems_flat() if v is not None)


def flatten_stats(stats):
    return OrderedDict(('.'.join(k), v)
                       for k, v in nested_dict(stats).iteritems_flat())


def batch_norm(x, params, stats, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=stats[base + '.running_mean'],
                        running_var=stats[base + '.running_var'],
                        training=mode)
