#!/usr/bin/env python

import torch
from torch.nn.parameter import Parameter

class BoundedParameter(torch.Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.
    This constructor returns a Parameter, but adds definable upper and lower bounds.

    Arguments:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details. Default: `True`
        lower_bound (float, optional): lower bound for the values.
            This is observed only by bounds-aware optimizers.
        upper_bound (float, optional): upper bound for the values.
            This is observed only by bounds-aware optimizers.
    """
    def __new__(cls, data=None, requires_grad=True, lower_bound=None, upper_bound=None):
        if data is None:
            data = torch.Tensor()
        p = torch.Tensor._make_subclass(Parameter, data, requires_grad)
        p.lower_bound = lower_bound
        p.upper_bound = upper_bound
        return p


class ParamBoundEnforcer(object):
    """Wrapper for any optimizer that enforces parameter bounds."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def enforce(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if hasattr(p, 'lower_bound'):
                    lower = getattr(p, 'lower_bound')
                    upper = getattr(p, 'upper_bound')
                    if lower is not None or upper is not None:
                        p.data.clamp_(min=lower, max=upper)
