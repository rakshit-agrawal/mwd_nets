#!/usr/bin/env python

# Abstract class for nets.

from torch import nn
from json_plus import Serializable


class GenericNet(nn.Module, Serializable):
    """Generic Neural net class architecture """

    best_loss = None # FIXME
    bounded_parameters = False # Uses optimizer that is aware of parameter bounds.

    def __init__(self):
        super(GenericNet, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def sensitivity(self):
        raise NotImplementedError

    def set_regular_deriv(self, b):
        pass

    def set_pseudo(self, b):
        pass

