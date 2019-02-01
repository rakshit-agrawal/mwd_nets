#!/usr/bin/env python

# Implementation of sigmoid net.

import torch

from json_plus import Serializable, Storage
from net_generic import GenericNet
from unit_linear_extended import LinearExtended


class Sigmoid(GenericNet):
    """Standard neural net, which can be sigmoid or RELU, depending on flags."""

    best_loss = 'sdl'
    bounded_parameters = False
    default_lr = 1.0
    default_optim = 'adadelta'

    def __init__(self, args, empty=False):
        super(Sigmoid, self).__init__()
        self.args = args
        layer_sizes = map(int, args.layers.split(',')) + [args.n_classes]  # 10 for the output.
        self.layers = []
        if not empty:
            previous_size = args.input_channels * args.input_y_size * args.input_x_size
            for i, size in enumerate(layer_sizes):
                l = LinearExtended(previous_size, size)
                self.layers.append(l)
                self.add_module('layer_%d' % i, l)
                previous_size = size

    def forward(self, x):
        # Flatten the input for fully connected layers instead of Conv
        x = x.flatten(1)
        for i, l in enumerate(self.layers):
            x = torch.sigmoid(l(x))
        return x

    def interval_forward(self, x_min, x_max):
        """Forwards an interval through the net, to test for robustness."""
        x_min, x_max = x_min.flatten(1), x_max.flatten(1)
        for i, l in enumerate(self.layers):
            x_min, x_max = l.interval_forward(x_min, x_max)
            x_min = torch.sigmoid(x_min)
            x_max = torch.sigmoid(x_max)
        return x_min, x_max

    def sensitivity(self):
        s = None
        for l in self.layers:
            s = l.sensitivity(s)
        return torch.max(s)

    def regularization(self):
        """This function is purely for homogeneity with MWDNet."""
        return torch.tensor(0)

    def set_regular_deriv(self, b):
        """This function is purely for homogeneity with MWDNet."""
        pass

    # Serialization.
    def dumps(self):
        d = dict(
            args=self.args.__dict__,
            layers=[l.dumps() for l in self.layers]
        )
        return Serializable.dumps(d)

    @staticmethod
    def loads(s, device):
        d = Serializable.loads(s)
        args = dict(n_classes=10)
        args.update(d['args'])
        args = Storage(args)
        m = Sigmoid(args, empty=True)
        for i, ms in enumerate(d['layers']):
            l = LinearExtended.loads(ms, device)
            m.layers.append(l)
            m.add_module('layer_%d' % i, l)
        return m


default_class = Sigmoid
