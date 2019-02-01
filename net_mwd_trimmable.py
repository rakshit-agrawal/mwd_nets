#!/usr/bin/env python

# MWD net.

import torch

from net_generic import GenericNet
from unit_mwd_trimmable import MWD_trimmable as MWD


class MWDNetTrimmable(GenericNet):
    """MWD Neural net."""

    best_loss = 'sdl'
    bounded_parameters = True # Uses optimizer that is aware of parameter bounds.
    default_lr = 5.0
    default_optim = 'adadelta'

    def __init__(self, args, empty=False):
        super(MWDNetTrimmable, self).__init__()
        self.args = args
        layer_sizes = map(int, args.layers.split(',')) + [args.n_classes]  # 10 for the output.
        self.layers = []
        if not empty:
            previous_size = args.input_channels * args.input_y_size * args.input_x_size
            for i, size in enumerate(layer_sizes):
                l = MWD(previous_size, size,
                        andor=args.andor[i],
                        pseudo=1.0,
                        min_slope=args.min_slope,
                        max_slope=args.max_slope)
                self.layers.append(l)
                self.add_module('layer_%d' % i, l)
                previous_size = size

    def forward(self, x):
        # Flatten the input for fully connected layers instead of Conv
        x = x.flatten(1)
        for l in self.layers:
            x = l(x)
        return x

    def interval_forward(self, x_min, x_max):
        """Forwards an interval through the net, to test for robustness."""
        x_min, x_max = x_min.flatten(1), x_max.flatten(1)
        for l in self.layers:
            x_min, x_max = l.interval_forward(x_min, x_max)
        return x_min, x_max

    def sensitivity(self):
        s = None
        for l in self.layers:
            s = l.sensitivity(s)
        return torch.max(s)

    def regularization(self):
        s = torch.mean(torch.abs(self.layers[0].u))
        for l in self.layers[2:]:
            s = s + torch.mean(torch.abs(l.u))
        return s

    def set_regular_deriv(self, b):
        for l in self.layers:
            l.regular_deriv = b

    def set_pseudo(self, p):
        for l in self.layers:
            l.pseudo = p

default_class = MWDNetTrimmable