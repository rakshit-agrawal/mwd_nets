#!/usr/bin/env python

# MWD net.

import torch

from json_plus import Serializable, Storage
from net_generic import GenericNet
from unit_mwd import MWD


class MWDNet(GenericNet):
    """MWD Neural net."""

    best_loss = 'sdl'
    bounded_parameters = True  # Uses optimizer that is aware of parameter bounds.
    default_lr = 5.0
    default_optim = 'adadelta'

    def __init__(self, args, empty=False):
        super(MWDNet, self).__init__()
        self.args = args
        layer_sizes = list(map(int, args.layers.split(','))) + [args.n_classes]  # 10 for the output.
        self.layers = []
        if not empty:
            previous_size = args.input_channels * args.input_y_size * args.input_x_size
            for i, size in enumerate(layer_sizes):
                l = MWD(previous_size, size,
                        andor=args.andor[i],
                        modinf=True,  # We always use modinf nowadays.
                        min_slope=args.min_slope,
                        max_slope=args.max_slope,
                        regular_deriv=args.regular_deriv)
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
        m = MWDNet(args, empty=True)
        for i, ms in enumerate(d['layers']):
            l = MWD.loads(ms, device)
            m.layers.append(l)
            m.add_module('layer_%d' % i, l)
        return m


default_class = MWDNet
