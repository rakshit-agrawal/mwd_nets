#!/usr/bin/env python

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd.function import Function

import numpy as np
from json_plus import Serializable

import unittest

from torch_bounded_parameters import BoundedParameter


class LargeAttractorExp(Function):
    """Implements e^-x with soft derivative."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(-x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return - grad_output / torch.sqrt(1. + x)


class SharedFeedbackMax(Function):

    @staticmethod
    def forward(ctx, x):
        y, _ = torch.max(x, -1)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        y_complete = y.view(list(y.shape) + [1])
        d_complete = grad_output.view(list(grad_output.shape) + [1])
        return d_complete * torch.exp(x - y_complete)


class MWD(nn.Module, Serializable):

    def __init__(self, in_features, out_features, andor="*",
                 modinf=False, regular_deriv=False,
                 min_input=0.0, max_input=1.0, min_slope=0.001, max_slope=10.0):
        """
        Implementation of MWD module with logloss.
        :param in_features: Number of input features.
        :param out_features: Number of output features.
        :param andor: '^' for and, 'v' for or, '*' for mixed.
        :param modinf: Whether to aggregate using max (if True) of sum (if False).
        :param regular_deriv: Whether to use regular derivatives or not.
        :param min_input: minimum value for w (and therefore min value for input)
        :param max_input: max, as above.
        :param min_slope: min value for u, defining the slope.
        :param max_slope: max value for u, defining the slope.
        """
        super(MWD, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.andor = andor
        self.modinf = modinf
        self.regular_deriv = regular_deriv
        self.w = BoundedParameter(torch.Tensor(out_features, in_features),
                                  lower_bound=min_input, upper_bound=max_input)
        self.u = BoundedParameter(torch.Tensor(out_features, in_features),
                                  lower_bound=min_slope, upper_bound=max_slope)
        if andor == 'v':
            self.andor01 = Parameter(torch.ones((1, out_features)))
        elif andor == '^':
            self.andor01 = Parameter(torch.zeros((1, out_features)))
        else:
            self.andor01 = Parameter(torch.Tensor(1, out_features))
            self.andor01.data.random_(0, 2)
        self.andor01.requires_grad = False
        self.w.data.uniform_(min_input, max_input)
        # Initialization of u.
        self.u.data.uniform_(0.2, 0.7)  # These could be parameters.
        self.u.data.clamp_(min_slope, max_slope)


    def forward(self, x):
        # Let n be the input size, and m the output size.
        # The tensor x is of shape * n. To make room for the output,
        # we view it as of shape * 1 n.
        # Aggregates into a modulus.
        xx = x.unsqueeze(-2)
        xuw = self.u * (xx - self.w)
        xuwsq = xuw * xuw
        if self.modinf:
            # We want to get the largest square, which is the min one as we changed signs.
            if self.regular_deriv:
                z, _ = torch.max(xuwsq, -1)
                y = torch.exp(- z)
            else:
                z = SharedFeedbackMax.apply(xuwsq)
                y = LargeAttractorExp.apply(z)
        else:
            z = torch.sum(xuwsq, -1)
            if self.regular_deriv:
                y = torch.exp(- z)
            else:
                y = LargeAttractorExp.apply(z)
        # Takes into account and-orness.
        if self.andor == '^':
            return y
        elif self.andor == 'v':
            return 1.0 - y
        else:
            return y + self.andor01 * (1.0 - 2.0 * y)


    def interval_forward(self, x_min, x_max):
        xx_min = x_min.unsqueeze(-2)
        xx_max = x_max.unsqueeze(-2)
        xuw1 = self.u * (xx_min - self.w)
        xuwsq1 = xuw1 * xuw1
        xuw2 = self.u * (xx_max - self.w)
        xuwsq2 = xuw2 * xuw2
        sq_max = torch.max(xuwsq1, xuwsq2)
        sq_min = torch.min(xuwsq1, xuwsq2)
        # If w is between x_min and x_max, then sq_min should be 0.
        # So we multiply sq_min by something that is 0 if x_min < w < x_max.
        sq_min = sq_min * ((xx_min > self.w) + (self.w > xx_max)).float()

        y_min = torch.exp(- torch.max(sq_max, -1)[0])
        y_max = torch.exp(- torch.max(sq_min, -1)[0])
        # Takes into account and-orness.
        if self.andor == '^':
            return y_min, y_max
        elif self.andor == 'v':
            return 1.0 - y_max, 1.0 - y_min
        else:
            y1 = y_min + self.andor01 * (1.0 - 2.0 * y_min)
            y2 = y_max + self.andor01 * (1.0 - 2.0 * y_max)
            y_min = torch.min(y1, y2)
            y_max = torch.max(y1, y2)
            return y_min, y_max


    def overall_sensitivity(self):
        """Returns the sensitivity to adversarial examples of the layer."""
        if self.modinf:
            s = torch.max(torch.max(self.u, -1)[0], -1)[0].item()
        else:
            s = torch.max(torch.sqrt(torch.sum(self.u * self.u, -1)))[0].item()
        s *= np.sqrt(2. / np.e)
        return s


    def sensitivity(self, previous_layer):
        """Given the sensitivity of the previous layer (a vector of length equal
        to the number of inputs), it computes the sensitivity to adversarial examples
         of the current layer, as a vector of length equal to the output size of the
         layer.  If the input sensitivity of the previous layer is None, then unit
         sensitivity is assumed."""
        if previous_layer is None:
            previous_layer = self.w.new(1, self.in_features)
            previous_layer.fill_(1.)
        else:
            previous_layer = previous_layer.view(1, self.in_features)
        u_prod = previous_layer * self.u
        if self.modinf:
            # s = torch.max(u_prod, -1)[0]
            s = SharedFeedbackMax.apply(u_prod)
        else:
            s = torch.sqrt(torch.sum(u_prod * u_prod, -1))
        s = s * np.sqrt(2. / np.e)
        return s


    def dumps(self):
        """Writes itself to a string."""
        # Creates a dictionary
        d = dict(
            in_features=self.in_features,
            out_features=self.out_features,
            min_input=self.w.lower_bound,
            max_input=self.w.upper_bound,
            min_slope=self.u.lower_bound,
            max_slope=self.u.upper_bound,
            modinf=self.modinf,
            regular_deriv=self.regular_deriv,
            andor=self.andor,
            andor01=self.andor01.cpu().numpy(),
            u=self.u.data.cpu().numpy(),
            w=self.w.data.cpu().numpy(),
        )
        return Serializable.dumps(d)


    @staticmethod
    def loads(s, device):
        """Reads itself from string s."""
        d = Serializable.loads(s)
        m = MWD(
            d['in_features'],
            d['out_features'],
            andor=d['andor'],
            modinf=d['modinf'],
            regular_deriv=d['regular_deriv'],
            min_input=d['min_input'],
            max_input=d['max_input'],
            min_slope=d['min_slope'],
            max_slope=d['max_slope']
        )
        m.u.data = torch.from_numpy(d['u']).to(device)
        m.w.data = torch.from_numpy(d['w']).to(device)
        m.andor01.data = torch.from_numpy(d['andor01']).to(device)
        return m


class TestMWD(unittest.TestCase):

    def setup_layer(self, andor="^", regular_deriv=False, modinf=False, single_norm=False):
        layer = MWD(2, 3, andor=andor, modinf=modinf, regular_deriv=regular_deriv,
                    single_normalization=single_norm)
        # I want to start from simple numbers.
        layer.w.data.mul_(0.)
        layer.u.data.mul_(0.)
        layer.u.data.add_(torch.tensor([[3., 2.], [4., 3.], [1., 2]]))
        layer.w.data.add_(torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]))
        return layer

    # @unittest.skip("later")
    def test_all_forwards(self):
        """The methods differ only for how they do backward propagation,
        so we should get the same results."""
        layer1 = self.setup_layer(regular_deriv=True)
        layer2 = self.setup_layer(single_norm=True)
        layer3 = self.setup_layer()
        x = torch.tensor([1.0, 1.0])
        y1 = layer1(x)
        y2 = layer2(x)
        y3 = layer3(x)
        self.assertAlmostEqual(torch.sum(torch.abs(y1 - y2)), 0.0, 3)
        self.assertAlmostEqual(torch.sum(torch.abs(y1 - y3)), 0.0, 3)

    # @unittest.skip("later")
    def test_forward(self):
        layer_and = self.setup_layer(regular_deriv=True)
        layer_or = self.setup_layer(andor="v")
        print("w:", layer_and.w)
        print("u:", layer_and.u)
        x = torch.tensor([1.0, 1.0])
        y_and = layer_and(x)
        y_or = layer_or(x)
        y = y_and + y_or
        self.assertAlmostEqual(torch.sum(torch.abs(y - torch.tensor([1., 1., 1.]))), 0.0, 3)
        print("y:", y)
        print("y_and:", y_and)
        print("y_or:", y_or)

    @unittest.skip("later")
    def test_naming(self):
        l = self.setup_layer()
        for n, p in l.named_parameters():
            if p.requires_grad:
                print(n, p, "l:", p.lower_bound, "u:", p.upper_bound)

    # @unittest.skip("later")
    def test_back(self):
        # With regular derivatives first.
        regular_layer = self.setup_layer(regular_deriv=True)
        x = torch.tensor([1.1, 1.2], requires_grad=True)
        y0 = regular_layer(x)
        y0.backward(torch.tensor([1., 1., 1.]))
        print("Normal x.grad:", x.grad)
        x1 = x - 0.01 * x.grad
        with torch.no_grad():
            y1 = regular_layer(x1)
        print("Normal y0 - y1:", y0 - y1)

        # Then, with faster derivatives:
        fast_layer = self.setup_layer()
        x = torch.tensor([1.1, 1.2], requires_grad=True)
        y0 = fast_layer(x)
        y0.backward(torch.tensor([1., 1., 1.]))
        print("Fast x.grad:", x.grad)
        x1 = x - 0.01 * x.grad
        with torch.no_grad():
            y1 = fast_layer(x1)
        print("Fast y0 - y1:", y0 - y1)

        # Then, with fast, single normalization derivatives:
        fast_layer = self.setup_layer(single_norm=True)
        x = torch.tensor([1.1, 1.2], requires_grad=True)
        y0 = fast_layer(x)
        y0.backward(torch.tensor([1., 1., 1.]))
        print("Fast single_norm x.grad:", x.grad)
        x1 = x - 0.01 * x.grad
        with torch.no_grad():
            y1 = fast_layer(x1)
        print("Fast single_norm y0 - y1:", y0 - y1)


if __name__ == '__main__':
    unittest.main()
