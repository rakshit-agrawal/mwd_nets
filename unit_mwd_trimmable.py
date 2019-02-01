#!/usr/bin/env python

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd.function import Function
import numpy as np
from json_plus import Serializable
from torch_bounded_parameters import BoundedParameter


class LargeAttractorExpPseudo2(Function):
    """For MWD: Implements e^-x with soft derivative, trimmable."""

    @staticmethod
    def forward(ctx, x, pseudo):
        p = x.new([pseudo])
        ctx.save_for_backward(x, p)
        return torch.exp(-x)

    @staticmethod
    def backward(ctx, grad_output):
        x, p = ctx.saved_tensors
        pp = p.item()
        c = 1. - pp
        return - grad_output * torch.exp(- x * c), None


class SharedFeedbackMaxExpPseudo(Function):
    """For MWD:
    This is a trimmable version of the shared feedback max used for MWD."""

    @staticmethod
    def forward(ctx, x, pseudo):
        p = x.new([pseudo])
        y, _ = torch.max(x, -1)
        ctx.save_for_backward(x, y, p)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y, p = ctx.saved_tensors
        pp = p.item()
        y_complete = y.view(list(y.shape) + [1])
        d_complete = grad_output.view(list(grad_output.shape) + [1])
        return d_complete * torch.exp((x - y_complete) / (pp + 0.001)), None


class MWD_trimmable(nn.Module, Serializable):

    def __init__(self, in_features, out_features, andor="*", pseudo=1.0,
                 min_input=0.0, max_input=1.0, min_slope=0.001, max_slope=10.0,
                 ):
        """
        Implementation of MWD.
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
        super(MWD_trimmable, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.andor = andor
        self.pseudo = pseudo
        self.w = BoundedParameter(torch.Tensor(out_features, in_features),
                                  lower_bound=min_input, upper_bound=max_input)
        self.u = BoundedParameter(torch.Tensor(out_features, in_features),
                                  lower_bound=min_slope, upper_bound=max_slope)
        if andor == 'v':
            self.andor01 = Parameter(torch.ones((1, out_features)))
        elif andor == '^':
            self.andor01 = Parameter(torch.zeros((1, out_features,)))
        else:
            self.andor01 = Parameter(torch.Tensor(1, out_features, ))
            self.andor01.data.random_(0, 2)
        self.andor01.requires_grad = False
        self.w.data.uniform_(min_input, max_input)
        # Initialization of u.
        self.u.data.uniform_(0., 0.7)  # These could be parameters.
        self.u.data.clamp_(min_slope, max_slope)


    def forward(self, x):
        # Let n be the input size, and m the output size.
        # The tensor x is of shape * n. To make room for the output,
        # we view it as of shape * 1 n.
        xx = x.unsqueeze(-2)
        xuw = self.u * (xx - self.w)
        xuwsq = xuw * xuw
        # Aggregates into a modulus.
        # We want to get the largest square, which is the min one as we changed signs.
        z = SharedFeedbackMaxExpPseudo.apply(xuwsq, self.pseudo)
        y = LargeAttractorExpPseudo2.apply(z, self.pseudo)
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
        s = torch.max(torch.max(self.u, -1)[0], -1)[0].item()
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
        s = SharedFeedbackMaxExpPseudo.apply(u_prod, 1.0)
        s = s * np.sqrt(2. / np.e)
        return s

