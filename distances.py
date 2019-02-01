#!/usr/bin/env python

import torch


def square_distance_loss(output, target):
    """Returns the square distacne loss between output and target, except that
    the real reference output is a vector with all 0, with a 1 in the position
    specified by target."""
    s = list(output.shape)
    n_classes = s[-1]
    out = output.view(-1, n_classes)
    ss = out.shape
    n_els = ss[0]
    idxs = target.view(-1)
    t = output.new(n_els, n_classes)
    t.requires_grad = False
    t.fill_(0.)
    t[range(n_els), idxs] = 1.
    d = out - t
    dd = d * d
    return torch.sum(dd) / n_els


def square_distance_loss_soft(output, target, non_zero=0.1):
    """Like square distance loss between output and a vector-valued
    one-hot target, except that one-hot vector is encoded as non_zero's and
    1 minus non_zero.  Since the vector outputs are normalized, this discourages
    the appearance of progress by making the "correct" output overly large."""
    s = list(output.shape)
    n_classes = s[-1]
    out = output.view(-1, n_classes)
    ss = out.shape
    n_els = ss[0]
    idxs = target.view(-1)
    t = output.new(n_els, n_classes)
    t.requires_grad = False
    t.fill_(0.)
    t[range(n_els), idxs] = 1.
    t = t * (1.0 - non_zero) + non_zero / float(n_classes)
    d = out - t
    dd = d * d
    return torch.sum(dd) / n_els


def square_distance_loss_distr(output, target, non_zero=0.1):
    """This version is similar to square_disance_loss_soft, except that
    it first computes a distribution from the output values, that have to
    be non-negative."""
    distr = output / (torch.sum(output, -1, keepdim=True) + 0.001)
    return square_distance_loss_soft(distr, target, non_zero=non_zero)

