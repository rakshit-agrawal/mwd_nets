#!/usr/bin/env python

import torch

# Resizing with binning.
# Public domain, from http://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
def rebin(a, k, do_sum=False):
    n = len(a)
    return a.reshape(k, n // k).mean(-1)

def lastmax(x):
    return torch.max(x, -1)[0]
