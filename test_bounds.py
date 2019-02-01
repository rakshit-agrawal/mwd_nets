#!/usr/bin/env python

# Bound guarantees

import numpy as np
import torch


def test_bounds(args, logger, model, device, test_loader):
    with torch.no_grad():
        result_list = []
        epsilons = np.arange(args.epsilon_min, args.epsilon_max + 0.01, 0.05)
        for epsilon in epsilons:
            correct = 0
            for i, (data, target) in enumerate(test_loader):
                batch_size = data.shape[0]
                data, target = data.to(device), target.to(device)
                # Generates the intervals from the input.
                data_min = torch.clamp(data - epsilon, 0., 1.)
                data_max = torch.clamp(data + epsilon, 0., 1.)
                # Forward through the net.
                y_min, y_max = model.interval_forward(data_min, data_max)
                # y_min for the true value.
                y_min_true = y_min[torch.arange(batch_size), target]
                # y_max for any other value. First, we put a low value in the destination.
                y_max[torch.arange(batch_size), target] = torch.min(y_max, -1)[0] - 1.
                y_max_false = torch.max(y_max, -1)[0]
                correct += (y_min_true > y_max_false).sum().item()
            c = 100. * float(correct) / float(len(test_loader.dataset))
            logger.info('Accuracy provable for eps={:.2f}: {:.3f}'.format(epsilon, c))
            result_list.append((epsilon, c))
        return result_list
