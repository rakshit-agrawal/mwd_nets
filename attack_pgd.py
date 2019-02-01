#!/usr/bin/env python

# Tests a model with adversarial attacks generated via PGD (Projected Gradient Descent).

from __future__ import print_function

import torch
from torch import nn
from torch.nn.parameter import Parameter

import torch.optim as optim
import numpy as np
from distances import square_distance_loss


class PGDAdversary(nn.Module):
    """Module implementing the PGD adversarial attack on a given model."""

    def __init__(self, model, x_batch, max_epsilon):
        """
        :param model: Model under attack.
        :param x_batch: Batch of inputs to attack.
        :param max_epsilon: Maximum norm of the L_\infinity attack.
        """
        super(PGDAdversary, self).__init__()
        self.batch_size, self.input_size = x_batch.shape
        # print("Shape of input to attack:", self.batch_size, self.input_size)
        self.model = model
        self.x_batch = Parameter(x_batch, requires_grad=False)
        self.x_batch.requires_grad = False
        self.max_epsilon_norm = max_epsilon
        # The model is NOT trainable.
        for p in model.parameters():
            p.requires_grad = False
        # The epsilon is the attack.
        # Notice that we have one epsilon per input, _and_ per batch element.
        # Of course, we fine tune the epsilon of the attack to each individual batch element.
        # So the batches are NOT batches of learning; we learn individually the epsilon for
        # each batch member, and we use batches purely to speed up the computation.
        self.epsilon = Parameter(torch.Tensor(self.batch_size, self.input_size), requires_grad=True)
        self.epsilon.retain_grad()
        # print("Epsilon shape", self.epsilon.shape)
        # For each element of self.epsilon, we compute the min and max value of its range.
        # These are given by the [0, 1] constraints on range of x + epsilon, and by the
        # max_epsilon_norm constraint on epsilon.
        self.min_epsilon = Parameter(torch.clamp(x_batch, 0., max_epsilon), requires_grad=False)
        self.max_epsilon = Parameter(torch.clamp(1. - x_batch, 0, max_epsilon), requires_grad=False)
        # We initialize epsilon.
        self.epsilon.data.uniform_(-max_epsilon, +max_epsilon)
        # And we clip it to its range.
        self.clip_epsilon_to_range()

    def clip_epsilon_to_range(self):
        """Clips epsilon to its predetermined range."""
        self.epsilon.data = torch.max(self.epsilon, self.min_epsilon)
        self.epsilon.data = torch.min(self.epsilon, self.max_epsilon)

    def forward(self):
        # print("Trying with:", torch.max(torch.abs(self.epsilon)))
        x_adversarial = torch.clamp(self.x_batch + self.epsilon, 0., 1.)
        return self.model.forward(x_adversarial)


def pgd_attack(logger, model, device, x_batch, target_batch, max_epsilon, loss_function,
               num_iterations=1000, num_restarts=10):
    """Measures whether a 1-element batch can be attacked via PGD. Returns True if so.
    :param model: The model under attack.
    :param x_batch: The batch of x to attack.  Must be a batch of size 1.
    :param target_batch: the batch of correct targets.
    :param max_epsilon: the max epsilon to use in the attack.
    :param loss_function: loss function used in training.
    :param num_iterations: the number of iterations for which to carry out the PGD attack.
    :param num_restarts: how many restarts of the PGD optimizer to do.
    """
    # We do several restarts.
    for start_idx in range(num_restarts):
        # Sets up the model.
        attack_model = PGDAdversary(model, x_batch, max_epsilon).to(device)
        target_batch = target_batch.to(device)
        # Creates an optimizer.
        params = filter(lambda p: p.requires_grad, attack_model.parameters())
        # for n, p in attack_model.named_parameters():
        #     print("We are optimizing on:", n, p.requires_grad)
        attack_model.train()
        optimizer = optim.Adadelta(params, lr=1.)
        epsilon0 = torch.tensor(attack_model.epsilon.data)
        # We count a successful attack any one that happened in the course of any iteration.
        for iteration_idx in range(num_iterations):
            # for n, p in attack_model.named_parameters():
            #     print("We are optimizing on 1:", n, p.requires_grad)
            optimizer.zero_grad()
            output = attack_model()
            # We use - loss, so we optimize in the direction of increasing loss.
            # Perhaps we could have set the learning rate of the optimizer to -1 :-)
            loss = - loss_function(output, target_batch) * 100.  # loss_function(output, target_batch)
            loss.backward()
            optimizer.step()
            attack_model.clip_epsilon_to_range()
            # Gets the prediction.
            _, prediction = output.max(1, keepdim=True)
            # We add the successful attacks (the wrong predictions) to the overall success.
            if target_batch.eq(prediction.view_as(target_batch)) < 1:
                logger.info("*************** Attack successful at iteration: {} Epsilon: {} L1: {}".format(
                    iteration_idx,
                    torch.max(torch.abs(attack_model.epsilon.data)).item(),
                    torch.sum(torch.abs(attack_model.epsilon.data - epsilon0)).item()
                ))
                return True
    logger.info("Failed attack")
    return False


def pgd_batch_attack(logger, model, device, x_batch, target_batch, max_epsilon, loss_function,
                     num_iterations=1000, num_restarts=10):
    """Same as above, but operates on an input batch, and returns the fraction of success
    of the batch (in [0, 1]). """
    # Prepares the tensor to keep track of successful attacks.
    success = torch.zeros_like(target_batch).byte().to(device)
    # We do several restarts.
    # We also keep track of the success rate, as a function of the number of restarts.
    accuracy_vs_restarts = []
    for start_idx in range(num_restarts):
        # Sets up the model.
        attack_model = PGDAdversary(model, x_batch, max_epsilon).to(device)
        target_batch = target_batch.to(device)
        # Creates an optimizer.
        params = filter(lambda p: p.requires_grad, attack_model.parameters())
        attack_model.train()
        optimizer = optim.Adadelta(params, lr=1.)
        # We count a successful attack any one that happened in the course of any iteration.
        for iteration_idx in range(num_iterations):
            optimizer.zero_grad()
            output = attack_model()
            # We use - loss, so we optimize in the direction of increasing loss.
            # Perhaps we could have set the learning rate of the optimizer to -1 :-)
            loss = - loss_function(output, target_batch) * 100.  # loss_function(output, target_batch)
            loss.backward()
            optimizer.step()
            attack_model.clip_epsilon_to_range()
            # Gets the prediction.
            _, prediction = output.max(1, keepdim=True)
            # We add the successful attacks (the wrong predictions) to the overall success.
            wrong = 1 - target_batch.eq(prediction.view_as(target_batch))
            success = torch.max(success, wrong)
            # print("    Iteration: {} success: {}".format(
            #     iteration_idx, float(torch.sum(success).item()) / len(target_batch)))
        # We compute the accuracy for this number of restarts.
        accuracy = 1. - float(torch.sum(success).item()) / len(target_batch)
        accuracy_vs_restarts.append(accuracy)
    logger.info("Batch success for epsilon {}: {}".format(
        max_epsilon, float(torch.sum(success).item()) / len(target_batch)))
    return accuracy_vs_restarts, float(torch.sum(success).item()) / float(len(target_batch))


def compute_pgd_example(model, device, x_batch, target_batch, max_epsilon, loss_function,
                        num_iterations=100):
    """Computes an adversarial example to be used in training."""
    # Remembers what required grad in the model.
    requires_grad_for_training = [p for p in model.parameters() if p.requires_grad]
    attack_model = PGDAdversary(model, x_batch, max_epsilon).to(device)
    target_batch = target_batch.to(device)
    # Creates an optimizer.
    params = filter(lambda p: p.requires_grad, attack_model.parameters())
    attack_model.train()
    optimizer = optim.Adadelta(params, lr=1.)
    for iteration_idx in range(num_iterations):
        optimizer.zero_grad()
        output = attack_model()
        # We use - loss, so we optimize in the direction of increasing loss.
        # Perhaps we could have set the learning rate of the optimizer to -1 :-)
        loss = - loss_function(output, target_batch) * 100.  # loss_function(output, target_batch)
        loss.backward()
        optimizer.step()
        attack_model.clip_epsilon_to_range()
    # Ok, we use the resulting attack. The clamp is there just for safety.
    # Makes the model learning again.
    for p in requires_grad_for_training:
        p.requires_grad = True
    return torch.clamp(x_batch.to(device) + attack_model.epsilon, 0., 1.)
