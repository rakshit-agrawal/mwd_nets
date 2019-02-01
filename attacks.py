#!/usr/bin/env python

# Adversarial attacks on neural networks

import numpy as np
import torch
from torch import nn
from constants import *

from data import DataHandling
from distances import square_distance_loss

from attack_pgd import pgd_attack, pgd_batch_attack
from distances import square_distance_loss, square_distance_loss_soft, square_distance_loss_distr

# Loss functions
LOSS = {
    'nll': F.nll_loss,
    'sdl': square_distance_loss,
    'sdl_soft': square_distance_loss_soft,
    'sdl_distr': square_distance_loss_distr,
}
loss_relu = F.nll_loss
loss_sigmoid = square_distance_loss

cross_entropy_loss = nn.CrossEntropyLoss()
nll_loss = nn.NLLLoss()


def loss_soft_dist(output, target):
    soft_output = F.softmax(output, dim=1)
    return square_distance_loss(soft_output, target)


def test_fgsm(args, logger, model, device, test_loader):
    epsilons = np.arange(args.epsilon_min, args.epsilon_max + 0.01, 0.05)
    correctnesses = []
    loss_name = model.best_loss if args.loss is None else args.loss
    loss_function = LOSS[loss_name]
    if not args.pseudo_adv:
        model.set_regular_deriv(True)
    for epsilon in epsilons:
        correct = 0.
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data.requires_grad = True  # We want the gradient wrt the input.
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            # Builds an adversarial input.
            s = torch.sign(data.grad)
            x_adversarial = torch.clamp(data + torch.mul(s, epsilon), 0., 1.)
            # And feeds it, measuring the correctness.
            output_adversarial = model(x_adversarial)
            pred_adversarial = output_adversarial.max(1, keepdim=True)[1]
            correct += pred_adversarial.eq(target.view_as(pred_adversarial)).sum().item()
        correctnesses.append(100. * correct / float(len(test_loader.dataset)))
    model.set_regular_deriv(args.regular_deriv)
    result = zip(epsilons, correctnesses)
    logger.info("Performance under adversarial:")
    for epsilon, c in result:
        logger.info("  Epsilon: {:.2f} Correctness: {}".format(epsilon, c))
    return result


def test_adversarial_multistep(args, logger, model, device, test_loader):
    epsilons = np.arange(args.epsilon_min, args.epsilon_max + 0.01, 0.05)
    correctnesses = []

    loss_name = model.best_loss if args.loss is None else args.loss
    loss_function = LOSS[loss_name]
    if not args.pseudo_adv:
        model.set_regular_deriv(True)
    for epsilon in epsilons:
        correct = 0.
        step_epsilon = epsilon / float(args.adversarial_steps)
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            x = data
            for adversarial_step in range(args.adversarial_steps):
                x_input = torch.tensor(x)
                x_input.requires_grad = True
                output = model(x_input)
                loss = loss_function(output, target)
                loss.backward()
                # Builds the next adversarial input.
                # The step length is bounded by step_epsilon.
                # We need to do the computation in this funny way to leave the tensor on CUDA.
                delta = torch.mul(torch.reciprocal(torch.max(torch.abs(x_input.grad), -1, keepdim=True)[0]),
                                  step_epsilon)
                x = torch.clamp(x + delta * x_input.grad, 0., 1.)
            # Measures the correctness.
            output_adversarial = model(x)
            pred_adversarial = output_adversarial.max(1, keepdim=True)[1]
            correct += pred_adversarial.eq(target.view_as(pred_adversarial)).sum().item()
        correctnesses.append(100. * correct / float(len(test_loader.dataset)))
    model.set_regular_deriv(args.regular_deriv)
    result = zip(epsilons, correctnesses)
    logger.info("Performance under multi adversarial:")
    for epsilon, c in result:
        logger.info("  Epsilon: {:.2f} Correctness: {}".format(epsilon, c))
    return result


def test_ifgsm(args, logger, model, device, test_loader):
    epsilons = np.arange(args.epsilon_min, args.epsilon_max + 0.01, 0.05)
    correctnesses = []
    loss_name = model.best_loss if args.loss is None else args.loss
    loss_function = LOSS[loss_name]
    if not args.pseudo_adv:
        model.set_regular_deriv(True)
    for epsilon in epsilons:
        correct = 0.
        step_epsilon = epsilon / float(args.adversarial_steps)
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            x = data
            for adversarial_step in range(args.adversarial_steps):
                x_input = torch.tensor(x)
                x_input.requires_grad = True
                output = model(x_input)
                loss = loss_function(output, target)
                loss.backward()
                # Builds the next adversarial input.
                # The step length is bounded by step_epsilon.
                s = torch.sign(x_input.grad)
                x = torch.clamp(x + torch.mul(s, step_epsilon), 0., 1.)
            # Measures the correctness.
            output_adversarial = model(x)
            pred_adversarial = output_adversarial.max(1, keepdim=True)[1]
            correct += pred_adversarial.eq(target.view_as(pred_adversarial)).sum().item()
        correctnesses.append(100. * correct / float(len(test_loader.dataset)))
    model.set_regular_deriv(args.regular_deriv)
    result = zip(epsilons, correctnesses)
    logger.info("Performance under multi adversarial modinf:")
    for epsilon, c in result:
        logger.info("  Epsilon: {:.2f} Correctness: {}".format(epsilon, c))
    return result


def test_perturbation(args, logger, model, device, test_loader):
    epsilons = np.arange(args.epsilon_min, args.epsilon_max + 0.01, 0.05)
    correctnesses = []
    for epsilon in epsilons:
        correct = 0.
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data = data
            dims = list(data.shape)
            s = data.new(*dims)
            s.uniform_(0., 1.)  # Uniform noise
            # Convex combination.
            x_perturbed = torch.clamp(torch.mul(data, 1. - epsilon) + torch.mul(s, epsilon), 0., 1.)
            # And feeds it, measuring the correctness.
            output_perturbed = model(x_perturbed)
            pred_perturbed = output_perturbed.max(1, keepdim=True)[1]
            correct += pred_perturbed.eq(target.view_as(pred_perturbed)).sum().item()
        correctnesses.append(100. * correct / float(len(test_loader.dataset)))
    result = zip(epsilons, correctnesses)
    logger.info("Performance under perturbations:")
    for epsilon, c in result:
        logger.info("  Epsilon: {:.2f} Correctness: {}".format(epsilon, c))
    return result


def test_pgd(args, logger, model, device, kwargs):
    epsilons = np.arange(args.epsilon_min, args.epsilon_max + 0.01, 0.05)
    if not args.pseudo_adv:
        model.set_regular_deriv(True)

    dh = DataHandling(args.dataset, args, kwargs, batch_size=args.pgd_batch)
    pgd_loader = dh.get_test_loader()
    loss_name = model.best_loss if args.loss is None else args.loss
    loss_function = LOSS[loss_name]
    success_rate = []  # Success rate of the model, not of the attack, to be homogeneous.
    accuracy_vs_restarts_by_epsilon = {}
    for epsilon in epsilons:
        if args.pgd_batch == 1:
            # We do one by one, stopping as soon as we get success.
            attempted_attacks, successful_attacks = 0., 0.
            for i, (data, target) in enumerate(pgd_loader):
                attempted_attacks += 1.
                data = data
                if pgd_attack(logger, model, device, data, target, epsilon, loss_function,
                              num_iterations=args.pgd_iterations,
                              num_restarts=args.pgd_restarts):
                    successful_attacks += 1.
                if i == args.pgd_inputs:
                    break
            success_rate.append(successful_attacks / attempted_attacks)
        else:
            # We do in batches.
            fractions = []
            # I need to average the accuracy vs restarts on all batches.
            # So I accumulate the results in a list of numpy vectors.
            accuracy_vs_restart_list = []
            for i, (data, target) in enumerate(pgd_loader):
                flat_data = data.view(-1, 28 * 28)
                avs, fraction = pgd_batch_attack(logger,
                                                 model, device, flat_data, target, epsilon, loss_function,
                                                 num_iterations=args.pgd_iterations,
                                                 num_restarts=args.pgd_restarts)
                fractions.append(fraction)
                accuracy_vs_restart_list.append(np.array(avs))
                if i + 1 >= args.pgd_inputs / args.pgd_batch:
                    break
            success_rate.append(float(np.mean(fractions)))
            # Now I need to average the success rate by epsilon.
            accuracy_vs_restart = list(np.average(np.vstack(accuracy_vs_restart_list), 0))
            accuracy_vs_restart = [float(x) for x in accuracy_vs_restart]
            accuracy_vs_restarts_by_epsilon[epsilon] = accuracy_vs_restart
    model.set_regular_deriv(args.regular_deriv)
    accuracy = zip(epsilons, success_rate)
    logger.info("Performance under PGD:")
    for epsilon, c in accuracy:
        logger.info("  Epsilon: {:.2f} Correctness: {}".format(epsilon, 1. - c))
    return accuracy, accuracy_vs_restarts_by_epsilon
