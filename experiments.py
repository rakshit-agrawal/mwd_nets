#!/usr/bin/env python

# Experiment interface. This should be used to run experiments directly

import argparse
import datetime
import importlib
import logging
import json
import os

from json_plus import Serializable

import torch
import torch.optim as optim

from attacks import test_fgsm, test_ifgsm, test_perturbation, test_pgd
from test_bounds import test_bounds
from data import DataHandling
from attack_pgd import compute_pgd_example
from torch_bounded_parameters import ParamBoundEnforcer
from constants import *

# Creates a logger for the output.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(message)s")
c_handler.setFormatter(formatter)
logger.addHandler(c_handler)


ESCAPE_TOKEN = '<esc'
ESCAPE_SPACE = '\ '

def loss_soft_dist(output, target):
    soft_output = F.softmax(output, dim=1)
    return square_distance_loss(soft_output, target)


def flatten(s):
    out = []
    for t in s:
        if isinstance(t, str):
            out.append(t)
        else:
            out += t
    return out


def split_arg_list(s, escape_token='<esc>'):
    # Space split check.
    # Find escape spaces and replace to a token
    s = s.replace('\ ', ESCAPE_TOKEN)

    return flatten([t.replace(ESCAPE_TOKEN, ' ').split("=") for t in s.split()])


def read_model(in_fn, device):
    with open(in_fn, "r") as f:
        d = Serializable.from_json(f.read(), device=device)
        if 'encoding' in d:
            # New style model.
            model = d['model']
            model = model.to(device)
        else:
            # Old style model
            model_kind = d['kind']
            model_file, model_cls = OLD_NET_IDS.get(model_kind, None)
            module = importlib.import_module(model_file)
            cls = getattr(module, model_cls)
            model = cls.loads(d['model'], device)
    return model


def write_model(m, out_fn):
    with open(out_fn, 'w') as f:
        d = dict(encoding=0, model=m)
        f.write(Serializable.dumps(d))


def train_once(args, loss_function, model, flat_data, target, meta_optimizer):
    if model.bounded_parameters:
        # For MWD, the optimizer also enforces parameter bounds.
        meta_optimizer.optimizer.zero_grad()
        output = model(flat_data)
        primary_loss = loss_function(output, target)
        loss = primary_loss
        if args.sensitivity_cost > 0.0:
            loss += args.sensitivity_cost * model.sensitivity()
        if args.l1_regularization > 0.0:
            loss += args.l1_regularization * model.regularization()
        loss.backward()
        meta_optimizer.optimizer.step()
        meta_optimizer.enforce()
    else:
        # Otherwise, it's standard as in pytorch.
        meta_optimizer.zero_grad()
        output = model(flat_data)
        model_loss = loss_function(output, target)
        loss = model_loss + args.sensitivity_cost * model.sensitivity()
        loss.backward()
        meta_optimizer.step()
    return output, loss


def train(args, model, device, loss_function, train_loader, meta_optimizer, epoch):
    model.train()
    correct = 0.
    for batch_idx, (data_cpu, target_cpu) in enumerate(train_loader):
        if args.single_batch_only:
            if batch_idx>0:
                break
        x_input, target = data_cpu.to(device), target_cpu.to(device)
        if args.train_adv > 0.:
            x_input.requires_grad = True
        output, loss = train_once(args, loss_function, model, x_input, target, meta_optimizer)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        # Adversarial training, if requested.
        if args.train_adv > 0.:
            if args.use_pgd_for_adv_training:
                x_input_adv = compute_pgd_example(model, device, data_cpu, target_cpu, args.train_adv,
                                                  loss_function, num_iterations=args.train_adv_steps)
                # We check that the input is in the proper distance ball.
                assert torch.max(torch.abs(x_input - x_input_adv)).item() < args.train_adv * 1.01
                train_once(args, loss_function, model, x_input_adv, target, meta_optimizer)
            else:
                step_epsilon = args.train_adv / float(args.train_adv_steps)
                for adversarial_step in range(args.train_adv_steps):
                    # Creates an adversarial example to use for training.
                    s = torch.sign(x_input.grad)
                    new_x_input = torch.clamp(x_input + torch.mul(s, step_epsilon), 0., 1.)
                    x_input = x_input.new(*(new_x_input.shape))
                    x_input.requires_grad = True
                    x_input.data = new_x_input.data
                    output = model(x_input)
                    if adversarial_step < args.train_adv_steps - 1:
                        loss = loss_function(output, target)
                        loss.backward()
                    else:
                        train_once(args, loss_function, model, x_input, target, meta_optimizer)
        if (batch_idx + 1) % args.log_interval == 0:
            logger.info('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAccuracy:{:.5f} Sensitivity: {}'.format(
                epoch, args.epochs, (batch_idx + 1) * len(data_cpu), len(train_loader.dataset),
                                    100. * (batch_idx + 1) / len(train_loader), loss.item(),
                                    100. * correct / float(args.log_interval * args.batch_size),
                model.sensitivity().item()
            ))
            correct = 0.


def test(args, model, device, test_loader):
    model = model.to(device)
    model.eval()
    correct = 0.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, prediction = output.max(1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    sensitivity = float(model.sensitivity().item())
    logger.info('Test set: Accuracy: {}/{} ({:.5f}%, Sensitivity: {})\n'.format(
        correct, len(test_loader.dataset),
        accuracy,
        sensitivity
    ))
    return accuracy, sensitivity


def print_device_config():
    if torch.cuda.is_available():
        for devid in range(torch.cuda.device_count()):
            logger.info(torch.cuda.get_device_properties(devid))
    else:
        logger.info("CUDA device not available. Using CPU")


def create_optimizer(model, args, lr=None):
    # Creates an optimizer.
    lr = lr or args.lr
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.opt == 'mom':
        optimizer = optim.SGD(params, lr=lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = optim.Adam(params, lr=lr)
    else:
        optimizer = optim.Adadelta(params, lr=lr)
    # For MWD nets, we wrap it into an enforcer.
    meta_optimizer = ParamBoundEnforcer(optimizer) if model.bounded_parameters else optimizer
    return meta_optimizer


def run_experiment(args):
    """ Run an experiment with a dataset, method, and attacks """

    date_str = datetime.datetime.now().isoformat().replace(':', '-')
    # Updates the logger.
    if args.logfile is not None:
        fn = args.logfile + "_" + date_str + "_log.txt"
        f_handler = logging.FileHandler(fn)
        f_handler.setFormatter(formatter)
        f_handler.setLevel(logging.DEBUG)
        logger.addHandler(f_handler)

    # Device helpers
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.show_device:
        print_device_config()

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Data object
    dh = DataHandling(args.dataset, args, kwargs)

    # Get dataset and metadata
    train_loader, test_loader = dh.get_train_and_test_loaders()
    data_shape, label_shape = dh.get_shapes()
    n_classes = dh.n_classes

    args.input_channels = data_shape[0]
    args.input_x_size = data_shape[-1]
    args.input_y_size = data_shape[-2]
    args.n_classes = n_classes

    if args.runs == 1:
        # We listen to the seed only for single runs.
        torch.manual_seed(args.seed)

    # Prepares dictionary of outputs.
    output = {'parameters': args.__dict__,
              'accuracy_test': [],
              'sensitivity': [],
              'accuracy_adv': [],
              'accuracy_adv_multi': [],
              'accuracy_adv_multi_modinf': [],
              'accuracy_pert': [],
              'accuracy_pgd': [],
              'accuracy_vs_pgd_restarts': [],
              'accuracy_bounds': [],
              }

    # Creates an output file name.
    if args.model_file is not None:
        clean_model_file = args.model_file.split('/')[-1]
        out_fn = "measure_{}_[[{}]]".format(date_str, clean_model_file)
    else:
        out_fn = "measure_{}[{}.{}]_{}runs_{}epochs".format(
            date_str,
            args.net,
            args.cls or 'std',
            args.runs,
            args.epochs
        )

    # If directories are specified, uses them.
    out_results_fn = out_fn
    if args.write_results and args.results_dir is not None:
        out_results_fn = os.path.join(args.results_dir, out_fn)
    out_models_fn = out_fn
    if args.write_model:
        out_models_fn = os.path.join(args.write_models_dir or args.results_dir, out_fn)
    if args.read_models_dir is not None and args.model_file is not None:
        args.model_file = os.path.join(args.read_models_dir, args.model_file)

    # Runs the experiment.
    for run_idx in range(args.runs):
        logger.info("====== Run {} ======".format(run_idx))
        logger.info(json.dumps(args.__dict__, indent=4))
        if args.runs > 1:
            torch.manual_seed(run_idx)

        # Creates or reads the model.
        if args.model_file is None:
            # We create a new model.
            module = importlib.import_module(args.net)
            if args.cls is None:
                # We import the default class
                cls = module.default_class
                logger.info("Using {}.{}".format(args.net, module.default_class.__name__))
            else:
                # We import the specified class.
                cls = getattr(module, args.cls)
                logger.info("Using {}.{}".format(args.net, args.cls))
            assert cls is not None, "No class found!"
            model = cls(args).to(device)
            # Sets some parameters.
            args.lr = args.lr or cls.default_lr
            args.opt = args.opt or cls.default_optim
        else:
            # We read a model.
            if args.runs > 1:
                in_fn = args.model_file + '_{}.model.json'.format(run_idx)
            else:
                in_fn = args.model_file + '.model.json'
            model = read_model(in_fn, device)
            # Takes care of a few flags.
            model.set_regular_deriv(args.regular_deriv)

        print(model)

        # Trains the model.
        if args.model_file is None or args.continue_training:
            meta_optimizer = create_optimizer(model, args)
            # Trains the model for given number of epochs.
            pseudo = args.pseudo
            for epoch in range(1, args.epochs + 1):
                if args.pseudo is not None:
                    model.set_pseudo(pseudo)
                if args.second_lr is not None and epoch == args.epochs_for_second_lr:
                    # Re-initializes the LR.
                    meta_optimizer = create_optimizer(model, args, lr=args.second_lr)
                if args.epochs_before_regular_deriv > 0 and epoch > args.epochs_before_regular_deriv:
                    model.set_regular_deriv(True)
                # Uses the loss function specified by the model.
                loss_name = model.best_loss if args.loss is None else args.loss
                loss_function = LOSS[loss_name]
                train(args, model, device, loss_function, train_loader, meta_optimizer, epoch)
                if args.pseudo is not None:
                    pseudo *= args.pseudo_decay
            # We write the model.
            if args.write_model:
                model_fn = out_models_fn + '_{}.model.json'.format(run_idx)
                write_model(model, model_fn)

        # Now, we do the tests.
        accuracy, sensitivity = test(args, model, device, test_loader)
        output['accuracy_test'].append(accuracy)
        output['sensitivity'].append(sensitivity)
        if args.test_fgsm:
            output['accuracy_adv'].append(test_fgsm(args, logger, model, device, test_loader))
        if args.test_ifgsm:
            output['accuracy_adv_multi_modinf'].append(test_ifgsm(args, logger, model, device, test_loader))
        if args.test_pert:
            output['accuracy_pert'].append(test_perturbation(args, logger, model, device, test_loader))
        if args.test_pgd:
            accuracy_pgd, accuracy_vs_restarts_by_epsilon = test_pgd(args, logger, model, device, kwargs)
            output['accuracy_pgd'].append(accuracy_pgd)
            output['accuracy_vs_pgd_restarts'].append(accuracy_vs_restarts_by_epsilon)
        if args.test_bounds:
            output['accuracy_bounds'].append(test_bounds(args, logger, model, device, test_loader))

    if args.write_results:
        with open(out_results_fn + '.json', 'w') as f:
            json.dump(output, f, sort_keys=True, indent=4, separators=(',', ': '))


def main(arg_string=None):
    """ Main entry point """
    parser = argparse.ArgumentParser(description="Robust ML experiments")

    # Data selection
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset for experiment. Default=MNIST')

    # Net selection
    parser.add_argument('--net', type=str, default=None,
                        help='net module to be used')
    parser.add_argument('--cls', type=str, default=None,
                        help='net class to be used, if more than one in module')

    # Model settings
    parser.add_argument('--layers', type=str, default="32,32,32",
                        help='comma-separated list of layer sizes')
    parser.add_argument('--andor', type=str, default='^v^v',
                        help='Type of neurons in MWD nets, ^ = and, v = or, * = random mix')
    parser.add_argument('--regular_deriv', action='store_true', default=False,
                        help='Use true derivative for training MWDs')
    parser.add_argument('--min_slope', type=float, default=0.01,
                        help='Minimum slope for MWD (default: 0.01)')
    parser.add_argument('--max_slope', type=float, default=3.0,
                        help='Maximum slope for MWD (default: 3.0)')
    parser.add_argument('--sensitivity_cost', type=float, default=0.,
                        help='Cost of sensitivity')
    parser.add_argument('--l1_regularization', type=float, default=0.,
                        help='L1 regularization for MWD only')
    parser.add_argument('--init_slope', type=float, default=0.25,
                        help='Coefficient for slope initialization')

    # Training settings
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--epochs_before_regular_deriv', type=int, default=0,
                        help='number of epochs to train with pseudoderivatives')
    parser.add_argument('--auto_flow', action='store_true', default=False,
                        help='Use autograd to figure out convolution')
    parser.add_argument('--opt', type=str, default=None,
                        help='"ada" for adadelta, "mom" for momentum, None for class default')
    parser.add_argument('--loss', type=str, default=None,
                        help='loss to use; None = default for net')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (None for class default)')
    parser.add_argument('--second_lr', type=float, default=None,
                        help='Second, generally smaller, learning rate')
    parser.add_argument('--epochs_for_second_lr', type=float, default=16,
                        help='')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--train_adv', type=float, default=0.0,
                        help='Train also on adversarial examples computed with given attack quantity')
    parser.add_argument('--train_adv_steps', type=int, default=1,
                        help='Number of steps ')
    parser.add_argument('--use_pgd_for_adv_training', action='store_true', default=False,
                        help='Use PGD for generating adversarial examples during training.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pseudo', type=float, default=None,
                        help='Amount of pseudo-derivative to use for trimmable learning.'
                             'If None, then no pseudo-training is used.')
    parser.add_argument('--pseudo_decay', type=float, default=0.9,
                        help='Decay of pseudo-derivative in training.')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs to perform')

    # Testing settings
    parser.add_argument('--test_fgsm', action='store_true', default=False,
                        help='Perform FGSM adversarial test.')
    parser.add_argument('--test_ifgsm', action='store_true', default=False,
                        help='Perform I-FGSM adversarial test.')
    parser.add_argument('--test_pert', action='store_true', default=False,
                        help='Perform random perturbation testing.')
    parser.add_argument('--test_pgd', action='store_true', default=False,
                        help='Perform PGD adversarial testing.')
    parser.add_argument('--test_bounds', action='store_true', default=False,
                        help="Compute accuracy boubds via interval propagation.")
    parser.add_argument('--pgd_iterations', type=int, default=100,
                        help="Number of PGD iterations"),
    parser.add_argument('--pgd_restarts', type=int, default=10,
                        help="Number of PGD restarts"),
    parser.add_argument('--pgd_inputs', type=int, default=1000,
                        help="Number of inputs to attack via PGD")
    parser.add_argument('--pgd_batch', type=int, default=100,
                        help="PGD batch size")
    parser.add_argument('--pseudo_adv', action='store_true', default=False,
                        help='Use pseudoderivative in adversarial attacks.')
    parser.add_argument('--epsilon_min', type=float, default=0.05,
                        help="Min epsilon for testing attacks (must be multiple of 0.05)")
    parser.add_argument('--epsilon_max', type=float, default=0.5,
                        help="Max epsilon for testing attacks (must be multiple of 0.05)")
    parser.add_argument('--adversarial_steps', type=int, default=10,
                        help='Number of steps in adversarial multistep testing')

    # Experiment handling
    parser.add_argument('--log-interval', type=int, default=60, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--continue_training', action='store_true', default=False,
                        help='Continue training a given model')
    parser.add_argument('--try_new_loss', action='store_true', default=False,
                        help='Try new loss function for training MWD.')
    parser.add_argument('--evaluate_after_each_epoch', action='store_true', default=True,
                        help='Perform test run after each epoch')
    parser.add_argument('--show_device', action='store_true', default=False,
                        help='Show configuration of compute devices')
    parser.add_argument('--single_batch_only', action='store_true', default=False,
                        help='Run only on a single batch. For testing')

    # Files and directories.
    parser.add_argument('--model_file', type=str, default=None,
                        help='Model file from which to read the models.')
    parser.add_argument('--write_model', action='store_true', default=False,
                        help='Writes the trained model')
    parser.add_argument('--write_results', action='store_true', default=False,
                        help='Writes file with results.')
    parser.add_argument('--read_models_dir', type=str, default=None,
                        help='Path of data directory')
    parser.add_argument('--write_models_dir', type=str, default=None,
                        help='Path of data directory')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Path of results directory')
    parser.add_argument('--logfile', type=str, default=None,
                        help="File where to write progress results.")
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory for the data')

    # Parse
    if arg_string is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(split_arg_list(arg_string))

    run_experiment(args)


if __name__ == "__main__":
    main()
