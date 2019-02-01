# Robust Neural Networks

## Contributors
* [Rakshit Agrawal](https://rakshitagrawal.com/)
* [Luca de Alfaro](https://sites.google.com/a/ucsc.edu/luca/)
* [David Helmbold](https://users.soe.ucsc.edu/~dph/)

## Requirements
* Python 2.7
* PyTorch
* Numpy

To install requirements:
```bash
pip install -r requirements.txt
```

## Instructions

The primary file to run any experiment is `experiments.py`.
Use this file to train, test, and attack the models.

### Training the networks

* Each network is identified by a specific file (such as net_mwd.py, or net_relu.py)
* In order to train a certain kind of net, the experiments need to use this name as identifier for the network.
* For example, to train an MWD network with layers of size 512,512,512 on MNIST dataset for 30 epochs, we will run
```bash
python experiments.py --net net_mwd --dataset MNIST --layers=512,512,512 --epochs 30
```

### Testing the networks under attacks

* Each attack can be applied on a trained network, by calling it's corresponding `test_<attack>` flag.
* For example, to trian and evaluate a model with I-FGSM and PGD attacks, we can use:
```bash
python experiments.py --net net_mwd --dataset MNIST --layers=512,512,512 --epochs 30 --test_ifgsm --test_pgd
```

### Testing the networks for interval bounds

* Each network can also be evaluated for the provable interval bounds.
* For example, to train and evaluate an MWD model with bounds, we can use:
```bash
python experiments.py --net net_mwd --dataset MNIST --layers=512,512,512 --epochs 30 --test_bounds
```

### Sample commands

* For training an MWD network, with I-FGSM and PGD attacks, and interval testing
```bash
python experiments.py --net net_mwd --dataset MNIST --layers=512,512,512 --epochs 30 --test_ifgsm --test_pgd --test_bounds
```
* For training a ReLU network, with I-FGSM and PGD attacks, and interval testing
```bash
python experiments.py --net net_relu --dataset MNIST --layers=512,512,512 --epochs 30 --test_ifgsm --test_pgd --test_bounds
```

### Identifiers

#### Networks
* MWD network: net_mwd
* ReLU network: net_relu
* Sigmoid network: net_sigmoid

#### Tests
* FGSM: test_fgsm
* I-FGDM: test_ifgsm
* PGD: test_pgd
* Interval bounds: test_bounds

### Usage of experiments.py
```
usage: experiments.py [-h] [--dataset DATASET] [--net NET] [--cls CLS]
                      [--layers LAYERS] [--andor ANDOR] [--regular_deriv]
                      [--min_slope MIN_SLOPE] [--max_slope MAX_SLOPE]
                      [--sensitivity_cost SENSITIVITY_COST]
                      [--l1_regularization L1_REGULARIZATION]
                      [--init_slope INIT_SLOPE] [--batch-size N]
                      [--test-batch-size N] [--epochs N]
                      [--epochs_before_regular_deriv EPOCHS_BEFORE_REGULAR_DERIV]
                      [--auto_flow] [--opt OPT] [--loss LOSS] [--lr LR]
                      [--second_lr SECOND_LR]
                      [--epochs_for_second_lr EPOCHS_FOR_SECOND_LR]
                      [--momentum M] [--train_adv TRAIN_ADV]
                      [--train_adv_steps TRAIN_ADV_STEPS]
                      [--use_pgd_for_adv_training] [--no-cuda] [--seed S]
                      [--pseudo PSEUDO] [--pseudo_decay PSEUDO_DECAY]
                      [--runs RUNS] [--test_fgsm] [--test_ifgsm] [--test_pert]
                      [--test_pgd] [--test_bounds]
                      [--pgd_iterations PGD_ITERATIONS]
                      [--pgd_restarts PGD_RESTARTS] [--pgd_inputs PGD_INPUTS]
                      [--pgd_batch PGD_BATCH] [--pseudo_adv]
                      [--epsilon_min EPSILON_MIN] [--epsilon_max EPSILON_MAX]
                      [--adversarial_steps ADVERSARIAL_STEPS]
                      [--log-interval N] [--continue_training]
                      [--try_new_loss] [--evaluate_after_each_epoch]
                      [--show_device] [--single_batch_only]
                      [--model_file MODEL_FILE] [--write_model]
                      [--write_results] [--read_models_dir READ_MODELS_DIR]
                      [--write_models_dir WRITE_MODELS_DIR]
                      [--results_dir RESULTS_DIR] [--logfile LOGFILE]
                      [--data_dir DATA_DIR]

Robust ML experiments

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset for experiment. Default=MNIST
  --net NET             net module to be used
  --cls CLS             net class to be used, if more than one in module
  --layers LAYERS       comma-separated list of layer sizes
  --andor ANDOR         Type of neurons in MWD nets, ^ = and, v = or, * =
                        random mix
  --regular_deriv       Use true derivative for training MWDs
  --min_slope MIN_SLOPE
                        Minimum slope for MWD (default: 0.01)
  --max_slope MAX_SLOPE
                        Maximum slope for MWD (default: 3.0)
  --sensitivity_cost SENSITIVITY_COST
                        Cost of sensitivity
  --l1_regularization L1_REGULARIZATION
                        L1 regularization for MWD only
  --init_slope INIT_SLOPE
                        Coefficient for slope initialization
  --batch-size N        input batch size for training (default: 100)
  --test-batch-size N   input batch size for testing (default: 1000)
  --epochs N            number of epochs to train (default: 1)
  --epochs_before_regular_deriv EPOCHS_BEFORE_REGULAR_DERIV
                        number of epochs to train with pseudoderivatives
  --auto_flow           Use autograd to figure out convolution
  --opt OPT             "ada" for adadelta, "mom" for momentum, None for class
                        default
  --loss LOSS           loss to use; None = default for net
  --lr LR               learning rate (None for class default)
  --second_lr SECOND_LR
                        Second, generally smaller, learning rate
  --epochs_for_second_lr EPOCHS_FOR_SECOND_LR
  --momentum M          SGD momentum (default: 0.5)
  --train_adv TRAIN_ADV
                        Train also on adversarial examples computed with given
                        attack quantity
  --train_adv_steps TRAIN_ADV_STEPS
                        Number of steps
  --use_pgd_for_adv_training
                        Use PGD for generating adversarial examples during
                        training.
  --no-cuda             disables CUDA training
  --seed S              random seed (default: 1)
  --pseudo PSEUDO       Amount of pseudo-derivative to use for trimmable
                        learning.If None, then no pseudo-training is used.
  --pseudo_decay PSEUDO_DECAY
                        Decay of pseudo-derivative in training.
  --runs RUNS           Number of runs to perform
  --test_fgsm           Perform FGSM adversarial test.
  --test_ifgsm          Perform I-FGSM adversarial test.
  --test_pert           Perform random perturbation testing.
  --test_pgd            Perform PGD adversarial testing.
  --test_bounds         Compute accuracy boubds via interval propagation.
  --pgd_iterations PGD_ITERATIONS
                        Number of PGD iterations
  --pgd_restarts PGD_RESTARTS
                        Number of PGD restarts
  --pgd_inputs PGD_INPUTS
                        Number of inputs to attack via PGD
  --pgd_batch PGD_BATCH
                        PGD batch size
  --pseudo_adv          Use pseudoderivative in adversarial attacks.
  --epsilon_min EPSILON_MIN
                        Min epsilon for testing attacks (must be multiple of
                        0.05)
  --epsilon_max EPSILON_MAX
                        Max epsilon for testing attacks (must be multiple of
                        0.05)
  --adversarial_steps ADVERSARIAL_STEPS
                        Number of steps in adversarial multistep testing
  --log-interval N      how many batches to wait before logging training
                        status
  --continue_training   Continue training a given model
  --try_new_loss        Try new loss function for training MWD.
  --evaluate_after_each_epoch
                        Perform test run after each epoch
  --show_device         Show configuration of compute devices
  --single_batch_only   Run only on a single batch. For testing
  --model_file MODEL_FILE
                        Model file from which to read the models.
  --write_model         Writes the trained model
  --write_results       Writes file with results.
  --read_models_dir READ_MODELS_DIR
                        Path of data directory
  --write_models_dir WRITE_MODELS_DIR
                        Path of data directory
  --results_dir RESULTS_DIR
                        Path of results directory
  --logfile LOGFILE     File where to write progress results.
  --data_dir DATA_DIR   Directory for the data


```

## License
BSD