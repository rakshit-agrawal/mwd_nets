#!/usr/bin/env python

# Pytorch wrapper layer for loading datasets

import torch
from torchvision import transforms, datasets
import os

DATASETS = {
    'mnist': {'cls': datasets.MNIST, 'n_classes': 10},
    'cifar10': {'cls': datasets.CIFAR10, 'n_classes': 10},
}


class DataHandling(object):
    """ Class for maintaining data handlers.
        TODO: Make functional for both torchvision, and normal datasets
    """

    def __init__(self,
                 dataset_name,
                 args, kwargs,
                 batch_size=None):
        """ Set the object for a specific dataset """
        self.dataset_name = dataset_name
        self.args = args
        self.kwargs = kwargs
        self.batch_size = batch_size if batch_size is not None else args.batch_size

        # Check if dataset is available
        assert DATASETS.get(dataset_name.lower(), None) is not None, "Dataset '%s' is not available" % dataset_name

        # Set the data class and n_classes
        dataset = DATASETS[dataset_name.lower()]
        self.data_cls = dataset['cls']
        self.n_classes = dataset['n_classes']
        self.train_loader = None
        self.test_loader = None

    def get_loader(self,
                   train=True,
                   download=True,
                   transform=transforms.ToTensor(),
                   shuffle=True):
        """ Get the loader for dataset """

        loader = torch.utils.data.DataLoader(
            self.data_cls(os.path.join(self.args.data_dir, self.dataset_name),
                          train=train, download=download,
                          transform=transform),
            batch_size=self.batch_size, shuffle=shuffle, **self.kwargs)

        return loader

    def get_train_loader(self,
                         download=True,
                         transform=transforms.ToTensor(),
                         shuffle=True):
        """ Get the training loader for dataset """
        loader = self.get_loader(train=True,
                                 download=download,
                                 transform=transform,
                                 shuffle=shuffle)
        self.train_loader = loader
        return loader

    def get_test_loader(self,
                        download=True,
                        transform=transforms.ToTensor(),
                        shuffle=True):
        """ Get the test loader for dataset """
        loader = self.get_loader(train=False,
                                 download=download,
                                 transform=transform,
                                 shuffle=shuffle)
        self.test_loader = loader
        return loader

    def get_train_and_test_loaders(self,
                         download=True,
                         transform=transforms.ToTensor(),
                         shuffle=True):
        """ Get both train and test loaders """

        train_loader = self.get_train_loader(download, transform, shuffle)
        test_loader = self.get_test_loader(download, transform, shuffle)

        return train_loader, test_loader

    def get_shapes(self):
        """ Infer shape of data using a sample """
        if self.train_loader is None:
            self.get_train_loader()

        sample, sample_target = next(iter(self.train_loader))
        data_shape, label_shape = sample[0].shape, sample_target.shape

        return data_shape, label_shape
