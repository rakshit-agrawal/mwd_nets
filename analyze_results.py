#!/usr/bin/env python

from __future__ import print_function

import argparse
from collections import OrderedDict as OrderedDict
import glob
import json
import numpy as np


def stats(x):
    """Computes means and standard deviation of a series of numbers"""
    return {
        'mean': np.mean(x),
        'std': np.std(x, ddof=1)  # We only have a sample.
    }


def noise_stats(pert_lists):
    """Computes means and standard deviation, for all amounts of
    perturbation added."""
    d = OrderedDict()
    for l in pert_lists:
        # l represents a list of (perturbation, performance)
        for eta, x in l:
            d[eta] = d.get(eta, []) + [x]
    # Now we can compute the statistics.
    etas = []
    means = []
    stdevs = []
    results = []
    for eta, xlist in d.iteritems():
        m = np.mean(xlist)
        s = np.std(xlist, ddof=1) if len(xlist) > 1 else None
        etas.append(eta)
        means.append(m)
        stdevs.append(s)
        results.append(dict(eta=eta, mean=m, std=s))
    return dict(etas=etas, means=means, stdevs=stdevs, results=results)


def analyze_file(d):
    """Analyze a file, read as a dictionary."""
    d['sensitivity'] = stats(d['sensitivity'])
    d['accuracy_test'] = stats(d['accuracy_test'])
    d['accuracy_adv'] = noise_stats(d['accuracy_adv'])
    d['accuracy_pert'] = noise_stats(d['accuracy_pert'])


def read_data(args):
    data = []
    if args.file:
        with open(args.file) as f:
            data.append(json.load(f))
    else:
        file_list = glob.glob(args.dir + '/*.json')
        for fn in file_list:
            with open(fn) as f:
                data.append(json.load(f))
    return data


def print_results(data):
    print("=======================================")
    print(json.dumps(data['parameters'], indent=2))
    r = data['accuracy_test']
    print("Accuracy: {:.2f} Stdev: {:.2f}".format(r['mean'], r['std']))
    r = data['sensitivity']
    print("Sensitivity: {:.2f} Stdev: {:.2f}".format(r['mean'], r['std']))
    print("Adversarial:")
    for r in data['accuracy_adv']['results']:
        print("  Eta: {:.2f} Acc: {:.2f} Std: {:.2f}".format(r['eta'], r['mean'], r['std']))
    print("Noise:")
    for r in data['accuracy_pert']['results']:
        print("  Eta: {:.2f} Acc: {:.2f} Std: {:.2f}".format(r['eta'], r['mean'], r['std']))


def analysis(args):
    files_data = read_data(args)
    map(analyze_file, files_data)
    map(print_results, files_data)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Analysis of run results')
    parser.add_argument('--dir', type=str, default='results',
                        help='Directory where the results are.')
    parser.add_argument('--file', type=str, default=None,
                        help='File to analyze')
    parser.add_argument('--out', type=str, default=None,
                        help='Directory where to write the results.')

    args = parser.parse_args()
    analysis(args)


if __name__ == '__main__':
    main()
