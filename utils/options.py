#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument("--algorithm", type=str, default="DecFus")

    # model arguments
    parser.add_argument('--model', type=str, default='vgg', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='svhn', help="name of dataset")
    parser.add_argument('--generate_data', type=int, default=1, help="whether generate new dataset")
    parser.add_argument('--iid', type=int, default=0, help='whether i.i.d or not')
    parser.add_argument('--noniid_case', type=int, default=5, help="non i.i.d case (1, 2, 3, 4)")
    parser.add_argument('--data_beta', type=float, default=1,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument("--test_round", type=int, default=5)

    # DFLMR arguments
    parser.add_argument("--first_stage_bound", type=int, default=0)

    # DecFus arguments
    parser.add_argument('--initial_threshold', type=float, default=0.75,
                        help='initial similarity cutoff (at the start of training)')
    parser.add_argument('--final_threshold', type=float, default=0.25,
                        help='final similarity cutoff (at the end of training)')
    parser.add_argument('--sigmoid_steepness', type=float, default=20.0,
                        help='steepness of the sigmoid; larger values give a sharper transition')
    parser.add_argument('--sigmoid_midpoint', type=float, default=0.4,
                        help='midpoint of the sigmoid: the progress (between 0 and 1) at which the cutoff changes fastest')
    parser.add_argument('--variance_window_size', type=int, default=5,
                        help='sliding window size of the variance stability detector')
    parser.add_argument('--variance_stability_threshold', type=float, default=0.3,
                        help='stability threshold on the relative variation of the variance; smaller is stricter')
    parser.add_argument('--variance_protection_ratio', type=float, default=0.2,
                        help='fraction of total rounds used as the protection period, during which the initial cutoff is kept')
    parser.add_argument('--variance_descent_ratio', type=float, default=0.5,
                        help='fraction of total rounds over which the sigmoid decay completes once stability is detected')
    parser.add_argument("--variance_reference_start_ratio", type=float, default=0.1)
    parser.add_argument("--variance_reference_end_ratio", type=float, default=0.2)
    parser.add_argument('--exchange_prob', type=float, default=0.4,
                        help='probability that a low-similarity parameter is taken from the least similar neighbor')

    args = parser.parse_args()
    return args
