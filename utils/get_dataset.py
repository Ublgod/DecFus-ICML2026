#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchvision import datasets, transforms
from utils.sampling import cifar_iid, cifar_noniid, svhn_iid, svhn_noniid
from utils.dataset_utils import separate_data,read_record,separate_data_hierarchical
from utils import mydata
import os
import json

def get_dataset(args):

    file = os.path.join("data", args.dataset + "_" + str(args.num_users))
    if args.iid:
        file += "_iid"
    else:
        file += "_noniidCase" + str(args.noniid_case)

    if args.noniid_case > 4:
        file += "_beta" + str(args.data_beta)

    file += ".json"
    # load dataset and split users
    if args.dataset == 'cifar10':

        trans_cifar10_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 5:
                dict_users = cifar_noniid(dataset_train,args.num_users,args.noniid_case)
            elif args.noniid_case == 6:
                dict_users = separate_data_hierarchical(
                    dataset_train, args.num_users, args.num_classes, args.data_beta, num_groups=10
                )
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'cifar100':
        trans_cifar100 = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset_train = mydata.CIFAR100_coarse('./data/cifar100_coarse', train=True, download=True,
                                               transform=trans_cifar100)
        dataset_test = mydata.CIFAR100_coarse('./data/cifar100_coarse', train=False, download=True,
                                              transform=trans_cifar100)
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 5:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'svhn':
        # SVHN specific transforms
        trans_svhn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # SVHN normalization
        ])

        # Load SVHN dataset
        # Note: SVHN uses 'split' parameter instead of 'train'
        dataset_train = datasets.SVHN('./data/svhn', split='train', download=True, transform=trans_svhn)
        dataset_test = datasets.SVHN('./data/svhn', split='test', download=True, transform=trans_svhn)

        if args.generate_data:
            if args.iid:
                dict_users = svhn_iid(dataset_train, args.num_users)
            elif args.noniid_case < 5:
                dict_users = svhn_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                # Use Dirichlet distribution for non-IID
                dict_users = separate_data(dataset_train, args.num_users, 10, args.data_beta)  # 10 classes (0-9)
        else:
            dict_users = read_record(file)
    else:
        exit('Error: unrecognized dataset')

    if args.generate_data:
        with open(file,'w') as f:
            dataJson = {"dataset":args.dataset,"num_users":args.num_users,"iid":args.iid,"noniid_case":args.noniid_case,"data_beta":args.data_beta,"train_data":dict_users}
            json.dump(dataJson,f)

    return dataset_train, dataset_test, dict_users
