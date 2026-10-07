#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np
import torch
import swanlab

from utils.options import args_parser
from utils.set_seed import set_random_seed
from models.Update import LocalUpdate_FedAvg
from models.Nets import ResNet50, VGG16
from models.Fed import Aggregation
from models.test import test_img
from utils.get_dataset import get_dataset
from utils.utils import save_result,save_model
from Algorithm.Training_DFLMR import DFLMR
from Algorithm.Training_DFLAvg import DFLAvg
from Algorithm.Training_DecFus import DecFus

def FedAvg(net_glob, dataset_train, dataset_test, dict_users):

    net_glob.train()

    # training
    acc = []
    loss = []
    train_loss=[]

    for iter in range(args.epochs):

        print('*'*80)
        print('Round {:3d}'.format(iter))


        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        if iter % args.test_round == 0:
            item_acc,item_loss = test_with_loss(net_glob, dataset_test, args)
            ta,tl = test_with_loss(net_glob, dataset_train, args)

            swanlab.log({
                "round": iter,
                "average_test_accuracy": item_acc,
                "average_test_loss": item_loss,
                "average_train_loss": tl
            })

            acc.append(item_acc)
            loss.append(item_loss)
            train_loss.append(tl)

    save_result(acc, 'test_acc', args)
    save_result(loss, 'test_loss', args)
    save_result(train_loss, 'test_train_loss', args)
    save_model(net_glob.state_dict(), 'test_model', args)


def test_with_loss(net_glob, dataset_test, args):
    
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item(), loss_test

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    swanlab.init(
        project="DecFus",
        name=f"{args.algorithm}_{args.dataset}_{args.model}_{args.noniid_case}_{args.data_beta}",
        config=args.__dict__
    )

    set_random_seed(args.seed)

    dataset_train, dataset_test, dict_users = get_dataset(args)

    if args.model == 'resnet50':
        net_glob = ResNet50(args=args)
    elif args.model == 'vgg':
        net_glob = VGG16(args=args)

    net_glob.to(args.device)
    print(net_glob)

    if args.algorithm == 'FedAvg':
        FedAvg(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'DFLMR':
        DFLMR(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'DFLAvg':
        DFLAvg(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'DecFus':
        DecFus(args, net_glob, dataset_train, dataset_test, dict_users)
