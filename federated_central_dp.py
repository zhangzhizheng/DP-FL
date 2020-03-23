#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch.distributions as tdist

import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import statistics

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

from differential_privacy.privacy_accountant.pytorch import accountant

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def gaussian_noise(bound, sigma):
    n = tdist.Normal(0, sigma * sigma)
    return n.sample()


def compute_local_model_update(global_weights, local_weights):
    if len(global_weights) != len(local_weights):
        raise AssertionError("Two models have different length in number of clients")

    norm_values = []

    for i in range(len(global_weights)):
        norm_values.append(torch.norm(global_weights[i][1] - local_weights[i][1]))

    return statistics.median(norm_values)


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    args.gpu = 0
    args.epochs = 20
    args.num_users = 100
    args.local_ep = 5
    # args.iid = False
    # args.unequal = True
    args.model = 'cnn'
    args.epsilon = 10
    args.max_epsilon = 8
    args.delta = 10 ** (-5)
    args.sigma = 0.8
    args.noise_scale = 3
    args.dataset = "cifar"
    args.frac = 0.3
    args.lr = 0.01

    exp_details(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    if args.dataset == 'mnist':
        NUM_TRAINING_IMAGES = 60000
    elif args.dataset == 'fmnist':
        NUM_TRAINING_IMAGES = 60000
    elif args.dataset == 'cifar':
        NUM_TRAINING_IMAGES = 60000

    priv_accountant = accountant.GaussianMomentsAccountant(args.num_users)

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        l2norms = []

        for idx in idxs_users:
            print("user id ", idx)
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss, l2norm = local_model.central_update_weights(copy.deepcopy(global_model), epoch, args)
            l2norms.append(l2norm)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        sensitivity = statistics.median(l2norms).item()
        args.sigma = (sensitivity * args.noise_scale) / (args.frac * args.num_users)

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        # add noise
        global_model.add_noise(args.sigma, sensitivity)

        priv_accountant.accumulate_privacy_spending(args.noise_scale, m)

        print("-----------", priv_accountant.get_privacy_spent(target_deltas=[args.delta]))
        # if priv_accountant.get_privacy_spent(target_deltas=[args.delta])[0].spent_delta > args.max_epsilon:
        #     break

        loss_avg = sum(local_losses) / len(local_losses)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        # if (epoch+1) % print_every == 0:
        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

        # added for test
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

# def gradient_clip(param):
#     """Clip gradient to ensure ||param.grad||2 < bound"""
#     torch.nn.utils.clip_grad_norm([param], bound)



# update global weights
#         global_weights = average_weights(local_weights)
#
#         l2norms = []
#         for local_weight in local_weights:
#             l2norms.append(compute_local_model_update(list(global_weights.items()), list(local_weight.items())))


# add noise
# for param in global_model.parameters():
#     noise = gaussian_noise(sensitivity, args.sigma)
#     param.data.add_(noise)