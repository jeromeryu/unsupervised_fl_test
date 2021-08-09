from threading import local
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.types import Device
from torch.utils.data.sampler import RandomSampler
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import linear 
import os
from torchvision.transforms.transforms import ToTensor
from tqdm import tqdm
import argparse
from localmodel import LocalModel
import copy
from model import Model
import utils


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--linear_epochs', type=int, default=100)
    parser.add_argument('--num_users', type=int, default=5)
    parser.add_argument('--fraction', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--num_alignment_sample', type=int, default=3200)

    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    # convert data to torch.FloatTensor
    # transform = transforms.ToTensor()
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    # load the training and test datasets
    train_data = utils.CIFAR10Pair(root='../data', train=True, 
                                   transform=utils.train_transform, download=True)

    alignment_data = datasets.STL10(root='../data', split='train', 
                                   transform=transform, download=True)

    alignment_sampler = RandomSampler(alignment_data, replacement=True,
                                   num_samples=args.num_alignment_sample)
    alignment_dataloader = torch.utils.data.DataLoader(alignment_data, batch_size=args.batch_size, sampler=alignment_sampler)

    
    user_groups = cifar_iid(train_data, args.num_users)
    print(user_groups)

    global_model = Model(args.feature_dim).to(device)
    global_dictionary = torch.zeros(1024, args.feature_dim).float().to(device)
    
    clients = []
    for i in range(args.num_users):
        local_model = LocalModel(args, train_data, user_groups[i], device, copy.deepcopy(alignment_dataloader))
        clients.append(local_model)
    
    bar = tqdm(range(args.epochs))
    for round in bar:
        local_weights = []
        global_model.train()
        m = max(int(args.fraction * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        loss = 0
        lst_dict = []
        for i in idxs_users: #since fraction is 1
            # local_model = LocalModel(args, train_data, user_groups[i], device)
            local_model = clients[i]
            w, dict_u, l = local_model.train(copy.deepcopy(global_model), global_dictionary, round)
            loss += l
            lst_dict += [dict_u]
            local_weights.append(copy.deepcopy(w))

        loss = loss / len(idxs_users)
        
        # loss = loss / args.num_users
        global_dictionary = torch.cat(lst_dict, dim = 1)
        
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
    
        bar.set_description('Mean Loss: {}'.format(loss))    

    
    train_data_linear = datasets.CIFAR10(root='../data', train=True,
                                    download=True, transform=utils.train_transform)
    test_data_linear = datasets.CIFAR10(root='../data', train=False,
                                    download=True, transform=utils.test_transform)

    train_loader_linear = DataLoader(train_data_linear, batch_size=args.batch_size, shuffle=True)
    test_loader_linear = DataLoader(test_data_linear, batch_size=args.batch_size, shuffle=True)

    net = linear.Net(num_class=len(train_data_linear.classes), net = global_model).to(device)
    for param in net.f.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(net.fc.parameters(), lr=1e-3, weight_decay=1e-6)

    
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
            'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}
 
    for epoch in range(1, args.linear_epochs + 1):
        train_loss, train_acc_1, train_acc_5 = linear.train_val(net, train_loader_linear, optimizer, device)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = linear.train_val(net, test_loader_linear, None, device)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        
        print('{} : {} {} {}'.format(epoch, test_loss, test_acc_1, test_acc_5))

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('../results/fedca_linear_statistics.csv', index_label='epoch')
