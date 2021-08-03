import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cae import ConvAutoencoder, ContrastiveLoss
# import ae.utils
import linear 
import os
from tqdm import tqdm
import argparse
from localmodel import LocalModel
import copy

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
    parser.add_argument('--num_users', type=int, default=10)
    parser.add_argument('--fraction', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # load the training and test datasets
    train_data = datasets.CIFAR10(root='data', train=True,
                                    download=True, transform=transform)
    test_data = datasets.CIFAR10(root='data', train=False,
                                    download=True, transform=transform)

    # prepare data loaders
    # train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)


    # non fl setting
    # loss = nn.MSELoss()
    # model = ConvAutoencoder().to(device)
    # optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
    # num_epochs = 50

    # for epoch in range(num_epochs):
    #     losses = []
    #     for data, target in train_loader:
    #         data = data.to(device)
    #         out = model(data)
    #         batch_loss = loss(data, out)
    #         losses.append(batch_loss.item())
    #         optimizer.zero_grad()
    #         batch_loss.backward()
    #         optimizer.step()
    #     running_loss = np.mean(losses)
    #     print(f'Epoch {epoch}: {running_loss}')
    #     if epoch % 10 == 9:
    #         ae.utils.display_output(data, out, 32, 32, '{} ori.png'.format(epoch/10), '{} recon.png'.format(epoch/10))

    user_groups = cifar_iid(train_data, args.num_users)


    global_model = ConvAutoencoder().to(device)
    for epoch in tqdm(range(args.epochs)):
        local_weights = []
        global_model.train()
        for i in range(args.num_users): #since fraction is 1
            local_model = LocalModel(args, train_data, user_groups[i], device)
            w = local_model.train(net = copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w))
    
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)



    train_data_linear = datasets.CIFAR10(root='data', train=True,
                                    download=True, transform=transform)
    test_data_linear = datasets.CIFAR10(root='data', train=False,
                                    download=True, transform=transform)

    train_loader_linear = DataLoader(train_data_linear, batch_size=args.batch_size, shuffle=True)
    test_loader_linear = DataLoader(test_data_linear, batch_size=args.batch_size, shuffle=True)

    net = cae.linear.Net(num_class=len(train_data_linear.classes), net = global_model).to(device)
    for param in net.f.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(net.fc.parameters(), lr=1e-3, weight_decay=1e-6)


    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
                'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    if not os.path.exists('results'):
        os.mkdir('results')

    for epoch in range(1, args.linear_epochs + 1):
        train_loss, train_acc_1, train_acc_5 = cae.linear.train_val(net, train_loader_linear, optimizer, device)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = cae.linear.train_val(net, test_loader_linear, None, device)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        
        print('{} : {} {} {}'.format(epoch, test_loss, test_acc_1, test_acc_5))

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/fedae_linear_statistics.csv', index_label='epoch')
        # if test_acc_1 > best_acc:
        #     best_acc = test_acc_1
        #     torch.save(model.state_dict(), 'results/linear_model.pth')


