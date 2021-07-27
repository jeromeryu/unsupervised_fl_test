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

from cae.cae import ConvAutoencoder, ContrastiveLoss
import ae.utils
import cae.linear 
import os

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test datasets
train_data = datasets.CIFAR10(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.CIFAR10(root='data', train=False,
                                  download=True, transform=transform)
num_workers = 0
# how many samples per batch to load
batch_size = 64

# prepare data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

lr = 1e-3
weight_decay=1e-6
loss = nn.MSELoss()

model = ConvAutoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
num_epochs = 50

for epoch in range(num_epochs):
    losses = []
    for data, target in train_loader:
        data = data.to(device)
        out = model(data)
        batch_loss = loss(data, out)
        losses.append(batch_loss.item())
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    running_loss = np.mean(losses)
    print(f'Epoch {epoch}: {running_loss}')
    if epoch % 10 == 9:
        ae.utils.display_output(data, out, 32, 32, '{} ori.png'.format(epoch/10), '{} recon.png'.format(epoch/10))



train_data_linear = datasets.CIFAR10(root='data', train=True,
                                   download=True, transform=transform)
test_data_linear = datasets.CIFAR10(root='data', train=False,
                                  download=True, transform=transform)

train_loader_linear = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader_linear = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

linear_epoch = 100
net = cae.linear.Net(num_class=len(train_data_linear.classes), net = model).cuda()
for param in net.f.parameters():
    param.requires_grad = False
optimizer = optim.Adam(net.fc.parameters(), lr=1e-3, weight_decay=1e-6)


results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
            'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

if not os.path.exists('results'):
    os.mkdir('results')

for epoch in range(1, linear_epoch + 1):
    train_loss, train_acc_1, train_acc_5 = cae.linear.train_val(net, train_loader_linear, optimizer)
    results['train_loss'].append(train_loss)
    results['train_acc@1'].append(train_acc_1)
    results['train_acc@5'].append(train_acc_5)
    test_loss, test_acc_1, test_acc_5 = cae.linear.train_val(net, test_loader_linear, None)
    results['test_loss'].append(test_loss)
    results['test_acc@1'].append(test_acc_1)
    results['test_acc@5'].append(test_acc_5)
    print(test_loss, test_acc_1, test_acc_5)
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
    data_frame.to_csv('results/cae_linear_statistics.csv', index_label='epoch')
    # if test_acc_1 > best_acc:
    #     best_acc = test_acc_1
    #     torch.save(model.state_dict(), 'results/linear_model.pth')

