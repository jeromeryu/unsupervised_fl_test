import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cae import ConvAutoencoder, ContrastiveLoss
from ae import utils

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test datasets
train_data = datasets.CIFAR10(root='../data', train=True,
                                   download=True, transform=transform)
test_data = datasets.CIFAR10(root='../data', train=False,
                                  download=True, transform=transform)
num_workers = 0
# how many samples per batch to load
batch_size = 20

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
        utils.display_output(data, out, 32, 32, '1.png', '2.png')



