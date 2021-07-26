import os
import sys

import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.decomposition import PCA
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset, dataset
from torchvision import datasets, transforms


# tell jupyter where are local modules are
module_path = os.path.abspath('.')
if module_path not in sys.path:
    sys.path.append(module_path)

from ae.dae import DAE, Naive_DAE
from ae.demo_train_utils import train_rbm
from ae.rbm import RBM
from ae.utils import *


# train = MNIST(MNIST_DIR, train=True, download=False, transform=torchvision.transforms.ToTensor())
# test = MNIST(MNIST_DIR, train=False, download=False, transform=torchvision.transforms.ToTensor())
# train_dl = DataLoader(train, batch_size=64, shuffle=False)
# test_dl = DataLoader(test, batch_size=64, shuffle=False)

# MNIST_NUM_PIXELS = 784 # 28x28
CIFAR_NUM_PIXELS = 1024 # 32X32

train = datasets.CIFAR10(root='data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test = datasets.CIFAR10(root='data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
train_dl = DataLoader(train, batch_size=64, shuffle=False)
test_dl = DataLoader(test, batch_size=64, shuffle=False)

def flatten_input(dl):
    flat_input = []
    labels = []
    for features, targets in train_dl:
        flat_input.append(features.view(-1, CIFAR_NUM_PIXELS).detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())
    return np.concatenate(flat_input), np.concatenate(labels)

flat_train_input, train_labels = flatten_input(train_dl)
flat_test_input, test_labels = flatten_input(test_dl)
train_dl_flat = DataLoader(
    TensorDataset(torch.Tensor(flat_train_input).to(DEVICE)),
    batch_size=64,
    shuffle=False
)

hidden_dimensions = [
    {
        "hidden_dim": 1000, 
        "num_epochs": 10, 
        "learning_rate": 0.1, 
        "display_dim1": 28, 
        "display_dim2": 28, 
        "use_gaussian": False
    }, 
    {
        "hidden_dim": 500, 
        "num_epochs": 10, 
        "learning_rate": 0.1, 
        "display_dim1": 25, 
        "display_dim2": 40, 
        "use_gaussian": False
    },
    {
        "hidden_dim": 250, 
        "num_epochs": 10, 
        "learning_rate": 0.1, 
        "display_dim1": 25, 
        "display_dim2": 20, 
        "use_gaussian": False
    },
    {
        "hidden_dim": 2, 
        "num_epochs": 30, 
        "learning_rate": 0.001, # use much lower LR for gaussian to avoid exploding gradient
        "display_dim1": 25, 
        "display_dim2": 10, 
        "use_gaussian": True # use a Gaussian distribution for the last hidden layer to let it take advantage of continuous values
    }
]

new_train_dl = train_dl_flat

visible_dim = CIFAR_NUM_PIXELS
hidden_dim = None
models = [] # trained RBM models
for configs in hidden_dimensions:
    print("config")    
    # parse configs
    hidden_dim = configs["hidden_dim"]
    num_epochs = configs["num_epochs"]
    lr = configs["learning_rate"]
    d1 = configs["display_dim1"]
    d2 = configs["display_dim2"]
    use_gaussian = configs["use_gaussian"]
    # train RBM
    print(f"{visible_dim} to {hidden_dim}")
    model, v, v_pred = train_rbm(new_train_dl, visible_dim, hidden_dim, k=1, num_epochs=num_epochs, lr=lr, use_gaussian=use_gaussian)
    models.append(model)
    
    # display sample output
    display_output(v, v_pred, d1, d2)
    print("display")
    # rederive new data loader based on hidden activations of trained model
    new_data = []
    for data_list in new_train_dl:
        p = model.sample_h(data_list[0])[0]
        new_data.append(p.detach().cpu().numpy())
    new_input = np.concatenate(new_data)
    new_train_dl = DataLoader(
        TensorDataset(torch.Tensor(new_input).to(DEVICE)), 
        batch_size=64, 
        shuffle=False
    )
    # update new visible_dim for next RBM
    visible_dim = hidden_dim


# fine-tune autoencoder
lr = 1e-3
dae = DAE(models).to(DEVICE)
loss = nn.MSELoss()
optimizer = optim.Adam(dae.parameters(), lr)
num_epochs = 50

# train
for epoch in range(num_epochs):
    losses = []
    for i, data_list in enumerate(train_dl_flat): 
        data = data_list[0]
        v_pred = dae(data)
        batch_loss = loss(data, v_pred) # difference between actual and reconstructed   
        losses.append(batch_loss.item())
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    running_loss = np.mean(losses)
    print(f"Epoch {epoch}: {running_loss}")
    if epoch % 10 == 9:
        # show visual progress every 10 epochs
        display_output(data, v_pred, v0_fname="images/original_digits.png", vk_fname="images/reconstructed_digits_dae.png")