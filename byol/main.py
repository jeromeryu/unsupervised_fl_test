import os
import argparse

import torch
import yaml
from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from trainer import BYOLTrainer
from tqdm import tqdm
import numpy as np
import copy
from torch.utils.data import DataLoader
import linear
import torch.optim as optim
import pandas as pd
from torchvision import transforms

print(torch.__version__)
torch.manual_seed(0)


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

def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_simclr_data_transforms(**config['data_transforms'])

    train_dataset = datasets.STL10('/home/thalles/Downloads/', split='train+unlabeled', download=True,
                                   transform=MultiViewDataInjector([data_transform, data_transform]))

    # online network
    online_network = ResNet18(**config['network']).to(device)
    pretrained_folder = config['network']['fine_tune_from']

    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = ResNet18(**config['network']).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])

    trainer.train(train_dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--linear_epochs', type=int, default=100)
    parser.add_argument('--num_users', type=int, default=10)
    parser.add_argument('--fraction', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_simclr_data_transforms(**config['data_transforms'])

    train_dataset = datasets.CIFAR10(root='../data', train=True, transform=MultiViewDataInjector([data_transform, data_transform]), download=True)
        # online network
    online_network = ResNet18(**config['network']).to(device)
    pretrained_folder = config['network']['fine_tune_from']

    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = ResNet18(**config['network']).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()), **config['optimizer']['params'])

    user_groups = cifar_iid(train_dataset, args.num_users)


    for epoch in tqdm(range(1, args.epochs + 1)):
        local_online_weights = []
        local_target_weights = []
        local_predictor_wrights = []
        m = max(int(args.fraction * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        loss = 0
        for i in idxs_users:
            local_trainer = BYOLTrainer(online_network=copy.deepcopy(online_network),
                          target_network=copy.deepcopy(target_network),
                          predictor=copy.deepcopy(predictor),
                          device=device,
                          dataset=train_dataset,
                          idxs = user_groups[i],
                          args = args,
                          **config)
            o, t, p, l = local_trainer.train()
            loss += l
            local_online_weights.append(o)
            local_target_weights.append(t)
            local_predictor_wrights.append(p)
        
        loss = loss / len(idxs_users)
        print("loss ", epoch, loss)
        
        global_online = average_weights(local_online_weights)
        global_target = average_weights(local_target_weights)
        global_predict = average_weights(local_predictor_wrights)
        online_network.load_state_dict(global_online)
        target_network.load_state_dict(global_target)
        predictor.load_state_dict(global_predict)
    
        
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


    train_data_linear = datasets.CIFAR10(root='../data', train=True,
                                    download=True, transform=train_transform)
    test_data_linear = datasets.CIFAR10(root='../data', train=False,
                                    download=True, transform=test_transform)

    train_loader_linear = DataLoader(train_data_linear, batch_size=args.batch_size, shuffle=True)
    test_loader_linear = DataLoader(test_data_linear, batch_size=args.batch_size, shuffle=True)

    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
            'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}
 

    net = linear.Net(num_class=len(train_data_linear.classes), net = global_online).to(device)
    for param in net.f.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(net.fc.parameters(), lr=1e-3, weight_decay=1e-6)


    for epoch in range(1, args.linear_epochs + 1):
        train_loss, train_acc_1, train_acc_5 = linear.train_val(net, train_loader_linear, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = linear.train_val(net, test_loader_linear, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        
        print('{} : {} {} {}'.format(epoch, test_loss, test_acc_1, test_acc_5))

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('../results/fedbyol_linear_statistics.csv', index_label='epoch')
