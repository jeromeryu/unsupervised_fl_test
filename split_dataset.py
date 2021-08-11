import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import json

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return self.dataset[self.idxs[item]]


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


if __name__=='__main__':
    data_path = 'data_test'
    num_split = 10
    save_path = 'data_test/cifar_10_split.json'

    train_data = datasets.CIFAR10(root=data_path, train=True, download=True)
    user_groups = cifar_iid(train_data, num_split)
    
    json_obj = dict()
    for i in range(num_split):
        # dataset = DatasetSplit(train_data, user_groups[i])
        # trainloader = DataLoader(dataset, batch_size=128, )
        # torch.save(dataset, save_path + str(i))
        json_obj[i] = user_groups[i]
    
    print(json_obj)
    
    with open(save_path, 'w') as json_file:
        json.dump(json_obj, json_file)
    
    
