from tarfile import NUL
from numpy import sign
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import json


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset=None, idxs=None):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return self.dataset[self.idxs[item]]
    
def get_dataset(dataset_path):
    import numpy as np
    import os
    
    with open(os.path.join(dataset_path, 'cifar10_split.json')) as json_file:
        json_data = json.load(json_file)
    for i in range(10):
        json_data[i] = np.array(json_data[i])
    return json_data
    
if __name__=='__main__':
    dataset_path = '/st1/jyryu/data'
    
    train_data = datasets.CIFAR10(root=dataset_path, train=True, download=True)
    user_dict = get_dataset(dataset_path)

    for key in user_dict.keys():
        print(key)
        
        
    
    for i in range(10):
        dataset = DatasetSplit(train_data, user_dict[i])
        trainloader = DataLoader(dataset, batch_size=128)
        for j, (a, b) in enumerate(trainloader):
            print(j)