from tarfile import NUL
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

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
    
if __name__=='__main__':
    num_split = 10
    save_path = 'data_test/'
    for i in range(num_split):
        dataset = DatasetSplit()
        dataset = torch.load(save_path + str(i))
        # dataset = dataset.load_state_dict(torch.load(save_path + str(i)))
        print(dataset)
        
        print(len(dataset))
        dataloader = DataLoader(dataset, batch_size = 128)
        for j in dataloader:
            print(j)