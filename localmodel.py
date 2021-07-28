from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import tqdm
import torch.nn as nn

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



class LocalModel(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.dataset = DatasetSplit(dataset, idxs)
        self.idxs = idxs
        self.trainloader = DataLoader(self.dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)

    def train(self, net):
        loss = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr = self.args.lr, weight_decay=self.args.weight_decay)
        net.train()
        for iter in range(self.args.local_epochs):
            total_loss, total_num = 0.0, 0
            for data, target in self.trainloader:
                data = data.cuda()
                out = net(data)
                batch_loss = loss(data, out)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
        return net.state_dict()
