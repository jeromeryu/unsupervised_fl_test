from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import tqdm
import torch.nn as nn
import torch

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
    def __init__(self, args, dataset, idxs, device):
        self.args = args
        self.dataset = DatasetSplit(dataset, idxs)
        self.idxs = idxs
        self.device = device
        self.trainloader = DataLoader(self.dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)

    def train(self, net):
        # loss = nn.TripletMarginLoss()
        # loss = nn.KLDivLoss()
        optimizer = optim.Adam(net.parameters(), lr = self.args.lr, weight_decay=self.args.weight_decay)
        net.train()
        for iter in range(self.args.local_epochs):
            total_loss, total_num = 0.0, 0
            for data, target in self.trainloader:
                data = data.to(self.device)
                target = target.to(self.device)
                out = net(data)


                outcat = torch.cat([torch.flatten(data), torch.flatten(out)], dim = 0)
                a = outcat.t().contiguous()
                b = torch.mm(outcat, a)
                sim_matrix = torch.exp(torch.mm(outcat, outcat.t().contiguous()) / 0.5)
                mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.args.batch_size, device=sim_matrix.device)).bool()
                sim_matrix = sim_matrix.masked_select(mask).view(2 * self.args.batch_size, -1)
                pos_sim = torch.exp(torch.sum(data * out, dim=-1) / 0.5)
                pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # batch_loss = loss(data, out)
                # optimizer.zero_grad()
                # batch_loss.backward()
                # optimizer.step()
        return net.state_dict()
