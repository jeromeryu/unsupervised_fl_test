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
        optimizer = optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        net.train()
        total_loss, total_num, train_bar = 0.0, 0, self.trainloader
        for iter in range(self.args.local_epochs):
            for pos_1, pos_2, target in train_bar:
                pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
                feature_1, out_1 = net(pos_1)
                feature_2, out_2 = net(pos_2)
                # [2*B, D]
                print(out_1)
                out = torch.cat([out_1, out_2], dim=0)
                # [2*B, 2*B]
                sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.args.temperature)
                mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.args.batch_size, device=sim_matrix.device)).bool()
                # [2*B, 2*B-1]
                sim_matrix = sim_matrix.masked_select(mask).view(2 * self.args.batch_size, -1)

                # compute loss
                pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.args.temperature)
                # [2*B]
                pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return net.state_dict()