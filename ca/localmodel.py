import itertools
from torch import functional, jit
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
from model import Model

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class LocalModel(object):
    def __init__(self, args, dataset, idxs, device, alignment_loader):
        self.args = args
        self.dataset = DatasetSplit(dataset, idxs)
        self.idxs = idxs
        self.device = device    
        self.trainloader = DataLoader(self.dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)
        self.z = torch.zeros(1024, self.args.feature_dim).float().to(self.device) # intermediate values
        self.Z = torch.zeros(1024, self.args.feature_dim).float().to(self.device) # temporal outputs
        self.outputs = torch.zeros(len(self.dataset), self.args.feature_dim).float().to(self.device)   # current outputs
        self.alignment_model = Model(args.feature_dim).to(device)
        # self.alignment_dataset = alignment_dataset
        self.alignment_loader = iter(alignment_loader)


    def train(self, net, global_dict, round):
        optimizer = optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
 
        criterion = nn.CrossEntropyLoss()

        net.train()
        total_loss, total_num, train_bar = 0.0, 0, self.trainloader
        a_idx = 0
        
        for iter in range(self.args.local_epochs):
            total_loss, total_num = 0.0, 0
            for i, (pos_1, pos_2, target) in enumerate(train_bar):
                pos_1, pos_2 = pos_1.to(self.device), pos_2.to(self.device)
                h_i, z_i = net(pos_1)
                h_j, z_j = net(pos_2)
                #contrastive loss
                self.outputs[i * self.args.batch_size: (i + 1) * self.args.batch_size] = z_i.data.clone()
                logit_batch = torch.mm(z_i, torch.t(z_j))
                logit_dict = torch.mm(z_i, torch.t(global_dict))
                # labels = range(self.args.batch_size)
                labels = torch.LongTensor(range(self.args.batch_size)).to(self.device)
                logit_total = torch.cat([logit_batch, logit_dict], dim = 1).to(self.device)
                # print(logit_batch.shape, logit_dict.shape, logit_total.shape)
                
                loss_c = criterion(logit_total, labels)
                
                loss_h = 0
                loss_z = 0
                
                pos, target = next(self.alignment_loader)
                pos = pos.to(self.device)
                h_a, z_a = net(pos)
                h = torch.norm(torch.sub(h_a, h_i), dim=1)
                loss_h += torch.vdot(h, h)
                z = torch.norm(torch.sub(z_a, z_i), dim=1)
                loss_z += torch.vdot(z, z)
        
                # for chk, (pos, target) in enumerate(self.alignment_loader):
                    # if chk==a_idx:
                        # pos = pos.to(self.device)
                        # h_a, z_a = net(pos)
                        # h = torch.norm(torch.sub(h_a, h_i), dim=1)
                        # loss_h += torch.vdot(h, h)
                        # z = torch.norm(torch.sub(z_a, z_i), dim=1)
                        # loss_z += torch.vdot(z, z)
                        # break
                    
                # a_idx += 1
                # if a_idx >= len(self.alignment_loader.dataset)/self.args.batch_size:
                #     a_idx = 0

                loss_a = loss_h + loss_z
                loss = loss_c + self.args.beta * loss_a
                
                # print("loss", loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        self.Z = self.args.alpha * self.Z + (1. - self.args.alpha) * self.outputs[:1024]
        # self.z = self.Z * (1. / (1. - self.args.alpha ** (round + 1)))
        self.z = F.normalize(self.z, dim = 1) # 각각?

        return net.state_dict(), self.z, loss