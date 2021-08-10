import itertools
from torch import functional, jit
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
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
        self.alignment_loader = alignment_loader

    def train_alignment(self, idx):
        optimizer = optim.Adam(self.alignment_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.alignment_model.train()
        
        for it in tqdm(range(50)):
            for pos_1, pos_2, target in self.alignment_loader:
                pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
                feature_1, out_1 = self.alignment_model(pos_1)
                feature_2, out_2 = self.alignment_model(pos_2)
                # [2*B, D]
                # print(out_1)
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
                
        torch.save(self.alignment_model.state_dict(0, './'+str(idx)+'_alignment.pth'))

        

    def train(self, net, global_dict, round):
        optimizer = optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        alignment_optimizer = optim.Adam(self.alignment_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
 
        criterion = nn.CrossEntropyLoss()

        net.train()
        total_loss, total_num, train_bar = 0.0, 0, self.trainloader
        a_idx = 0
        
        it = iter(self.alignment_loader)
        
        for i in range(self.args.local_epochs):
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
                
                loss_c = criterion(torch.div(logit_total, self.args.temperature), labels)
                
                loss_h = 0
                loss_z = 0
                
                try:
                    pos, pos2, target = next(it)
                except StopIteration:
                    it = iter(self.alignment_loader)
                    pos, pos2, target = next(it)
                
                    
                pos = pos.to(self.device)
                h_net, z_net = net(pos)
                h_a, z_a = self.alignment_model(pos)
                h = torch.norm(torch.sub(h_a, h_net), dim=1)
                loss_h += torch.vdot(h, h)
                z = torch.norm(torch.sub(z_a, z_net), dim=1)
                loss_z += torch.vdot(z, z)

                
                # out = torch.cat([z_net, z_a], dim=0)
                # # [2*B, 2*B]
                # sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.args.temperature)
                # mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.args.batch_size, device=sim_matrix.device)).bool()
                # # [2*B, 2*B-1]
                # sim_matrix = sim_matrix.masked_select(mask).view(2 * self.args.batch_size, -1)

                # # compute loss
                # pos_sim = torch.exp(torch.sum(z_net * z_a, dim=-1) / self.args.temperature)
                # # [2*B]
                # pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                # alignment_loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
                # alignment_optimizer.zero_grad()
                # alignment_loss.backward(retain_graph=True)
                # alignment_optimizer.step()
                
                
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
                
                print(loss_c, loss_a, loss)    
                    
                # print("loss", loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        self.Z = self.args.alpha * self.Z + (1. - self.args.alpha) * self.outputs[:1024]
        # self.z = self.Z * (1. / (1. - self.args.alpha ** (round + 1)))
        self.z = F.normalize(self.z, dim = 1) # 각각?

        return net.state_dict(), self.z, loss