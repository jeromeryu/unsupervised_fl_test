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

                bsz = target.shape[0]
                f1, f2 = torch.split(out, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                mask = torch.eye(self.args.batch_size, dtype=torch.float32).to(self.device)
                contrast_count = features.shape[1]
                contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
                anchor_feature = contrast_feature
                anchor_count = contrast_count
                anchor_dot_contrast = torch.div(
                    torch.matmul(anchor_feature, contrast_feature.T),
                    0.5)
                # for numerical stability
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast - logits_max.detach()
                # tile mask
                mask = mask.repeat(anchor_count, contrast_count)
                # mask-out self-contrast cases
                logits_mask = torch.scatter(
                    torch.ones_like(mask),
                    1,
                    torch.arange(self.args.batch_size * anchor_count).view(-1, 1).to(self.device),
                    0
                )
                mask = mask * logits_mask

                # compute log_prob
                exp_logits = torch.exp(logits) * logits_mask
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

                # compute mean of log-likelihood over positive
                mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

                # loss
                loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                loss = loss.view(anchor_count, self.args.batch_size).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # outcat = torch.cat([data, out], dim = 0)
                # sim_matrix = torch.exp(torch.mm(outcat, outcat.t().contiguous()) / 0.5)
                # mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.args.batch_size, device=sim_matrix.device)).bool()
                # sim_matrix = sim_matrix.masked_select(mask).view(2 * self.args.batch_size, -1)
                # pos_sim = torch.exp(torch.sum(data * out, dim=-1) / 0.5)
                # pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                # loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                
                # batch_loss = loss(data, out)
                # optimizer.zero_grad()
                # batch_loss.backward()
                # optimizer.step()
        return net.state_dict()
