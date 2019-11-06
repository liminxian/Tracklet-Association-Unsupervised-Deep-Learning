from __future__ import absolute_import

import torch
import math
from torch import nn
from torch.autograd import Variable
from ..utils.meters import AverageMeter
import time
import ipdb


class PosetLoss_G2G(nn.Module):
    def __init__(self, margin=0):
        super(PosetLoss_G2G, self).__init__()
        self.margin = margin
        # self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def unique_index(self, L, e):
        return [i for i, v in enumerate(L) if v == e]

    def forward(self, batch_features, labels, camid):
        sample_num = batch_features.size(0)  # sample_num is batchsize
        dim = batch_features.size(1)
        task_num = len(set(camid))

        # batch_features to group_features
        num_instances = 0
        for i_sample in range(sample_num):
            if camid[i_sample] == camid[0]:
                num_instances += 1
            else:
                break
        group_num = int(sample_num / num_instances)  # m is the num of classes in minibatch

        # Compute the mask via camid
        mask = torch.ByteTensor(group_num, sample_num).zero_().cuda()
        for i_group in range(group_num):
            for j_instance in range(num_instances):
                mask[i_group][i_group * num_instances + j_instance] = 1
        group_features = []
        group_labels = []
        group_camid = []
        for i in range(group_num):
            feature = batch_features[mask[i].nonzero().squeeze(), :]
            feature_mean = torch.mean(feature, 0).unsqueeze(0)

            label_mean = labels[mask[i].nonzero().squeeze()][0]
            camid_mean = camid[mask[i].nonzero().squeeze()][0]

            group_features.append(feature_mean)
            group_labels.append(label_mean)
            group_camid.append(camid_mean)
        group_features = torch.cat(group_features)
        group_labels = torch.LongTensor(group_labels).cuda()
        group_camid = torch.LongTensor(group_camid).cuda()

        # Group to Group of the batch
        x = group_features
        y = group_features
        n = group_num
        m = group_num
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, m) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, n).t()
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        sigma = 4
        exp_dist = torch.exp(-dist / (2 * sigma))

        mask = torch.ByteTensor(n, m).zero_().cuda()
        dist_near = Variable(torch.FloatTensor(n).zero_().cuda())
        K = int(task_num / 2)
        for i in range(n):
            taskID = group_camid[i]
            mask[i] = (group_camid != taskID)
            # len of dist_neib_cross less than len of exp_dist
            dist_neib_cross = torch.masked_select(exp_dist[i].data, mask[i])
            value_cross, index_cross = torch.sort(dist_neib_cross, 0, descending=True) # True: big2small False: small2big
            dist_near[i] = torch.sum(value_cross[:K])
        dist_all = torch.sum(exp_dist, 1)

        # compute poset_loss
        quotient = dist_near / dist_all
        # entropy_loss = -torch.sum(quotient * torch.log(quotient))/sample_num
        poset_loss = torch.sum(-torch.log(quotient)) / n
        assert poset_loss.item() >= 0
        return poset_loss
