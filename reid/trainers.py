from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
from .evaluation_metrics import accuracy
from .loss import PosetLoss_G2G
from .utils.meters import AverageMeter
import ipdb

class BaseTrainer(object):
    def __init__(self, model, criterion_1, criterion_2, num_task):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion_1 = criterion_1
        self.criterion_2 = criterion_2
        self.num_task = num_task

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            # inputs is the image data, targets is the pid label, camid is the cam
            inputs, targets, camid = self._parse_data(inputs)
            loss, loss_1, loss_2, prec1, loss_1_time, loss_2_time = self._forward(inputs, targets, camid)

            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Loss_1 {:.3f} \t'
                      'Loss_2 {:.3f} \t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      'Loss_1_Time {:.3f} ({:.3f})\t'
                      'Loss_2_Time {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              loss_1.item(),
                              loss_2.item(),
                              # 0,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg,
                              loss_1_time.val, loss_1_time.avg,
                              loss_2_time.val, loss_2_time.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets, camid):
        raise NotImplementedError

def unique_index(L, e):
    return [i for i, v in enumerate(L) if v == e]

class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, fname, _, _, tids_pc, _, camids = inputs  # image_data, image_names, tids, pids, tids_pc, pids_pc, cams

        inputs = imgs.cuda()
        targets = tids_pc.cuda()
        camids = camids.cuda()
        return inputs, targets, camids

    def _forward(self, inputs, labels, camid):
        if isinstance(self.criterion_1, torch.nn.CrossEntropyLoss) and isinstance(self.criterion_2, PosetLoss_G2G):
            loss_1_time = AverageMeter()
            loss_2_time = AverageMeter()

            end = time.time()
            # compute loss_1
            loss_1 = Variable(torch.FloatTensor(1).zero_().cuda())
            prec = 0
            batch_features = []
            for t in range(self.num_task):
                sample_index = torch.LongTensor(unique_index(camid, t)).cuda()
                if len(sample_index) > 0:
                    labels_t = torch.index_select(labels, 0, sample_index)  # labels in task t
                    inputs_t = Variable(torch.index_select(inputs, 0, sample_index))
                    prelogits_cam_i, features_cam_i = self.model(inputs_t, t)

                    # loss_1
                    loss_1 += self.criterion_1(prelogits_cam_i, Variable(labels_t))
                    prec_1, = accuracy(prelogits_cam_i.data, labels_t)
                    prec += prec_1[0]

                    batch_features.append(features_cam_i)   # concentrate the features for computing loss_2
            batch_features = torch.cat(batch_features)
            prec = prec / self.num_task
            loss_1_time.update(time.time() - end)

            # compute loss_2
            end = time.time()
            loss_2 = self.criterion_2(batch_features, labels, camid)

            # sum
            lamda = 0.7
            loss = (1 - lamda) * loss_1 + lamda * loss_2
        else:
            raise ValueError("Unsupported loss:", self.criterion_1)
        return loss, (1 - lamda) * loss_1, lamda * loss_2, prec, loss_1_time, loss_2_time
