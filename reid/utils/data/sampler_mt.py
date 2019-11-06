from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)

import ipdb

def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))

class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1, num_task=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.num_task = num_task
        self.index_dic_mt = nested_dict(2, list)

        for index, (_, _, _, tid_pc, _, camid) in enumerate(data_source):
            self.index_dic_mt[camid][tid_pc].append(index)         # camid equals taskid

        self.pids = [0]*num_task
        self.num_samples = [0]*num_task
        for cam_index in range(num_task):
            self.pids[cam_index] = list(self.index_dic_mt[cam_index].keys())
            self.num_samples[cam_index] = len(self.pids[cam_index])

    def __len__(self):
        num_samples = 0
        for t in range(self.num_task):
            num_samples += self.num_samples[t]
        return num_samples * self.num_instances

    def __iter__(self):
        ret = []
        indices = nested_dict(self.num_task, list)
        for t in range(self.num_task):
            indices[t] = torch.randperm(self.num_samples[t])
        # train_num = max(self.num_samples)
        train_num = min(self.num_samples)

        for tkl_index in range(train_num):
            for cam_index in range(self.num_task):
                loop_tkl_index = tkl_index % len(self.pids[cam_index])
                i = indices[cam_index][loop_tkl_index]   # pid_index has been shuffled
                pid = self.pids[cam_index][i]
                fid = self.index_dic_mt[cam_index][pid]
                if len(fid) >= self.num_instances:
                    fid = np.random.choice(fid, size=self.num_instances, replace=False)
                else:
                    fid = np.random.choice(fid, size=self.num_instances, replace=True)
                ret.extend(fid)
        return iter(ret)
