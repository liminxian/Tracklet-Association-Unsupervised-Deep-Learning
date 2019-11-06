from __future__ import absolute_import
import numpy as np
import torch
from PIL import Image

import ipdb

class Preprocessor_Video(object):
    """
    This class deals with video-reid where each tracklet has a number
    of images and only a fixed number of images is selected.
    """
    _sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, root=None, transform=None, seq_len=1, sample='evenly'):
        super(Preprocessor_Video, self).__init__()
        assert sample in self._sample_methods, "sample must be within {}".format(self._sample_methods)
        assert transform is not None
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.seq_len = seq_len
        self.sample = sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fpaths, tid, pid, tidpc, pidpc, camid = self.dataset[index]
        num = len(fpaths) # the frame num of a tracklet

        if self.sample == 'random':
            """
            Randomly sample seq_len items from num items, if num is smaller than
            seq_len, replicating items is adopted
            """
            indices = np.arange(num)
            if num >= self.seq_len:
                indices = np.random.choice(indices, size=self.seq_len, replace=False)
            else:
                indices = np.random.choice(indices, size=self.seq_len, replace=True)
            # TODO: disable the sorting to achieve order-agnostic
            #indices = np.sort(indices)
        elif self.sample == 'evenly':
            """
            Evenly sample seq_len items from num items
            """
            assert num >= self.seq_len, "condition failed: num ({}) >= self.seq_len ({})".format(num, self.seq_len)
            num -= num % self.seq_len
            indices = np.arange(0, num, num/self.seq_len)
            assert len(indices) == self.seq_len
        elif self.sample == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs
            to be set to 1 otherwise error will occur
            """
            indices = np.arange(num)
        else:
            raise KeyError("unknown sample method: {}".format(self.sample))

        imgs = []
        fnames = []
        for idx in indices:
            fpath = fpaths[idx]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None: img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
            fnames.append(fpath)
        imgs = torch.cat(imgs, dim=0)

        # return imgs, fpaths[0], tid, pid, camid
        return imgs, fnames, tid, pid, tidpc, pidpc, camid
