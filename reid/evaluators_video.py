from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter

import ipdb

def pooling(inputs, method='average'):
    _pooling_methods = ['average', 'max']
    assert method in _pooling_methods, "method must be within {}".format(_pooling_methods)

    # num_frames = inputs.shape[0]
    dim = inputs.shape[1]

    if method == 'average':
        feature = torch.mean(inputs, 0)
        assert feature.shape[0] == dim
        return feature
    elif method == 'max':
        feature, _ = torch.max(inputs, 0)
        assert feature.shape[0] == dim
        return feature


def extract_features(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features_tkl = OrderedDict()
    labels = OrderedDict()
    cams = OrderedDict()
    end = time.time()

    # VERY IMPORTANT: tid is the key in video Re-ID (as the fname in image Re-ID)
    for i, (imgs, fnames, tid, pid, _, _, cam) in enumerate(data_loader): # traverse all data in data_loader(test_data)
        data_time.update(time.time() - end)

        tkl_batch_size = imgs.size(0)
        num_instances = imgs.size(1)
        img_batch_size = tkl_batch_size*num_instances

        imgs = imgs.view(img_batch_size, imgs.size(2), imgs.size(3), imgs.size(4))
        features = extract_cnn_feature(model, imgs, cam)

        for j in range(tkl_batch_size):
            index = tid[j]
            features_tkl[index] = features[j*num_instances:(j+1)*num_instances]
            labels[index] = pid[j]
            cams[index] = cam[j]

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features_tkl, labels, cams

def extract_features_pooling(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features_tkl = OrderedDict()
    labels = OrderedDict()
    cams = OrderedDict()
    end = time.time()

    # VERY IMPORTANT: tid is the key in video Re-ID (as the fname in image Re-ID)
    for i, (imgs, fnames, tid, pid, _, _, cam) in enumerate(data_loader): # traverse all data in data_loader(test_data)
        data_time.update(time.time() - end)

        tkl_batch_size = imgs.size(0)
        num_instances = imgs.size(1)
        img_batch_size = tkl_batch_size * num_instances
        imgs = imgs.view(img_batch_size, imgs.size(2), imgs.size(3), imgs.size(4))

        features = extract_cnn_feature(model, imgs, cam)
        for j in range(tkl_batch_size):
            index = tid[j].item()
            features_tkl[index] = pooling(features[j*num_instances:(j+1)*num_instances], 'average')
            labels[index] = pid[j]
            cams[index] = cam[j]

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features_tkl, labels, cams


def pairwise_distance(features, query=None, gallery=None, metric=None):
    x = torch.cat([features[tid].unsqueeze(0) for _, tid, _, _, _, _ in query], 0)
    y = torch.cat([features[tid].unsqueeze(0) for _, tid, _, _, _, _ in gallery], 0)

    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def set_min_distance(features, query=None, gallery=None, metric=None):
    x = torch.cat([features[tid] for _, tid, _, _, _, _ in query], 0)
    query_set_num = len(query)
    y = torch.cat([features[tid] for _, tid, _, _, _, _ in gallery], 0)
    gallery_set_num = len(gallery)

    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())

    num_instances = m / query_set_num
    set_dist = torch.FloatTensor(query_set_num, gallery_set_num).zero_().cuda()
    start = time.time()
    for i in range(query_set_num):
        for j in range(gallery_set_num):
            set_dist[i, j] = torch.min(dist[i*num_instances:(i+1)*num_instances, j*num_instances:(j+1)*num_instances])
    end = time.time()
    print('Get set_dist time:{:.1f}s'.format(end-start))

    return set_dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, _, pid, _, _, _ in query]
        gallery_ids = [pid for _, _, pid, _, _, _ in gallery]
        query_cams = [cam for _, _, _, _, _, cam in query]
        gallery_cams = [cam for _, _, _, _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True, # In cuhk03, query and gallery sets are from different camera views.
                       single_gallery_shot=True, # The gallery just includes a camera view
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False, # In Market-1501, query and gallery sets could have same camera views.
                           single_gallery_shot=False, # The gallery includes multiple camera views
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['allshots'][0]


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None):
        features_tkl, labels, cams = extract_features_pooling(self.model, data_loader)
        distmat = pairwise_distance(features_tkl, query, gallery, metric=metric)
        return evaluate_all(distmat, query=query, gallery=gallery)

    # def evaluate(self, data_loader, query, gallery, metric=None):
    #     features_tkl, labels, cams = extract_features(self.model, data_loader)
    #     distmat = set_min_distance(features_tkl, query, gallery, metric=metric)
    #     return evaluate_all(distmat, query=query, gallery=gallery)
