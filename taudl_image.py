from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
import time
import datetime
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import PosetLoss_G2G
from reid.trainers import Trainer
from reid.evaluators_image import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor_image import Preprocessor_Image
from reid.utils.data.sampler_mt import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
import ipdb


def flatten_dataset(dataset):
    new_dataset = []
    for tracklet in dataset:
        img_names, tid, pid, tid_pc, pid_pc, camid = tracklet
        for img_name in img_names:
            new_dataset.append((img_name, tid, pid, tid_pc, pid_pc, camid))
    return new_dataset


def get_data(name, split_id, data_dir, height, width, batch_size, num_instances, workers, combine_trainval):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root, split_id=split_id)
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # load train_set, query_set, gallery_set
    mt_train_set = dataset.train
    mt_num_classes = dataset.num_train_tids_sub
    query_set = dataset.query
    gallery_set = dataset.gallery

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    # Random ID
    mt_train_set = flatten_dataset(mt_train_set)
    num_task = len(mt_num_classes)  # num_task equals camera number, each camera is a task
    mt_train_loader = DataLoader(
        Preprocessor_Image(mt_train_set, root=dataset.dataset_dir, transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(mt_train_set, num_instances, num_task),  # Here is different between softmax_loss
        pin_memory=True, drop_last=True)

    query_set = flatten_dataset(query_set)
    gallery_set = flatten_dataset(gallery_set)
    test_set = list(set(query_set) | set(gallery_set))
    test_loader = DataLoader(
        Preprocessor_Image(test_set, root=dataset.dataset_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return mt_train_loader, mt_num_classes, test_loader, query_set, gallery_set


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    start = time.time()

    # Redirect print to both console and log file
    if not args.evaluate:
        dt = datetime.datetime.now()
        sys.stdout = Logger(osp.join(args.logs_dir, 'log_'
                                     + str(dt.month).zfill(2)
                                     + str(dt.day).zfill(2)
                                     + str(dt.hour).zfill(2)
                                     + str(dt.minute).zfill(2) + '.txt'))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
    mt_train_loader, mt_num_classes, test_loader, query_set, gallery_set = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instances, args.workers,
                 args.combine_trainval)

    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=mt_num_classes, double_loss=True)
    model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model)

    # Criterion
    criterion_1 = nn.CrossEntropyLoss().cuda()
    criterion_2 = PosetLoss_G2G(margin=args.margin).cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Trainer
    num_task = len(mt_num_classes)  # num_task equals camera number, each camera is a task
    trainer = Trainer(model, criterion_1, criterion_2, num_task)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr if epoch <= 100 else \
            args.lr * (0.001 ** ((epoch - 100) / 50.0))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    start_epoch = best_top1 = 0
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, mt_train_loader, optimizer)
        if (epoch % args.start_save == (args.start_save - 1)):
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
            }, 0, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            # Final test
            print('Test with the model after epoch {:d}:'.format(epoch + 1))
            checkpoint = load_checkpoint(osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            model.module.load_state_dict(checkpoint['state_dict'])
            metric.train(model, mt_train_loader)
            evaluator.evaluate(test_loader, query_set, gallery_set, metric)
    end = time.time()
    print('Total time: {:.1f}s'.format(end-start))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation", default=True)
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    # loss
    parser.add_argument('--margin', type=float, default=0.1,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4) #0.1?
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    main(parser.parse_args())
