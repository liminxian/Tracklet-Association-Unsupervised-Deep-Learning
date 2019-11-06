from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave
from ..utils.iotools import mkdir_if_missing, write_json, read_json

import ipdb


class CUHK03():
    """
    CUHK03
    Reference:
    Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.
    URL: http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!

    Dataset statistics:
    # identities: 1360
    # images: 13164
    # cameras: 6
    # splits: 20 (classic)
    Args:
        split_id (int): split index (default: 0)
        cuhk03_labeled (bool): whether to load labeled images; if false, detected images are loaded (default: False)
    """

    def __init__(self, root, split_id=0, cuhk03_labeled=False, cuhk03_classic_split=True, min_seq_len=0):
        self.dataset_dir = osp.join(root, '')
        self.data_dir = osp.join(self.dataset_dir, 'cuhk03_release')
        self.raw_mat_path = osp.join(self.data_dir, 'cuhk-03.mat')

        self.imgs_detected_dir = osp.join(self.dataset_dir, 'images_detected')
        self.imgs_labeled_dir = osp.join(self.dataset_dir, 'images_labeled')

        self.split_classic_det_json_path = osp.join(self.dataset_dir, 'splits_classic_detected.json')
        self.split_classic_lab_json_path = osp.join(self.dataset_dir, 'splits_classic_labeled.json')

        self.split_new_det_json_path = osp.join(self.dataset_dir, 'splits_new_detected.json')
        self.split_new_lab_json_path = osp.join(self.dataset_dir, 'splits_new_labeled.json')

        self.split_new_det_mat_path = osp.join(self.dataset_dir, 'cuhk03_new_protocol_config_detected.mat')
        self.split_new_lab_mat_path = osp.join(self.dataset_dir, 'cuhk03_new_protocol_config_labeled.mat')

        self._check_before_run()
        self._preprocess()

        if cuhk03_labeled:
            image_type = 'labeled'
            split_path = self.split_classic_lab_json_path if cuhk03_classic_split else self.split_new_lab_json_path
        else:
            image_type = 'detected'
            split_path = self.split_classic_det_json_path if cuhk03_classic_split else self.split_new_det_json_path

        splits = read_json(split_path)
        assert split_id < len(splits), "Condition split_id ({}) < len(splits) ({}) is false".format(split_id, len(splits))
        split = splits[split_id]
        print("Split index = {}".format(split_id))

        # the raw format (based on imgs) of element is [img_path, pid, cam]
        train = split['train']
        query = split['query']
        gallery = split['gallery']

        # get train set
        tid_start = 0
        train_set, num_tkls_train, num_persons_train, \
        num_tkls_pc_train, num_persons_pc_train, \
        trainval_len_pertkl, trainval_len_pertkl_percam = \
            self.Build_Set(train, relabel=True, min_seq_len=min_seq_len, tid_start=tid_start)
        # get query set
        tid_start = 0
        query_set, num_tkls_query, num_persons_query, \
        num_tkls_pc_query, num_pids_pc_query, \
        query_len_pertkl, query_len_pertkl_percam = \
            self.Build_Set(query, relabel=False, min_seq_len=min_seq_len, tid_start=tid_start)
        # get gallery set
        tid_start = num_tkls_query
        gallery_set, num_tkls_gallery, num_persons_gallery, \
        num_tkls_pc_gallery, num_pids_pc_gallery, \
        gallery_len_pertkl, gallery_len_pertkl_percam = \
            self.Build_Set(gallery, relabel=False, min_seq_len=min_seq_len, tid_start=tid_start)

        num_total_pids = num_persons_train + num_persons_query
        num_train_imgs = sum(trainval_len_pertkl)
        num_query_imgs = sum(query_len_pertkl)
        num_gallery_imgs = sum(gallery_len_pertkl)
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        self.train = train_set
        self.num_train_pids = num_persons_train
        self.num_train_pids_sub = num_persons_pc_train
        self.num_train_tids_sub = num_tkls_pc_train
        self.query = query_set
        self.gallery = gallery_set

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.raw_mat_path):
            raise RuntimeError("'{}' is not available".format(self.raw_mat_path))
        if not osp.exists(self.split_new_det_mat_path):
            raise RuntimeError("'{}' is not available".format(self.split_new_det_mat_path))
        if not osp.exists(self.split_new_lab_mat_path):
            raise RuntimeError("'{}' is not available".format(self.split_new_lab_mat_path))

    def _preprocess(self):
        """
        This function is a bit complex and ugly, what it does is
        1. Extract data from cuhk-03.mat and save as png images.
        2. Create 20 classic splits. (Li et al. CVPR'14)
        3. Create new split. (Zhong et al. CVPR'17)
        """
        print(
            "Note: if root path is changed, the previously generated json files need to be re-generated (delete them first)")
        if osp.exists(self.imgs_labeled_dir) and \
                osp.exists(self.imgs_detected_dir) and \
                osp.exists(self.split_classic_det_json_path) and \
                osp.exists(self.split_classic_lab_json_path) and \
                osp.exists(self.split_new_det_json_path) and \
                osp.exists(self.split_new_lab_json_path):
            return

        mkdir_if_missing(self.imgs_detected_dir)
        mkdir_if_missing(self.imgs_labeled_dir)

        print("Extract image data from {} and save as png".format(self.raw_mat_path))
        mat = h5py.File(self.raw_mat_path, 'r')

        def _deref(ref):
            return mat[ref][:].T

        def _process_images(img_refs, campid, pid, save_dir):
            img_paths = []  # Note: some persons only have images for one view
            for imgid, img_ref in enumerate(img_refs):
                img = _deref(img_ref)
                # skip empty cell
                if img.size == 0 or img.ndim < 3: continue
                # images are saved with the following format, index-1 (ensure uniqueness)
                # campid: index of camera pair (1-5)
                # pid: index of person in 'campid'-th camera pair
                # viewid: index of view, {1, 2}
                # imgid: index of image, (1-10)
                viewid = 1 if imgid < 5 else 2
                img_name = '{:01d}_{:03d}_{:01d}_{:02d}.png'.format(campid + 1, pid + 1, viewid, imgid + 1)
                img_path = osp.join(save_dir, img_name)
                imsave(img_path, img)
                img_paths.append(img_path)
            return img_paths

        def _extract_img(name):
            print("Processing {} images (extract and save) ...".format(name))
            meta_data = []
            imgs_dir = self.imgs_detected_dir if name == 'detected' else self.imgs_labeled_dir
            for campid, camp_ref in enumerate(mat[name][0]):
                camp = _deref(camp_ref)
                num_pids = camp.shape[0]
                for pid in range(num_pids):
                    img_paths = _process_images(camp[pid, :], campid, pid, imgs_dir)
                    assert len(img_paths) > 0, "campid{}-pid{} has no images".format(campid, pid)
                    meta_data.append((campid + 1, pid + 1, img_paths))
                print("done camera pair {} with {} identities".format(campid + 1, num_pids))
            return meta_data

        meta_detected = _extract_img('detected')
        meta_labeled = _extract_img('labeled')

        def _extract_classic_split(meta_data, test_split):
            train, test = [], []
            num_train_pids, num_test_pids = 0, 0
            num_train_imgs, num_test_imgs = 0, 0
            for i, (campid, pid, img_paths) in enumerate(meta_data):

                if [campid, pid] in test_split:
                    for img_path in img_paths:
                        camid = int(osp.basename(img_path).split('_')[2])
                        test.append((img_path, num_test_pids, camid))
                    num_test_pids += 1
                    num_test_imgs += len(img_paths)
                else:
                    for img_path in img_paths:
                        camid = int(osp.basename(img_path).split('_')[2])
                        train.append((img_path, num_train_pids, camid))
                    num_train_pids += 1
                    num_train_imgs += len(img_paths)
            return train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs

        print("Creating classic splits (# = 20) ...")
        splits_classic_det, splits_classic_lab = [], []
        for split_ref in mat['testsets'][0]:
            test_split = _deref(split_ref).tolist()

            # create split for detected images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_detected, test_split)
            splits_classic_det.append({
                'train': train, 'query': test, 'gallery': test,
                'num_train_pids': num_train_pids, 'num_train_imgs': num_train_imgs,
                'num_query_pids': num_test_pids, 'num_query_imgs': num_test_imgs,
                'num_gallery_pids': num_test_pids, 'num_gallery_imgs': num_test_imgs,
            })

            # create split for labeled images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_labeled, test_split)
            splits_classic_lab.append({
                'train': train, 'query': test, 'gallery': test,
                'num_train_pids': num_train_pids, 'num_train_imgs': num_train_imgs,
                'num_query_pids': num_test_pids, 'num_query_imgs': num_test_imgs,
                'num_gallery_pids': num_test_pids, 'num_gallery_imgs': num_test_imgs,
            })

        write_json(splits_classic_det, self.split_classic_det_json_path)
        write_json(splits_classic_lab, self.split_classic_lab_json_path)

        def _extract_set(filelist, pids, pid2label, idxs, img_dir, relabel):
            tmp_set = []
            unique_pids = set()
            for idx in idxs:
                img_name = filelist[idx][0]
                camid = int(img_name.split('_')[2])
                pid = pids[idx]
                if relabel: pid = pid2label[pid]
                img_path = osp.join(img_dir, img_name)
                tmp_set.append((img_path, int(pid), camid))
                unique_pids.add(pid)
            return tmp_set, len(unique_pids), len(idxs)

        def _extract_new_split(split_dict, img_dir):
            train_idxs = split_dict['train_idx'].flatten() - 1  # index-0
            pids = split_dict['labels'].flatten()
            train_pids = set(pids[train_idxs])
            pid2label = {pid: label for label, pid in enumerate(train_pids)}
            query_idxs = split_dict['query_idx'].flatten() - 1
            gallery_idxs = split_dict['gallery_idx'].flatten() - 1
            filelist = split_dict['filelist'].flatten()
            train_info = _extract_set(filelist, pids, pid2label, train_idxs, img_dir, relabel=True)
            query_info = _extract_set(filelist, pids, pid2label, query_idxs, img_dir, relabel=False)
            gallery_info = _extract_set(filelist, pids, pid2label, gallery_idxs, img_dir, relabel=False)
            return train_info, query_info, gallery_info

        print("Creating new splits for detected images (767/700) ...")
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_det_mat_path),
            self.imgs_detected_dir,
        )
        splits = [{
            'train': train_info[0], 'query': query_info[0], 'gallery': gallery_info[0],
            'num_train_pids': train_info[1], 'num_train_imgs': train_info[2],
            'num_query_pids': query_info[1], 'num_query_imgs': query_info[2],
            'num_gallery_pids': gallery_info[1], 'num_gallery_imgs': gallery_info[2],
        }]
        write_json(splits, self.split_new_det_json_path)

        print("Creating new splits for labeled images (767/700) ...")
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_lab_mat_path),
            self.imgs_labeled_dir,
        )
        splits = [{
            'train': train_info[0], 'query': query_info[0], 'gallery': gallery_info[0],
            'num_train_pids': train_info[1], 'num_train_imgs': train_info[2],
            'num_query_pids': query_info[1], 'num_query_imgs': query_info[2],
            'num_gallery_pids': gallery_info[1], 'num_gallery_imgs': gallery_info[2],
        }]
        write_json(splits, self.split_new_lab_json_path)

    def Build_Set(self, dataset_raw, relabel=False, min_seq_len=0, tid_start=0):
        # the format of dataset_raw is [img_path, pid, cam]

        # Step_1: compute the tracklet number of dataset_raw
        pid_container = set()
        cam_container = set()
        for i, (_, pid, cam) in enumerate(dataset_raw):
            pid_container.add(pid)
            cam_container.add(cam)
        num_pids = len(pid_container)
        num_cams = len(cam_container)
        num_imgs = len(dataset_raw)

        # Step_2: get the dataset(based on tkl)
        dataset = []
        len_pertkl = []
        tid = tid_start
        for pid_idx, pid in enumerate(pid_container):
            for cam_idx, cam in enumerate(cam_container):
                img_names = []
                for i in range(num_imgs):
                    if dataset_raw[i][1] == pid and dataset_raw[i][2] == cam:
                        img_names.append(dataset_raw[i][0])
                if len(img_names) > min_seq_len:
                    img_names = tuple(img_names)
                    tid_pc = -1
                    pid_pc = -1
                    cam = cam - 1 # cam start from 0
                    dataset.append((img_names, tid, pid, tid_pc, pid_pc, cam))
                    tid += 1
                    len_pertkl.append(len(img_names))
        num_tkls = len(dataset)

        # ----------------------------- Next: get the tid_pc and pid_pc -----------------------------#
        num_tkls_pc = []
        num_pids_pc = []
        len_pertkl_pc = []
        for c in range(num_cams):
            # count tid per camera
            tkl_index_pc = [index for index, (_, _, _, _, _, camid) in enumerate(dataset) if camid == c]
            num_tkls_pc.append(len(tkl_index_pc))

            # count pid per camera
            pid_list_pc = [dataset[i][2] for i in tkl_index_pc]
            unique_pid_list_pc = list(set(pid_list_pc))
            num_pids_pc.append(len(unique_pid_list_pc))

            # count image number per tracklet
            len_pertkl_pc.append([len_pertkl[i] for i in tkl_index_pc])

            pid_percam2label = {pid: label for label, pid in enumerate(unique_pid_list_pc)}
            for i, tkl_index in enumerate(tkl_index_pc):
                tid = dataset[tkl_index][1]
                pid = dataset[tkl_index][2]
                tid_pc = i
                pid_pc = pid_percam2label[pid]
                cam = dataset[tkl_index][5]
                dataset[tkl_index] = (dataset[tkl_index][0], tid, pid, tid_pc, pid_pc, cam)
        assert num_tkls == sum(num_tkls_pc)
        # # check if pid starts from 0 and increments with 1
        # for idx, pid in enumerate(pid_container):
        #     assert idx == pid, "See code comment for explanation"
        return dataset, num_tkls, num_pids, num_tkls_pc, num_pids_pc, len_pertkl, len_pertkl_pc