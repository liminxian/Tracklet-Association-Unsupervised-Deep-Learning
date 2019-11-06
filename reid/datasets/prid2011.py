from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path as osp
import numpy as np
from ..utils.iotools import mkdir_if_missing, write_json, read_json
import ipdb


class PRID2011(object):
    """
        PRID2011
        Reference:
        Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.
        URL: https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/

        Dataset statistics:
        # identities: 200
        # tracklets: 400
        # cameras: 2
    """
    dataset_dir = ''
    def __init__(self, root, split_id=10, min_seq_len=0):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_path = osp.join(self.dataset_dir, 'info/splits_prid2011.json')
        self.cam_a_path = osp.join(self.dataset_dir, 'multi_shot', 'cam_a')
        self.cam_b_path = osp.join(self.dataset_dir, 'multi_shot', 'cam_b')

        self._check_before_run()
        splits = read_json(self.split_path)
        print("{}/{}".format(split_id, len(splits)))
        if split_id >= len(splits):
            raise ValueError(
                "split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits) - 1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
            self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
            self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
            self._process_data(test_dirs, cam1=False, cam2=True)

        # get train set
        tid_start = 0
        train_set, num_tkls_train, num_persons_train, \
        num_tkls_pc_train, num_persons_pc_train, \
        trainval_len_pertkl, trainval_len_pertkl_percam = \
            self.Build_Set(train, relabel=True, min_seq_len=min_seq_len, tid_start=tid_start)
        # get test set
        tid_start = 0
        query_set, query_num_tracklets, query_num_pids, \
        query_num_tracklets_percam, query_num_pids_percam, \
        query_num_len_pertkl, query_len_pertkl_percam = \
            self.Build_Set(query, relabel=False, min_seq_len=min_seq_len, tid_start=tid_start)
        tid_start = query_num_tracklets
        gallery_set, gallery_num_tracklets, gallery_num_pids, \
        gallery_num_tracklets_percam, gallery_num_pids_percam, \
        gallery_len_pertkl, gallery_len_pertkl_percam = \
            self.Build_Set(gallery, relabel=False, min_seq_len=min_seq_len, tid_start=tid_start)

        num_imgs_per_tracklet = trainval_len_pertkl + query_num_len_pertkl + gallery_len_pertkl
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)
        print(" number of images per tracklet: {} ~ {}, average {}".format(min_num, max_num, avg_num))

        self.train = train_set
        self.num_train_pids = num_persons_train
        self.num_train_pids_sub = num_persons_pc_train
        self.num_train_tids_sub = num_tkls_pc_train
        self.query = query_set
        self.gallery = gallery_set
        return

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


    def Build_Set(self, dataset_raw, relabel=False, min_seq_len=0, tid_start=0):
        pid_container = set()
        cam_container = set()
        for i, (_, pid, cam) in enumerate(dataset_raw):
            pid_container.add(pid)
            cam_container.add(cam)
        num_pids = len(pid_container)
        num_cams = len(cam_container)
        num_tkls = len(dataset_raw)

        dataset = []
        len_pertkl = []
        tid = tid_start  # tid is the key in video Re-ID (as the fname in image Re-ID)
        for i in range(num_tkls):
            img_names = dataset_raw[i][0]
            pid = dataset_raw[i][1]
            cam = dataset_raw[i][2]
            assert 0 <= cam <= 1
            tid_percam = -1
            pid_percam = -1
            if len(img_names) >= min_seq_len:
                dataset.append((img_names, tid, pid, tid_percam, pid_percam, cam))
                tid += 1
                len_pertkl.append(len(img_names))

        # ----------------------------- Next: get the tid_pc and pid_pc -----------------------------#
        num_tkls_pc = []
        num_pids_pc = []
        len_pertkl_pc = []
        for i, c in enumerate(cam_container):
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














