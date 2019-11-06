from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import torch
import random
from scipy.io import loadmat
from collections import defaultdict

def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n - 1, type))


def flatten_dataset(dataset):
    new_dataset = []
    for tracklet in dataset:
        img_names, tid, pid, camid = tracklet
        for img_name in img_names:
            new_dataset.append((img_name, tid, pid, camid))
    return new_dataset


class Mars(object):
    def __init__(self, root, split_id=0, min_seq_len=0):
        super(Mars, self).__init__()

        # configure path
        self.dataset_dir = osp.join(root, '')
        train_name_path = osp.join(self.dataset_dir, 'info/train_name.txt')
        test_name_path = osp.join(self.dataset_dir, 'info/test_name.txt')
        track_train_path = osp.join(self.dataset_dir, 'info/tracks_train_info.mat')
        track_test_path = osp.join(self.dataset_dir, 'info/tracks_test_info.mat')
        query_IDX_path = osp.join(self.dataset_dir, 'info/query_IDX.mat')

        # build train & test set according to the configure files (unit: tracklet)
        track_train = loadmat(track_train_path)['track_train_info']  # numpy.ndarray (8298, 4)
        track_test = loadmat(track_test_path)['track_test_info']    # numpy.ndarray (12180, 4)
        query_IDX = loadmat(query_IDX_path)['query_IDX'].squeeze()
        query_IDX -= 1  # index from 0
        track_query = track_test[query_IDX, :]                       # numpy.ndarray (1980,4)
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]                   # numpy.ndarray (10200,4)
        train_names = self._get_names(train_name_path)
        test_names = self._get_names(test_name_path)

        # get train set
        tid_start = 0
        # trainval_set, trainval_num_tracklets, trainval_num_pids, \
        # trainval_num_tracklets_percam, trainval_num_pids_percam, \
        # trainval_len_pertkl, trainval_len_pertkl_percam = \
        #     self.Get_Set(train_names, track_train, home_dir=osp.join(self.dataset_dir,'bbox_train'),
        #                  relabel=True, min_seq_len=min_seq_len, tid_start=tid_start)
        trainval_set, trainval_num_tracklets, trainval_num_pids, \
        trainval_num_tracklets_percam, trainval_num_pids_percam, \
        trainval_len_pertkl = \
            self.Get_Set_1T4P(train_names, track_train, home_dir=osp.join(self.dataset_dir, 'bbox_train'),
                         multitask=True, relabel=True, min_seq_len=min_seq_len, tid_start=tid_start)

        # get test set
        tid_start = 0
        query_set, query_num_tracklets, query_num_pids, \
        query_num_tracklets_percam, query_num_pids_percam, \
        query_num_len_pertkl, query_len_pertkl_percam = \
            self.Get_Set(test_names, track_query, home_dir=osp.join(self.dataset_dir,'bbox_test'),
                         relabel=False, min_seq_len=min_seq_len, tid_start=tid_start)

        tid_start = query_num_tracklets
        gallery_set, gallery_num_tracklets, gallery_num_pids, \
        gallery_num_tracklets_percam, gallery_num_pids_percam, \
        gallery_len_pertkl, gallery_len_pertkl_percam = \
            self.Get_Set(test_names, track_gallery, home_dir=osp.join(self.dataset_dir,'bbox_test'),
                         relabel=False, min_seq_len=min_seq_len, tid_start=tid_start)

        # display info
        print("=========================================================================================")
        print("", self.__class__.__name__, "dataset loaded")
        print("      subset   | # ids | # tracklets | Cam 1 | Cam 2 | Cam 3 | Cam 4 | Cam 5 | Cam 6 |")
        print("     ---------------------------------------------------------------------------------")
        print("      trainval | {:5d} | {:8d}    | {:5d} | {:5d} | {:5d} | {:5d} | {:5d} | {:5d} |"
              .format(trainval_num_pids, trainval_num_tracklets,
                      trainval_num_tracklets_percam[0],trainval_num_tracklets_percam[1],
                      trainval_num_tracklets_percam[2], trainval_num_tracklets_percam[3],
                      trainval_num_tracklets_percam[4], trainval_num_tracklets_percam[5]))
        print("      query    | {:5d} | {:8d}    | {:5d} | {:5d} | {:5d} | {:5d} | {:5d} | {:5d} |"
              .format(query_num_pids, query_num_tracklets,
                      query_num_tracklets_percam[0], query_num_tracklets_percam[1],
                      query_num_tracklets_percam[2], query_num_tracklets_percam[3],
                      query_num_tracklets_percam[4], query_num_tracklets_percam[5]))
        print("      gallery  | {:5d} | {:8d}    | {:5d} | {:5d} | {:5d} | {:5d} | {:5d} | {:5d} |"
              .format(gallery_num_pids, gallery_num_tracklets,
                      gallery_num_tracklets_percam[0], gallery_num_tracklets_percam[1],
                      gallery_num_tracklets_percam[2], gallery_num_tracklets_percam[3],
                      gallery_num_tracklets_percam[4], gallery_num_tracklets_percam[5]))

        num_imgs_per_tracklet = trainval_len_pertkl + query_num_len_pertkl + gallery_len_pertkl
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)
        print(" number of images per tracklet: {} ~ {}, average {}".format(min_num, max_num, avg_num))

        self.train = trainval_set
        self.train_num_pids = trainval_num_pids
        self.num_train_pids_sub = trainval_num_pids_percam
        self.num_train_tids_sub = trainval_num_tracklets_percam
        self.query = query_set
        self.gallery = gallery_set
        return

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def Get_Set(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0, tid_start = 0):
        # assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        raw_pid_list = list(set(meta_data[:, 2]))
        num_pids = len(raw_pid_list)
        num_cams = len(set(meta_data[:, 3]))
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(raw_pid_list)}
        else:
            pid2label = {pid: pid for label, pid in enumerate(raw_pid_list)}

        dataset = []
        len_pertkl = []
        tid = tid_start  # tid is the key in video Re-ID (as the fname in image Re-ID)
        for i in range(num_tracklets):
            data = meta_data[i,...]
            start_index, end_index, raw_pid, camid = data
            assert 1 <= camid <= 6
            camid -= 1
            if raw_pid == -1: continue  # junk images are just ignored
            img_names = names[start_index-1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "error: a single tracklet contains different person images!"
            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "error: images are captured under different cameras!"

            # append image names with directory information
            img_names = [osp.join(home_dir, img_name[:4], img_name) for img_name in img_names]
            img_names = tuple(img_names)
            pid = pid2label[raw_pid]
            tid_percam = -1
            pid_percam = -1
            if len(img_names) >= min_seq_len:
                dataset.append((img_names, tid, pid, tid_percam, pid_percam, camid))
                tid += 1
                len_pertkl.append(len(img_names))
        num_tracklets = len(dataset)

        num_tracklets_percam = []
        num_pids_percam = []
        len_pertkl_percam = []
        for i in range(num_cams):
            # count tid per camera
            tkl_index_list = [index for index, (_, _, _, _, _, camid) in enumerate(dataset) if camid == i]
            num_tracklets_percam.append(len(tkl_index_list))
            # count pid per camera
            pid_list_percam = [dataset[j][2] for j in tkl_index_list]
            unique_pid_list_percam = list(set(pid_list_percam))
            num_pids_percam.append(len(unique_pid_list_percam))
            # count image number per tracklet
            len_pertkl_percam.append([len_pertkl[j] for j in tkl_index_list])

            pid_percam2label = {pid: label for label, pid in enumerate(unique_pid_list_percam)}

            for j, tkl_index in enumerate(tkl_index_list):
                tid = dataset[tkl_index][1]
                pid = dataset[tkl_index][2]
                tid_percam = j
                pid_percam = pid_percam2label[pid]
                camid = dataset[tkl_index][5]
                dataset[tkl_index] = (dataset[tkl_index][0], tid, pid, tid_percam, pid_percam, camid)
        assert num_tracklets == sum(num_tracklets_percam)
        return dataset, num_tracklets, num_pids, num_tracklets_percam, num_pids_percam, len_pertkl, len_pertkl_percam


    def Get_Set_1T4P(self, names, meta_data, home_dir=None, multitask=False, relabel=False, min_seq_len=0, tid_start=0):
        # assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        raw_pid_list = list(set(meta_data[:, 2]))
        num_pids = len(raw_pid_list)
        num_cams = len(set(meta_data[:, 3]))
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(raw_pid_list)}
        else:
            pid2label = {pid: pid for label, pid in enumerate(raw_pid_list)}

        dataset = []
        len_pertkl = []
        tid = tid_start  # tid is the key in video Re-ID (as the fname in image Re-ID)
        for i in range(num_tracklets):
            data = meta_data[i, ...]
            start_index, end_index, raw_pid, camid = data
            assert 1 <= camid <= 6
            camid -= 1
            img_names = names[start_index - 1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "error: a single tracklet contains different person images!"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "error: images are captured under different cameras!"

            # append image names with directory information
            img_names = [osp.join(home_dir, img_name[:4], img_name) for img_name in img_names]
            img_names = tuple(img_names)
            pid = pid2label[raw_pid]
            tid_percam = -1
            pid_percam = -1
            if len(img_names) >= min_seq_len:
                dataset.append((img_names, tid, pid, tid_percam, pid_percam, camid))
                tid += 1
                len_pertkl.append(len(img_names))
        num_tracklets = len(dataset)

        if multitask == False:
            return dataset, num_tracklets, num_pids, len_pertkl
        else:
            mt_num_tracklets = []
            num_pids_pc = []
            mt_num_imgs_per_tracklet = []
            for i in range(num_cams):
                tkl_index_list = [index for index, (_, _, _, _, _, camid) in enumerate(dataset) if camid == i]
                mt_num_tracklets.append(len(tkl_index_list))
                mt_num_imgs_per_tracklet.append([len_pertkl[j] for j in tkl_index_list])

                id_cami_list = [dataset[j][2] for j in tkl_index_list]
                id_unique_cami_list = list(set(id_cami_list))
                num_pids_pc.append(len(id_unique_cami_list))

            # Scheme 3: choose 1 tracklet per pid, rate of pid have more than 1 tracklet (test duplicate rate)
            tindex = 0
            new_dataset = []
            num_tids_pc_new = []
            for t in range(num_cams):
                tkl_index_list = [index for index, (_, _, _, _, _, camid) in enumerate(dataset) if camid == t]
                assert len(tkl_index_list) == mt_num_tracklets[t]

                num_sampling = num_pids_pc[t]
                tkl_per_pid = 1
                id_cami_list = [dataset[j][2] for j in tkl_index_list]
                id_unique_cami_list = list(set(id_cami_list))
                id_indices = torch.randperm(len(id_unique_cami_list))
                tkl_sample_no = 0
                for pid_sample_no in range(num_pids_pc[t]):
                    # Step_1: after shuffling the pid order, pick up the pid one by one
                    pid_anchor = id_unique_cami_list[id_indices[pid_sample_no]]
                    # Step_2: get all tracklets belong to pid_anchor
                    tid_list = []
                    for tid in tkl_index_list:
                        if dataset[tid][2] == pid_anchor:
                            tid_list.append(tid)
                    # Step_3: pick up (tkl_per_pid) tracklets randomly
                    if len(tid_list) >= tkl_per_pid:
                        random.shuffle(tid_list)  # shuffle the tid order
                        for j in range(tkl_per_pid):
                            tkl_index = tid_list[j]
                            fnames = dataset[tkl_index][0]
                            tid = tindex
                            pid = dataset[tkl_index][2]
                            tid_percam = tkl_sample_no
                            pid_percam = pid_sample_no
                            camid = dataset[tkl_index][5]

                            assert camid == t
                            new_dataset.append((fnames, tid, pid, tid_percam, pid_percam, camid))
                            tindex += 1
                            tkl_sample_no += 1
                    else:
                        for j in range(len(tid_list)):
                            tkl_index = tid_list[j]
                            fnames = dataset[tkl_index][0]
                            tid = tindex
                            pid = dataset[tkl_index][2]
                            tid_percam = tkl_sample_no
                            pid_percam = pid_sample_no
                            camid = dataset[tkl_index][5]

                            assert camid == t
                            new_dataset.append((fnames, tid, pid, tid_percam, pid_percam, camid))
                            tindex += 1
                            tkl_sample_no += 1
                        for j in range(tkl_per_pid - len(tid_list)):
                            tkl_index = tid_list[j]
                            fnames = dataset[tkl_index][0]
                            tid = tindex
                            pid = dataset[tkl_index][2]
                            tid_percam = tkl_sample_no
                            pid_percam = pid_sample_no
                            camid = dataset[tkl_index][5]

                            assert camid == t
                            new_dataset.append((fnames, tid, pid, tid_percam, pid_percam, camid))
                            tindex += 1
                            tkl_sample_no += 1
                    if tkl_sample_no == tkl_per_pid * num_sampling:
                        break
                assert tkl_sample_no == tkl_per_pid * num_sampling

                # # Increment sampling
                # tkl_sample_no = tkl_sample_no # keep tkl_sample_no
                # rate = 0.2
                # Inc_num_pids = int (mt_num_pids[t]*rate)
                # tkl_per_pid = 1
                # # select pid
                # id_cami_list = [dataset[j][2] for j in index_list]
                # id_unique_cami_list = list(set(id_cami_list))
                # # id_indices = torch.randperm(len(id_unique_cami_list))
                # for pid_sample_no in range(Inc_num_pids):  # Increment pids
                #     pid_index = id_unique_cami_list[id_indices[pid_sample_no]]
                #
                #     # get all tracklets of pid_index
                #     tkl_list = []
                #     for index in index_list:
                #         if dataset[index][2] == pid_index:
                #             tkl_list.append(index)
                #     if len(tkl_list) >= tkl_per_pid:
                #         # random.shuffle(tkl_list)  # randomly select (tkl_per_pid) tracklets
                #         for j in range(tkl_per_pid):
                #             fnames = dataset[tkl_list[j]][0]
                #             camid = dataset[tkl_list[j]][3]
                #             assert camid == t
                #             new_dataset.append((fnames, tindex, tkl_sample_no, camid))
                #             tindex += 1
                #             tkl_sample_no += 1
                #     else:
                #         for j in range(len(tkl_list)):
                #             fnames = dataset[tkl_list[j]][0]
                #             camid = dataset[tkl_list[j]][3]
                #             assert camid == t
                #             new_dataset.append((fnames, tindex, tkl_sample_no, camid))
                #             tindex += 1
                #             tkl_sample_no += 1
                #         for j in range(tkl_per_pid - len(tkl_list)):
                #             fnames = dataset[tkl_list[0]][0]
                #             camid = dataset[tkl_list[0]][3]
                #             assert camid == t
                #             new_dataset.append((fnames, tindex, tkl_sample_no, camid))
                #             tindex += 1
                #             tkl_sample_no += 1
                # assert tkl_sample_no == mt_num_pids[t] + Inc_num_pids
                num_tids_pc_new.append(tkl_sample_no)
            num_tracklets_percam = num_tids_pc_new
            num_pids_percam = num_pids_pc
            num_tracklets = sum(num_tracklets_percam)

            return new_dataset, num_tracklets, num_pids, num_tracklets_percam, num_pids_percam, len_pertkl






