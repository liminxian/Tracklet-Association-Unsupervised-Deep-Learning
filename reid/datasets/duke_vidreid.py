from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import zipfile
import os.path as osp
from reid.utils.iotools import mkdir_if_missing, write_json, read_json
from .bases import BaseVideoDataset
import ipdb


class DukeMTMC_VidReID(BaseVideoDataset):
    """
    DukeMTMCVidReID

    Reference:
    Wu et al. Exploit the Unknown Gradually: One-Shot Video-Based Person
    Re-Identification by Stepwise Learning. CVPR 2018.

    URL: https://github.com/Yu-Wu/DukeMTMC-VideoReID
    
    Dataset statistics:
    # identities: 702 (train) + 702 (test)
    # tracklets: 2196 (train) + 2636 (test)
    """
    dataset_dir = ''

    def __init__(self, root='data', min_seq_len=0, verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip'
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self.split_train_json_path = osp.join(self.dataset_dir, 'info/split_train_ext.json')
        self.split_query_json_path = osp.join(self.dataset_dir, 'info/split_query_ext.json')
        self.split_gallery_json_path = osp.join(self.dataset_dir, 'info/split_gallery_ext.json')

        self.min_seq_len = min_seq_len
        self._download_data()
        self._check_before_run()
        print("Note: if root path is changed, the previously generated json files need to be re-generated (so delete them first)")

        train = self._process_dir(self.train_dir, self.split_train_json_path, relabel=True)
        query = self._process_dir(self.query_dir, self.split_query_json_path, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.split_gallery_json_path, relabel=False)

        if verbose:
            print("=> DukeMTMC-VideoReID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_tkls, self.num_train_cams, self.num_train_pids_sub, self.num_train_tids_sub \
            = self.get_videodata_info(self.train)
        self.num_query_pids, self.num_query_tkls, self.num_query_cams, _, _ = self.get_videodata_info(self.query)
        self.num_gallery_pids, self.num_gallery_tkls, self.num_gallery_cams, _, _ = self.get_videodata_info(self.gallery)

    def _download_data(self):
        if osp.exists(self.dataset_dir):
            print("This dataset has been downloaded.")
            return

        print("Creating directory {}".format(self.dataset_dir))
        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        # print("Downloading DukeMTMC-VideoReID dataset")
        # urllib.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(self.dataset_dir)
        zip_ref.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, json_path, relabel):
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return split['tracklets']

        print("=> Automatically generating split (might take a while for the first time, have a coffe)")
        pdirs = sorted(glob.glob(osp.join(dir_path, '*'))) # avoid .DS_Store
        print("Processing '{}' with {} person identities".format(dir_path, len(pdirs)))

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        cam_list = []
        tracklets = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            if relabel: pid = pid2label[pid]
            tdirs = sorted(glob.glob(osp.join(pdir, '*')))

            for tdir in tdirs:
                tid = int(osp.basename(tdir)) - 1
                raw_img_paths = sorted(glob.glob(osp.join(tdir, '*.jpg')))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                img_paths = []
                for img_idx in range(num_imgs):
                    # some tracklet starts from 0002 instead of 0001
                    img_idx_name = 'F' + str(img_idx+1).zfill(4)
                    res = sorted(glob.glob(osp.join(tdir, '*' + img_idx_name + '*.jpg')))
                    if len(res) == 0:
                        print("Warn: index name {} in {} is missing, jump to next".format(img_idx_name, tdir))
                        continue
                    img_paths.append(res[0])
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1:
                    # old naming format: 0001C6F0099X30823.jpg
                    camid = int(img_name[5]) - 1
                else:
                    # new naming format: 0001_C6_F0099_X30823.jpg
                    camid = int(img_name[6]) - 1
                img_paths = tuple(img_paths)
                tid_sub = -1
                pid_sub = -1
                tracklets.append((img_paths, tid, pid, tid_sub, pid_sub, camid))
                cam_list.append(camid)

        num_cams = len(list(set(cam_list)))
        start_tid_uic = 0
        for cam_index in range(num_cams):
            # count tid per camera
            tkl_index_list = [index for index, (_, _, _, _, _, camid) in enumerate(tracklets) if camid == cam_index]
            # count pid per camera
            pid_list_sub = [tracklets[j][2] for j in tkl_index_list]
            unique_pid_list_percam = list(set(pid_list_sub))
            start_tid_uic += len(unique_pid_list_percam)

            pid_percam2label = {pid: label for label, pid in enumerate(unique_pid_list_percam)}

            for index, tkl_index in enumerate(tkl_index_list):
                img_paths = tracklets[tkl_index][0]
                tid = tracklets[tkl_index][1]
                pid = tracklets[tkl_index][2]
                tid_sub = index
                pid_sub = pid_percam2label[pid]
                camid = tracklets[tkl_index][5]
                tracklets[tkl_index] = (img_paths, tid, pid, tid_sub, pid_sub, camid)

        print("Saving split to {}".format(json_path))
        split_dict = {
            'tracklets': tracklets,
        }
        write_json(split_dict, json_path)
        return tracklets