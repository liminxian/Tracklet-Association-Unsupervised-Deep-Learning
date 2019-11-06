from __future__ import print_function, absolute_import
import os.path as osp


class MSMT17(object):
    """
    MSMT17
    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """

    def __init__(self, root='data', split_id=0, min_seq_len=0, **kwargs):
        self.dataset_dir = osp.join(root, '')
        self.train_dir = osp.join(self.dataset_dir, 'mask_train_v2')
        self.test_dir = osp.join(self.dataset_dir, 'mask_test_v2')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()
        # get train set
        tid_start = 0
        train_set, num_tkls_train, num_persons_train, \
        num_tkls_pc_train, num_persons_pc_train, \
        trainval_len_pertkl, trainval_len_pertkl_percam = \
            self.Build_Set(self.train_dir, self.list_train_path, min_seq_len=min_seq_len, tid_start=tid_start)
        # get query set
        tid_start = 0
        query_set, num_tkls_train_query, num_persons_query, \
        num_tkls_pc_query, num_pids_pc_query, \
        query_len_pertkl, query_len_pertkl_percam = \
            self.Build_Set(self.test_dir, self.list_query_path, min_seq_len=min_seq_len, tid_start=tid_start)
        # get gallery set
        tid_start = num_tkls_train_query
        gallery_set, num_tkls_train_gallery, num_persons_gallery, \
        num_tkls_pc_gallery, num_pids_pc_gallery, \
        gallery_len_pertkl, gallery_len_pertkl_percam = \
            self.Build_Set(self.test_dir, self.list_gallery_path, min_seq_len=min_seq_len, tid_start=tid_start)

        num_total_pids = num_persons_train + num_persons_query
        num_train_imgs = sum(trainval_len_pertkl)
        num_query_imgs = sum(query_len_pertkl)
        num_gallery_imgs = sum(gallery_len_pertkl)
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> MSMT17 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_persons_train, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_persons_query, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_persons_gallery, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

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
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def Build_Set(self, dir_path, list_path, min_seq_len=0, tid_start=0):
        # Get the dataset_img
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset_img = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            cam = int(img_path.split('_')[2])
            img_name = osp.join(dir_path, img_path)

            # dataset.append((img_path, pid, camid))
            dataset_img.append((img_name, pid, cam))
            pid_container.add(pid)
            cam_container.add(cam)
        num_imgs = len(dataset_img)
        num_pids = len(pid_container)
        num_cams = len(cam_container)

        # Get the dataset(tkl)
        dataset = []
        len_pertkl = []
        tid = tid_start  # tid is the key in video Re-ID (as the fname in image Re-ID)
        for pid_idx, pid in enumerate(pid_container):
            for cam_idx, cam in enumerate(cam_container):
                img_names = []
                for i in range(num_imgs):
                    if dataset_img[i][1] == pid and dataset_img[i][2] == cam:
                        img_names.append(dataset_img[i][0])
                if len(img_names) > min_seq_len:
                    img_names = tuple(img_names)
                    tid_pc = -1
                    pid_pc = -1
                    cam = cam - 1 # cam start from 0
                    dataset.append((img_names, tid, pid, tid_pc, pid_pc, cam))
                    tid += 1
                    len_pertkl.append(len(img_names))
        num_tkls = len(dataset)

        #----------------------------- Next: get the tid_pc and pid_pc -----------------------------#
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

            pid_percam2label = {pid:label for label, pid in enumerate(unique_pid_list_pc)}
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

    # def _process_dir(self, dir_path, list_path):
    #     with open(list_path, 'r') as txt:
    #         lines = txt.readlines()
    #     dataset = []
    #     pid_container = set()
    #     for img_idx, img_info in enumerate(lines):
    #         img_path, pid = img_info.split(' ')
    #         pid = int(pid)  # no need to relabel
    #         camid = int(img_path.split('_')[2])
    #         img_path = osp.join(dir_path, img_path)
    #         dataset.append((img_path, pid, camid))
    #         pid_container.add(pid)
    #     num_imgs = len(dataset)
    #     num_pids = len(pid_container)
    #     # check if pid starts from 0 and increments with 1
    #     for idx, pid in enumerate(pid_container):
    #         assert idx == pid, "See code comment for explanation"
    #     return dataset, num_pids, num_imgs