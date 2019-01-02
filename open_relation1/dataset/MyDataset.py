import os
import random
import pickle
import numpy as np
import h5py
import torch


class MyDataset():
    def __init__(self, raw_feature_root, flabel_list_path, label_embedding_path, org2path_path, org2weight_path, minibatch_size=64):
        # whole dataset
        self._minibatch_size = minibatch_size
        self._raw_feature_root = raw_feature_root
        self._feature_indexes = []      # index feature: [feature file name, offset]
        self._label_indexes = []        # label index
        # cached feature package
        self._curr_package = dict()
        # number of image_feature file
        self._curr_package_capacity = 4000
        # package bounds
        self._curr_package_start_fid = 0
        self._next_package_start_fid = 0
        # _curr_package_cursor indexes _curr_package_feature_indexes
        self._curr_package_cursor = -1
        # random current package feature indexes of the whole feature list
        self._curr_package_feature_indexes = []
        # word2vec
        label_embedding_file = h5py.File(label_embedding_path, 'r')
        self._label_embedding = np.array(label_embedding_file['label_vec'])
        self._label_feature_length = len(self._label_embedding[0])
        # label2path
        self._org2path = pickle.load(open(org2path_path, 'rb'))
        self._org2weight = pickle.load(open(org2weight_path, 'rb'))
        with open(flabel_list_path, 'r') as list_file:
            flabel_list = list_file.read().splitlines()
        for item in flabel_list:
            # image id, offset, hier_label_index, vg_label_index
            item_info = item.split(' ')
            item_feature_file = item_info[0]
            item_id = int(item_info[1])
            item_label_index = int(item_info[2])
            item_vg_index = int(item_info[3])
            # label indexes [hier_label_index, vg_label_index]
            self._label_indexes.append([item_label_index, item_vg_index])
            # feature indexes [feature file name, offset]
            self._feature_indexes.append([item_feature_file, item_id])

    def init_package(self):
        self._next_package_start_fid = 0
        self._curr_package_start_fid = 0
        self._curr_package_cursor = 0
        self._curr_package_feature_indexes = []

    def __len__(self):
        return len(self._feature_indexes)

    def has_next_minibatch(self):
        if self._next_package_start_fid == len(self._feature_indexes):
            # the last package
            if self._curr_package_cursor == len(self._curr_package_feature_indexes):
                # the last minibatch
                return False
        return True

    def load_next_feature_package(self):
        print('Loading features into memory ......')
        del self._curr_package          # release memory
        self._curr_package = dict()     # feature_file -> [f1,f2,f3,...]
        self._curr_package_start_fid = self._next_package_start_fid
        while len(self._curr_package.keys()) < self._curr_package_capacity:
            if self._next_package_start_fid == len(self._feature_indexes):
                # all features already loaded
                break
            # fill feature package
            next_feature_file, _ = self._feature_indexes[self._next_package_start_fid]
            if next_feature_file not in self._curr_package.keys():
                feature_path = os.path.join(self._raw_feature_root, next_feature_file)
                with open(feature_path, 'rb') as feature_file:
                    features = pickle.load(feature_file)
                    self._curr_package[next_feature_file] = features
            self._next_package_start_fid += 1
        self._curr_package_feature_indexes = np.arange(self._curr_package_start_fid, self._next_package_start_fid)
        # shuffle the feature indexes of current feature package
        random.shuffle(self._curr_package_feature_indexes)
        # init package index cursor
        self._curr_package_cursor = 0

    def minibatch_acc(self, negative_label_num=1000):
        negative_label_num = min(negative_label_num, len(self._label_embedding))
        vfs = np.zeros((self._minibatch_size, 4096))
        p_lfs = np.zeros((self._minibatch_size, self._label_feature_length))
        v_actual_num = 0
        p_label_set = set()
        for v in range(0, self._minibatch_size):
            if self._curr_package_cursor == len(self._curr_package_feature_indexes):
                # current package finished, load another 4000 feature files
                self.load_next_feature_package()
            if self._curr_package_cursor == len(self._curr_package_feature_indexes):
                vfs = vfs[:v_actual_num]
                p_lfs = p_lfs[:v_actual_num]
                break
            fid = self._curr_package_feature_indexes[self._curr_package_cursor]
            feature_file, offset = self._feature_indexes[fid]
            vfs[v] = self._curr_package[feature_file][offset]
            p_label_index = self._label_indexes[fid][0]
            p_lfs[v] = self._label_embedding[p_label_index]
            p_label_set = p_label_set | set(self._org2path[self._label_indexes[fid][1]])
            self._curr_package_cursor += 1
            v_actual_num += 1
        all_n_labels = list(set(range(0, len(self._label_embedding))) - p_label_set)
        n_labels = random.sample(all_n_labels, min(negative_label_num, len(all_n_labels)))
        n_lfs = self._label_embedding[n_labels]
        #  vfs: minibatch_size | p_lfs: minibatch_size | n_lfs: negative_label_num
        vfs = torch.from_numpy(np.array(vfs)).float()
        p_lfs = torch.from_numpy(np.array(p_lfs)).float()
        n_lfs = torch.from_numpy(np.array(n_lfs)).float()
        return vfs, p_lfs, n_lfs

    def minibatch_acc1(self, negative_label_num=330):
        vfs = np.zeros((self._minibatch_size, 4096))
        pls = np.zeros(self._minibatch_size).astype(np.int)
        nls = np.zeros((self._minibatch_size, negative_label_num)).astype(np.int)
        pws = np.zeros(self._minibatch_size)
        v_actual_num = 0
        for v in range(0, self._minibatch_size):
            if self._curr_package_cursor == len(self._curr_package_feature_indexes):
                # current package finished, load another 4000 feature files
                self.load_next_feature_package()
            if self._curr_package_cursor == len(self._curr_package_feature_indexes):
                vfs = vfs[:v_actual_num]
                pls = pls[:v_actual_num]
                nls = nls[:v_actual_num]
                pws = pws[:v_actual_num]
                break
            fid = self._curr_package_feature_indexes[self._curr_package_cursor]
            feature_file, offset = self._feature_indexes[fid]
            vfs[v] = self._curr_package[feature_file][offset]
            pls[v] = self._label_indexes[fid][0]
            all_nls = list(set(range(0, len(self._label_embedding))) - set(self._org2path[self._label_indexes[fid][1]]))
            nls[v] = random.sample(all_nls, negative_label_num)
            pws[v] = self._org2weight[self._label_indexes[fid][1]]
            self._curr_package_cursor += 1
            v_actual_num += 1
        #  vfs: minibatch_size | pls: minibatch_size | nls: minibatch_size
        vfs = torch.from_numpy(vfs).float()
        pls = torch.from_numpy(pls)
        nls = torch.from_numpy(nls)
        pws = torch.from_numpy(pws).float()
        label_vecs = torch.from_numpy(self._label_embedding).float()

        # Tensor to Variable
        label_vecs = torch.autograd.Variable(label_vecs).cuda()
        vfs = torch.autograd.Variable(vfs).cuda()
        pws = torch.autograd.Variable(pws).cuda()
        return vfs, pls, nls, label_vecs, pws


    def minibatch_eval(self):
        # generate minibatch from current feature package
        vfs = []
        p_lfs = []
        n_lfs = []
        if self._curr_package_cursor == len(self._curr_package_feature_indexes):
            # current package finished, load another 4000 feature files
            self.load_next_feature_package()
        if self._curr_package_cursor < len(self._curr_package_feature_indexes):
            fid = self._curr_package_feature_indexes[self._curr_package_cursor]
            feature_file, offset = self._feature_indexes[fid]
            vf = self._curr_package[feature_file][offset]
            positive_label_index = self._label_indexes[fid][0]
            p_lf = self._label_embedding[positive_label_index]
            self._curr_package_cursor += 1
            positive_labels = self._org2path[self._label_indexes[fid][1]]
            all_negative_labels = list(set(range(0, len(self._label_embedding))) -
                                       set(positive_labels))
            vfs = [vf]
            n_lfs = self._label_embedding[all_negative_labels]
            p_lfs = [p_lf]
        vfs = torch.from_numpy(np.array(vfs)).float()
        p_lfs = torch.from_numpy(np.array(p_lfs)).float()
        n_lfs = torch.from_numpy(np.array(n_lfs)).float()
        return vfs, p_lfs, n_lfs




