import os
import random
import pickle
import numpy as np
import h5py
import torch


class MyDataset():
    def __init__(self, raw_feature_root, flabel_list_path, label2path_path, minibatch_size=64):
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
        # label2path
        self._label2path = pickle.load(open(label2path_path, 'rb'))
        with open(flabel_list_path, 'r') as list_file:
            flabel_list = list_file.read().splitlines()
        for item in flabel_list:
            # image id, offset, hier_label_index, vg_label_index
            item_info = item.split(' ')
            item_feature_file = item_info[0]
            item_id = int(item_info[1])
            item_vg_index = int(item_info[2])
            # label indexes [hier_label_index, vg_label_index]
            self._label_indexes.append(item_vg_index)
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

    def minibatch_acc(self):
        vfs = np.zeros((self._minibatch_size, 4096))
        label_vecs = np.zeros(self._minibatch_size, len(self._label2path.keys()))
        v_actual_num = 0
        for v in range(0, self._minibatch_size):
            if self._curr_package_cursor == len(self._curr_package_feature_indexes):
                # current package finished, load another 4000 feature files
                self.load_next_feature_package()
            if self._curr_package_cursor == len(self._curr_package_feature_indexes):
                vfs = vfs[:v_actual_num]
                break
            fid = self._curr_package_feature_indexes[self._curr_package_cursor]
            feature_file, offset = self._feature_indexes[fid]
            vfs[v] = self._curr_package[feature_file][offset]
            label_index = self._label_indexes[fid]
            label_vecs[v][label_index] = 1
            self._curr_package_cursor += 1
            v_actual_num += 1

        #  vfs: minibatch_size | lfs: one-hot label vec
        vfs = torch.from_numpy(np.array(vfs)).float()
        lfs = torch.from_numpy(np.array(label_vecs))
        return vfs, lfs

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
            positive_labels = self._label2path[self._label_indexes[fid][1]]
            all_negative_labels = list(set(range(0, len(self._label_embedding))) -
                                       set(positive_labels))
            vfs = [vf]
            n_lfs = self._label_embedding[all_negative_labels]
            p_lfs = [p_lf]
        vfs = torch.from_numpy(np.array(vfs)).float()
        p_lfs = torch.from_numpy(np.array(p_lfs)).float()
        n_lfs = torch.from_numpy(np.array(n_lfs)).float()
        return vfs, p_lfs, n_lfs




