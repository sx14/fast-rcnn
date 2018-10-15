import os
import random
import pickle
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, raw_data_root, list_path, wn_embedding_path, minibatch_size=128):
        # whole dataset
        self._minibatch_size = minibatch_size
        self._raw_data_root = raw_data_root
        self._index = []            # index feature: [image_id, offset]
        self._word_indexes = []     # label index
        self._gt = []               # gt [1/-1]
        # cached feature package
        self._curr_package = dict()
        self._curr_package_capacity = 2000
        # package bounds
        self._curr_package_start_fid = 0
        self._next_package_start_fid = 0
        # _curr_package_cursor indexes _package_random_feature_list
        self._curr_package_cursor = 0
        # random current package feature indexes of the whole feature list
        self._curr_package_feature_indexes = []
        wn_embedding_file = h5py.File(wn_embedding_path, 'r')
        # word2vec
        self._wn_embedding = wn_embedding_file['word_vec']
        with open(list_path, 'r') as list_file:
            list = list_file.read().splitlines()
        for item in list:
            item_info = item.split(' ')
            feature_file = item_info[0]
            item_id = int(item_info[1])
            item_word_index = int(item_info[2])
            item_gt = int(item_info[3])
            # image id, offset, hyper, gt(1,-1)
            # label numbers
            self._word_indexes.append(item_word_index)
            # feature indexes
            self._index.append([feature_file, item_id])
            # gts
            self._gt.append(item_gt)

    def init_package(self):
        self._next_package_start_fid = 0
        self._curr_package_start_fid = 0
        self._curr_package_cursor = 0
        self._curr_package_feature_indexes = []

    def __getitem__(self, index):
        feature_name = self._index[index][0]
        feature_offset = self._index[index][1]
        feature_path = os.path.join(self._raw_data_root, feature_name)
        with open(feature_path) as feature_file:
            features = pickle.load(feature_file)
        vf_raw = features[feature_offset]
        vf = torch.from_numpy(vf_raw)
        word_index = self._word_indexes[index]
        wf_raw = self._wn_embedding[word_index]
        wf = torch.from_numpy(wf_raw)
        wf = wf.float()
        gt = torch.FloatTensor(1)
        gt[0] = self._gt[index]
        return vf, wf, gt

    def __len__(self):
        return len(self._index)

    def __load_next_feature_package(self):
        del self._curr_package          # release memory
        self._curr_package = dict()     # feature_file -> [f1,f2,f3,...]
        self._curr_package_start_fid = self._next_package_start_fid
        while len(self._curr_package.keys()) < self._curr_package_capacity:
            # fill feature package
            next_feature_file, _ = self._index[self._next_package_start_fid]
            if next_feature_file not in self._curr_package.keys():
                feature_path = os.path.join(self._raw_data_root, next_feature_file)
                with open(feature_path, 'rb') as feature_file:
                    features = pickle.load(feature_file)
                    self._curr_package[next_feature_file] = features
            self._next_package_start_fid += 1
        self._curr_package_feature_indexes = np.arange(self._curr_package_start_fid, self._next_package_start_fid)
        # shuffle the feature indexes of current feature package
        random.shuffle(self._curr_package_feature_indexes)
        # init package index cursor
        self._curr_package_cursor = 0

    def minibatch(self):
        # generate minibatch from current feature package
        if self._curr_package_cursor == len(self._curr_package_feature_indexes):
            # current package finished
            # load another 2000 feature files
            self.__load_next_feature_package()
        batch_start_index = self._curr_package_cursor
        batch_end_index = min(batch_start_index + self._minibatch_size, len(self._curr_package_feature_indexes))
        vfs = []
        wfs = []
        gts = []
        for i in range(batch_start_index, batch_end_index):
            fid = self._curr_package_feature_indexes[i]
            feature_file, offset = self._index[fid]
            vf = self._curr_package[feature_file][offset]
            vfs.append(vf)
            word_index = self._word_indexes[fid]
            wf = self._wn_embedding[word_index]
            wfs.append(wf)
            gts.append([self._gt[fid]])
        self._curr_package_cursor = batch_end_index
        vfs = torch.from_numpy(np.array(vfs)).float()
        wfs = torch.from_numpy(np.array(wfs)).float()
        gts = torch.from_numpy(np.array(gts)).float()
        return vfs, wfs, gts

    def has_next_minibatch(self):
        if self._next_package_start_fid == len(self._index):
            # the last package
            if self._curr_package_cursor == len(self._curr_package_feature_indexes):
                return False
        return True

