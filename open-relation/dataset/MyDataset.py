import os
import pickle
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, raw_data_root, list_path, wn_embedding_path):
        self._raw_data_root = raw_data_root
        self._index = []
        self._word_indexs = []
        self._gt = []
        wn_embedding_file = h5py.File(wn_embedding_path, 'r')
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
            self._word_indexs.append(item_word_index)
            self._index.append([feature_file, item_id])
            self._gt.append(item_gt)
        pass

    def __getitem__(self, index):
        feature_name = self._index[index][0]
        feature_offset = self._index[index][1]
        feature_path = os.path.join(self._raw_data_root, feature_name)
        with open(feature_path) as feature_file:
            features = pickle.load(feature_file)
        vf_raw = features[feature_offset]
        vf = torch.from_numpy(vf_raw)
        word_index = self._word_indexs[index]
        wf_raw = self._wn_embedding[word_index]
        wf = torch.from_numpy(wf_raw)
        wf = wf.float()
        gt = torch.FloatTensor(1)
        gt[0] = self._gt[index]
        return vf, wf, gt

    def __len__(self):
        return len(self._index)
