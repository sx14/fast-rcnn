import copy
import h5py
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from lang_config import train_params
from open_relation.dataset.dataset_config import DatasetConfig
from open_relation.dataset.dataset_config import DatasetConfig
from open_relation.dataset.vrd.label_hier.obj_hier import objnet
from open_relation.dataset.vrd.label_hier.pre_hier import prenet

vrd_config = DatasetConfig('vrd')

class LangDataset(Dataset):

    def get_gt_vecs(self):
        return self._pre_vecs

    def update_pos_neg_pairs(self):
        obj_label_num = objnet.label_sum()
        random_obj_labels = np.random.choice(range(obj_label_num), self._pos_rlts.shape[0])
        neg_rlts = copy.deepcopy(self._pos_rlts)
        neg_rlts[:neg_rlts.shape[0] / 2, 0] = random_obj_labels[:neg_rlts.shape[0] / 2]
        neg_rlts[neg_rlts.shape[0] / 2:, 2] = random_obj_labels[neg_rlts.shape[0] / 2:]
        self._rlt_pairs = [self._pos_rlts, neg_rlts]

    def __init__(self, rlt_path):
        obj_vec_file = h5py.File(vrd_config.extra_config['object'].config['label_vec_path'], 'r')
        self._obj_vecs = torch.from_numpy(np.array(obj_vec_file['label_vec']))

        pre_vec_file = h5py.File(vrd_config.extra_config['predicate'].config['label_vec_path'], 'r')
        self._pre_vecs = torch.from_numpy(np.array(pre_vec_file['label_vec']))

        pos_rlts = np.load(rlt_path+'.npy')
        self._pos_rlts = np.array(pos_rlts)
        self._rlt_pairs = []

        self._raw2path = prenet.raw2path()
        self._label_sum = prenet.label_sum()
        self._neg_sample_num = train_params['neg_sample_num']

        self.update_pos_neg_pairs()

    def __getitem__(self, item):
        rlt1 = self._rlt_pairs[0][item]
        rlt2 = self._rlt_pairs[1][item]
        rlts = np.append(rlt1, rlt2)

        sbj_vec1 = self._obj_vecs[rlt1[0]]
        sbj_vec2 = self._obj_vecs[rlt2[0]]

        pre_vec1 = self._pre_vecs[rlt1[1]]
        pre_vec2 = self._pre_vecs[rlt2[1]]

        obj_vec1 = self._obj_vecs[rlt1[2]]
        obj_vec2 = self._obj_vecs[rlt2[2]]

        pos_inds = self._raw2path[rlt1[3]]
        all_neg_inds = list(set(range(self._label_sum)) - set(pos_inds))
        neg_inds = random.sample(all_neg_inds, self._neg_sample_num)
        pos_neg_inds = np.array([rlt1[1]] + neg_inds)

        return [sbj_vec1, pre_vec1, obj_vec1, sbj_vec2, pre_vec2, obj_vec2, rlts, pos_neg_inds]

    def __len__(self):
        return len(self._rlt_pairs[0])
