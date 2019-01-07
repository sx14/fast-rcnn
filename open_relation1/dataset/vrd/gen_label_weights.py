"""
weights for original labels only
"""

import os
import pickle
import numpy as np
from open_relation1 import vrd_data_config



# label maps
vrd2path_path = vrd_data_config.vrd_object_config['vrd2path_path']
vrd2path = pickle.load(open(vrd2path_path, 'rb'))
index2label_path = vrd_data_config.vrd_object_config['index2label_path']
index2label = pickle.load(open(index2label_path, 'rb'))
label2index_path = vrd_data_config.vrd_object_config['label2index_path']
label2index = pickle.load(open(label2index_path, 'rb'))



# org data
feature_root = vrd_data_config.vrd_object_feature_root
box_label_path = os.path.join(feature_root, 'prepare', 'train_box_label.bin')
box_labels = pickle.load(open(box_label_path, 'rb'))


def gen_weights(box_labels, vrd2path, index2label, label2index, weights_save_path, mode):
    # WARNING: deprecated
    # counter
    label_counter = np.zeros(len(index2label))
    vrd_counter = np.zeros(len(index2label))

    # counting
    for img_id in box_labels:
        img_box_labels = box_labels[img_id]
        for box_label in img_box_labels:
            vrd_label = box_label[4]
            vrd_counter[label2index[vrd_label]] += 1
            label_path = vrd2path[label2index[vrd_label]]
            for l in label_path:
                label_counter[l] += 1

    if mode == 'org':
        show_counter = vrd_counter
    else:
        show_counter = label_counter

    ranked_counts = np.sort(show_counter).tolist()
    ranked_counts.reverse() # large -> small
    ranked_counts = np.array(ranked_counts)
    ranked_counts = ranked_counts[ranked_counts > 0]

    ranked_inds = np.argsort(show_counter).tolist()
    ranked_inds.reverse()   # large -> small
    ranked_inds = ranked_inds[:len(ranked_counts)]

    count_sum = ranked_counts.sum()
    min_weight = 1.0 / (ranked_counts.max() / count_sum)
    vrd2weight = dict()
    for i in range(len(ranked_counts)):
        w = 1.0 / (ranked_counts[i] / count_sum) / min_weight
        vrd2weight[ranked_inds[i]] = w
    pickle.dump(vrd2weight, open(weights_save_path, 'wb'))


def gen_weights1(box_labels, vrd2path, index2label, label2index, weights_save_path, mode):
    # counter
    label_counter = np.zeros(len(index2label))
    vrd_counter = np.zeros(len(index2label))

    # counting
    for img_id in box_labels:
        img_box_labels = box_labels[img_id]
        for box_label in img_box_labels:
            vrd_label = box_label[4]
            vrd_counter[label2index[vrd_label]] += 1
            label_path = vrd2path[label2index[vrd_label]]
            for l in label_path:
                label_counter[l] += 1

    if mode == 'org':
        show_counter = vrd_counter
    else:
        show_counter = label_counter

    ranked_counts = np.sort(show_counter).tolist()
    ranked_counts.reverse() # large -> small
    ranked_counts = np.array(ranked_counts)
    ranked_counts = ranked_counts[ranked_counts > 0]

    ranked_inds = np.argsort(show_counter).tolist()
    ranked_inds.reverse()   # large -> small
    ranked_inds = ranked_inds[:len(ranked_counts)]

    count_sum = ranked_counts.sum()
    max_weight = 1.0 / (ranked_counts.min() / count_sum)
    min_weight = 1.0 / (ranked_counts.max() / count_sum)
    vrd2weight = dict()
    # expected max weight: 4
    for i in range(len(ranked_counts)):
        w = (1.0 / (ranked_counts[i] / count_sum)) / (max_weight / 3) + 1
        vrd2weight[ranked_inds[i]] = w
    pickle.dump(vrd2weight, open(weights_save_path, 'wb'))


if __name__ == '__main__':
    # label maps
    vrd2path_path = vrd_data_config.vrd_object_config['vrd2path_path']
    vrd2path = pickle.load(open(vrd2path_path, 'rb'))
    index2label_path = vrd_data_config.vrd_object_config['index2label_path']
    index2label = pickle.load(open(index2label_path, 'rb'))
    label2index_path = vrd_data_config.vrd_object_config['label2index_path']
    label2index = pickle.load(open(label2index_path, 'rb'))

    # org data
    feature_root = vrd_data_config.vrd_object_feature_root
    box_label_path = os.path.join(feature_root, 'prepare', 'train_box_label.bin')
    box_labels = pickle.load(open(box_label_path, 'rb'))

    # weight save path
    weights_save_path = vrd_data_config.vrd_object_config['vrd2weight_path']
    gen_weights1(box_labels, vrd2path, index2label, label2index, weights_save_path, 'org')
