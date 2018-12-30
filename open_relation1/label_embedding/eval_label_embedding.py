import os
import pickle
import json
import h5py
import numpy as np
from nltk.corpus import wordnet as wn
from open_relation1.vrd_data_config import vrd_object_config


dataset_name = 'vrd'
target = 'object'


def eval2(label_vecs, index2label, label2index, vg2wn):


    vg_labels = vg2wn.keys()
    for vg_label in vg_labels:
        # vg_label_index = label2index[vg_label]
        # vg_label_vec = label_vecs[vg_label_index]

        wn_label_index = label2index[vg2wn[vg_label][0]]
        wn_label_vec = label_vecs[wn_label_index]

        sub = label_vecs - wn_label_vec
        sub_zero = np.stack((sub, np.zeros(sub.shape)), axis=2)
        relu = np.max(sub_zero, axis=2)
        relu = relu * relu
        E = np.sum(relu, axis=1)
        pred = np.argsort(E)[:20]
        print('\n===== '+vg_label+' =====')
        print('---answer---')
        wn_labels = vg2wn[vg_label]
        wn_node = wn.synset(wn_labels[0])
        hypernym_path = wn_node.hypernym_paths()[0]
        for i in range(0, len(hypernym_path)):
            print(hypernym_path[i])
        print('---prediction---')
        for p in pred:
            print(index2label[p]+'| %f' % E[p])


def eval3(label_vecs, index2label, label2index, vg2wn, label):
    label_vec = label_vecs[label2index[label]]
    sub = label_vecs - label_vec
    sub[sub < 0] = 0
    sub_square = sub * sub
    E = np.sum(sub_square, axis=1)
    pred = np.argsort(E)[:20]
    print('\n===== '+label+' =====')
    print('---answer---')
    if label in vg2wn:
        wn_labels = vg2wn[label]
        wn_node = wn.synset(wn_labels[0])
    else:
        wn_node = wn.synset(label)
    hypernym_path = wn_node.hypernym_paths()[0]
    for i in range(0, len(hypernym_path)):
        print(hypernym_path[i].name())
    print('---prediction---')
    for p in pred:
        print(index2label[p] + '| %f' % E[p])


if __name__ == '__main__':
    # label vectors
    weight_path = vrd_object_config['label_vec_path']
    label_vec_file = h5py.File(weight_path, 'r')
    label_vecs = np.array(label_vec_file['label_vec'])

    # label mapping
    label2index_path = vrd_object_config['label2index_path']
    label2index = pickle.load(open(label2index_path, 'rb'))
    index2label_path = vrd_object_config['index2label_path']
    index2label = pickle.load(open(index2label_path, 'rb'))
    vg2wn_path = vrd_object_config['vrd2wn_path']
    vg2wn = pickle.load(open(vg2wn_path, 'rb'))

    eval2(label_vecs, index2label, label2index, vg2wn)
    eval3(label_vecs, index2label, label2index, vg2wn, 'living_thing.n.01')
