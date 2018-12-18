import os
import pickle
import json
import h5py
import numpy as np
from nltk.corpus import wordnet as wn
from open_relation1.vg_data_config import vg_object_config


dataset_name = 'vg'
target = 'object'


def eval2():
    # label vectors
    weight_path = target+'/label_vec_'+dataset_name+'.h5'
    label_vec_file = h5py.File(weight_path, 'r')
    label_vecs = np.array(label_vec_file['label_vec'])
    # label_vecs = np.array(pickle.load(open(weight_path, 'rb')))
    label2index_path = vg_object_config['label2index_path']
    label2index = pickle.load(open(label2index_path, 'rb'))
    index2label_path = vg_object_config['index2label_path']
    index2label = pickle.load(open(index2label_path, 'rb'))
    vg2wn_path = vg_object_config['vg2wn_path']
    vg2wn = pickle.load(open(vg2wn_path, 'rb'))

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





if __name__ == '__main__':
    eval2()