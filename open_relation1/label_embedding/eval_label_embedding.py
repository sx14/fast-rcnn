import os
import pickle
import json
import h5py
import numpy as np
from nltk.corpus import wordnet as wn
from open_relation1.vg_data_config import vg_object_config


dataset_name = 'vg'
target = 'object'

def eval1():
    weight_path = target+'/label_vec_'+dataset_name+'.h5'
    wn_embedding_file = h5py.File(weight_path, 'r')
    wn_embedding = np.array(wn_embedding_file['label_vec'])
    label2index_path = vg_object_config['label2index_path']
    label2index = pickle.load(open(label2index_path, 'rb'))
    labels_path = vg_object_config['index2label_path']
    labels = json.load(open(labels_path, 'rb'))
    vg2wn_path = vg_object_config['vg2wn_path']
    vg2wn = json.load(open(vg2wn_path, 'rb'))


    positive = 0
    negative = 0
    test_positive = True
    for vg_label in vg2wn:
        wn_label = vg2wn[vg_label][0]
        hyper = label2index[wn_label]
        hypo = label2index[vg_label]
        if test_positive:
            hyper_v = np.array(wn_embedding[hyper])
            hypo_v = np.array(wn_embedding[hypo])
        else:
            hyper_v = np.array(wn_embedding[hypo])
            hypo_v = np.array(wn_embedding[hyper])

        h_zero = np.stack((hyper_v, np.zeros(hyper_v.shape)), axis=1)
        relu = np.max(h_zero, axis=1)
        relu = relu * relu
        E = np.sum(relu)
        # print('word norm' + str(E))
        sub = hyper_v - hypo_v
        sub_zero = np.stack((sub, np.zeros(sub.shape)), axis=1)
        relu = np.max(sub_zero, axis=1)
        relu = relu * relu
        E = np.sum(relu)
        # print('sub norm' + str(E))
        if E < 0.5:
            print(wn_label +'->'+vg_label+'| yes')
            positive += 1
        else:
            print(wn_label + '->' + vg_label + '| no')
            negative += 1
    all = len(vg2wn.keys())
    if test_positive:
        print('acc :' + str(positive * 1.0 / all))
    else:
        print('acc :' + str(negative * 1.0 / all))


def eval2():
    # label vectors
    weight_path = target+'/label_vec_'+dataset_name+'.h5'
    label_vec_file = h5py.File(weight_path, 'r')
    label_vecs = np.array(label_vec_file['label_vec'])

    label2index_path = vg_object_config['label2index_path']
    label2index = pickle.load(open(label2index_path, 'rb'))
    index2label_path = vg_object_config['index2label_path']
    index2label = pickle.load(open(index2label_path, 'rb'))
    vg2wn_path = vg_object_config['vg2wn_path']
    vg2wn = pickle.load(open(vg2wn_path, 'rb'))

    vg_labels = vg2wn.keys()
    for vg_label in vg_labels:
        vg_label_index = label2index[vg_label]
        vg_label_vec = label_vecs[vg_label_index]
        sub = label_vecs - vg_label_vec
        sub_zero = np.stack((sub, np.zeros(sub.shape)), axis=2)
        relu = np.max(sub_zero, axis=2)
        relu = relu * relu
        E = np.sum(relu, axis=1)
        pred = np.argsort(E)[1:20]
        print('\n===== '+vg_label+'=====')
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