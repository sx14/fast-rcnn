import os
import json
import h5py
import numpy as np
from nltk.corpus import wordnet as wn


dataset_name = 'vs'


def eval1():
    if dataset_name == 'pascal':
        dataset_root = '/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007'
    elif dataset_name == 'vs':
        dataset_root = '/media/sunx/Data/dataset/visual genome'

    weight_path = '../label_embedding/object/word_vec_'+dataset_name+'.h5'
    wn_embedding_file = h5py.File(weight_path, 'r')
    wn_embedding = np.array(wn_embedding_file['word_vec'])
    wn2index_path = os.path.join(dataset_root, 'feature', 'object', 'prepare', 'wn2index.json')
    with open(wn2index_path, 'r') as wn2index_file:
        wn2index = json.load(wn2index_file)
    synset_name_path = '../label_embedding/object/'+dataset_name+'_dataset/synset_names_with_'+dataset_name+'.json'
    with open(synset_name_path, 'r') as synset_name_file:
        synset_names = json.load(synset_name_file)
    label2wn_path = os.path.join(dataset_root, 'feature', 'object', 'prepare', 'label2wn.json')
    with open(label2wn_path, 'r') as label2wn_file:
        label2wn = json.load(label2wn_file)

    all = len(label2wn.keys())
    positive = 0
    negative = 0
    entity = 0
    test_positive = True
    for label in label2wn:
        wn = label2wn[label][0]
        hyper = wn2index[wn]
        if wn == 'entity.n.01':
            entity += 1
        hypo = wn2index[label]
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
            print(wn +'->'+label+'| yes')
            positive += 1
        else:
            print(wn + '->' + label + '| no')
            negative += 1
    if test_positive:
        print('acc :' + str(positive * 1.0 / all))
    else:
        print('acc :' + str(negative * 1.0 / all))
    print('entity :' + str(entity * 1.0 / all))


def eval2():
    if dataset_name == 'pascal':
        dataset_root = '/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007'
    elif dataset_name == 'vs':
        dataset_root = '/media/sunx/Data/dataset/visual genome'

    weight_path = '../label_embedding/object/word_vec_'+dataset_name+'.h5'
    wn_embedding_file = h5py.File(weight_path, 'r')
    wn_embedding = np.array(wn_embedding_file['word_vec'])
    wn2index_path = os.path.join(dataset_root, 'feature', 'object', 'prepare', 'wn2index.json')
    with open(wn2index_path, 'r') as wn2index_file:
        wn2index = json.load(wn2index_file)
    synset_name_path = '../label_embedding/object/'+dataset_name+'_dataset/synset_names_with_'+dataset_name+'.json'
    with open(synset_name_path, 'r') as synset_name_file:
        synset_names = json.load(synset_name_file)
    label2wn_path = os.path.join(dataset_root, 'feature', 'object', 'prepare', 'label2wn.json')
    with open(label2wn_path, 'r') as label2wn_file:
        label2wn = json.load(label2wn_file)

    label = 'dog'
    index = wn2index[label]
    embedding = wn_embedding[index]
    sub = wn_embedding - embedding
    sub_zero = np.stack((sub, np.zeros(sub.shape)), axis=2)
    relu = np.max(sub_zero, axis=2)
    relu = relu * relu
    E = np.sum(relu, axis=1)
    pred = np.argsort(E)[1:20]
    for p in pred:
        print(synset_names[p]+'| %f' % E[p])





if __name__ == '__main__':
    eval2()