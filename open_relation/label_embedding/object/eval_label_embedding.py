import pickle
import h5py
import numpy as np
from nltk.corpus import wordnet as wn
from open_relation.dataset.dataset_config import vrd_object_config
from open_relation.dataset.vrd.label_hier.obj_hier import objnet


dataset_name = 'vrd'
target = 'object'


def eval2(label_vecs, index2label, label2index, vg2wn):


    vg_labels = vg2wn.keys()
    for vg_label in vg_labels:
        vg_label_index = label2index[vg_label]
        vg_label_vec = label_vecs[vg_label_index]
        sub = label_vecs - vg_label_vec

        # wn_label_index = label2index[vg2wn[vg_label][0]]
        # wn_label_vec = label_vecs[wn_label_index]
        # sub = label_vecs - wn_label_vec

        sub_zero = np.stack((sub, np.zeros(sub.shape)), axis=2)
        relu = np.max(sub_zero, axis=2)
        relu = relu * relu
        E = np.sum(relu, axis=1)
        pred = np.argsort(E)[:20]
        print('\n===== '+vg_label+' =====')
        print('---answer---')
        wn_labels = vg2wn[vg_label]
        wn_node = wn.synset(wn_labels[0])
        wn_label_set = set()
        for hypernym_path in wn_node.hypernym_paths():
            for w in hypernym_path:
                if w.name() not in wn_label_set:
                    print(w.name())
                    wn_label_set.add(w.name())
        print('---prediction---')
        for p in pred:
            print(index2label[p]+'| %f' % E[p])


def eval3(label_vecs, index2label, label2index, vg2wn, label):
    label_vec = label_vecs[label2index[label]]
    sub = label_vecs - label_vec
    sub[sub < 0] = 0
    sub_square = sub * sub
    E = np.sum(sub_square, axis=1)
    pred = np.argsort(E)[:50]
    print('\n===== '+label+' =====')
    print('---answer---')
    if label in vg2wn:
        wn_labels = vg2wn[label]
        wn_node = wn.synset(wn_labels[0])
    else:
        wn_node = wn.synset(label)
    wn_label_set = set()
    for hypernym_path in wn_node.hypernym_paths():
        for w in hypernym_path:
            if w.name() not in wn_label_set:
                print(w.name())
                wn_label_set.add(w.name())
    print('---prediction---')
    for p in pred:
        print(index2label[p] + '| %f' % E[p])


def eval4(label_vecs, label2index, label1, label2):
    ind1 = label2index[label1]
    ind2=  label2index[label2]
    vec1 = label_vecs[ind1]
    vec2 = label_vecs[ind2]
    sub = vec1 - vec2
    sub[sub < 0] = 0
    sub_square = sub * sub
    E = np.sqrt(np.sum(sub_square))
    print('P( %s , %s ) = %.2f' % (label1, label2, E))



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

    # eval2(label_vecs, index2label, label2index, vg2wn)
    # eval3(label_vecs, index2label, label2index, vg2wn, 'jeans')
    # eval4(label_vecs, label2index, 'shirt', 'garment.n.01')
    objnet.get_node_by_name('person').show_hyper_paths()
