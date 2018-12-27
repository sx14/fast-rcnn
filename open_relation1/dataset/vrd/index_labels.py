"""
step3: VRD original label mapping WordNet synset
next: split_dataset.py
"""
import os
import json
import pickle
from nltk.corpus import wordnet as wn
from open_relation1.vrd_data_config import vrd_config, vrd_object_config


def index_labels(vg2wn, label2index_path, index2label_path):
    vg_labels = sorted(vg2wn.keys())
    wn_label_set = set()
    label2index = dict()
    index2label = []
    next_label_index = 0
    for vg_label in vg_labels:
        # vg_label is unique
        label2index[vg_label] = next_label_index
        next_label_index += 1
        index2label.append(vg_label)
        wn_labels = vg2wn[vg_label]
        for wn_label in wn_labels:
            if wn_label not in wn_label_set:
                wn_node = wn.synset(wn_label)
                hypernym_path = wn_node.hypernym_paths()[0]
                for w in hypernym_path:
                    if w.name() not in wn_label_set:
                        wn_label_set.add(w.name())
                        label2index[w.name()] = next_label_index
                        index2label.append(w.name())
                        next_label_index += 1
    pickle.dump(label2index, open(label2index_path, 'wb'))
    pickle.dump(index2label, open(index2label_path, 'wb'))
    return label2index


def vg2path(vg2wn, label2index, vg2path_path):
    vg2path = dict()
    for vg_label in vg2wn:
        # path_indexes = set()
        # add vg_label index
        # path_indexes.add(label2index[vg_label])
        wn_labels = vg2wn[vg_label]
        for wn_label in wn_labels:
            wn_node = wn.synset(wn_label)
            hypernym_path = wn_node.hypernym_paths()[0]
            # WordNet indexes on the hyper path of vg_label
            wn_indexes = []
            for w in hypernym_path:
                wn_index = label2index[w.name()]
                wn_indexes.append(wn_index)
            vg2path[label2index[wn_label]] = wn_indexes
        # vg2path[label2index[vg_label]] = list(path_indexes)
    pickle.dump(vg2path, open(vg2path_path, 'wb'))
    return vg2path


if __name__ == '__main__':
    anno_root = vrd_config['clean_anno_root']
    # object
    obj_vrd2wn_path = vrd_object_config['vrd2wn_path']
    vrd2wn = pickle.load(open(obj_vrd2wn_path, 'rb'))

    obj_label2index_path = vrd_object_config['label2index_path']
    obj_label_list_path = vrd_object_config['index2label_path']
    label2index = index_labels(vrd2wn, obj_label2index_path, obj_label_list_path)

    vrd2path_path = vrd_object_config['vrd2path_path']
    vg2path(vrd2wn, label2index, vrd2path_path)

    # relation
    # rlt_vg2wn_path = vg_relation_config['vg2wn_path']
    # vglabel2wnleaf(anno_root, rlt_vg2wn_path, 'relationships')


