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
                hypernym_paths = wn_node.hypernym_paths()
                for hypernym_path in hypernym_paths:
                    for w in hypernym_path:
                        if w.name() not in wn_label_set:
                            wn_label_set.add(w.name())
                            label2index[w.name()] = next_label_index
                            index2label.append(w.name())
                            next_label_index += 1
    pickle.dump(label2index, open(label2index_path, 'wb'))
    pickle.dump(index2label, open(index2label_path, 'wb'))
    return label2index


def vrd2path(vrd2wn, label2index, vrd2path_path, vrd2pw_path):
    vrd2path = dict()
    vrd2pd = dict()
    for vrd_label in vrd2wn:
        path_indexes = []
        path_weights = []
        wn_labels = vrd2wn[vrd_label]
        wn_label_set = set()
        for wn_label in wn_labels:
            wn_node = wn.synset(wn_label)
            hypernym_paths = wn_node.hypernym_paths()
            # WordNet indexes on the hyper paths of vg_label
            for hypernym_path in hypernym_paths:
                # weight = a * depth^2 + 1
                w_min = 1.0
                w_max = 10.0
                a = ((w_max - w_min) / (len(hypernym_path)) ** 2)
                for d, w in enumerate(hypernym_path):
                    if w.name() not in wn_label_set:
                        wn_label_set.add(w.name())
                        wn_index = label2index[w.name()]
                        path_indexes.append(wn_index)
                        path_weights.append(a * d ** 2 + w_min)
        # add vg_label index
        path_indexes.append(label2index[vrd_label])
        path_weights.append(w_max)
        vrd2path[label2index[vrd_label]] = path_indexes
        vrd2pd[label2index[vrd_label]] = path_weights

    pickle.dump(vrd2path, open(vrd2path_path, 'wb'))
    pickle.dump(vrd2pd, open(vrd2pw_path, 'wb'))
    return vrd2path


if __name__ == '__main__':
    anno_root = vrd_config['clean_anno_root']
    # object
    obj_vrd2wn_path = vrd_object_config['vrd2wn_path']
    vrd2wn = pickle.load(open(obj_vrd2wn_path, 'rb'))

    obj_label2index_path = vrd_object_config['label2index_path']
    obj_label_list_path = vrd_object_config['index2label_path']
    label2index = index_labels(vrd2wn, obj_label2index_path, obj_label_list_path)

    vrd2path_path = vrd_object_config['vrd2path_path']
    vrd2pw_path = vrd_object_config['vrd2pw_path']
    vrd2path(vrd2wn, label2index, vrd2path_path, vrd2pw_path)


