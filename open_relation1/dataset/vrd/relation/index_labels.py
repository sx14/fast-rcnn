"""
step3: VRD original label mapping WordNet synset
next: split_dataset.py
"""
import os
import json
import pickle
from pre_hier import PreNet
from open_relation1.vrd_data_config import vrd_config, vrd_predicate_config


def index_labels(pre_net, label2index_path, index2label_path):
    # all raw labels
    raw_labels = pre_net.get_raw_labels()

    # result
    label2index = dict()
    index2label = []

    hier_label_set = set()  # record unique labels
    next_label_index = 0
    # index: root(0) -> raw(n)
    for raw_label in raw_labels:
        pre = pre_net.get_pre(raw_label)
        hyper_paths = pre.hyper_paths()
        for hyper_path in hyper_paths:
            # hyper path contains raw label
            # from root to raw
            for i in range(len(hyper_path)):
                n = hyper_path[i]
                if n.name() not in hier_label_set:
                    hier_label_set.add(n.name())
                    label2index[n.name()] = next_label_index
                    index2label.append(n.name())
                    next_label_index += 1

    pickle.dump(label2index, open(label2index_path, 'wb'))
    pickle.dump(index2label, open(index2label_path, 'wb'))

    return label2index, index2label


def raw2path(pre_net, label2index, vrd2path_path, vrd2pw_path):
    # raw label 2 label path
    raw2path = dict()
    # raw label 2 path weight
    raw2pw = dict()

    for raw_label in pre_net.get_raw_labels():
        path_indexes = []
        path_weights = []

        pre = pre_net.get_pre(raw_label)
        hier_label_set = set()    # record unique labels
        hyper_paths = pre.hyper_paths()

        # path = [root, ... , raw]
        # hyper path contains raw label
        w_min = 1.0
        w_max = 10.0
        for hyper_path in hyper_paths:
            # weight = a * depth^2 + 1
            a = ((w_max - w_min) / (len(hyper_path)) ** 2)

            # hyper path contains raw label
            # from root to raw's father
            for d in range(len(hyper_path)-1):
                w = hyper_path[d]
                if w.name() not in hier_label_set:
                    hier_label_set.add(w.name())
                    wn_index = label2index[w.name()]
                    path_indexes.append(wn_index)
                    path_weights.append(a * d ** 2 + w_min)
        # add raw label index
        path_indexes.append(label2index[raw_label])
        path_weights.append(w_max)

        raw2path[label2index[raw_label]] = path_indexes
        raw2pw[label2index[raw_label]] = path_weights

    pickle.dump(raw2path, open(vrd2path_path, 'wb'))
    pickle.dump(raw2pw, open(vrd2pw_path, 'wb'))
    return raw2path


if __name__ == '__main__':
    # predicate
    pn = PreNet()
    pre_label2index_path = vrd_predicate_config['label2index_path']
    pre_label_list_path = vrd_predicate_config['index2label_path']
    label2index, index2label = index_labels(pn, pre_label2index_path, pre_label_list_path)

    raw2path_path = vrd_predicate_config['raw2path_path']
    raw2pw_path = vrd_predicate_config['raw2pw_path']
    raw2path(pn, label2index, raw2path_path, raw2pw_path)


