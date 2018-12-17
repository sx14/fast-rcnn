"""
step3: VG original label mapping WordNet synset
next: split_dataset.py
"""
import os
import json
import pickle
from nltk.corpus import wordnet as wn
from open_relation1.vg_data_config import vg_config, vg_object_config, vg_relation_config


def vglabel2wnleaf(anno_root, vg2wn_path, target_key):
    vg2wn = dict()  # vg_label: [wn_label]
    anno_list = os.listdir(anno_root)
    anno_total = len(anno_list)
    for i in range(0, anno_total):  # collect
        anno_file_name = anno_list[i]
        print('processing vg2wn [%d/%d]' % (anno_total, (i+1)))
        anno_path = os.path.join(anno_root, anno_file_name)
        with open(anno_path, 'r') as anno_file:
            anno = json.load(anno_file)
        items = anno[target_key]
        for item in items:
            if target_key == 'objects':
                item_name = item['name']
            elif target_key == 'relationships':
                item_name = item['predicate']
            else:
                print('target_key is expected to be "objects" or "relationships"')
                return
            item_synsets = item['synsets']
            if item_name not in vg2wn:
                vg2wn[item_name] = dict()
                for syn in item_synsets:
                    vg2wn[item_name][syn] = 1
            else:
                existed_syns = vg2wn[item_name]
                for syn in item_synsets:  # count
                    if syn in existed_syns:
                        vg2wn[item_name][syn] += 1
                    else:
                        vg2wn[item_name][syn] = 1
    for vg_label in vg2wn:
        syn_counter = vg2wn[vg_label]
        if target_key == 'objects':
            max_time_syn = syn_counter.keys()[0]
            max_times = syn_counter[max_time_syn]
            for syn in syn_counter:
                if syn_counter[syn] > max_times:
                    max_times = syn_counter[syn]
                    max_time_syn = syn
            vg2wn[vg_label] = [max_time_syn]        # 1 - [1]
        elif target_key == 'relationships':
            # vg2wn[vg_label] = syn_counter.keys()    # 1 - [n]
            vg2wn[vg_label] = sorted(syn_counter.keys())[0] # 1 - [1]
        else:
            print('target_key is expected to be "objects" or "relationships"')
            return
    pickle.dump(vg2wn, open(vg2wn_path, 'wb'))
    return vg2wn


def index_labels(vg2wn, label2index_path, index2label_path):
    vg_labels = sorted(vg2wn.keys())
    wn_label_set = set()
    label2index = dict()
    index2label = []
    next_label_index = 0
    for vg_label in vg_labels:
        # vg_label is unique
        # label2index[vg_label] = next_label_index
        # next_label_index += 1
        # index2label.append(vg_label)
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
            path_indexes = set()
            wn_node = wn.synset(wn_label)
            hypernym_path = wn_node.hypernym_paths()[0]
            # WordNet indexes on the hyper path of vg_label
            wn_indexes = set()
            for w in hypernym_path:
                wn_index = label2index[w.name()]
                wn_indexes.add(wn_index)
            path_indexes = path_indexes | wn_indexes
            vg2path[label2index[wn_label]] = list(path_indexes)
        # vg2path[label2index[vg_label]] = list(path_indexes)
    pickle.dump(vg2path, open(vg2path_path, 'wb'))
    return vg2path


if __name__ == '__main__':
    anno_root = vg_config['clean_anno_root']
    # object
    obj_vg2wn_path = vg_object_config['vg2wn_path']
    vg2wn = vglabel2wnleaf(anno_root, obj_vg2wn_path, 'objects')

    obj_label2index_path = vg_object_config['label2index_path']
    obj_label_list_path = vg_object_config['index2label_path']
    label2index = index_labels(vg2wn, obj_label2index_path, obj_label_list_path)

    vg2path_path = vg_object_config['vg2path_path']
    vg2path(vg2wn, label2index, vg2path_path)

    # relation
    # rlt_vg2wn_path = vg_relation_config['vg2wn_path']
    # vglabel2wnleaf(anno_root, rlt_vg2wn_path, 'relationships')


