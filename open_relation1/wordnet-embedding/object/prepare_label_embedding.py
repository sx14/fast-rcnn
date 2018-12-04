import os
import copy
import json
import pickle
import numpy as np
from nltk.corpus import wordnet as wn
from open_relation1.vg_data_config import vg_object_config

vg2wn_path = vg_object_config['vg2wn_path']
vg2wn = json.load(open(vg2wn_path, 'r'))
wn_syn_lists = vg2wn.values()
wn_nodes = set()
for wn_syns in wn_syn_lists:
    for wn_syn in wn_syns:
        wn_synset = wn.synset(wn_syn)
        for p in wn_synset.hypernym_paths():
            for w in p:
                wn_nodes.add(w)
wn_nodes = list(wn_nodes)
# wn_nodes = list(wn.all_synsets('n'))


# map synset to index
wn2index = {}
for i in range(len(wn_nodes)):
    wn2index[wn_nodes[i].name()] = i

# gen wordnet hypernym relations
hypernyms = []
for wn_node in wn_nodes:
    for hyp in wn_node.hypernyms() + wn_node.instance_hypernyms():
        # [hypo, hyper]
        hypernyms.append([wn2index[wn_node.name()], wn2index[hyp.name()]])

# ==== append vg labels, append vg -> syn ====
next_label_index = len(wn2index)
label2index = copy.deepcopy(wn2index)
for vg_label in vg2wn:
    label2index[vg_label] = next_label_index
    wn_syns = vg2wn[vg_label]
    for syn in wn_syns:
        # [hypo, hyper]
        hypernyms.append([next_label_index, label2index[syn]])
    next_label_index = next_label_index + 1
# ===============================

hypernyms = np.array(hypernyms)
# save hypernym dataset
import h5py
f = h5py.File('vg_dataset/wordnet_with_vg.h5', 'w')
f.create_dataset('hypernyms', data=hypernyms)
f.close()

# vg_index: path_index_list
vg2path = dict()
for vg_label in vg2wn:
    path_indexes = set()
    path_indexes.add(label2index[vg_label])
    wn_syns = vg2wn[vg_label]
    for wn_syn in wn_syns:
        wn_node = wn.synset(wn_syn)
        hyper_paths = wn_node.hypernym_paths()
        hyper_indexes = set()
        for hyper_path in hyper_paths:
            for hyper in hyper_path:
                hyper_index = label2index[hyper.name()]
                hyper_indexes.add(hyper_index)
        path_indexes = path_indexes + hyper_indexes
    vg2path[label2index[vg_label]] = list(path_indexes)

# save list of labels
# names = map(lambda s: s.name(), wn_nodes)
labels = vg2wn.values() + vg2wn.keys()

label2index_path = vg_object_config['label2index_path']
pickle.dump(label2index, open(label2index_path, 'wb'))
label2path_path = vg_object_config['vg2path_path']
pickle.dump(vg2path, open(label2path_path, 'wb'))
json.dump(labels, open('vg_dataset/synset_names_with_vg.json', 'w'))