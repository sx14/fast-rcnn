import os
import json
import numpy as np
from nltk.corpus import wordnet as wn
from path_config import config

path_config = config['pascal']

label2wn_path = os.path.join(path_config['object_prepare_root'], 'label2wn.json')
with open(label2wn_path, 'r') as label2wn_file:
    label2wn = json.load(label2wn_file)
# wns = label2wn.values()
# wn_nodes = set()
# for wn_names in wns:
#     for wn_name in wn_names:
#         wn_synset = wn.synset(wn_name)
#         for p in wn_synset.hypernym_paths():
#             for w in p:
#                 wn_nodes.add(w)
# wn_nodes = list(wn_nodes)
wn_nodes = list(wn.all_synsets('n'))



# get mapping of synset id to index
wn2index = {}
for i in range(len(wn_nodes)):
    wn2index[wn_nodes[i].name()] = i

# get wordnet part hypernym relations
hypernyms = []
for synset in wn_nodes:
    for h in synset.hypernyms() + synset.instance_hypernyms():
        hypernyms.append([wn2index[synset.name()], wn2index[h.name()]])

# ==== append object labels ====
labels = []
next_id2index_id = len(wn2index)
for label in label2wn:
    labels.append(label)
    wn2index[label] = next_id2index_id
    wns = label2wn[label]
    for h in wns:
        hypernyms.append([next_id2index_id, wn2index[h]])
    next_id2index_id = next_id2index_id + 1
# ===============================
hypernyms = np.array(hypernyms)
# save hypernyms
import h5py
f = h5py.File(path_config['hypernym_data_path'], 'w')
f.create_dataset('hypernyms', data=hypernyms)
f.close()

# index2path_index_list
label2path = dict()
for label in label2wn:
    path_indexes = list([wn2index[label]])
    wn_names = label2wn[label]
    for wn_name in wn_names:
        syn = wn.synset(wn_name)
        hyper_paths = syn.hypernym_paths()
        hyper_indexes = []
        for p in hyper_paths:
            for w in p:
                w_index = wn2index[w.name()]
                hyper_indexes.append(w_index)
        path_indexes = path_indexes + hyper_indexes
        label2path[wn2index[label]] = path_indexes




# save list of synset names
names = map(lambda s: s.name(), wn_nodes)
names = names + labels

wn2index_save_path = os.path.join(path_config['object_prepare_root'], 'wn2index.json')
json.dump(wn2index, open(wn2index_save_path, 'w'))
label2path_save_path = os.path.join(path_config['object_prepare_root'], 'label2path.json')
json.dump(label2path, open(label2path_save_path, 'w'))
json.dump(names, open(path_config['synset_names_path'], 'w'))

