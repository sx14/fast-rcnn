import os
import json
import numpy as np
from nltk.corpus import wordnet as wn
import path_config

FOR_VS = True
if FOR_VS:
    vs2wn_path = os.path.join(path_config.OBJECT_SAVE_ROOT, 'label2wn.json')
    with open(vs2wn_path, 'r') as vs2wn_file:
        vs2wn = json.load(vs2wn_file)
    vs_wns = vs2wn.values()
    wn_nouns = set()
    for wn_names in vs_wns:
        for wn_name in wn_names:
            wn_synset = wn.synset(wn_name)
            for p in wn_synset.hypernym_paths():
                for w in p:
                    wn_nouns.add(w)
    wn_nouns = list(wn_nouns)
else:
    wn_nouns = list(wn.all_synsets('n'))

# get mapping of synset id to index
id2index = {}
for i in range(len(wn_nouns)):
    id2index[wn_nouns[i].name()] = i

# get wordnet part hypernym relations
hypernyms = []
for synset in wn_nouns:
    for h in synset.hypernyms() + synset.instance_hypernyms():
        hypernyms.append([id2index[synset.name()], id2index[h.name()]])
if FOR_VS:
    # ==== append Visual Genome object classes ====
    vs_nouns = []
    next_id2index_id = len(id2index)
    for vs_label in vs2wn:
        vs_nouns.append(vs_label)
        id2index[vs_label] = next_id2index_id
        wns = vs2wn[vs_label]
        for h in wns:
            hypernyms.append([next_id2index_id, id2index[h]])
        next_id2index_id = next_id2index_id + 1
    # ====
hypernyms = np.array(hypernyms)
# save hypernyms
import h5py
if FOR_VS:
    f = h5py.File('exp_dataset/wordnet_with_VS.h5', 'w')
    # index2path_index_list
    label2path = dict()
    vs_labels = vs2wn.keys()
    for vs_label in vs_labels:
        path_indexes = list([id2index[vs_label]])
        wn_names = vs2wn[vs_label]
        for wn_name in wn_names:
            syn = wn.synset(wn_name)
            hyper_paths = syn.hypernym_paths()
            hyper_indexes = []
            for p in hyper_paths:
                for w in p:
                    w_index = id2index[w.name()]
                    hyper_indexes.append(w_index)
            path_indexes = path_indexes + hyper_indexes
            label2path[id2index[vs_label]] = path_indexes
else:
    f = h5py.File('dataset/wordnet.h5', 'w')
f.create_dataset('hypernyms', data=hypernyms)
f.close()


# save list of synset names
names = map(lambda s: s.name(), wn_nouns)
if FOR_VS:
    names = names + vs_nouns
if FOR_VS:
    wn2index_save_path = os.path.join(path_config.OBJECT_SAVE_ROOT, 'wn2index.json')
    json.dump(id2index, open(wn2index_save_path, 'w'))
    label2path_save_path = os.path.join(path_config.OBJECT_SAVE_ROOT, 'label2path.json')
    json.dump(label2path, open(label2path_save_path, 'w'))
    json.dump(names, open('exp_dataset/synset_names_with_VS.json', 'w'))
else:
    json.dump(names, open('dataset/synset_names.json', 'w'))
