from nltk.corpus import wordnet as wn
all_nouns = list(wn.all_synsets('n'))
import numpy as np
import json


# get mapping of synset id to index
id2index = {}
for i in range(len(all_nouns)):
    id2index[all_nouns[i].name()] = i

# get hypernym relations
hypernyms = []
for synset in all_nouns:
    for h in synset.hypernyms() + synset.instance_hypernyms():
        hypernyms.append([id2index[synset.name()], id2index[h.name()]])


# ==== append Visual Genome object classes ====
vs2wn_path = 'exp_dataset/label2wn.json'
# visual genome object labels
with open(vs2wn_path, 'r') as vs2wn_file:
    vs2wn = json.load(vs2wn_file)
    next_id2index_id = len(id2index)
    for vs_object in vs2wn:
        temp_node = wn.synset('car.n.01')
        temp_node._name = vs_object
        all_nouns.append(temp_node)
        id2index[vs_object] = next_id2index_id
        wns = vs2wn[vs_object]
        for h in wns:
            h_parts = h.split('.')
            if h_parts[0] != vs_object:
                hypernyms.append([next_id2index_id, id2index[h]])
        next_id2index_id = next_id2index_id + 1
# ====

hypernyms = np.array(hypernyms)
# save hypernyms
import h5py
f = h5py.File('exp_dataset/wordnet_with_VS.h5', 'w')
# f = h5py.File('dataset/wordnet.h5', 'w')
f.create_dataset('hypernyms', data = hypernyms)
f.close()
# save list of synset names
names = map(lambda s: s.name(), all_nouns)
import json
# json.dump(names, open('dataset/synset_names.json', 'w'))
json.dump(names, open('exp_dataset/synset_names_with_VS.json', 'w'))
