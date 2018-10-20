import h5py
import numpy as np
import label_map

weight_path = '../wordnet-embedding/object/word_vec_wn.h5'
wn_embedding_file = h5py.File(weight_path, 'r')
wn_embedding = wn_embedding_file['word_vec']
wn_synsets_path = '/media/sunx/Data/linux-workspace/python-workspace/' \
                  'hierarchical-relationship/open-relation/wordnet-embedding/object/dataset/synset_names.json'
wn2index = label_map.wn2index(wn_synsets_path)


hypers = ['animal.n.01']

hypos = ['cat.n.01']

for h in range(0, len(hypers)):
    hyper = wn2index[hypers[h]]
    hypo = wn2index[hypos[h]]
    hyper_v = np.array(wn_embedding[hyper])
    hypo_v = np.array(wn_embedding[hypo])
    sub = hyper_v - hypo_v
    sub_zero = np.stack((sub, np.zeros(sub.shape)), axis=1)
    relu = np.max(sub_zero, axis=1)
    relu = relu * relu
    E = np.sum(relu)
    if E < 0.01:
        print(hypers[h]+'->'+hypos[h]+'| yes')
    else:
        print(hypers[h] + '->' + hypos[h]+'| no')
