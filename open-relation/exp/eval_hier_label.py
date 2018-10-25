import json
import h5py
import numpy as np

weight_path = '../wordnet-embedding/object/word_vec_vs.h5'
wn_embedding_file = h5py.File(weight_path, 'r')
wn_embedding = wn_embedding_file['word_vec']
wn2index_path = '/media/sunx/Data/dataset/visual genome/feature/object/prepare/wn2index.json'
with open(wn2index_path, 'r') as wn2index_file:
    wn2index = json.load(wn2index_file)
synset_name_path = '../wordnet-embedding/object/exp_dataset/synset_names_with_VS.json'
with open(synset_name_path, 'r') as synset_name_file:
    synset_names = json.load(synset_name_file)
label2wn_path = '/media/sunx/Data/dataset/visual genome/feature/object/prepare/label2wn.json'
with open(label2wn_path, 'r') as label2wn_file:
    label2wn = json.load(label2wn_file)

all = len(wn2index.keys())
positive = 0
entity = 0
for label in label2wn:
    wn = label2wn[label][0]

    hyper = wn2index[wn]
    if wn == 'entity.n.01':
        entity += 1
    hypo = wn2index[label]
    hyper_v = np.array(wn_embedding[hyper])
    hypo_v = np.array(wn_embedding[hypo])

    h_zero = np.stack((hyper_v, np.zeros(hyper_v.shape)), axis=1)
    relu = np.max(h_zero, axis=1)
    relu = relu * relu
    E = np.sum(relu)
    # print('word norm' + str(E))

    sub = hyper_v - hypo_v
    sub_zero = np.stack((sub, np.zeros(sub.shape)), axis=1)
    relu = np.max(sub_zero, axis=1)
    relu = relu * relu
    E = np.sum(relu)
    # print('sub norm' + str(E))

    if E < 0.5:
        print(wn +'->'+label+'| yes')
        positive += 1
    else:
        print(wn + '->' + label + '| no')
print('acc :' + str(positive * 1.0 / all))
print('entity :' + str(entity * 1.0 / all))