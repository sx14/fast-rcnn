import pickle
from nltk.corpus import wordnet as wn
from open_relation1.vg_data_config import vg_object_config

# pass
vg2wn_path = vg_object_config['vg2wn_path']
vg2wn = pickle.load(open(vg2wn_path, 'rb'))
label2index_path = vg_object_config['label2index_path']
label2index = pickle.load(open(label2index_path, 'rb'))
index2label_path = vg_object_config['index2label_path']
index2label = pickle.load(open(index2label_path, 'rb'))
vg2path_path = vg_object_config['vg2path_path']
vg2path = pickle.load(open(vg2path_path, 'rb'))

for vg_label in vg2wn:
    wn_label = vg2wn[vg_label][0]
    wn_node = wn.synset(wn_label)
    hyper_path = wn_node.hypernym_paths()[0]
    # print(hyper_path)
    wn_label_ind = label2index[wn_label]
    wn_path_inds = vg2path[wn_label_ind]
    hyper_path1 = []
    for i in wn_path_inds:
        hyper_path1.append(index2label[i])
    # print(hyper_path1)
    if len(hyper_path) != len(hyper_path1):
        print(wn_label_ind)
        print('wrong!!!\n')