import os
import pickle
import numpy as np
from nltk.corpus import wordnet as wn
from open_relation.dataset.dataset_config import vrd_object_config
from open_relation import global_config


def generate_direct_hypernyms(vg2wn, label2index, hypernym_save_path):
    # ==== generate direct hypernym relations ====
    hypernyms = []
    all_wn_nodes = set()
    # [[hypo, hyper]]
    for vg_label in vg2wn:
        wn_labels = vg2wn[vg_label]
        for wn_label in wn_labels:
            # vg_label -> wn_label
            hypernyms.append([label2index[vg_label], label2index[wn_label]])
            wn_node = wn.synset(wn_label)
            for wn_path in wn_node.hypernym_paths():
                for w in wn_path:
                    all_wn_nodes.add(w)
    for wn_node in all_wn_nodes:
        for h in wn_node.hypernyms() + wn_node.instance_hypernyms():
            if h.name() in label2index:
                hypernyms.append([label2index[wn_node.name()], label2index[h.name()]])
    # save hypernym dataset
    hypernyms = np.array(hypernyms)
    import h5py
    f = h5py.File(hypernym_save_path, 'w')
    f.create_dataset('hypernyms', data=hypernyms)
    f.close()


if __name__ == '__main__':
    vrd2wn_path = vrd_object_config['vrd2wn_path']
    vrd2wn = pickle.load(open(vrd2wn_path, 'r'))

    label2index_path = vrd_object_config['label2index_path']
    label2index = pickle.load(open(label2index_path, 'rb'))

    hypernym_save_path = os.path.join(global_config.project_root,
                                      'open_relation1', 'label_embedding', 'object', 'vrd_dataset', 'wordnet_with_vrd.h5')
    generate_direct_hypernyms(vrd2wn, label2index, hypernym_save_path)