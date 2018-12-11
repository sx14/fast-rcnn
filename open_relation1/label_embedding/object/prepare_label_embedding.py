import os
import pickle
import numpy as np
from nltk.corpus import wordnet as wn
from open_relation1.vg_data_config import vg_object_config
from open_relation1 import global_config


def generate_direct_hypernyms(vg2wn, label2index, hypernym_save_path):
    # ==== generate direct hypernym relations ====
    hypernyms = []
    # [[hypo, hyper]]
    for vg_label in vg2wn:
        wn_labels = vg2wn[vg_label]
        for wn_label in wn_labels:
            # vg_label -> wn_label
            hypernyms.append([label2index[vg_label], label2index[wn_label]])
            wn_node = wn.synset(wn_label)
            for hyper in wn_node.hypernyms() + wn_node.instance_hypernyms():
                if hyper in label2index:
                    # wn_label -> hyper_label
                    hypernyms.append([label2index[wn_node.name()], label2index[hyper.name()]])
    # save hypernym dataset
    hypernyms = np.array(hypernyms)
    import h5py
    f = h5py.File(hypernym_save_path, 'w')
    f.create_dataset('hypernyms', data=hypernyms)
    f.close()


if __name__ == '__main__':
    vg2wn_path = vg_object_config['vg2wn_path']
    vg2wn = pickle.load(open(vg2wn_path, 'r'))

    label2index_path = vg_object_config['label2index_path']
    label2index = pickle.load(open(label2index_path, 'rb'))

    hypernym_save_path = os.path.join(global_config.project_root,
                                      'open_relation1', 'label_embedding', 'object', 'vg_dataset')
    generate_direct_hypernyms(vg2wn, label2index, hypernym_save_path)