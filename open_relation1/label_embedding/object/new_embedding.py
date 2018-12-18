import h5py
import pickle
import numpy as np
from nltk.corpus import wordnet as wn
from open_relation1.vg_data_config import vg_object_config

def max_children_num(vg2wn):
    # count children
    fathers = dict()
    for vg_label in vg2wn:
        wn_labels = vg2wn[vg_label]
        wn_node = wn.synset(wn_labels[0])
        hyper_path = wn_node.hypernym_paths()[0]
        for h in range(0, len(hyper_path)):
            h_node = hyper_path[h]
            if h_node.name() in fathers:
                children = fathers[h_node.name()]
            else:
                children = set()
                fathers[h_node.name()] = children
            if h < len(hyper_path)-1:
                children.add(hyper_path[h+1].name())
    children_max = 0
    for f in fathers:
        children = fathers[f]
        children_max = max(children_max, len(children))
    print('children max: %d' % children_max)
    return children_max


def max_height(vg2wn):
    max_height = 0
    for vg_label in vg2wn:
        wn_labels = vg2wn[vg_label]
        wn_node = wn.synset(wn_labels[0])
        hyper_path = wn_node.hypernym_paths()[0]
        max_height = max(max_height, len(hyper_path))
    print('max height: %d' % max_height)
    return max_height


def direct_relation(vg2wn):
    child2father = dict()
    father2children = dict()
    for vg_label in vg2wn:
        wn_label = vg2wn[vg_label][0]
        wn_node = wn.synset(wn_label)
        hyper_path = wn_node.hypernym_paths()[0]
        hyper_path.reverse()
        for h in range(0, len(hyper_path)-1):
            h_node = hyper_path[h]
            if h_node.name() not in child2father:
                father_node = hyper_path[h+1]
                child2father[h_node.name()] = father_node.name()
                if father_node.name() not in father2children:
                    father2children[father_node.name()] = set()
                children = father2children[father_node.name()]
                children.add(h_node.name())
    return child2father, father2children


def order_embedding_by_dfs(father2children, father, father_vec, label2vec):
    label2vec[father] = father_vec
    if father not in father2children:
        return
    children = list(father2children[father])
    max_value = np.max(father_vec)
    avaliable_poses = np.where(father_vec == max_value)[0]
    for i, child in enumerate(children):
        child_vec = np.copy(father_vec)
        for p in range(0, len(avaliable_poses)):
            if p != i:
                child_vec[avaliable_poses[p]] += 1
        order_embedding_by_dfs(father2children, child, child_vec, label2vec)


def save_label_vecs_as_h5(label_vec_path, label2vec, label2index):
    label_vecs = np.zeros((len(label2index), embedding_length))
    for label in label2vec:
        label_vec = label2vec[label]
        label_index = label2index[label]
        label_vecs[label_index] = label_vec
    h5_file = h5py.File(label_vec_path)
    h5_file.create_dataset('label_vec', data=label_vecs.tolist())
    h5_file.close()


if __name__ == '__main__':
    vg2wn_path = vg_object_config['vg2wn_path']
    vg2wn = pickle.load(open(vg2wn_path, 'rb'))
    max_height = max_height(vg2wn)
    max_width = max_children_num(vg2wn)
    embedding_length = 150
    c2f, f2c = direct_relation(vg2wn)
    label2vec = dict()
    order_embedding_by_dfs(f2c, 'entity.n.01', np.zeros(embedding_length), label2vec)
    label2index_path = vg_object_config['label2index_path']
    label2index = pickle.load(open(label2index_path, 'rb'))
    if len(label2index.keys()) != len(label2vec.keys()):
        print('WRONG !!!')
        exit(-1)
    label_vec_path = vg_object_config['label_vec_path1']
    save_label_vecs_as_h5(label_vec_path, label2vec, label2index)




