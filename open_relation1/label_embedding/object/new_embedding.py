import copy
import pickle
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
            else:
                children.add(vg_label)
    children_max = 0
    for f in fathers:
        children = fathers[f]
        children_max = max(children_max, len(children))

    print('children max: %d' % children_max)


def max_height(vg2wn):
    max_height = 0
    for vg_label in vg2wn:
        wn_labels = vg2wn[vg_label]
        wn_node = wn.synset(wn_labels[0])
        hyper_path = wn_node.hypernym_paths()[0]
        max_height = max(max_height, len(hyper_path) + 1)
    print('max height: %d' % max_height)


if __name__ == '__main__':
    vg2wn_path = vg_object_config['vg2wn_path']
    vg2wn = pickle.load(open(vg2wn_path, 'rb'))
    max_height(vg2wn)