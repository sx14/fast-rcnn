import copy
import pickle
from nltk.corpus import wordnet as wn
from open_relation1.vg_data_config import vg_object_config

class TreeNode():
    def __init__(self, id):
        self.id = id
        self.child = []
        self.embedding = ''

label_tree_info = dict()
vg2wn_path = vg_object_config['vg2wn_path']
vg2wn = pickle.load(open(vg2wn_path, 'r'))
wn_syn_lists = vg2wn.values()
wn_nodes = set()
for wn_syns in wn_syn_lists:
    for wn_syn in wn_syns:
        wn_node = wn.synset(wn_syn)
        # for p in wn_node.hypernym_paths():
        p = wn_node.hypernym_paths()[0]
        for w in p:
            wn_nodes.add(w)
wn_nodes = list(wn_nodes)
# wn_nodes = list(wn.all_synsets('n'))


# wn -> index
wn2index = dict()
for i in range(len(wn_nodes)):
    wn2index[wn_nodes[i].name()] = i
    level_nChild = []
    level_nChild[0] = len(wn_nodes[i].hypernym_paths()[0]) - 1
    level_nChild[1] = 0

# gen wordnet hypernym relations
hypernyms = []
for wn_node in wn_nodes:
    for hyper in wn_node.hypernyms() + wn_node.instance_hypernyms():
        if hyper.name() in wn2index:
            # [hypo, hyper]
            hypernyms.append([wn2index[wn_node.name()], wn2index[hyper.name()]])

# ==== append vg labels ====
vg_labels = []
next_label_index = len(wn2index)
label2index = copy.deepcopy(wn2index)
for vg_label in vg2wn:
    vg_labels.append(vg_label)
    label2index[vg_label] = next_label_index
    next_label_index = next_label_index + 1
# save list of labels
wn_labels = map(lambda s: s.name(), wn_nodes)
labels = wn_labels + vg_labels

label2index_path = vg_object_config['label2index_path']
pickle.dump(label2index, open(label2index_path, 'wb'))

labels_path = vg_object_config['labels_path']
pickle.dump(labels, open(labels_path, 'w'))

