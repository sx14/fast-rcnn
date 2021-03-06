import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from open_relation.dataset.dataset_config import DatasetConfig
from open_relation.dataset.vrd.label_hier.obj_hier import objnet
from open_relation.dataset.vrd.label_hier.pre_hier import prenet


dataset_config = DatasetConfig('vrd')
target = 'object'


if target == 'object':
    labelnet = objnet
elif target == 'predicate':
    labelnet = prenet

# parent-children
p2c = dict()
for i in range(labelnet.label_sum()):
    n = labelnet.get_node_by_index(i)
    p2c[n.name()] = {}
for i in range(labelnet.label_sum()):
    n = labelnet.get_node_by_index(i)
    hypers = n.hypers()
    for h in hypers:
        p2c[h.name()][n.name()] = 0

# label maps

raw2path = labelnet.raw2path()
index2label = labelnet.index2label()
label2index = labelnet.label2index()

# counter
label_counter = np.zeros(len(index2label))
raw_counter = np.zeros(len(index2label))

# org data
extra_root = dataset_config.extra_config[target].prepare_root
box_label_path = os.path.join(extra_root, 'train_box_label.bin')
box_labels = pickle.load(open(box_label_path, 'rb'))

# counting
for img_id in box_labels:
    img_box_labels = box_labels[img_id]
    for box_label in img_box_labels:
        raw_label = box_label[4]
        raw_counter[raw_label] += 1
        label_path = raw2path[raw_label]
        for l in label_path:
            label_counter[l] += 1


show_counter = label_counter

ranked_counts = np.sort(show_counter)[::-1]
ranked_counts = ranked_counts[ranked_counts > 0]

rank = np.argsort(show_counter)[::-1]
rank = rank[:len(ranked_counts)]

for i in range(len(rank)):
    node = labelnet.get_node_by_index(rank[i])
    node.weight = ranked_counts[i]
    print(index2label[rank[i]] + ' : ' + str(ranked_counts[i]))

for p in p2c:
    children = p2c[p]
    for c in children:
        ci = label2index[c]
        ccount = label_counter[ci]
        p2c[p][c] = ccount

ranked_labels = [index2label[i] for i in rank]

plt.plot(ranked_labels, ranked_counts)
plt.title('distribution')
plt.xlabel('label')
plt.ylabel('count')
plt.show()
