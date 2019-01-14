import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from open_relation1 import vrd_data_config

feature_root = vrd_data_config.vrd_predicate_feature_root

# label maps
raw2path_path = vrd_data_config.vrd_predicate_config['raw2path_path']
raw2path = pickle.load(open(raw2path_path, 'rb'))
index2label_path = vrd_data_config.vrd_predicate_config['index2label_path']
index2label = pickle.load(open(index2label_path, 'rb'))
label2index_path = vrd_data_config.vrd_predicate_config['label2index_path']
label2index = pickle.load(open(label2index_path, 'rb'))

# counter
label_counter = np.zeros(len(index2label))
raw_counter = np.zeros(len(index2label))

# org data
box_label_path = os.path.join(feature_root, 'prepare', 'train_box_label.bin')
box_labels = pickle.load(open(box_label_path, 'rb'))

# counting
for img_id in box_labels:
    img_box_labels = box_labels[img_id]
    for box_label in img_box_labels:
        raw_label = box_label[4]
        raw_counter[label2index[raw_label]] += 1
        label_path = raw2path[label2index[raw_label]]
        for l in label_path:
            label_counter[l] += 1


show_counter = label_counter

ranked_counts = np.sort(show_counter).tolist()
ranked_counts.reverse()
ranked_counts = np.array(ranked_counts)
ranked_counts = ranked_counts[ranked_counts > 0]

rank = np.argsort(show_counter).tolist()
rank.reverse()
rank = rank[:len(ranked_counts)]

for i in range(len(rank)):
    print(index2label[rank[i]] + ' : ' + str(ranked_counts[i]))


ranked_labels = [index2label[i] for i in rank]

plt.plot(ranked_labels, ranked_counts)
plt.title('distribution')
plt.xlabel('label')
plt.ylabel('count')
plt.show()
