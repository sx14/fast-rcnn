import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from open_relation1 import vrd_data_config

feature_root = vrd_data_config.vrd_object_feature_root

# label maps
vrd2wn_path = vrd_data_config.vrd_object_config['vrd2wn_path']
vrd2wn = pickle.load(open(vrd2wn_path, 'rb'))
vrd2path_path = vrd_data_config.vrd_object_config['vrd2path_path']
vrd2path = pickle.load(open(vrd2path_path, 'rb'))
index2label_path = vrd_data_config.vrd_object_config['index2label_path']
index2label = pickle.load(open(index2label_path, 'rb'))
label2index_path = vrd_data_config.vrd_object_config['label2index_path']
label2index = pickle.load(open(label2index_path, 'rb'))

# counter
label_counter = np.zeros(len(index2label))
vrd_counter = np.zeros(len(index2label))

# org data
box_label_path = os.path.join(feature_root, 'prepare', 'train_box_label.bin')
box_labels = pickle.load(open(box_label_path, 'rb'))

# counting
for img_id in box_labels:
    img_box_labels = box_labels[img_id]
    for box_label in img_box_labels:
        vrd_label = box_label[4]
        vrd_counter[label2index[vrd_label]] += 1
        label_path = vrd2path[label2index[vrd_label]]
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
