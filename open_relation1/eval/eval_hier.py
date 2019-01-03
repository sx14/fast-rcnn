import os
import pickle
import h5py
import numpy as np
import torch
from open_relation1.infer import infer
from open_relation1.model import model
from open_relation1 import vrd_data_config
from open_relation1.train.train_config import hyper_params


# prepare feature
config = hyper_params['vrd']
test_list_path = os.path.join(vrd_data_config.vrd_object_feature_prepare_root, 'test_box_label.bin')
test_box_label = pickle.load(open(test_list_path))
label_vec_path = config['label_vec_path']
label_embedding_file = h5py.File(label_vec_path, 'r')
label_vecs = np.array(label_embedding_file['label_vec'])

# prepare label maps
org2wn_path = vrd_data_config.vrd_object_config['vrd2wn_path']
org2wn = pickle.load(open(org2wn_path))
org2path_path = config['vrd2path_path']
org2path = pickle.load(open(org2path_path))
label2index_path = vrd_data_config.vrd_object_config['label2index_path']
label2index = pickle.load(open(label2index_path))
index2label_path = vrd_data_config.vrd_object_config['index2label_path']
index2label = pickle.load(open(index2label_path))

org_indexes = [label2index[i] for i in org2wn.keys()]

mode = 'org'
# mode = 'hier'

# load model with best weights
best_weights_path = config['best_weight_path']
net = model.HypernymVisual_acc(config['visual_d'], config['embedding_d'])
if os.path.isfile(best_weights_path):
    net.load_state_dict(torch.load(best_weights_path))
    print('Loading weights success.')
net.cuda()
print(net)

# eval
# simple TF counter
counter = 0
T = 0.0
# expected -> actual
e_p = []

visual_feature_root = config['visual_feature_root']
for feature_file_id in test_box_label:
    box_labels = test_box_label[feature_file_id]
    if len(box_labels) == 0:
        continue
    feature_file_name = feature_file_id+'.bin'
    feature_file_path = os.path.join(visual_feature_root, feature_file_name)
    features = pickle.load(open(feature_file_path, 'rb'))
    for i, box_label in enumerate(test_box_label[feature_file_id]):
        counter += 1
        vf = features[i]
        vf_v = torch.autograd.Variable(torch.from_numpy(vf).float()).cuda()
        lfs_v = torch.autograd.Variable(torch.from_numpy(label_vecs).float()).cuda()
        org_label = box_label[4]
        scores = net.forward2(vf_v, lfs_v).cpu().data
        print(inf)

