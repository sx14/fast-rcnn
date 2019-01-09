import os
import json
import pickle
import scipy.io as sio
from nltk.corpus import wordnet as wn
from open_relation1 import vrd_data_config


def try_map_wn(org_labels):
    vrd2wn = dict()
    for vrd_label in org_labels:
        vrd_label = vrd_label.strip()
        syns = wn.synsets(vrd_label)
        if len(syns) > 0:
            vrd2wn[vrd_label] = [syns[0].name()]
        else:
            vrd2wn[vrd_label] = ['']
            # print(vrd_label)

    # fix auto annotation
    vrd2wn['shoes'] = ['shoe.n.01']
    vrd2wn['bike'] = ['bicycle.n.01']
    vrd2wn['plate'] = ['plate.n.04']
    vrd2wn['trash can'] = ['ashcan.n.01']
    vrd2wn['traffic light'] = ['traffic_light.n.01']
    vrd2wn['truck'] = ['truck.n.01']
    vrd2wn['van'] = ['van.n.05']
    vrd2wn['mouse'] = ['mouse.n.04']
    vrd2wn['hydrant'] = ['fireplug.n.01']
    vrd2wn['pants'] = ['trouser.n.01']
    vrd2wn['jeans'] = ['trouser.n.01']
    vrd2wn['monitor'] = ['monitor.n.04']
    vrd2wn['post'] = ['post.n.04']
    return vrd2wn


# ====== label 2 wn ======
vrd_root = vrd_data_config.vrd_root
prepare_root = vrd_data_config.vrd_object_feature_prepare_root

# load object labels from mat
obj_label_path = os.path.join(vrd_root, 'objectListN.mat')
obj_label_file = sio.loadmat(obj_label_path)
obj_labels = obj_label_file['objectListN'][0]
obj_label_list = []
for obj_label in obj_labels:
    obj_label_list.append(obj_label[0]+'\n')
# save labels as readable format
obj_label_path = os.path.join(prepare_root, 'object_labels.txt')
with open(obj_label_path, 'w') as f:
    f.writelines(list(obj_label_list))

# label mapping wn
vrd2wn_json_path = os.path.join(prepare_root, 'vrd2wn.json')
vrd2wn_bin_path = os.path.join(prepare_root, 'vrd2wn.bin')
vrd2wn = try_map_wn(obj_label_list)
# save as readable json
json.dump(vrd2wn, open(vrd2wn_json_path, 'w'), indent=4)
# save as binary file
pickle.dump(vrd2wn, open(vrd2wn_bin_path, 'wb'))


# load predicate labels from mat
pred_label_path = os.path.join(vrd_root, 'relationListN.mat')
pred_label_file = sio.loadmat(pred_label_path)
pred_labels = pred_label_file['relationListN'][0]
pred_label_list = []
for pred_label in pred_labels:
    pred_label_list.append(pred_label[0]+'\n')
# TODO predicate label mapping
# save labels as readable format
pre_label_path = os.path.join(prepare_root, 'predicate_labels.txt')
with open(pre_label_path, 'w') as f:
    f.writelines(list(pred_label_list))