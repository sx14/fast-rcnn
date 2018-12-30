import os
import pickle
import h5py
import numpy as np
import torch
from traditional.model import model
from open_relation1 import vrd_data_config
from train_config import hyper_params


# prepare feature
config = hyper_params['vrd']
test_list_path = os.path.join(vrd_data_config.vrd_object_feature_prepare_root, 'test_box_label.bin')
test_box_label = pickle.load(open(test_list_path))

# prepare label maps
vg2wn_path = vrd_data_config.vrd_object_config_t['vrd2wn_path']
vg2wn = pickle.load(open(vg2wn_path))
vg2path_path = config['vrd2path_path']
vg2path = pickle.load(open(vg2path_path))
label2index_path = vrd_data_config.vrd_object_config_t['label2index_path']
label2index = pickle.load(open(label2index_path))
index2label_path = vrd_data_config.vrd_object_config_t['index2label_path']
index2label = pickle.load(open(index2label_path))

# load model with best weights
best_weights_path = config['best_weight_path']
net = model.HypernymVisual_acc(config['visual_d'], config['class_num'])
if os.path.isfile(best_weights_path):
    net.load_state_dict(torch.load(best_weights_path))
    print('Loading weights success.')
net.cuda()
net.eval()
print(net)

# eval
visual_feature_root = config['visual_feature_root']
counter = 0
TP = 0.0
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
        label = box_label[4]
        with torch.no_grad():
            scores = net.forward(vf_v).cpu().data
            pred_ind = np.argmax(scores.numpy())
            if pred_ind == label2index[label]:
                TP += 1
                print('T: ' + label + ' : ' + index2label[pred_ind])
            else:
                print('F: ' + label + ' : ' + index2label[pred_ind])

print('accuracy: %.2f' % (TP/counter))



