import os
import pickle
import h5py
import numpy as np
import torch
from nltk.corpus import wordnet as wn
from open_relation1.infer import tree_infer2, simple_infer
from open_relation1.model.object import model
from open_relation1 import vrd_data_config
from open_relation1.train.train_config import hyper_params
from open_relation1.dataset.vrd.label_hier.obj_hier import objnet

def score_pred(pred_ind, org_label_ind, pred_label, wn_label, org2path):
    if pred_ind == org_label_ind:
        return 1
    elif pred_ind not in org2path[org_label_ind]:
        return 0
    else:
        wn_node = wn.synset(wn_label)
        hyper_paths = wn_node.hypernym_paths()
        best_ratio = 0
        for h_path in hyper_paths:
            for i, node in enumerate(h_path):
                if node.name() == pred_label:
                    best_ratio = max((i+1) * 1.0 / (len(h_path)+1), best_ratio)
                    break
        return best_ratio



# prepare feature
config = hyper_params['vrd']['object']
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
org2pw_path = vrd_data_config.vrd_object_config['vrd2pw_path']
org2pw = pickle.load(open(org2pw_path))
label2index_path = vrd_data_config.vrd_object_config['label2index_path']
label2index = pickle.load(open(label2index_path))
index2label_path = vrd_data_config.vrd_object_config['index2label_path']
index2label = pickle.load(open(index2label_path))

org_indexes = [label2index[i] for i in org2wn.keys()]


# load model with best weights
best_weights_path = config['latest_weight_path']
net = model.HypernymVisual_acc(config['visual_d'], config['hidden_d'], config['embedding_d'])
if os.path.isfile(best_weights_path):
    net.load_state_dict(torch.load(best_weights_path))
    print('Loading weights success.')
net.cuda()
net.eval()
print(net)

# eval
# simple TF counter
counter = 0
T = 0.0
# expected -> actual
e_p = []

rank_scores = tree_infer2.cal_rank_scores1(len(index2label))
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
        org_label_ind = label2index[org_label]
        scores = net.forward2(vf_v, lfs_v).cpu().data\

        if counter == 6:
            a = 1

        pred_ind, cands = tree_infer2.my_infer(objnet, scores, rank_scores)
        # pred_ind, cands = tree_infer.my_infer(scores, org2path, org2pw, label2index, index2label, rank_scores)
        # pred_ind, cands = simple_infer.simple_infer(scores, org2path, label2index)
        pred_score = score_pred(pred_ind, org_label_ind, index2label[pred_ind], org2wn[org_label][0], org2path)
        T += pred_score
        if pred_score > 0:
            result = str(counter).ljust(5) + ' T: '
        else:
            result = str(counter).ljust(5) + ' F: '

        pred_str = (result + org_label + ' -> ' + index2label[pred_ind]).ljust(40) + ' %.2f | ' % pred_score
        cand_str = ' [%s(%d) , %s(%d)]' % (index2label[cands[0][0]], cands[0][1], index2label[cands[1][0]], cands[1][1])
        print(pred_str + cand_str)

print('\n=========================================')
print('accuracy: %.2f' % (T / counter))


