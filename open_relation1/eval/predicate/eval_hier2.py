import os
import pickle
import h5py
import numpy as np
import torch
from open_relation1.model.predicate import model
from open_relation1 import vrd_data_config
from open_relation1.train.train_config import hyper_params
from open_relation1.dataset.vrd.label_hier.pre_hier import prenet
from open_relation1.dataset.vrd.label_hier.obj_hier import objnet
from open_relation1.language.infer.model import RelationEmbedding
from open_relation1.language.infer.lang_config import train_params
from open_relation1.infer import tree_infer2


def score_pred(pred_ind, raw_label_ind, pred_label, raw_label, raw2path, pre_net):
    if pred_ind == raw_label_ind:
        return 1
    elif pred_ind not in raw2path[raw_label_ind]:
        return 0
    else:
        pre = pre_net.get_node_by_name(raw_label)
        hyper_paths = pre.hyper_paths()
        best_ratio = 0
        for h_path in hyper_paths:
            for i, node in enumerate(h_path):
                if node.name() == pred_label:
                    best_ratio = max((i+1) * 1.0 / (len(h_path)+1), best_ratio)
                    break
        return best_ratio



# prepare feature
pre_config = hyper_params['vrd']['predicate']
obj_config = hyper_params['vrd']['object']
test_list_path = os.path.join(vrd_data_config.vrd_predicate_feature_prepare_root, 'test_box_label.bin')
test_box_label = pickle.load(open(test_list_path))


pre_label_vec_path = pre_config['label_vec_path']
label_embedding_file = h5py.File(pre_label_vec_path, 'r')
pre_label_vecs = np.array(label_embedding_file['label_vec'])
label_embedding_file.close()

obj_label_vec_path = obj_config['label_vec_path']
label_embedding_file = h5py.File(obj_label_vec_path, 'r')
obj_label_vecs = np.array(label_embedding_file['label_vec'])
label_embedding_file.close()

# prepare label maps
org2path_path = pre_config['vrd2path_path']
org2path = pickle.load(open(org2path_path))
label2index_path = vrd_data_config.vrd_predicate_config['label2index_path']
label2index = pickle.load(open(label2index_path))
index2label_path = vrd_data_config.vrd_predicate_config['index2label_path']
index2label = pickle.load(open(index2label_path))

mode = 'org'
# mode = 'hier'

# load visual model with best weights
vmodel_best_weights_path = pre_config['latest_weight_path']
vmodel = model.PredicateVisual_acc()
if os.path.isfile(vmodel_best_weights_path):
    vmodel.load_state_dict(torch.load(vmodel_best_weights_path))
    print('Loading visual model weights success.')
else:
    print('Weights not found !')
    exit(1)
vmodel.cuda()
vmodel.eval()
# print(vmodel)

# load language model with best weights
lmodel_best_weights_path = train_params['best_model_path']
lmodel = RelationEmbedding(train_params['embedding_dim'] * 2, train_params['embedding_dim'], pre_label_vec_path)
if os.path.isfile(lmodel_best_weights_path):
    lmodel.load_state_dict(torch.load(lmodel_best_weights_path))
    print('Loading language model weights success.')
else:
    print('Weights not found !')
    exit(1)
lmodel.cuda()
lmodel.eval()
# print(lmodel)

# eval
# simple TF counter
counter = 0
T = 0.0
T_C = 0.0
# expected -> actual
e_p = []
T_ranks = []
F_ranks = []


rank_scores = tree_infer2.cal_rank_scores(len(index2label))
visual_feature_root = pre_config['visual_feature_root']
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
        vf = vf[np.newaxis, :]
        vf_v = torch.autograd.Variable(torch.from_numpy(vf).float()).cuda()
        pre_lfs_v = torch.autograd.Variable(torch.from_numpy(pre_label_vecs).float()).cuda()
        obj_lfs_v = torch.autograd.Variable(torch.from_numpy(obj_label_vecs).float()).cuda()

        # visual prediction
        v_pre_scores, _, _ = vmodel.forward2(vf_v, pre_lfs_v, obj_lfs_v)

        # language prediction
        sbj_label = box_label[9]
        sbj_ind = objnet.get_node_by_name(sbj_label).index()
        sbj_vec = obj_lfs_v[sbj_ind].unsqueeze(0)

        obj_label = box_label[14]
        obj_ind = objnet.get_node_by_name(obj_label).index()
        obj_vec = obj_lfs_v[obj_ind].unsqueeze(0)

        l_pre_scores = lmodel(sbj_vec, obj_vec)[0]

        pre_scores = (v_pre_scores * 0.6 + l_pre_scores * 0.4) / 2

        pred_ind, cands = tree_infer2.my_infer(prenet, pre_scores.cpu().data, rank_scores, 'pre')
        org_label = box_label[4]
        org_label_ind = prenet.get_node_by_name(org_label).index()
        pred_score = score_pred(pred_ind, org_label_ind, index2label[pred_ind], org_label, org2path, prenet)
        T += pred_score
        if pred_score > 0:
            result = str(counter).ljust(5) + ' T: '
            T_C += 1
        else:
            result = str(counter).ljust(5) + ' F: '

        pred_str = (result + org_label + ' -> ' + index2label[pred_ind]).ljust(40) + ' %.2f | ' % pred_score
        cand_str = ' [%s(%d) , %s(%d)]' % (index2label[cands[0][0]], cands[0][1], index2label[cands[1][0]], cands[1][1])
        print(pred_str + cand_str)


print('\n=========================================')
print('accuracy: %.4f (%.4f)' % ((T / counter), (T_C / counter)))