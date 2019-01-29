import os
import pickle
import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from open_relation.model.predicate import model
from open_relation.dataset import dataset_config
from open_relation.train.train_config import hyper_params
from open_relation.dataset.vrd.label_hier.pre_hier import prenet
from open_relation.dataset.vrd.label_hier.obj_hier import objnet
from open_relation.language.infer.model import RelationEmbedding
from open_relation.language.infer.lang_config import train_params

# prepare feature
pre_config = hyper_params['vrd']['predicate']
obj_config = hyper_params['vrd']['object']
test_list_path = os.path.join(dataset_config.vrd_predicate_feature_prepare_root, 'test_box_label.bin')
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
label2index_path = dataset_config.vrd_predicate_config['label2index_path']
label2index = pickle.load(open(label2index_path))
index2label_path = dataset_config.vrd_predicate_config['index2label_path']
index2label = pickle.load(open(index2label_path))

mode = 'raw'
# mode = 'hier'

# load visual model with best weights
vmodel_best_weights_path = pre_config['best_weight_path']
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
T1 = 0.0
# expected -> actual
e_p = []
T_ranks = []
F_ranks = []

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
        v_ranked_inds = np.argsort(v_pre_scores.cpu().data).tolist()    # ascend
        v_ind2score = np.zeros(len(v_ranked_inds))
        for s, ind in enumerate(v_ranked_inds):     # ascending rank as score
            v_ind2score[ind] = s

        # language prediction
        sbj_label = box_label[9]
        sbj_ind = objnet.get_node_by_name(sbj_label).index()
        sbj_vec = obj_lfs_v[sbj_ind].unsqueeze(0)

        obj_label = box_label[14]
        obj_ind = objnet.get_node_by_name(obj_label).index()
        obj_vec = obj_lfs_v[obj_ind].unsqueeze(0)

        l_pre_scores = lmodel(sbj_vec, obj_vec)[0]
        l_ranked_inds = np.argsort(l_pre_scores.cpu().data).tolist()
        l_ranked_inds.reverse()  # descending
        l_ind2score = np.zeros(len(l_ranked_inds))
        for s, ind in enumerate(l_ranked_inds):
            l_ind2score[ind] = s

        ind2score = np.max([v_ind2score, l_ind2score])
        ranked_inds = np.argsort(ind2score).tolist()
        ranked_inds.reverse()   # descending

        # pre_scores = (v_pre_scores * 0.6 + l_pre_scores * 0.4) / 2
        # ranked_inds = np.argsort(pre_scores.cpu().data).tolist()
        # ranked_inds.reverse()   # descending

        # ====== hier label =====
        pre_label = box_label[4]
        if mode == 'hier':
            label_inds = org2path[label2index[pre_label]]
            print('\n===== ' + pre_label + ' =====')
            print('\n----- answer -----')
            for label_ind in label_inds:
                print(index2label[label_ind])
            preds = ranked_inds[:20]
            print('----- prediction -----')
            for p in preds:
                print('%s : %f' % (index2label[p], v_pre_scores[p]))
            if counter == 100:
                exit(0)
        # ====== raw label only =====
        else:
            org_indexes = set([label2index[l] for l in prenet.get_raw_labels()])
            org_pred_counter = 0
            print('\n===== ' + pre_label + ' =====')
            for j, pred in enumerate(ranked_inds):
                if pred in org_indexes:
                    expected = label2index[pre_label]

                    if org_pred_counter == 0:
                        e_p.append([expected, pred])

                    if pred == expected:
                        result = 'T: '
                        if org_pred_counter == 0:
                            T += 1
                            T_ranks.append(j + 1)
                        elif org_pred_counter == 1:
                            T1 += 1
                    else:
                        result = 'F: '
                        F_ranks.append(j + 1)
                    if org_pred_counter < 2:
                        print(result + index2label[pred] + '('+str(j+1)+')')
                    else:
                        break
                    org_pred_counter += 1

print('\naccuracy: %.2f' % (T / counter))
print('potential accuracy increment: %.2f' % (T1 / counter))
pickle.dump(e_p, open('e_p.bin', 'wb'))


plt.hist(F_ranks, 100)
plt.xlabel('rank')
plt.ylabel('num')
plt.show()