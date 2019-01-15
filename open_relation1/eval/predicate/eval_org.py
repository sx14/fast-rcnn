import os
import pickle
import h5py
import numpy as np
import torch
from open_relation1.model import model
from open_relation1 import vrd_data_config
from open_relation1.train.train_config import hyper_params
from open_relation1.dataset.vrd.predicate.pre_hier import PreNet


# prepare feature
config = hyper_params['vrd']
test_list_path = os.path.join(vrd_data_config.vrd_predicate_feature_prepare_root, 'test_box_label.bin')
test_box_label = pickle.load(open(test_list_path))
label_vec_path = config['label_vec_path']
label_embedding_file = h5py.File(label_vec_path, 'r')
label_vecs = np.array(label_embedding_file['label_vec'])

# prepare label maps
pn = PreNet()
org2path_path = config['vrd2path_path']
org2path = pickle.load(open(org2path_path))
label2index_path = vrd_data_config.vrd_predicate_config['label2index_path']
label2index = pickle.load(open(label2index_path))
index2label_path = vrd_data_config.vrd_predicate_config['index2label_path']
index2label = pickle.load(open(index2label_path))



mode = 'org'
# mode = 'hier'

# load model with best weights
best_weights_path = config['latest_weight_path']
net = model.HypernymVisual_acc(config['visual_d'], config['embedding_d'])
if os.path.isfile(best_weights_path):
    net.load_state_dict(torch.load(best_weights_path))
    print('Loading weights success.')
else:
    print('Weights not found !')
    exit(1)
net.cuda()
net.eval()
print(net)

# eval
# simple TF counter
counter = 0
T = 0.0
T1 = 0.0
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
        ranked_inds = np.argsort(scores).tolist()
        ranked_inds.reverse()   # descending

        # ====== hier label =====
        if mode == 'hier':
            label_inds = org2path[label2index[org_label]]
            print('\n===== ' + org_label + ' =====')
            print('\n----- answer -----')
            for label_ind in label_inds:
                print(index2label[label_ind])


            preds = ranked_inds[:20]
            print('----- prediction -----')
            for p in preds:
                print('%s : %f' % (index2label[p], scores[p]))
            if counter == 100:
                exit(0)
        # ====== org label only =====
        else:
            org_indexes = set([label2index[l] for l in pn.get_raw_labels()])
            org_pred_counter = 0
            print('\n===== ' + org_label + ' =====')
            for j, pred in enumerate(ranked_inds):
                if pred in org_indexes:
                    expected = label2index[org_label]

                    if org_pred_counter == 0:
                        e_p.append([expected, pred])

                    if pred == expected:
                        result = 'T: '
                        if org_pred_counter == 0:
                            T += 1
                        elif org_pred_counter == 1:
                            T1 += 1
                    else:
                        result = 'F: '
                    if org_pred_counter < 2:
                        print(result + index2label[pred] + '('+str(j+1)+')')
                    else:
                        break
                    org_pred_counter += 1

print('accuracy: %.2f' % (T / counter))
print('potential accuracy increment: %.2f' % (T1 / counter))
pickle.dump(e_p, open('e_p.bin', 'wb'))
