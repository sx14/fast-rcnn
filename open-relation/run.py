import os
import numpy as np
import pickle
import h5py
import torch
from nltk.corpus import wordnet as wn
from model import model
from train_config import hyper_params
from dataset.pascaltest import label_map
config = hyper_params['pascal']
print('Loading model ...')
model_weight_path = config['best_weight_path']
net = model.HypernymVisual1(config['visual_d'], config['embedding_d'])
print(net)
net.load_state_dict(torch.load(model_weight_path))
net.eval()
print('Loading label map ...')
feature_root = config['visual_feature_root']
label_list_path = os.path.join(config['dataset_root'], 'feature', 'prepare', 'val_labels.bin')
wn_synsets_path = os.path.join('wordnet-embedding', 'object', 'dataset', 'synset_names.json')
word_vec_path = config['word_vec_path']
p_result = []
n_result = []
sample_sum = 0
corr = 0
label2wn = label_map.label2wn()
wn2index = label_map.wn2index(wn_synsets_path)
print('Preparing word feature vectors ...')
wn_embedding_file = h5py.File(word_vec_path, 'r')
word_embedding = wn_embedding_file['word_vec']
wfs = []
# wns = label2wn.values()
wns = wn2index.keys()
for wn in wns:
    wn_index = wn2index[wn]
    wfs.append(word_embedding[wn_index])
wfs = np.array(wfs)
wfs = torch.from_numpy(wfs).float()
print('Predicting ...')
with open(label_list_path, 'r') as label_list_file:
    img_labels = pickle.load(label_list_file)
for img_id in img_labels:
    img_feature_path = os.path.join(feature_root, img_id+'.bin')
    with open(img_feature_path, 'r') as img_feature_file:
        img_features = pickle.load(img_feature_file)
    labels = img_labels[img_id]
    sample_sum += len(img_features)
    # for each object
    for i in range(0, len(img_features)):
        label = labels[i]
        raw_vf = img_features[i]
        raw_vfs = np.tile(raw_vf, (len(wns), 1))
        vfs = torch.from_numpy(raw_vfs)
        train_vfs = torch.autograd.Variable(vfs)
        train_wfs = torch.autograd.Variable(wfs)
        E = net(train_vfs, train_wfs)
        E = E.data.numpy()
        E = np.reshape(E, (E.size))
        pred_label_index = np.where(E < 0.1)[0]
        pred_wn = wns[pred_label_index]
        output_info = img_id+'.jpg ' + str(i+1) + ' ' + label + ' | ' + pred_wn
        print(output_info)
        if pred_wn == label:
            p_result.append(output_info + '\n')
            corr += 1
        else:
            n_result.append(output_info + '\n')
print(corr * 1.0/ sample_sum)
with open('n_result.txt', 'w') as n_result_file:
    n_result_file.writelines(n_result)
with open('p_result.txt', 'w') as p_result_file:
    p_result_file.writelines(p_result)


def predict_path(predict_wns):
    pred_paths = []
    tail2path = dict()
    tails = set()
    for i in range(0, len(predict_wns)):
        for j in range(0, len(predict_wns)):
            if isHypo(predict_wns[i], predict_wns[j]):
                if i in tail2path:
                    path = pred_paths[tail2path[i]]
                    path.add(j)
                    pred_paths[tail2path[i]] = path
                else:
                    tail2path[i] = len(pred_paths)
                    pred_paths.append(set([i, j]))
                    tails.add(i)


def isHypo(i,j):
    #  i is j's hypo ?
    syn_i = wn.synset(i)
    syn_j = wn.synset(j)
    i_hyper_paths = syn_i.hypernym_paths()
    i_hyper_syns = set()
    for p in i_hyper_paths:
        i_hyper_syns = i_hyper_syns | set(p)
    return syn_j in i_hyper_syns