import os
import json
import numpy as np
import pickle
import h5py
import torch
from model import model
from train_config import hyper_params
from dataset.pascaltest import label_map

dataset_name = 'vs'
config = hyper_params[dataset_name]
print('Loading model ...')
model_weight_path = config['best_weight_path']
net = model.HypernymVisual1(config['visual_d'], config['embedding_d'])
print(net)
net.load_state_dict(torch.load(model_weight_path))
net.eval()

print('Loading label map ...')
dataset_root = config['dataset_root']
label_list_path = config['label_path']  # img,offset -> label
label2path_path = config['label2path_path']
word_vec_path = config['word_vec_path']  # label embedding
with open(label_list_path, 'r') as label_list_file:
    img_labels = pickle.load(label_list_file)
with open(label2path_path, 'r') as label2path_file:
    label2path = json.load(label2path_file)
p_result = []
n_result = []
sample_sum = 0
corr = 0
wn_embedding_file = h5py.File(word_vec_path, 'r')
word_embedding = wn_embedding_file['word_vec']
wfs = []
wn2index_path = os.path.join(dataset_root, 'feature', 'object', 'prepare', 'wn2index.json')
wn2index = json.load(open(wn2index_path, 'r'))
wn_synsets_path = os.path.join('wordnet-embedding', 'object', dataset_name+'_dataset', 'synset_names_with_'+dataset_name+'.json')   # all labels
index2label = json.load(open(wn_synsets_path, 'r'))
for w in range(0, len(index2label)):
    wfs.append(word_embedding[w])
wfs = np.array(wfs)
wfs = torch.from_numpy(wfs).float()
# label2wn_path = os.path.join(dataset_root, 'feature', 'object', 'prepare', 'label2wn.json')
# label2wn = json.load(open(label2wn_path, 'r'))
# index2label = label2wn.keys()
# for label in index2label:
#     wfs.append(word_embedding[wn2index[label]])
# wfs = np.array(wfs)
# wfs = torch.from_numpy(wfs).float()




print('Predicting ...')
feature_root = config['visual_feature_root']    # get visual feature
img_list_path = os.path.join(config['list_root'], 'val_small.txt')  # val image list
with open(img_list_path, 'r') as img_list_file:
    img_list = img_list_file.read().splitlines()
for line in img_list:
    img_id = line.split(' ')[0].replace('.bin', '')
    img_feature_path = os.path.join(feature_root, img_id+'.bin')
    with open(img_feature_path, 'r') as img_feature_file:
        img_features = pickle.load(img_feature_file)
    labels = img_labels[img_id]
    sample_sum += len(img_features)
    # for each object
    for i in range(0, len(img_features)):
        print('------------------')
        label = labels[i]
        raw_vf = img_features[i]
        vfs = torch.from_numpy(raw_vf)
        train_vfs = torch.autograd.Variable(vfs)
        train_wfs = torch.autograd.Variable(wfs)
        E = net(train_vfs, train_wfs)
        E = E.data.numpy()
        E = np.reshape(E, (E.size))
        pred_label_indexes = np.argsort(E)
        for pred_label_index in pred_label_indexes[:1]:
            pred_wn = index2label[pred_label_index]
            label_index = wn2index[label]
            label_path = label2path[str(label_index)]
            output_info = img_id+'.jpg ' + str(i+1) + ' ' + label + ' | ' + pred_wn
            print(output_info)
            hit = 0
            # if pred_label_index in label_path:
            if label == pred_wn:
                p_result.append(output_info + ' | y\n')
                hit = 1
            else:
                n_result.append(output_info + ' | n\n')
            corr += hit
print(corr * 1.0 / sample_sum)
with open('n_result.txt', 'w') as n_result_file:
    n_result_file.writelines(n_result)
with open('p_result.txt', 'w') as p_result_file:
    p_result_file.writelines(p_result)


