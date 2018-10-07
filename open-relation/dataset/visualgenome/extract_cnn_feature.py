import os
import copy
import json
import pickle
import random
import numpy as np
import caffe
import cv2
from nltk.corpus import wordnet as wn
import _init_paths
from fast_rcnn.test import im_detect
import data_config
import vs_anno_2_dict as org


def prepare_boxes_labels(anno_root, anno_list_path, box_save_path, label_save_path, synset_save_path):
    boxes = dict()
    labels = dict()
    synsets = dict()
    with open(anno_list_path, 'r') as anno_list_file:
        anno_list = anno_list_file.read().splitlines()
    for i in range(0, len(anno_list)):
        anno_file_id = anno_list[i]
        print('prepare processing[%d/%d] %s' % (len(anno_list), (i + 1), anno_file_id))
        anno = org.vs_anno_2_dict(os.path.join(anno_root, anno_list[i]+'.json'))
        image_id = anno['filename'].split('.')[0]
        anno_objects = anno['objects']
        box_list = []
        label_list = []
        synset_list = []
        for o in anno_objects:
            xmin = int(o['xmin'])
            ymin = int(o['ymin'])
            xmax = int(o['xmax'])
            ymax = int(o['ymax'])
            if len(o['synsets']) > 0:
                synset = o['synsets'][0]
            else:
                synset = None
            label_list.append(o['name'])
            box_list.append([xmin, ymin, xmax, ymax])
            synset_list.append(synset)
        boxes[image_id] = box_list
        labels[image_id] = label_list
        synsets[image_id] = synset_list
    with open(label_save_path, 'wb') as label_file:
        pickle.dump(labels, label_file)
    with open(box_save_path, 'wb') as boxes_file:
        pickle.dump(boxes, boxes_file)
    with open(synset_save_path, 'wb') as synset_file:
        pickle.dump(synsets, synset_file)


def extract_fc7_features(net, boxes, labels, synsets, img_root, list_path,  feature_root, label_path, vs_wn2index):
    label_list = []
    with open(list_path, 'r') as list_file:
        image_list = list_file.read().splitlines()
    for i in range(0, len(image_list)):
        image_id = image_list[i]
        print('fc7 processing[%d/%d] %s' % (len(image_list), (i + 1), image_id))
        if image_id not in boxes:
            continue
        curr_img_boxes = np.array(boxes[image_id])
        curr_img_labels = labels[image_id]
        curr_img_synsets = synsets[image_id]
        box_num = curr_img_boxes.shape[0]
        if box_num == 0:  # no object
            continue
        img = cv2.imread(os.path.join(img_root, image_id+'.jpg'))
        im_detect(net, img, curr_img_boxes)
        fc7s = np.array(net.blobs['fc7'].data)
        feature_id = image_id + '.bin'
        feature_path = os.path.join(feature_root, feature_id)
        with open(feature_path, 'w') as feature_file:  # save fc7 feature
            pickle.dump(fc7s, feature_file)
        for f in range(0, len(fc7s)):
            label = curr_img_labels[f]
            wn_index = vs_wn2index[label]
            label_list.append(feature_id + ' ' + str(f) + ' ' + str(wn_index) + ' 1\n')
            syn = curr_img_synsets[f]
            if syn is not None:
                synset = wn.synset(syn)
                hypernym_paths = synset.hypernym_paths()
                for s in hypernym_paths[0]:
                    wn_index = vs_wn2index[s.name()]
                    label_list.append(feature_id + ' ' + str(f) + ' ' + str(wn_index) + ' 1\n')

    label_list_path = os.path.join(label_path)
    with open(label_list_path, 'w') as label_file:
        label_file.writelines(label_list)


# def extract_fc7_features1(net, boxes, labels, img_root, anno_root,  feature_root, label_root, image_sum):
#     label_list = []
#     package_id = 0
#     package_capacity = 1000
#     package_fcs = []
#     package_item_counter = 0
#     anno_list = os.listdir(anno_root)
#     anno_list = np.sort(anno_list, axis=0)
#     for i in range(0, len(anno_list)):
#         if i == image_sum:
#             break
#         image_id = anno_list[i]
#         image_id = image_id.split('.')[0] + '.jpg'
#         img = cv2.imread(os.path.join(img_root, image_id))
#         if image_id not in boxes:
#             continue
#         box_list = np.array(boxes[image_id])
#         curr_img_labels = labels[image_id]
#         box_num = box_list.shape[0]
#         if box_num == 0:
#             continue
#         im_detect(net, img, box_list)
#         fc7s = np.array(net.blobs['fc7'].data)
#         print('processing[%d/%d] %s' % (len(anno_list), (i + 1), image_id))
#         for f in range(0, len(fc7s)):
#             label = curr_img_labels[f]
#             label_list.append(str(package_id)+'.bin/'+str(package_item_counter)+' '+label+'\n')
#             fc = fc7s[f]
#             if package_item_counter < package_capacity:
#                 package_fcs.append(fc)
#             elif package_item_counter == package_capacity:
#                 # save package
#                 package_save_path = os.path.join(feature_root, str(package_id)+'.bin')
#                 with open(package_save_path, 'wb') as output_file:
#                     pickle.dump(package_fcs, output_file)
#                 # new package
#                 package_id += 1
#                 package_fcs = []
#                 package_item_counter = 0
#                 package_fcs.append(fc)
#             package_item_counter += 1
#         if i == (len(anno_list) - 1) and len(package_fcs) > 0:
#             # last image, save package
#             package_save_path = os.path.join(feature_root, str(package_id) + '.bin')
#             with open(package_save_path, 'wb') as output_file:
#                 pickle.dump(package_fcs, output_file)
#     label_list_path = os.path.join(label_root, 'labels.txt')
#     with open(label_list_path, 'w') as label_file:
#         label_file.writelines(label_list)


def generate_negative_data(list_path, wn_synset_sum):
    print('generating negative items ......')
    with open(list_path, 'r') as list_file:
        data_list = list_file.readlines()
    new_data_list = copy.copy(data_list)
    for line in data_list:
        item = line.split(' ')
        negative_label = random.randint(0, wn_synset_sum - 1)
        new_line = item[0]+' '+item[1]+' '+str(negative_label)+' -1\n'
        new_data_list.append(new_line)
    with open(list_path, 'w') as list_file:
        list_file.writelines(new_data_list)


if __name__ == '__main__':
    wn_synsets_path = data_config.WN_SYNSETS_PATH
    vs_root = data_config.VS_ROOT
    prototxt = data_config.FAST_PROTOTXT_PATH
    caffemodel = data_config.FAST_CAFFEMODEL_PATH
    datasets = ['train', 'val', 'test']
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    for d in datasets:
        label_save_path = os.path.join(vs_root, 'feature/prepare/' + d + '_labels.bin')
        box_save_path = os.path.join(vs_root, 'feature/prepare/' + d + '_boxes.bin')
        synset_save_path = os.path.join(vs_root, 'feature/prepare/' + d + '_synsets.bin')
        fc7_save_root = os.path.join(vs_root, 'feature/fc7')
        label_save_root = os.path.join(vs_root, 'feature/label/' + d + '.txt')
        anno_root = os.path.join(vs_root, 'anno')
        anno_list = os.path.join(vs_root, 'ImageSets/Main/' + d + '.txt')
        img_root = os.path.join(vs_root, 'JPEGImages')
        prepare_boxes_labels(anno_root, anno_list, box_save_path, label_save_path, synset_save_path)
        with open(box_save_path, 'rb') as box_file:
            boxes = pickle.load(box_file)
        with open(label_save_path, 'rb') as label_file:
            labels = pickle.load(label_file)
        with open(data_config.VS_WN2INDEX_PATH, 'rb') as vs_wn2index_file:
            vs_wn2index = pickle.load(vs_wn2index_file)
        with open(data_config.LABEL2WN_PATH, 'r') as label2wn_file:
            label2wn = json.load(label2wn_file)
        extract_fc7_features(net, boxes, labels, img_root, anno_list, fc7_save_root, label_save_root,
                             vs_wn2index, label2wn)
        generate_negative_data(label_save_root, len(vs_wn2index.keys()))