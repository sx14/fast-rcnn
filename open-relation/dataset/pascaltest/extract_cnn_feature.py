import os
import json
import copy
import random
import pickle
import pascal_anno_2_dict as org
import caffe
import cv2
import numpy as np
from nltk.corpus import wordnet as wn
import _init_paths
from fast_rcnn.test import im_detect
import label_map
import data_config


def prepare_boxes_labels(anno_root, anno_list_path, box_save_path, label_save_path):
    boxes = dict()
    wn_labels = dict()
    label2wn = label_map.label2wn()
    with open(anno_list_path, 'r') as anno_list_file:
        anno_list = anno_list_file.read().splitlines()
    for i in range(0, len(anno_list)):
        anno_file_id = anno_list[i]
        print('prepare processing[%d/%d] %s' % (len(anno_list), (i+1), anno_file_id))
        anno = org.pascal_anno_2_dict(os.path.join(anno_root, anno_file_id+'.xml'), label2wn)
        image_id = anno['filename'].split('.')[0]
        anno_objects = anno['objects']
        box_list = []
        label_list = []
        for o in anno_objects:
            xmin = int(o['xmin'])
            ymin = int(o['ymin'])
            xmax = int(o['xmax'])
            ymax = int(o['ymax'])
            label_list.append(o['name'])
            box_list.append([xmin, ymin, xmax, ymax])
        boxes[image_id] = box_list
        wn_labels[image_id] = label_list
    with open(label_save_path, 'wb') as label_file:
        pickle.dump(wn_labels, label_file)
    with open(box_save_path, 'wb') as boxes_file:
        pickle.dump(boxes, boxes_file)


def extract_fc7_features(net, boxes, wn_labels, img_root, list_path, feature_root, label_path, wn2index):
    label_list = []
    with open(list_path, 'r') as list_file:
        image_list = list_file.read().splitlines()
    for i in range(0, len(image_list)):
        image_id = image_list[i]
        print('fc7 processing[%d/%d] %s' % (len(image_list), (i + 1), image_id))
        if image_id not in boxes:
            continue
        box_list = np.array(boxes[image_id])
        curr_img_labels = wn_labels[image_id]
        box_num = box_list.shape[0]
        if box_num == 0:
            continue
        img = cv2.imread(os.path.join(img_root, image_id+'.jpg'))
        im_detect(net, img, box_list)
        fc7s = np.array(net.blobs['fc7'].data)
        feature_id = image_id + '.bin'
        feature_path = os.path.join(feature_root, feature_id)
        with open(feature_path, 'w') as feature_file:
            pickle.dump(fc7s, feature_file)
        for f in range(0, len(fc7s)):
            label = curr_img_labels[f]
            synset = wn.synset(label)
            hypernym_paths = synset.hypernym_paths()
            for s in hypernym_paths[0]:
                wn_index = wn2index[s.name()]
                # feature.bin box_id wn_label 1
                # 1 means positive
                label_list.append(feature_id+' '+str(f)+' '+str(wn_index)+' 1\n')
    label_list_path = os.path.join(label_path)
    with open(label_list_path, 'w') as label_file:
        label_file.writelines(label_list)


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
    voc_root = data_config.VOC_ROOT
    prototxt = data_config.FAST_PROTOTXT_PATH
    caffemodel = data_config.FAST_CAFFEMODEL_PATH
    datasets = ['train', 'val', 'test']
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    for d in datasets:
        label_save_path = os.path.join(voc_root, 'feature/prepare/'+d+'_labels.bin')
        box_save_path = os.path.join(voc_root, 'feature/prepare/'+d+'_boxes.bin')
        fc7_save_root = os.path.join(voc_root, 'feature/fc7')
        label_save_root = os.path.join(voc_root, 'feature/label/'+d+'.txt')
        anno_root = os.path.join(voc_root, 'Annotations')
        anno_list = os.path.join(voc_root, 'ImageSets/Main/'+d+'.txt')
        img_root = os.path.join(voc_root, 'JPEGImages')
        prepare_boxes_labels(anno_root, anno_list, box_save_path, label_save_path)
        with open(box_save_path, 'rb') as box_file:
            boxes = pickle.load(box_file)
        with open(label_save_path, 'rb') as label_file:
            wn_labels = pickle.load(label_file)
        wn2index = label_map.wn2index(wn_synsets_path)
        extract_fc7_features(net, boxes, wn_labels, img_root, anno_list, fc7_save_root, label_save_root, wn2index)
        generate_negative_data(label_save_root, len(wn2index.keys()))