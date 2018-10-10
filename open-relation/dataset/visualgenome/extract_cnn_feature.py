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


def prepare_relation_boxes_labels(anno_root, anno_list_path, box_save_path, label_save_path):
    boxes = dict()
    labels = dict()
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
        for o in anno_objects:
            xmin = int(o['xmin'])
            ymin = int(o['ymin'])
            xmax = int(o['xmax'])
            ymax = int(o['ymax'])
            label_list.append(o['name'])
            box_list.append([xmin, ymin, xmax, ymax])
        boxes[image_id] = box_list
        labels[image_id] = label_list
    with open(label_save_path, 'wb') as label_file:
        pickle.dump(labels, label_file)
    with open(box_save_path, 'wb') as boxes_file:
        pickle.dump(boxes, boxes_file)


def prepare_object_boxes_labels(anno_root, anno_list_path, box_save_path, label_save_path):
    boxes = dict()
    labels = dict()
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
        for o in anno_objects:
            xmin = int(o['xmin'])
            ymin = int(o['ymin'])
            xmax = int(o['xmax'])
            ymax = int(o['ymax'])
            label_list.append(o['name'])
            box_list.append([xmin, ymin, xmax, ymax])
        boxes[image_id] = box_list
        labels[image_id] = label_list
    with open(label_save_path, 'wb') as label_file:
        pickle.dump(labels, label_file)
    with open(box_save_path, 'wb') as boxes_file:
        pickle.dump(boxes, boxes_file)


def extract_fc7_features(net, boxes, labels, img_root, list_path,  feature_root, label_list_path, wn2index, label2wn):
    label_list = []
    wn_synset_sum = len(wn2index.keys())
    with open(list_path, 'r') as list_file:
        image_list = list_file.read().splitlines()
    for i in range(0, len(image_list)):
        image_id = image_list[i]
        print('fc7 processing[%d/%d] %s' % (len(image_list), (i + 1), image_id))
        if image_id not in boxes:
            continue
        box_list = np.array(boxes[image_id])
        curr_img_labels = labels[image_id]
        box_num = box_list.shape[0]
        if box_num == 0:
            continue
        feature_id = image_id + '.bin'
        feature_path = os.path.join(feature_root, feature_id)
        if not os.path.exists(feature_path): # extract fc7
            img = cv2.imread(os.path.join(img_root, image_id+'.jpg'))
            im_detect(net, img, box_list)
            fc7s = np.array(net.blobs['fc7'].data)
            with open(feature_path, 'w') as feature_file:
                pickle.dump(fc7s, feature_file)
        for f in range(0, len(box_list)):
            wn_label = curr_img_labels[f]
            wn_index = wn2index[wn_label]
            label_list.append(feature_id + ' ' + str(f) + ' ' + str(wn_index) + ' 1\n')
            label_list.append(feature_id + ' ' + str(f) + ' ' + str(random.randint(0, wn_synset_sum-1)) + ' -1\n')
            syns = label2wn[wn_label]
            for syn in syns:
                synset = wn.synset(syn)
                hypernym_paths = synset.hypernym_paths()
                for s in hypernym_paths[0]:
                    wn_index = wn2index[s.name()]
                    label_list.append(feature_id + ' ' + str(f) + ' ' + str(wn_index) + ' 1\n')
                    label_list.append(feature_id + ' ' + str(f) + ' ' + str(random.randint(0, wn_synset_sum-1)) + ' -1\n')
    with open(label_list_path, 'w') as label_file:
        label_file.writelines(label_list)


# def generate_negative_data(list_path, wn_synset_sum):
#     print('generating negative items ......')
#     with open(list_path, 'r') as list_file:
#         data_list = list_file.readlines()
#     new_data_list = copy.copy(data_list)
#     for line in data_list:
#         item = line.split(' ')
#         negative_label = random.randint(0, wn_synset_sum - 1)
#         new_line = item[0]+' '+item[1]+' '+str(negative_label)+' -1\n'
#         new_data_list.append(new_line)
#     with open(list_path, 'w') as list_file:
#         list_file.writelines(new_data_list)


def split_a_small_val(val_list_path, length, small_val_path):
    small_val = []
    with open(val_list_path, 'r') as val_list_file:
        val_list = val_list_file.readlines()
        val_list_length = len(val_list)
    for i in range(0, length):
        ind = random.randint(0, val_list_length - 1)
        small_val.append(val_list[ind])
    with open(small_val_path, 'w') as small_val_file:
        small_val_file.writelines(small_val)


if __name__ == '__main__':
    vs_root = data_config.VS_ROOT
    prototxt = data_config.FAST_PROTOTXT_PATH
    caffemodel = data_config.FAST_CAFFEMODEL_PATH
    # datasets = ['train', 'val', 'test']
    datasets = ['train']
    target = 'object'  # relation
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    for d in datasets:
        label_save_path = os.path.join(vs_root, 'feature', target, 'prepare', d + '_labels.bin')
        box_save_path = os.path.join(vs_root, 'feature', target, 'prepare', d + '_boxes.bin')
        fc7_save_root = os.path.join(vs_root, 'feature', target, 'fc7')
        label_save_root = os.path.join(vs_root, 'feature', target, 'label', d + '.txt')
        anno_root = os.path.join(vs_root, 'anno')
        anno_list = os.path.join(vs_root, 'ImageSets', 'Main', d + '.txt')
        img_root = os.path.join(vs_root, 'JPEGImages')
        # if target == 'object':
        #     prepare_object_boxes_labels(anno_root, anno_list, box_save_path, label_save_path)
        # else:
        #     prepare_relation_boxes_labels(anno_root, anno_list, box_save_path, label_save_path)
        with open(box_save_path, 'rb') as box_file:
            boxes = pickle.load(box_file)
        with open(label_save_path, 'rb') as label_file:
            labels = pickle.load(label_file)
        wn2index_path = os.path.join(vs_root, 'feature', target, 'prepare', 'wn2index.json')
        with open(wn2index_path, 'rb') as wn2index_file:
            wn2index = json.load(wn2index_file)
        label2wn_path = os.path.join(vs_root, 'feature', target, 'prepare', 'label2wn.json')
        with open(label2wn_path, 'r') as label2wn_file:
            label2wn = json.load(label2wn_file)
        extract_fc7_features(net, boxes, labels, img_root, anno_list, fc7_save_root, label_save_root,
                              wn2index, label2wn)
        # generate_negative_data(label_save_root, len(wn2index.keys()))
    small_val_list_path = os.path.join(vs_root, 'feature', target, 'label', 'val_small.txt')
    val_list_path = os.path.join(vs_root, 'feature', target, 'label', 'val.txt')
    split_a_small_val(val_list_path, 1000, small_val_list_path)
