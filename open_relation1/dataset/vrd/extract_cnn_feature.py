"""
step5: extract CNN feature for image region using pretrained CNN
"""
import os
import json
import random
import pickle
import numpy as np
import caffe
import cv2
from nltk.corpus import wordnet as wn
from lib.fast_rcnn.test import im_detect
from open_relation1 import vrd_data_config
from open_relation1 import global_config


def prepare_object_boxes_and_labels(anno_root, anno_list_path, box_label_path):
    objs = dict()
    with open(anno_list_path, 'r') as anno_list_file:
        anno_list = anno_list_file.read().splitlines()
    for i in range(0, len(anno_list)):
        anno_file_id = anno_list[i]
        print('prepare processing[%d/%d] %s' % (len(anno_list), (i + 1), anno_file_id))
        anno_path = os.path.join(anno_root, anno_list[i]+'.json')
        anno = json.load(open(anno_path, 'r'))
        image_id = anno_list[i]
        anno_objects = anno['objects']
        obj_info = []
        for o in anno_objects:
            xmin = int(o['xmin'])
            ymin = int(o['ymin'])
            xmax = int(o['xmax'])
            ymax = int(o['ymax'])
            obj_info.append([xmin, ymin, xmax, ymax, o['name']])
        objs[image_id] = obj_info
    with open(box_label_path, 'wb') as box_label_file:
        pickle.dump(objs, box_label_file)


def extract_fc7_features(net, img_box_label, img_root, list_path, feature_root, label_list_path, label2index, vrd2wn, vrd2path):
    if os.path.exists(label_list_path):
        os.remove(label_list_path)
    label_list = []

    # loading image file list of current dataset
    with open(list_path, 'r') as list_file:
        image_list = list_file.read().splitlines()

    # for each image file
    for i in range(0, len(image_list)):
        image_id = image_list[i]
        print('fc7 processing[%d/%d] %s' % (len(image_list), (i + 1), image_id))
        if image_id not in img_box_label:
            continue
        # get boxes
        curr_img_boxes = np.array(img_box_label[image_id])
        box_num = curr_img_boxes.shape[0]
        if box_num == 0:
            continue
        feature_id = image_id + '.bin'
        feature_path = os.path.join(feature_root, feature_id)
        if not os.path.exists(feature_path):
            # extract fc7
            img = cv2.imread(os.path.join(img_root, image_id+'.jpg'))
            im_detect(net, img, curr_img_boxes[:, :4])
            fc7s = np.array(net.blobs['fc7'].data)
            with open(feature_path, 'w') as feature_file:
                pickle.dump(fc7s, feature_file)
        for box_id in range(0, len(curr_img_boxes)):
            vrd_label = curr_img_boxes[box_id, 4]
            vrd_label_index = label2index[vrd_label]
            # img_id.bin offset label_index vrd_label_index
            label_list.append(feature_id+' '+str(box_id)+' '+str(vrd_label_index)+' '+str(vrd_label_index)+'\n')

            wn_leaf_label = vrd2wn[vrd_label]
            wn_leaf_index = label2index[wn_leaf_label]
            label_list.append(feature_id+' '+str(box_id)+' '+str(wn_leaf_index)+' '+str(vrd_label_index)+'\n')

            label_indexes = vrd2path[vrd_label_index]
            label_indexes = random.sample(label_indexes, int(len(label_indexes) / 3))
            for label_index in label_indexes:
                label_list.append(feature_id + ' ' + str(box_id) + ' ' + str(label_index) + ' ' + str(vrd_label_index) + '\n')
        if (i+1) % 10000 == 0 or (i+1) == len(image_list):
            with open(label_list_path, 'a') as label_file:
                label_file.writelines(label_list)
            del label_list
            label_list = []
    if len(label_list) > 0:
        with open(label_list_path, 'a') as label_file:
            label_file.writelines(label_list)


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
    # load cnn
    prototxt = global_config.fast_prototxt_path
    caffemodel = global_config.fast_caffemodel_path
    datasets = ['train', 'test']
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    # prepare
    target = 'object'
    # target = 'relation'
    if target == 'object':
        label2index_path = vrd_data_config.vrd_object_config['label2index_path']
        vrd2wn_path = vrd_data_config.vrd_object_config['vrd2wn_path']
        vrd2path_path = vrd_data_config.vrd_object_config['vrd2path_path']
        feature_root = vrd_data_config.vrd_object_feature_root
        fc7_save_root = vrd_data_config.vrd_object_fc7_root
        label_save_root = vrd_data_config.vrd_object_label_root
    else:
        label2index_path = ''
        vrd2wn_path = ''
        feature_root = ''
        fc7_save_root = ''

    # extracting feature
    anno_root = vrd_data_config.vrd_config['clean_anno_root']
    img_root = os.path.join(vrd_data_config.vrd_pascal_format['JPEGImages'])
    for d in datasets:
        # prepare labels and boxes
        label_save_path = os.path.join(label_save_root, d + '.txt')
        anno_list = os.path.join(vrd_data_config.vrd_pascal_format['ImageSets'], d + '.txt')
        box_label_path = os.path.join(feature_root, 'prepare', d + '_box_label.bin')
        prepare_object_boxes_and_labels(anno_root, anno_list, box_label_path)

        # extract cnn feature
        box_label = pickle.load(open(box_label_path, 'rb'))
        label2index = pickle.load(open(label2index_path, 'rb'))
        vrd2wn = pickle.load(open(vrd2wn_path, 'rb'))
        vrd2path = pickle.load(open(vrd2path_path, 'rb'))
        extract_fc7_features(net, box_label, img_root, anno_list, fc7_save_root, label_save_path, label2index, vrd2wn, vrd2path)

    # split a small val list for quick evaluation
    small_val_path = os.path.join(label_save_root, 'small_val.txt')
    val_path = os.path.join(label_save_root, 'test.txt')
    split_a_small_val(val_path, 1000, small_val_path)