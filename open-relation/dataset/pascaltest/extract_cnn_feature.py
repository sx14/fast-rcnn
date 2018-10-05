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

def prepare_boxes_labels(anno_root, anno_list_path, box_save_path, label_save_path, image_sum):
    boxes = dict()
    labels = dict()
    label2wn = label_map.label2wn()
    with open(anno_list_path, 'r') as anno_list_file:
        anno_list = anno_list_file.read().splitlines()
    for i in range(0, len(anno_list)):
        anno_file_name = anno_list[i]
        if i == image_sum:
            break
        print('prepare processing[%d/%d] %s' % (len(anno_list), (i+1), anno_file_name))
        anno = org.pascal_anno_2_dict(os.path.join(anno_root, anno_file_name+'.xml'), label2wn)
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


def extract_fc7_features(net, boxes, labels, img_root, list_path, feature_root, label_path, image_sum, wn2index):
    label_list = []
    with open(list_path, 'r') as list_file:
        image_list = list_file.read().splitlines()
    for i in range(0, len(image_list)):
        if i == image_sum:
            break
        image_id = image_list[i]
        print('fc7 processing[%d/%d] %s' % (len(image_list), (i + 1), image_id))
        if image_id not in boxes:
            continue
        box_list = np.array(boxes[image_id])
        curr_img_labels = labels[image_id]
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
    wn_synsets_path = '/media/sunx/Data/linux-workspace/python-workspace/' \
                      'hierarchical-relationship/open-relation/wordnet-embedding/dataset/synset_names.json'
    datasets = ['train', 'val', 'test']
    for d in datasets:
        label_save_path = '/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007/feature/prepare/'+d+'_labels.bin'
        box_save_path = '/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007/feature/prepare/'+d+'_boxes.bin'
        fc7_save_root = '/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007/feature/fc7/'
        label_save_root = '/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007/feature/label/'+d+'.txt'
        anno_root = '/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007/Annotations'
        anno_list = '/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007/ImageSets/Main/'+d+'.txt'
        img_root = '/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007/JPEGImages'
        prepare_image_sum = 10000000
        prepare_boxes_labels(anno_root, anno_list, box_save_path, label_save_path, prepare_image_sum)
        with open(box_save_path, 'rb') as box_file:
            boxes = pickle.load(box_file)
        with open(label_save_path, 'rb') as label_file:
            labels = pickle.load(label_file)
        prototxt = '/media/sunx/Data/linux-workspace/python-workspace/hierarchical-relationship/models/VGG16/test.prototxt'
        caffemodel = '/media/sunx/Data/linux-workspace/python-workspace/hierarchical-relationship/data/fast_rcnn_models/vgg16_fast_rcnn_iter_40000.caffemodel'
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        wn2index = label_map.wn2index(wn_synsets_path)
        extract_fc7_features(net, boxes, labels, img_root, anno_list, fc7_save_root, label_save_root, prepare_image_sum, wn2index)
        generate_negative_data(label_save_root, len(wn2index.keys()))