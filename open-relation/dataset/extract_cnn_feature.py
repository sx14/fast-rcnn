import os
import pickle
import vs_anno_2_dict as org
import caffe
import cv2
import numpy as np
import _init_paths
from fast_rcnn.test import im_detect


def prepare_boxes_labels(anno_root, box_save_path, label_save_path, image_sum):
    boxes = dict()
    labels = dict()
    counter = 0
    for anno_file_name in os.listdir(anno_root):
        if counter == image_sum:
            break
        counter += 1
        anno = org.vs_anno_2_dict(os.path.join(anno_root, anno_file_name))
        image_id = anno['filename']
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


def extract_fc7_features(net, boxes, labels, img_root, anno_root,  feature_root, label_root, image_sum):
    label_list = []
    anno_list = os.listdir(anno_root)
    anno_list = np.sort(anno_list, axis=0)
    for i in range(0, len(anno_list)):
        if i == image_sum:
            break
        image_id = anno_list[i]
        image_id = image_id.split('.')[0] + '.jpg'
        print('processing[%d/%d] %s' % (len(anno_list), (i + 1), image_id))
        if image_id not in boxes:
            continue
        box_list = np.array(boxes[image_id])
        curr_img_labels = labels[image_id]
        box_num = box_list.shape[0]
        if box_num == 0:
            continue
        img = cv2.imread(os.path.join(img_root, image_id))
        im_detect(net, img, box_list)
        fc7s = np.array(net.blobs['fc7'].data)
        feature_id = image_id.split('.')[0] + '.bin'
        feature_path = os.path.join(feature_root, feature_id)
        with open(feature_path, 'w') as feature_file:
            pickle.dump(fc7s, feature_file)
        for f in range(0, len(fc7s)):
            label = curr_img_labels[f]
            label_list.append(feature_id+' '+str(f)+' '+label+'\n')

    label_list_path = os.path.join(label_root, 'labels.txt')
    with open(label_list_path, 'w') as label_file:
        label_file.writelines(label_list)


def extract_fc7_features1(net, boxes, labels, img_root, anno_root,  feature_root, label_root, image_sum):
    label_list = []
    package_id = 0
    package_capacity = 1000
    package_fcs = []
    package_item_counter = 0
    anno_list = os.listdir(anno_root)
    anno_list = np.sort(anno_list, axis=0)
    for i in range(0, len(anno_list)):
        if i == image_sum:
            break
        image_id = anno_list[i]
        image_id = image_id.split('.')[0] + '.jpg'
        img = cv2.imread(os.path.join(img_root, image_id))
        if image_id not in boxes:
            continue
        box_list = np.array(boxes[image_id])
        curr_img_labels = labels[image_id]
        box_num = box_list.shape[0]
        if box_num == 0:
            continue
        im_detect(net, img, box_list)
        fc7s = np.array(net.blobs['fc7'].data)
        print('processing[%d/%d] %s' % (len(anno_list), (i + 1), image_id))
        for f in range(0, len(fc7s)):
            label = curr_img_labels[f]
            label_list.append(str(package_id)+'.bin/'+str(package_item_counter)+' '+label+'\n')
            fc = fc7s[f]
            if package_item_counter < package_capacity:
                package_fcs.append(fc)
            elif package_item_counter == package_capacity:
                # save package
                package_save_path = os.path.join(feature_root, str(package_id)+'.bin')
                with open(package_save_path, 'wb') as output_file:
                    pickle.dump(package_fcs, output_file)
                # new package
                package_id += 1
                package_fcs = []
                package_item_counter = 0
                package_fcs.append(fc)
            package_item_counter += 1
        if i == (len(anno_list) - 1) and len(package_fcs) > 0:
            # last image, save package
            package_save_path = os.path.join(feature_root, str(package_id) + '.bin')
            with open(package_save_path, 'wb') as output_file:
                pickle.dump(package_fcs, output_file)
    label_list_path = os.path.join(label_root, 'labels.txt')
    with open(label_list_path, 'w') as label_file:
        label_file.writelines(label_list)


if __name__ == '__main__':
    label_save_path = '/media/sunx/Data/dataset/visual genome/feature/prepare/labels.bin'
    box_save_path = '/media/sunx/Data/dataset/visual genome/feature/prepare/boxes.bin'
    fc7_save_root = '/media/sunx/Data/dataset/visual genome/feature/fc7'
    label_save_root = '/media/sunx/Data/dataset/visual genome/feature/label'
    anno_root = '/media/sunx/Data/dataset/visual genome/anno'
    img_root = '/media/sunx/Data/dataset/visual genome/JPEGImages'
    prepare_image_sum = 10000
    # prepare_boxes_labels(anno_root, box_save_path, label_save_path, prepare_image_sum)
    with open(box_save_path, 'rb') as box_file:
        boxes = pickle.load(box_file)
    with open(label_save_path, 'rb') as label_file:
        labels = pickle.load(label_file)
    prototxt = '/media/sunx/Data/linux-workspace/python-workspace/fast-rcnn-master/models/VGG16/test.prototxt'
    caffemodel = '/media/sunx/Data/linux-workspace/python-workspace/fast-rcnn-master/data/fast_rcnn_models/vgg16_fast_rcnn_iter_40000.caffemodel'
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    extract_fc7_features(net, boxes, labels, img_root, anno_root, fc7_save_root, label_save_root, prepare_image_sum)