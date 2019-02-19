import os
import sys
import cv2
import numpy as np
os.environ['GLOG_minloglevel'] = '3'
import caffe
from lib.fast_rcnn.test import im_detect
from open_relation import global_config


# load cnn
os.environ['GLOG_minloglevel'] = '3'
prototxt = global_config.fast_prototxt_path
caffemodel = global_config.fast_caffemodel_path
caffe.set_mode_gpu()
caffe.set_device(0)
cnn = caffe.Net(prototxt, caffemodel, caffe.TEST)


def ext_cnn_feat(im, boxes):
    im_detect(cnn, im, boxes)
    fc7s = np.array(cnn.blobs['fc7'].data)
    return fc7s



args = sys.argv
img_path = args[1]
temp_box_name = args[2]
temp_fc7_name = args[3]

boxes = np.load(temp_box_name+'.npy')
im = cv2.imread(img_path)
fc7 = ext_cnn_feat(im, boxes)
np.save(temp_fc7_name, fc7)
