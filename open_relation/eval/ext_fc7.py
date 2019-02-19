import sys
import pickle
import cv2
import numpy as np
import caffe
from lib.fast_rcnn.test import im_detect
from open_relation import global_config


# load cnn
prototxt = global_config.fast_prototxt_path
caffemodel = global_config.fast_caffemodel_path
caffe.set_mode_gpu()
caffe.set_device(0)
cnn = caffe.Net(prototxt, caffemodel, caffe.TEST)


def ext_cnn_feat(im, boxes):
    im_detect(cnn, im, boxes)
    fc7s = np.array(cnn.blobs['fc7'].data)
    return fc7s


if __name__ == '__init__':
    args = sys.argv
    img_path = args[1]
    temp_box_path = args[2]
    temp_fc7_path = args[3]

    boxes = pickle.load(open(temp_box_path))
    im = cv2.imread(img_path)
    fc7 = ext_cnn_feat(im, boxes)
    pickle.dump(fc7, open(temp_fc7_path))
