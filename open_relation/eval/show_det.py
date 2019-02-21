import os
import pickle
import cv2
import numpy as np
from open_relation.dataset.dataset_config import DatasetConfig
from open_relation.dataset.show_box import show_boxes
from open_relation.train.train_config import hyper_params


dataset = 'vrd'
dataset_config = DatasetConfig(dataset)
pre_config = hyper_params[dataset]['predicate']
obj_config = hyper_params[dataset]['object']
det_roidb_path = dataset_config.extra_config['object'].det_box_path
det_roidb = pickle.load(open(det_roidb_path))

img_root = dataset_config.data_config['img_root']
for img_id in det_roidb:
    img_path = os.path.join(img_root, img_id+'.jpg')
    im = cv2.imread(img_path)
    dets = det_roidb[img_id]
    dets_temp = np.copy(dets)
    dets_temp[:, 2] = dets[:, 2] - dets[:, 0]   # width
    dets_temp[:, 3] = dets[:, 3] - dets[:, 1]   # height
    confs = dets[:, 4]
    show_boxes(im, dets_temp[:, :4], confs)