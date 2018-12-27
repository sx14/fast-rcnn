import os
import json
import numpy as np
from open_relation1.vrd_data_config import vrd_config, vrd_object_config

org_anno_root = vrd_config['dirty_anno_root']
dst_anno_root = vrd_config['clean_anno_root']

# load vrd label index 2 label
obj_label_list_path = vrd_object_config['vrd_label_list']
with open(obj_label_list_path, 'r') as f:
    index2label = f.readlines()

anno_list = os.listdir(org_anno_root)
for i, anno_name in enumerate(anno_list):
    print('processing [%d/%d]' % (len(anno_list), i+1))

    org_anno_path = os.path.join(org_anno_root, anno_name)
    org_anno = json.load(open(org_anno_path, 'r'))

    label_boxes = []
    # objects, relationships
    for rlt in org_anno:
        objs = [rlt['object'], rlt['subject']]
        for obj in objs:
            # left top, right bottom
            # ymin, ymax, xmin, xmax, category
            label_box = obj['bbox']
            label_box.append(obj['category'])
            label_boxes.append(label_box)

    # remove redundant objects
    objs = []
    if len(label_boxes) > 0:
        label_boxes = np.array(label_boxes)
        unique_label_boxes = np.unique(label_boxes, axis=0)
        for label_box in unique_label_boxes:
            obj = dict()
            obj['name'] = index2label[int(label_box[4])].strip()
            obj['ymin'] = int(label_box[0])
            obj['ymax'] = int(label_box[1])
            obj['xmin'] = int(label_box[2])
            obj['xmax'] = int(label_box[3])
            objs.append(obj)

    new_anno = dict()
    new_anno['objects'] = objs
    # TODO fill relationships
    save_path = os.path.join(dst_anno_root, anno_name)
    json.dump(new_anno, open(save_path, 'w'))
