import numpy as np
import cv2
import os
import json
from matplotlib import pyplot as plt
from open_relation1.vrd_data_config import vrd_config, vrd_pascal_format


def show_boxes(im, dets, cls):
    """Draw detected bounding boxes."""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(0, len(dets)):
        bbox = dets[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2],
                          bbox[3], fill=False,
                          edgecolor='red', linewidth=1.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{}'.format(cls[i]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_objects(img_root, anno_root, img_name):
    img_path = os.path.join(img_root, img_name)
    im = cv2.imread(img_path)
    anno_path = os.path.join(anno_root, img_name.split('.')[0] + '.json')
    with open(anno_path, 'r') as anno_file:
        anno = json.load(anno_file)
    objects = anno['objects']
    cls = []
    boxes = []
    for o in objects:
        cls.append(o['name'])
        boxes.append([o['xmin'], o['ymin'], o['xmax']-o['xmin'], o['ymax']-o['ymin']])
    return im, cls, boxes


if __name__ == '__main__':
    img_root = vrd_config['img_root']
    anno_root = vrd_config['clean_anno_root']
    img_id = '1602315_961e6acf72_b'
    for img_name in os.listdir(img_root):
        im, cls, boxes = get_objects(img_root, anno_root, img_name)
        if u'post' in cls:
            show_boxes(im, boxes, cls)
