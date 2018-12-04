import numpy as np
import cv2
import os
import json
from matplotlib import pyplot as plt
from open_relation1.vg_data_config import vg_config


def show_proposals(im, dets, cls):
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


def get_objects(img_root, anno_root, img_id):
    img_path = os.path.join(img_root, img_id + '.jpg')
    im = cv2.imread(img_path)
    anno_path = os.path.join(anno_root, img_id + '.json')
    with open(anno_path, 'r') as anno_file:
        anno = json.load(anno_file)
    objects = anno['objects']
    cls = []
    boxes = []
    for o in objects:
        cls.append(o['name'])
        boxes.append([o['x'], o['y'], o['w'], o['h']])
    return im, cls, boxes


if __name__ == '__main__':
    img_root = vg_config['img_root']
    anno_root = vg_config['dirty_anno_root']
    img_id = '1'
    im, cls, boxes = get_objects(img_root, anno_root, img_id)
    show_proposals(im, boxes, cls)
