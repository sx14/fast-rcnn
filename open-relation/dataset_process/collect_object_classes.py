import os
import json


def object_class2wn_leaf(anno_root, output_path):
    label2wn_path = os.path.join(output_path, 'label2wn.json')
    wn2label_path = os.path.join(output_path, 'wn2label.json')
    label2wn = dict()
    wn2label = dict()
    anno_total = len(os.listdir(anno_root))
    counter = 0
    for anno_id in os.listdir(anno_root):
        counter = counter + 1
        print('processing[%d/%d] : %s\n' % (anno_total, counter, anno_id))
        anno_path = os.path.join(anno_root, anno_id)
        with open(anno_path, 'r') as anno_file:
            anno = json.load(anno_file)
            objects = anno['objects']['objects']
            for o in objects:
                o_names = o['names']
                o_synsets = o['synsets']
                for s in o_synsets:
                    if s not in wn2label.keys():
                        wn2label[s] = set(o_names)
                    else:
                        wn2label[s] = wn2label[s] | set(o_names)
                for n in o_names:
                    if n not in label2wn.keys():
                        label2wn[n] = set(o_synsets)
                    else:
                        label2wn[n] = label2wn[n] | set(o_synsets)
    for k in wn2label:
        wn2label[k] = list(wn2label[k])
    for k in label2wn:
        label2wn[k] = list(label2wn[k])
    with open(wn2label_path, 'w') as out:
        json.dump(wn2label, out, sort_keys=False, indent=4)
    with open(label2wn_path, 'w') as out:
        json.dump(label2wn, out, sort_keys=False, indent=4)


if __name__ == '__main__':
    anno_path = '/media/sunx/Data/dataset/visual genome/anno'
    output_path = '/media/sunx/Data/dataset/visual genome/my_output'
    object_class2wn_leaf(anno_path, output_path)