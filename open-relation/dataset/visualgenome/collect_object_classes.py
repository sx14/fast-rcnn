import os
import json
import data_config


def object_class2wn_leaf(anno_root, output_path):
    label2wn_path = os.path.join(output_path, 'label2wn.json')
    wn2label_path = os.path.join(output_path, 'wn2label.json')
    label2wn = dict()
    wn2label = dict()
    anno_total = len(os.listdir(anno_root))
    counter = 0
    for anno_file_name in os.listdir(anno_root):
        counter = counter + 1
        print('processing[%d/%d] : %s' % (anno_total, counter, anno_file_name))
        anno_path = os.path.join(anno_root, anno_file_name)
        with open(anno_path, 'r') as anno_file:
            anno = json.load(anno_file)
            objects = anno['objects']
            for o in objects:
                o_names = o['names']
                o_synsets = o['synsets']
                for s in o_synsets:
                    if s not in wn2label.keys():
                        wn2label[s] = set(o_names)
                    else:
                        wn2label[s] = wn2label[s] | set(o_names)
                for n in o_names:
                    if n not in label2wn:
                        label2wn[n] = dict()
                        for s in o_synsets:
                            label2wn[n][s] = 1
                    else:
                        existed_syns = label2wn[n]
                        for s in o_synsets:
                            # time statistic
                            if s in existed_syns:
                                label2wn[n][s] += 1
                            else:
                                label2wn[n][s] = 1
    for k in wn2label:
        wn2label[k] = list(wn2label[k])
    for n in label2wn:
        syns = label2wn[n]
        max_times = 0
        max_time_syn = ''
        for s in syns:
            if syns[s] > max_times:
                max_times = syns[s]
                max_time_syn = s
        label2wn[n] = [max_time_syn]
    with open(wn2label_path, 'w') as out:
        json.dump(wn2label, out, sort_keys=False, indent=4)
    with open(label2wn_path, 'w') as out:
        json.dump(label2wn, out, sort_keys=False, indent=4)


if __name__ == '__main__':
    anno_root = os.path.join(data_config.VS_ROOT, 'anno')
    output_root = os.path.join(data_config.VS_ROOT, 'feature', 'prepare')
    object_class2wn_leaf(anno_root, output_root)
