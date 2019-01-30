import os
import json
import matplotlib.pyplot as plt
from open_relation.dataset.dataset_config import DatasetConfig


vg_config = DatasetConfig('vg')


# counter
obj_counter = dict()
pre_counter = dict()
obj2wn = dict()
pre2wn = dict()

# counting
clean_anno_root = vg_config.data_config['dirty_anno_root']
anno_list = os.listdir(clean_anno_root)
anno_num = len(anno_list)
for i, anno_name in enumerate(anno_list):
    print('counting [%d/%d]' % (anno_num, i+1))
    anno_path = os.path.join(clean_anno_root, anno_name)
    anno = json.load(open(anno_path, 'r'))
    objs = anno['objects']
    for obj in objs:
        synsets = set(obj['synsets'])
        name = obj['name']
        if name in obj_counter:
            obj2wn[name] = obj2wn[name] | synsets
            obj_counter[name] += 1
        else:
            obj2wn[name] = synsets
            obj_counter[name] = 1
    relations = anno['relations']
    for rlt in relations:
        synsets = set(rlt['predicate']['synsets'])
        predicate = rlt['predicate']['name']
        if predicate in pre_counter:
            pre2wn[predicate] = pre2wn[predicate] | synsets
            pre_counter[predicate] += 1
        else:
            pre2wn[predicate] = synsets
            pre_counter[predicate] = 1


counters = {
    'object': (obj_counter, obj2wn, 1000),
    'predicate': (pre_counter, pre2wn, 500)
}


for target, (counter, raw2wn, top) in counters:

    label_list = []
    sorted_obj_count = sorted(obj_counter.items(), key=lambda a: a[1])
    sorted_obj_count.reverse()
    obj_counts = [item[1] for item in sorted_obj_count]
    for i, (name, c) in enumerate(sorted_obj_count):
        # retain top N
        if i < top:
            line = '%s|' % name
            syns = raw2wn[name]
            for syn in syns:
                line = line + syn + ' '
            label_list.append(line+'\n')
            print('%d %s: %d' % (i + 1, name, c))
        else:
            break

    # save label list
    label_list_path = os.path.join(vg_config.dataset_root, target+'_labels.txt')
    with open(label_list_path, 'w') as f:
        f.writelines(label_list)

    plt.plot(range(len(label_list)), obj_counts[:len(label_list)])
    plt.title('distribution')
    plt.xlabel('object')
    plt.ylabel('count')
    plt.show()