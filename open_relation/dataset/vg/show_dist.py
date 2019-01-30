import os
import json
import matplotlib.pyplot as plt
from open_relation.dataset.dataset_config import DatasetConfig


vg_config = DatasetConfig('vg')


# counter
obj_counter = dict()
pre_counter = dict()

# counting
clean_anno_root = vg_config.data_config['clean_anno_root']
anno_list = os.listdir(clean_anno_root)
anno_num = len(anno_list)
for i, anno_name in enumerate(anno_list):
    if i == 100:
        break
    print('counting [%d / %d]' % (anno_num, i+1))
    anno_path = os.path.join(clean_anno_root, anno_name)
    anno = json.load(open(anno_path, 'r'))
    objs = anno['objects']
    for obj in objs:
        name = obj['name']
        if name in obj_counter:
            obj_counter[name] += 1
        else:
            obj_counter[name] = 1
    relations = anno['relations']
    for rlt in relations:
        predicate = rlt['predicate']['name']
        if predicate in pre_counter:
            pre_counter[predicate] += 1
        else:
            pre_counter[predicate] = 1


sorted_obj_count = sorted(obj_counter.items(), key=lambda a: a[1])
sorted_obj_count.reverse()
obj_counts = [item[1] for item in sorted_obj_count]
for name, c in sorted_obj_count:
    print('%s: %d' % (name, c))

plt.plot(range(len(obj_counts)), obj_counts)
plt.title('distribution')
plt.xlabel('object')
plt.ylabel('count')
plt.show()


sorted_pre_count = sorted(pre_counter.items(), key=lambda a: a[1])
sorted_pre_count.reverse()
pre_counts = [item[1] for item in sorted_pre_count]
for name, c in sorted_pre_count:
    print('%s: %d' % (name, c))

plt.plot(range(len(pre_counts)), pre_counts)
plt.title('distribution')
plt.xlabel('predicate')
plt.ylabel('count')
plt.show()
