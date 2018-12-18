"""
step4: split the whole dataset as train, val and test.
next: extract_cnn_feature.py
"""
import os
import random
from open_relation1.vg_data_config import vg_config, vg_split, vg_pascal_format

if __name__ == '__main__':
    anno_root = vg_config['clean_anno_root']
    anno_sum = len(os.listdir(anno_root))

    val_capacity = vg_split['val']
    test_capacity = vg_split['test']
    train_capacity = anno_sum - val_capacity - test_capacity
    anno_list = os.listdir(anno_root)
    # random.shuffle(anno_list)
    dataset_list = {
        'train': anno_list[0:train_capacity],
        'val': anno_list[train_capacity:train_capacity+val_capacity],
        'test': anno_list[train_capacity+val_capacity:anno_sum]
    }
    for d in dataset_list:
        image_id_list = []
        ls = dataset_list[d]
        for l in ls:
            image_id_list.append(l.split('.')[0]+'\n')
        list_file_path = os.path.join(vg_pascal_format['ImageSets'], 'Main', d+'.txt')
        with open(list_file_path, 'w') as list_file:
            list_file.writelines(image_id_list)