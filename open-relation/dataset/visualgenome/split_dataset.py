import os
import random
import data_config

if __name__ == '__main__':
    anno_root = os.path.join(data_config.VS_ROOT, 'anno')
    anno_sum = len(os.listdir(anno_root))

    val_capacity = data_config.DATASET_SPLIT_CONFIG['val']
    test_capacity = data_config.DATASET_SPLIT_CONFIG['test']
    train_capacity = anno_sum - val_capacity - test_capacity
    anno_list = os.listdir(anno_root)
    random.shuffle(anno_list)
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
        list_file_path = os.path.join(data_config.VS_ROOT, 'ImageSets', 'Main', d+'.txt')
        with open(list_file_path, 'w') as list_file:
            list_file.writelines(image_id_list)
