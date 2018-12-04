import os
import global_config

#global config
vg_root = global_config.vg_root
vg_feature_root = os.path.join(vg_root, 'feature')
vg_config = {
    # dataset preprocess
    'img_root': os.path.join(vg_root, 'JPEGImages'),
    'org_anno_root': os.path.join(vg_root, 'org_anno_2'),
    'dirty_anno_root': os.path.join(vg_root, 'dirty_anno'),
    'clean_anno_root': os.path.join(vg_root, 'anno')
}

# object part config
vg_object_feature_root = os.path.join(vg_feature_root, 'object')
vg_object_config = {
    'vg2wn_path': os.path.join(vg_object_feature_root, 'prepare', 'vg2wn.bin'),
    'label2index_path': os.path.join(vg_object_feature_root, 'prepare', 'label2index.bin'),
    'vg2path_path': os.path.join(vg_object_feature_root, 'prepare', 'vg2path.bin'),
}

# relation part config
vg_relation_feature_root = os.path.join(vg_feature_root, 'relation')
vg_relation_config = {
    'vg2wn_path': os.path.join(vg_object_feature_root, 'prepare', 'vg2wn.bin'),
    'label2index_path': os.path.join(vg_object_feature_root, 'prepare', 'label2index.bin'),
    'vg2path_path': os.path.join(vg_object_feature_root, 'prepare', 'vg2path.bin'),
}

vg_split = {
    'val': 1000,
    'test': 10000,
}

vg_pascal_format = {
    'ImageSets': 'ImageSets',
    'Annotations': 'Annotations'
}