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
vg_object_fc7_root = os.path.join(vg_object_feature_root, 'fc7')
vg_object_label_root = os.path.join(vg_object_feature_root, 'label')
vg_object_feature_prepare_root = os.path.join(vg_object_feature_root, 'prepare')
vg_object_config = {
    'vg2wn_path': os.path.join(vg_object_feature_prepare_root, 'vg2wn.bin'),
    'label2index_path': os.path.join(vg_object_feature_prepare_root, 'label2index.bin'),
    'vg2path_path': os.path.join(vg_object_feature_prepare_root, 'vg2path.bin'),
    'index2label_path': os.path.join(vg_object_feature_prepare_root, 'index2label.bin'),
}

# relation part config
vg_relation_feature_root = os.path.join(vg_feature_root, 'relation')
vg_relation_config = {
    'vg2wn_path': os.path.join(vg_relation_feature_root, 'prepare', 'vg2wn.bin'),
    'label2index_path': os.path.join(vg_relation_feature_root, 'prepare', 'label2index.bin'),
    'vg2path_path': os.path.join(vg_relation_feature_root, 'prepare', 'vg2path.bin'),
}

vg_split = {
    'val': 1000,
    'test': 97077,
}

vg_pascal_format = {
    'JPEGImages': os.path.join(vg_root, 'JPEGImages'),
    'ImageSets': os.path.join(vg_root, 'ImageSets'),
    'Annotations': os.path.join(vg_root, 'Annotations')
}
