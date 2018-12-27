import os
import global_config

#global config
vrd_root = global_config.vrd_root
vrd_feature_root = os.path.join(vrd_root, 'feature')

vrd_config = {
    # dataset preprocess
    'img_root': os.path.join(vrd_root, 'JPEGImages'),
    'org_anno_root': os.path.join(vrd_root, 'json_dataset'),
    'dirty_anno_root': os.path.join(vrd_root, 'dirty_anno'),
    'clean_anno_root': os.path.join(vrd_root, 'anno')
}

# object part config
vrd_object_feature_root = os.path.join(vrd_feature_root, 'object')
vrd_object_fc7_root = os.path.join(vrd_object_feature_root, 'fc7')
vrd_object_label_root = os.path.join(vrd_object_feature_root, 'label')
vrd_object_feature_prepare_root = os.path.join(vrd_object_feature_root, 'prepare')
vrd_object_config = {
    'vrd_label_list': os.path.join(vrd_object_feature_prepare_root, 'object_labels.txt'),
    'vrd2wn_path': os.path.join(vrd_object_feature_prepare_root, 'vrd2wn.bin'),
    'label2index_path': os.path.join(vrd_object_feature_prepare_root, 'label2index.bin'),
    'vrd2path_path': os.path.join(vrd_object_feature_prepare_root, 'vrd2path.bin'),
    'index2label_path': os.path.join(vrd_object_feature_prepare_root, 'index2label.bin'),
    'label_vec_path': os.path.join(global_config.project_root, 'open_relation1', 'label_embedding', 'object', 'label_vec_vrd.h5'),
    'label_vec_path1': os.path.join(global_config.project_root, 'open_relation1', 'label_embedding', 'object', 'label_vec_vrd1.h5'),
}


vrd_pascal_format = {
    'JPEGImages': os.path.join(vrd_root, 'JPEGImages'),
    'ImageSets': os.path.join(vrd_root, 'ImageSets', 'Main'),
    'Annotations': os.path.join(vrd_root, 'Annotations')
}
