import os

VS_ROOT = '/media/sunx/Data/dataset/visual genome'
PASCAL_ROOT = '/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007'

vs_config = {
    'object_prepare_root': os.path.join(VS_ROOT, 'feature', 'object', 'prepare'),
    'hypernym_data_path': 'vs_dataset/wordnet_with_vs.h5',
    'synset_names_path': 'vs_dataset/synset_names_with_vs.json'
}

pascal_config = {
    'object_prepare_root': os.path.join(PASCAL_ROOT, 'feature', 'object', 'prepare'),
    'hypernym_data_path': 'pascal_dataset/wordnet_with_pascal.h5',
    'synset_names_path': 'pascal_dataset/synset_names_with_pascal.json'
}

config = {
    'vs': vs_config,
    'pascal': pascal_config
}