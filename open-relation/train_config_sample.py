pascal_hyper_params = {
    'visual_d': 4096,
    'embedding_d': 300,
    'epoch': 30,
    'batch_size': 200,
    'eval_freq': 100,
    'visual_feature_root': "/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007/feature/fc7",
    'list_root': "/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007/feature/label",
    'word_vec_path': 'wordnet-embedding/object/word_vec_wn.h5',
    'weight_path': 'model/pascal_weights.pkl'
}

vs_hyper_params = {
    'visual_d': 4096,
    'embedding_d': 300,
    'epoch': 30,
    'batch_size': 200,
    'eval_freq': 100,
    'visual_feature_root': "/media/sunx/Data/dataset/visual genome/object/feature/fc7",
    'list_root': "/media/sunx/Data/dataset/visual genome/object/feature/label",
    'word_vec_path': 'wordnet-embedding/object/word_vec_vs.h5',
    'weight_path': 'model/vs_weights.pkl'
}

hyper_prams = {
    'pascal': pascal_hyper_params,
    'visual genome': vs_hyper_params
}