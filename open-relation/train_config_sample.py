pascal_hyper_params = {
    'visual_d': 4096,
    'embedding_d': 300,
    'epoch': 30,
    'batch_size': 200,
    'eval_freq': 100,
    'visual_feature_root': "/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007/feature/fc7",
    'list_root': "/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007/feature/label",
    'word_vec_path': 'wordnet-embedding/dataset/object/word_vec_wn.h5',
    'latest_weight_path': 'model/pascal_weights.pkl',
    'best_weight_path': 'model/pascal_weights_best.pkl',
}

vs_hyper_params = {
    'visual_d': 4096,
    'embedding_d': 300,
    'epoch': 30,
    'batch_size': 200,
    'eval_freq': 100,
    'visual_feature_root': "/media/sunx/Data/dataset/visual genome/feature/object/fc7",
    'list_root': "/media/sunx/Data/dataset/visual genome/feature/object/label",
    'word_vec_path': 'wordnet-embedding/dataset/object/word_vec_vs.h5',
    'latest_weight_path': 'model/vs_weights.pkl',
    'best_weight_path': 'model/vs_weights_best.pkl',
}

hyper_params = {
    'pascal': pascal_hyper_params,
    'visual genome': vs_hyper_params
}