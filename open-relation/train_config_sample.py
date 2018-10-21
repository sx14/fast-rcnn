pascal_hyper_params = {
    'visual_d': 4096,
    'embedding_d': 300,
    'epoch': 10,
    'batch_size': 128,
    'eval_freq': 1000,
    'print_freq': 10,
    'lr': 0.0001,
    'log_path': 'log/pascal_training_log.txt',
    'dataset_root': "/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007",
    'visual_feature_root': "/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007/feature/fc7",
    'list_root': "/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007/feature/label",
    'word_vec_path': 'wordnet-embedding/object/word_vec_wn.h5',
    'label2path_path': '/media/sunx/Data/dataset/visual genome/feature/object/prepare/label2path_path.json',
    'latest_weight_path': 'model/pascal_weights.pkl',
    'best_weight_path': 'model/pascal_weights_best.pkl',
}

vs_hyper_params = {
    'visual_d': 4096,
    'embedding_d': 300,
    'epoch': 3,
    'batch_size': 128,
    'eval_freq': 100,
    'print_freq': 10,
    'lr': 0.0001,
    'log_root': 'log/',
    'log_path': 'log/vs_training',
    'log_loss_path': 'log/vs_loss.bin',
    'log_acc_path': 'log/vs_acc.bin',
    'visual_feature_root': "/media/sunx/Data/dataset/visual genome/feature/object/fc7",
    'list_root': "/media/sunx/Data/dataset/visual genome/feature/object/label",
    'word_vec_path': 'wordnet-embedding/object/word_vec_vs.h5',
    'label2path_path': '/media/sunx/Data/dataset/visual genome/feature/object/prepare/label2path_path.json',
    'latest_weight_path': 'model/vs_weights.pkl',
    'best_weight_path': 'model/vs_weights_best.pkl',
}

hyper_params = {
    'pascal': pascal_hyper_params,
    'visual genome': vs_hyper_params
}