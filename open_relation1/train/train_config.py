import os
from open_relation1 import vg_data_config, global_config


log_root = 'open_relation1/log'
vg_hyper_params = {
    'visual_d': 4096,
    'embedding_d': 600,
    'epoch': 30,
    'batch_size': 128,
    'eval_freq': 4000,
    'print_freq': 10,
    'lr': 0.0001,
    'log_root': os.path.join(global_config.project_root, log_root),
    'log_path': os.path.join(global_config.project_root, log_root, 'vg_loss.txt'),
    'log_loss_path': os.path.join(global_config.project_root, log_root, 'vg_loss.bin'),
    'log_acc_path': os.path.join(global_config.project_root, log_root, 'vg_acc.bin'),
    'visual_feature_root': vg_data_config.vg_object_fc7_root,
    'list_root': vg_data_config.vg_object_label_root,
    'label_vec_path': os.path.join(global_config.project_root, 'open_relation1', 'label_embedding', 'object', 'label_vec_vg.h5'),
    'vg2path_path': vg_data_config.vg_object_config['vg2path_path'],
    'latest_weight_path': os.path.join(global_config.our_model_root, 'object', 'vg_weights.pkl'),
    'best_weight_path': os.path.join(global_config.our_model_root, 'object', 'vg_weights_best.pkl'),
}

hyper_params = {
    'vg': vg_hyper_params
}