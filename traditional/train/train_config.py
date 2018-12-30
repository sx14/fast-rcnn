import os
from open_relation1 import vg_data_config, global_config
from open_relation1 import vrd_data_config

log_root = 'traditional/log'

vrd_hyper_params = {
    'class_num': 100,
    'visual_d': 4096,
    'epoch': 100,
    'batch_size': 256,
    'eval_freq': 1000,
    'print_freq': 10,
    'lr': 0.01,
    'log_root': os.path.join(global_config.project_root, log_root),
    'log_path': os.path.join(global_config.project_root, log_root, 'vrd_loss.txt'),
    'log_loss_path': os.path.join(global_config.project_root, log_root, 'vrd_loss.bin'),
    'log_acc_path': os.path.join(global_config.project_root, log_root, 'vrd_acc.bin'),
    'visual_feature_root': vrd_data_config.vrd_object_fc7_root,
    'list_root': vrd_data_config.vrd_object_label_root_t,
    'vrd2path_path': vrd_data_config.vrd_object_config_t['vrd2path_path'],
    'latest_weight_path': os.path.join(global_config.our_model_root, 'object', 't_vrd_weights.pkl'),
    'best_weight_path': os.path.join(global_config.our_model_root, 'object', 't_vrd_weights_best.pkl'),
}

hyper_params = {
    'vrd': vrd_hyper_params
}