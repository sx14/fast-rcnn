import os
from open_relation1 import vg_data_config, global_config
from open_relation1 import vrd_data_config

log_root = 'open_relation1/log'
vg_hyper_params = {
    'visual_d': 4096,
    'embedding_d': 600,
    'epoch': 500,
    'batch_size': 128,
    'eval_freq': 1000,
    'print_freq': 10,
    'lr': 0.1,
    'log_root': os.path.join(global_config.project_root, log_root),
    'log_path': os.path.join(global_config.project_root, log_root, 'vg_loss.txt'),
    'log_loss_path': os.path.join(global_config.project_root, log_root, 'vg_loss.bin'),
    'log_acc_path': os.path.join(global_config.project_root, log_root, 'vg_acc.bin'),
    'visual_feature_root': vg_data_config.vg_object_fc7_root,
    'list_root': vg_data_config.vg_object_label_root,
    'label_vec_path': os.path.join(global_config.project_root, 'open_relation1', 'label_embedding', 'object', 'label_vec_vg.h5'),
    'label_vec_path1': os.path.join(global_config.project_root, 'open_relation1', 'label_embedding', 'object', 'label_vec_vg1.h5'),
    'vg2path_path': vg_data_config.vg_object_config['vg2path_path'],
    'latest_weight_path': os.path.join(global_config.our_model_root, 'object', 'vg_weights.pkl'),
    'best_weight_path': os.path.join(global_config.our_model_root, 'object', 'vg_weights_best.pkl'),
}

vrd_obj_hyper_params = {
    'visual_d': 4096,
    'embedding_d': 600,
    'epoch': 200,
    'batch_size': 256,
    'eval_freq': 1000,
    'print_freq': 10,
    'lr': 0.01,
    'log_root': os.path.join(global_config.project_root, log_root),
    'log_path': os.path.join(global_config.project_root, log_root, 'vrd_loss.txt'),
    'log_loss_path': os.path.join(global_config.project_root, log_root, 'vrd_loss.bin'),
    'log_acc_path': os.path.join(global_config.project_root, log_root, 'vrd_acc.bin'),
    'visual_feature_root': vrd_data_config.vrd_object_fc7_root,
    'list_root': vrd_data_config.vrd_object_label_root,
    'label_vec_path': os.path.join(global_config.project_root, 'open_relation1', 'label_embedding', 'object', 'label_vec_vrd.h5'),
    'label_vec_path1': os.path.join(global_config.project_root, 'open_relation1', 'label_embedding', 'object', 'label_vec_vrd1.h5'),
    'vrd2path_path': vrd_data_config.vrd_object_config['vrd2path_path'],
    'vrd2weight_path': vrd_data_config.vrd_object_config['vrd2weight_path'],
    'latest_weight_path': os.path.join(global_config.our_model_root, 'object', 'vrd_weights.pkl'),
    'best_weight_path': os.path.join(global_config.our_model_root, 'object', 'vrd_weights_best.pkl'),
    'eval_weight_path':os.path.join(global_config.our_model_root, 'best', 'vrd_obj_weights.pkl'),
}

vrd_pre_hyper_params = {
    'visual_d': 4096,
    'embedding_d': 600,
    'epoch': 200,
    'batch_size': 256,
    'eval_freq': 1000,
    'print_freq': 10,
    'lr': 0.01,
    'log_root': os.path.join(global_config.project_root, log_root),
    'log_path': os.path.join(global_config.project_root, log_root, 'vrd_loss.txt'),
    'log_loss_path': os.path.join(global_config.project_root, log_root, 'vrd_loss.bin'),
    'log_acc_path': os.path.join(global_config.project_root, log_root, 'vrd_acc.bin'),
    'visual_feature_root': vrd_data_config.vrd_predicate_fc7_root,
    'list_root': vrd_data_config.vrd_predicate_label_root,
    'label_vec_path': os.path.join(global_config.project_root, 'open_relation1', 'label_embedding', 'predicate', 'label_vec_vrd.h5'),
    'label_vec_path1': os.path.join(global_config.project_root, 'open_relation1', 'label_embedding', 'predicate', 'label_vec_vrd1.h5'),
    'vrd2path_path': vrd_data_config.vrd_predicate_config['raw2path_path'],
    'vrd2weight_path': vrd_data_config.vrd_predicate_config['raw2weight_path'],
    'latest_weight_path': os.path.join(global_config.our_model_root, 'predicate', 'vrd_weights.pkl'),
    'best_weight_path': os.path.join(global_config.our_model_root, 'predicate', 'vrd_weights_best.pkl'),
    'eval_weight_path':os.path.join(global_config.our_model_root, 'best', 'vrd_obj_weights.pkl'),
}

hyper_params = {
    'vg': vg_hyper_params,
    'vrd': vrd_pre_hyper_params
}