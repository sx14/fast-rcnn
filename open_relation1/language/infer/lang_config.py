lang_config = {
    'train': {
        'rlt_save_path': 'train_ext_rlts',
        'raw_rlt_path': 'train_raw_rlts'
    },
    'test': {
        'rlt_save_path': 'test_ext_rlts',
        'raw_rlt_path': 'test_raw_rlts',
    }
}

train_params = {
    'lr': 0.1,
    'epoch_num': 400,
    'batch_size': 1000,
    'embedding_dim': 600,
    'model_save_path': 'lan_weights.pkl',
    'best_model_path': 'lan_weights_best.pkl'
}