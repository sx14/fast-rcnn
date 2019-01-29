from open_relation.dataset.vrd.process import ext_pre_cnn_feat, ext_obj_cnn_feat

if __name__ == '__main__':
    # split_anno_pkg()
    # convert_anno()
    ext_obj_cnn_feat.ext_cnn_feat()
    ext_pre_cnn_feat.ext_cnn_feat()
