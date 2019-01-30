from process.split_anno_pkg import split_anno_pkg
from process.reformat_anno import reformat_anno
from process.collect_labels import collect_labels
from process.filter_anno import filter_anno
from process.split_dataset import split_dataset
from process import ext_obj_cnn_feat, ext_pre_cnn_feat

if __name__ == '__main__':
    # split_anno_pkg()
    # clean_anno()
    # collect_labels()
    filter_anno()
    # split_dataset()
    # ext_obj_cnn_feat.ext_cnn_feat()
    # TODO: predicate part
    pass