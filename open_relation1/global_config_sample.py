import os
vg_root = '/media/sunx/Data/dataset/visual genome'
project_root = '/media/sunx/Data/linux-workspace/python-workspace/hierarchical-relationship'

fast_prototxt_path = os.path.join(project_root, 'models', 'VGG16', 'test.prototxt')
fast_caffemodel_path = os.path.join(project_root, 'data', 'fast_rcnn_models', 'vgg16_fast_rcnn_iter_40000.caffemodel')
