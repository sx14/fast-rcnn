import os
import global_config
vg_root = global_config.vg_root
vg_config = {
    'img_root': os.path.join(vg_root, 'JPEGImages'),
    'org_anno_root': os.path.join(vg_root, 'org_anno'),
    'dirty_anno_root': os.path.join(vg_root, 'dirty_anno'),
    'clean_anno_root': os.path.join(vg_root, 'anno')
}