import os
import copy
import json
from open_relation1.data_config import vg_config


def clean_anno(dirty_anno_path, clean_anno_path):
    dirty_anno = json.load(open(dirty_anno_path, 'r'))
    clean_anno = copy.deepcopy(dirty_anno)
    cleaned_objs = []
    cleaned_rlts = []
    # clean objects
    objId2name = dict()
    obj_annos = dirty_anno['objects']
    for obj in obj_annos:
        oid = obj['object_id']
        oname = obj['names'][0]
        o_syns = obj['synsets']
        if len(o_syns) > 0:
            # object has wn synset
            objId2name[oid] = oname
            cleaned_objs.append(obj)
    clean_anno['objects'] = cleaned_objs
    # clean relationships
    rlt_dict = dict()
    rlt_annos = dirty_anno['relationships']
    for rlt in rlt_annos:
        sid = rlt['subject']['object_id']
        oid = rlt['object']['object_id']
        pre_syn = rlt['synsets']
        if sid not in objId2name:
            # subject has no wn synset
            continue
        if oid not in objId2name:
            # object has no wn synset
            continue
        if len(pre_syn) == 0:
            # predicate has no wn synset
            continue
        predicate = rlt['predicate']
        if sid in rlt_dict:
            obj_rlts = rlt_dict[sid]
            if oid in obj_rlts:
                rlts = obj_rlts[oid]
            else:
                rlts = list()
            # s -> o
            if predicate not in rlts:
                rlts.append(predicate)
                cleaned_rlts.append(rlt)
            obj_rlts[oid] = rlts
        else:
            obj_rlts = dict()
            obj_rlts[oid] = [predicate]
            rlt_dict[sid] = obj_rlts
            cleaned_rlts.append(rlt)
    clean_anno['relationships'] = cleaned_rlts
    json.dump(clean_anno, open(clean_anno_path, 'w'), indent=4)


if __name__ == '__main__':
    dirty_anno_root = vg_config['dirty_anno_root']
    clean_anno_root = vg_config['clean_anno_root']
    anno_list = os.listdir(dirty_anno_root)
    anno_list = sorted(anno_list)
    anno_sum = len(anno_list)
    # anno_sum = 1000
    for i in range(0, anno_sum):
        print('processing [%d/%d]' % (anno_sum, i+1))
        dirty_anno_path = os.path.join(dirty_anno_root, anno_list[i])
        clean_anno_path = os.path.join(clean_anno_root, anno_list[i])
        clean_anno(dirty_anno_path, clean_anno_path)


