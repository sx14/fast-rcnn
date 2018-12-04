import os
import copy
import json
from open_relation1.data_config import vg_config


def clean_anno1(dirty_anno_path, clean_anno_path):
    # use objects in relationships
    dirty_anno = json.load(open(dirty_anno_path, 'r'))
    clean_anno = copy.deepcopy(dirty_anno)
    cleaned_objs = []
    cleaned_rlts = []
    # clean relationships
    objIds = set()
    rlt_dict = dict()
    rlt_annos = dirty_anno['relationships']
    for rlt in rlt_annos:
        # clean objects
        objs = [rlt['subject'], rlt['object']]
        for o in objs:
            if o['object_id'] not in objIds and len(o['synsets']) > 0:
                objIds.add(o['object_id'])
                if 'names' in o:
                    o['name'] = o['names'][0]
                    o.pop('names')
                cleaned_objs.append(o)
        sid = rlt['subject']['object_id']
        oid = rlt['object']['object_id']
        pre_syn = rlt['synsets']
        if sid not in objIds:
            # subject has no wn synset
            continue
        if oid not in objIds:
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
    clean_anno['objects'] = cleaned_objs
    clean_anno['relationships'] = cleaned_rlts
    json.dump(clean_anno, open(clean_anno_path, 'w'), indent=4)


def clean_anno(dirty_anno_path, clean_anno_path):
    # use objects in objects.json
    dirty_anno = json.load(open(dirty_anno_path, 'r'))
    clean_anno = copy.deepcopy(dirty_anno)
    cleaned_objs = []
    cleaned_rlts = []
    # clean objects
    obj_annos = dirty_anno['objects']
    for obj in obj_annos:
        obj[u'name'] = obj[u'names'][0]
        obj.pop(u'names')
        o_syns = obj['synsets']
        if len(o_syns) > 0:
            # object has wn synset
            cleaned_objs.append(obj)
    clean_anno['objects'] = cleaned_objs
    # clean relationships
    rlt_dict = dict()
    rlt_annos = dirty_anno['relationships']
    for rlt in rlt_annos:
        sbj = rlt['subject']
        if u'names' in sbj:
            sbj[u'name'] = sbj['names'][0]
            sbj.pop('names')
            rlt[u'subject'] = sbj
        obj = rlt['object']
        if u'names' in obj:
            obj[u'name'] = obj['names'][0]
            obj.pop('names')
            rlt[u'object'] = obj
        sid = sbj['object_id']
        oid = obj['object_id']
        s_syns = sbj['synsets']
        o_syns = obj['synsets']
        p_syns = rlt['synsets']
        if len(s_syns) == 0:
            # subject has no wn synset
            continue
        if len(o_syns) == 0:
            # object has no wn synset
            continue
        if len(p_syns) == 0:
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
                # not redundant
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
        print('processing wash_anno [%d/%d]' % (anno_sum, i+1))
        dirty_anno_path = os.path.join(dirty_anno_root, anno_list[i])
        clean_anno_path = os.path.join(clean_anno_root, anno_list[i])
        clean_anno(dirty_anno_path, clean_anno_path)


