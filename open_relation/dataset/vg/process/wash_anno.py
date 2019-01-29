# -*- coding: utf-8 -*-
"""
step2 retain the objects and relationships with WordNet annotation
next: index_labels.py
"""
import os
import copy
import json
import nltk
from open_relation.dataset.dataset_config import DatasetConfig


legal_pos_tags = {
    # 名词
    'object': {'NN'},
    # 介词，比较级，虚词，to，VB
    'predicate': {'IN', 'JJR', 'RP', 'TO'}
}


def regularize_label(label, type):
    # type : object, predicate
    pos_tags = legal_pos_tags[type]
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(label)
    legal_tokens = []
    for token in tokens:
        raw_token = lemmatizer.lemmatize(token)
        if raw_token in pos_tags:
            legal_tokens.append(raw_token)
    return ' '.join(legal_tokens)


def rlt_reformat(rlt_anno):

    def obj_reformat(obj_anno):
        obj = dict()
        obj['name'] = obj_anno['name']
        obj['ymin'] = int(obj_anno['y'])
        obj['ymax'] = int(obj_anno['y']+int(obj_anno['h']))
        obj['xmin'] = int(obj_anno['x'])
        obj['xmax'] = int(obj_anno['x']+int(obj_anno['w']))
        obj['synsets'] = obj_anno['synsets']
        return obj

    sbj_anno = rlt_anno['subject']
    obj_anno = rlt_anno['object']
    sbj = obj_reformat(sbj_anno)
    obj = obj_reformat(obj_anno)
    pre = dict()
    pre['name'] = rlt_anno['predicate']
    # predicate box is union of obj box and sbj box
    pre['ymin'] = min(obj['ymin'], sbj['ymin'])
    pre['ymax'] = max(obj['ymax'], sbj['ymax'])
    pre['xmin'] = min(obj['xmin'], sbj['xmin'])
    pre['xmax'] = max(obj['xmax'], sbj['xmax'])
    new_rlt = dict()
    new_rlt['object'] = obj
    new_rlt['subject'] = sbj
    new_rlt['predicate'] = pre
    return new_rlt




def clean_anno(dirty_anno_path, clean_anno_path):
    # use objects in relationships
    dirty_anno = json.load(open(dirty_anno_path, 'r'))
    clean_anno = dict()

    # extract unique objects from relationships
    id2obj = dict()
    id2rlt = dict()

    rlts = dirty_anno['relationships']
    for rlt in rlts:
        new_rlt = rlt_reformat(rlt)
        objs_have_synset = True
        objs = [new_rlt['subject'], new_rlt['object']]
        for obj in objs:
            if len('synsets') > 0:
                # object must have wn synset
                reg_label = regularize_label(obj['name'], 'object')
                print('%s | %s' % (obj['name'], reg_label))
                obj['name'] = reg_label
                id2obj[obj['object_id']] = obj
            else:
                objs_have_synset = False
        if not objs_have_synset:
            reg_label = regularize_label(new_rlt['predicate']['name'], 'predicate')
            print('%s | %s' % (new_rlt['predicate']['name'], reg_label))
            new_rlt['predicate']['name'] = reg_label
            id2rlt[new_rlt['relationship_id']] = new_rlt

    clean_anno['objects'] = id2obj.values()
    clean_anno['relations'] = id2rlt.values()
    json.dump(clean_anno, open(clean_anno_path, 'w'), indent=4)


if __name__ == '__main__':
    vg_config = DatasetConfig('vg')
    dirty_anno_root = vg_config.data_config['dirty_anno_root']
    clean_anno_root = vg_config.data_config['clean_anno_root']
    anno_list = os.listdir(dirty_anno_root)
    anno_list = sorted(anno_list)
    anno_sum = len(anno_list)
    for i in range(0, anno_sum):
        print('processing wash_anno [%d/%d]' % (anno_sum, i+1))
        dirty_anno_path = os.path.join(dirty_anno_root, anno_list[i])
        clean_anno_path = os.path.join(clean_anno_root, anno_list[i])
        clean_anno(dirty_anno_path, clean_anno_path)


