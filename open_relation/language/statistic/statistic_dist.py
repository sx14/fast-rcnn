import os
import json
import pickle
from copy import deepcopy
import numpy as np
from open_relation.dataset import dataset_config
from open_relation.dataset.vrd.label_hier.obj_hier import objnet
from open_relation.dataset.vrd.label_hier.pre_hier import prenet
from open_relation.infer.tree_infer2 import TreeNode


def construct_tree(label_hier):
    ind2node = dict()
    for label in label_hier.get_all_labels():
        hnode = label_hier.get_node_by_name(label)
        tnode = TreeNode(label, hnode.index())
        ind2node[hnode.index()] = tnode

    for label in label_hier.get_all_labels():
        hnode = label_hier.get_node_by_name(label)
        tnode = ind2node[hnode.index()]
        hypers = hnode.hypers()
        for hyper in hypers:
            pnode = ind2node[hyper.index()]
            pnode.add_children(tnode)
            tnode.append_parent(pnode)
    return ind2node


def fill_dist(s, o, p, raw_dist, obj_tree, pre_tree):
    count = raw_dist[s, o, p]
    raw_dist[s, o, p] = 0

    s_node = obj_tree[s]
    s_children = s_node.children()
    if len(s_children) > 0:
        for sc in s_children:
            count += fill_dist(sc.index(), o, p, raw_dist, obj_tree, pre_tree)
    else:
        o_node = obj_tree[o]
        o_children = o_node.children()
        if len(o_children) > 0:
            for oc in o_children:
                count += fill_dist(s, oc.index(), p, raw_dist, obj_tree, pre_tree)
        else:
            p_node = pre_tree[p]
            p_children = p_node.children()
            if len(p_children) > 0:
                for pc in p_children:
                    count += fill_dist(s, o, pc.index(), raw_dist, obj_tree, pre_tree)
    return count


anno_root = dataset_config.data_config['clean_anno_root']
anno_list_path = os.path.join(dataset_config.pascal_format['ImageSets'], 'train.txt')
with open(anno_list_path, 'r') as anno_list_file:
    anno_list = anno_list_file.read().splitlines()

raw_dist = np.zeros((objnet.label_sum(), objnet.label_sum(), prenet.label_sum()))
anno_num = len(anno_list)
# collect raw
for i in range(anno_num):
    print('processsing [%d/%d]' % (anno_num, i+1))
    anno_path = os.path.join(anno_root, anno_list[i]+'.json')
    anno = json.load(open(anno_path, 'r'))
    image_id = anno_list[i]
    anno_rlts = anno['relations']

    rlt_info_list = []
    for rlt in anno_rlts:
        anno_obj = rlt['object']
        anno_sbj = rlt['subject']
        anno_pre = rlt['predicate']
        obj_ind = objnet.get_node_by_name(anno_obj['name']).index()
        sbj_ind = objnet.get_node_by_name(anno_sbj['name']).index()
        pre_ind = prenet.get_node_by_name(anno_pre['name']).index()
        raw_dist[sbj_ind, obj_ind, pre_ind] += 1

s = raw_dist.max()
print('sum: %d' % raw_dist.sum())

dist = np.zeros((objnet.label_sum(), objnet.label_sum(), prenet.label_sum()))

obj_tree = construct_tree(objnet)
pre_tree = construct_tree(prenet)

rlt_class_num = (objnet.label_sum()-1) * (objnet.label_sum()-1) * (prenet.label_sum()-1)
proc_count = 0.0
for s in range(1, objnet.label_sum()):
    for o in range(1, objnet.label_sum()):
        for p in range(1, prenet.label_sum()):
            proc_count += 1
            s_node = objnet.get_node_by_index(s)
            o_node = objnet.get_node_by_index(o)
            p_node = prenet.get_node_by_index(p)
            raw_dist_copy = deepcopy(raw_dist)
            dist[s, o, p] = fill_dist(s, o, p, raw_dist_copy, obj_tree, pre_tree)
            per = proc_count / rlt_class_num
            if proc_count % 1000 == 0:
                print('processing [%d/%d]' % (rlt_class_num, proc_count))
            print('<%s, %s, %s> = %d' % (s_node.name(), p_node.name(), o_node.name(), dist[s, o, p]))


pickle.dump(dist, open('dist.bin', 'wb'))


for s in range(objnet.label_sum()):
    for o in range(objnet.label_sum()):
        rlt_num = dist[s, o, 1]
        dist[s, o, :] = dist[s, o, :] / rlt_num

pickle.dump(dist, open('cond_probs.bin', 'wb'))



