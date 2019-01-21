# -*- coding: utf-8 -*-
import sys
import numpy as np
from open_relation1.dataset.vrd.label_hier.pre_hier import prenet


def cal_rank_scores(label_num):
    # rank scores [1 - 10]
    # s = a(x - b)^2 + c
    # if rank is 0, score is 10
    # b = num-1
    s_min = 1.0
    s_max = 10.0
    b = label_num - 1
    c = s_min
    a = (s_max - c) / b ** 2
    rank_scores = [0] * label_num
    for r in range(label_num):
        rank_scores[r] = a*(r-b)**2 + c
    return rank_scores

def cal_rank_scores1(n_item):
    s_max = 10
    ranks = np.arange(1, n_item+1).astype(np.float)

    s = (np.cos(ranks / n_item * np.pi) + 1) * (s_max * 1.0 / 2)
    return s

class TreeNode:
    def __init__(self, name, index):
        self._rank = -1
        self._name = name
        self._index = index
        self._parents = []
        self._children = []

    def __str__(self):
        return '%s[%d]' % (self._name, self._rank)

    def add_children(self, child):
        self._children.append(child)

    def children(self):
        return self._children

    def append_parent(self, parent):
        self._parents.append(parent)

    def set_rank(self, rank):
        self._rank = rank


def construct_tree(label_hier):
    name2node = dict()
    for label in label_hier.get_all_labels():
        hnode = label_hier.get_node_by_name(label)
        tnode = TreeNode(label, hnode.index())
        name2node[label] = tnode

    for label in label_hier.get_all_labels():
        hnode = label_hier.get_node_by_name(label)
        tnode = name2node[label]
        hypers = hnode.hypers()
        for hyper in hypers:
            pnode = name2node[hyper.name()]
            pnode.add_children(tnode)
            tnode.append_parent(pnode)

    return name2node



def my_infer(label_hier, scores, rank_scores):
    label2tnode = construct_tree(label_hier)
    # label_ind 2 rank
    ind2ranks = [0] * label_hier.label_sum()
    ranked_inds = np.argsort(scores).tolist()
    ranked_inds.reverse()  # descending
    for r, ind in enumerate(ranked_inds):
        rank = r+1  # 1 based
        ind2ranks[ind] = rank
        label = label_hier.get_node_by_index(ind).name()
        tnode = label2tnode[label]
        tnode.set_rank(rank)





    # cands = [[pred_ind, pred_rank], [cand_ind, cand_rank]]
    return None, None
