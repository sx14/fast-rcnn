# -*- coding: utf-8 -*-
import sys
import copy
from math import exp
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.functional import softmax

class TreeNode:
    def __init__(self, name, index):
        self._score = -1
        self._name = name
        self._index = index
        self._parents = []
        self._children = []

    def __str__(self):
        return '%s[%.2f]' % (self._name, self._score)

    def depth(self):
        min_p_depth = 0
        for p in self._parents:
            min_p_depth = max(min_p_depth, p.depth())
        return min_p_depth + 1

    def add_children(self, child):
        self._children.append(child)

    def children(self):
        return self._children

    def append_parent(self, parent):
        self._parents.append(parent)

    def set_score(self, score):
        self._score = score

    def score(self):
        return self._score

    def index(self):
        return self._index

    def name(self):
        return self._name


def construct_tree(label_hier, scores):
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

    ranked_inds = np.argsort(scores)[::-1]
    for r, ind in enumerate(ranked_inds):
        # rank = r + 1.0  # 1 based
        # score = (len(ranked_inds) - rank) / len(ranked_inds)
        score = scores[ind]
        tnode = ind2node[ind]
        tnode.set_score(score)

    return ind2node


def top_down_search(root, threshold):
    node = root
    while len(node.children()) > 0:
        c_scores = []
        for c in node.children():
            c_scores.append(c.score())
        c_scores_v = Variable(Tensor(c_scores))
        c_scores_s = softmax(c_scores_v, 0)
        pred_c_ind = torch.argmax(c_scores_s)
        pred_c_scr = c_scores_s[pred_c_ind]
        threshold = min(exp(node.depth() - 10), 0.9)
        if pred_c_scr < threshold:
            break
        node = node.children()[pred_c_ind]
        node.set_score(pred_c_scr.data.numpy().tolist())
    return node


def my_infer(label_hier, scores, target):
    tnodes = construct_tree(label_hier, scores)
    choice = top_down_search(tnodes[label_hier.root().index()], 0.4)
    return [[choice.index(), choice.score()], [choice.index(), choice.score()]]

