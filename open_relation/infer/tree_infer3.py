# -*- coding: utf-8 -*-
import sys
import copy
import numpy as np


class TreeNode:
    def __init__(self, name, index):
        self._score = -1
        self._name = name
        self._index = index
        self._parents = []
        self._children = []

    def __str__(self):
        return '%s[%.2f]' % (self._name, self._score)

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


def construct_tree(label_hier, ranked_inds):
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

    for r, ind in enumerate(ranked_inds):
        rank = r + 1.0  # 1 based
        score = (len(ranked_inds) - rank) / len(ranked_inds)
        tnode = ind2node[ind]
        tnode.set_score(score)

    return ind2node


def top_down_search(root):
    node = root
    while len(node.children()) > 0:
        max_c_score = -1
        choice = None
        for c in node.children():
            if c.score() > max_c_score:
                max_c_score = c.score()
                choice = c
        if max_c_score < 0.0:
            break
        node = choice
    return node


def my_infer(label_hier, scores, target):
    ranked_inds = np.argsort(scores)[::-1]
    tnodes = construct_tree(label_hier, ranked_inds)
    choice = top_down_search(tnodes[label_hier.root().index()])
    return [[choice.index(), choice.score()], [choice.index(), choice.score()]]

