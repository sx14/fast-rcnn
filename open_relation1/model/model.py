import torch
import numpy as np
from torch import nn


class PartialOrderSimilarity:
    def __init__(self, norm):
        self._norm = norm
        self.activate = nn.ReLU()

    def forward(self, hypers, hypos):
        sub = hypers - hypos
        act = self.activate.forward(sub)
        partial_order_dis = act.norm(p=self._norm, dim=1)
        partial_order_sim = -partial_order_dis
        return partial_order_sim


class HypernymVisual_acc(nn.Module):
    def __init__(self, visual_feature_dimension, embedding_dimension):
        super(HypernymVisual_acc, self).__init__()
        self.cls_score = nn.Linear(visual_feature_dimension, embedding_dimension)

    def forward(self, vfs):
        scores = self.cls_score.forward(vfs)
        return scores


