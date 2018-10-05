import torch
from torch import nn


class HypernymVisual(nn.Module):
    def __init__(self, visual_feature_dimension, embedding_dimension):
        super(HypernymVisual, self).__init__()
        self.embedding = nn.Linear(visual_feature_dimension, embedding_dimension)
        self.activate = nn.ReLU()

    def forward(self, vf, wf):
        vf_embedding = self.embedding.forward(vf)
        sub = wf - vf_embedding
        act = self.activate.forward(sub)
        act_pow = act * act
        e = act_pow.sum(1)
        return e.view(len(e.data), 1)
