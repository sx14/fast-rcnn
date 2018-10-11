import torch
from torch import nn


class HypernymVisual(nn.Module):
    def __init__(self, visual_feature_dimension, embedding_dimension):
        super(HypernymVisual, self).__init__()
        hidden_layer_unit_num = 1500
        self.hidden = nn.Linear(visual_feature_dimension, hidden_layer_unit_num)
        self.embedding = nn.Linear(hidden_layer_unit_num, embedding_dimension)
        self.activate = nn.ReLU()

    def forward(self, vf, wf):
        hidden_out = self.hidden.forward(vf)
        vf_embedding = self.embedding.forward(hidden_out)
        sub = wf - vf_embedding
        act = self.activate.forward(sub)
        act_pow = act * act
        e = act_pow.sum(1)
        return e.view(len(e.data), 1)
