import torch
from torch import nn


class HypernymVisual(nn.Module):
    def __init__(self, visual_feature_dimension, embedding_dimension):
        super(HypernymVisual, self).__init__()
        self.embedding = nn.Linear(visual_feature_dimension, embedding_dimension)
        self.activate = nn.ReLU()

    def forward(self, vf, p_wfs, n_wfs):
        vf_embedding = self.embedding.forward(vf)
        p_sub = p_wfs - vf_embedding
        p_act = self.activate.forward(p_sub)
        p_act_pow = p_act * p_act
        p_e = p_act_pow.sum(1)

        n_sub = n_wfs - vf_embedding
        n_act = self.activate.forward(n_sub)
        n_act_pow = n_act * n_act
        n_e = n_act_pow.sum(1)
        return p_e.view(len(p_e.data), 1), n_e.view(len(n_e.data), 1)


class HypernymVisual3(nn.Module):
    def __init__(self, visual_feature_dimension, embedding_dimension):
        super(HypernymVisual3, self).__init__()
        hidden_layer_unit_num = 1500
        self.hidden = nn.Linear(visual_feature_dimension, hidden_layer_unit_num)
        self.embedding = nn.Linear(hidden_layer_unit_num, embedding_dimension)
        self.activate = nn.ReLU()

    def forward(self, vf, p_wfs, n_wfs):
        # preserve values of vf positive
        vf = self.activate(vf)
        hidden_out = self.hidden.forward(vf)
        vf_embedding = self.embedding.forward(hidden_out)
        p_sub = p_wfs - vf_embedding
        p_act = self.activate.forward(p_sub)
        p_act_pow = p_act * p_act
        p_e = p_act_pow.sum(1)

        n_sub = n_wfs - vf_embedding
        n_act = self.activate.forward(n_sub)
        n_act_pow = n_act * n_act
        n_e = n_act_pow.sum(1)
        return p_e.view(len(p_e.data), 1), n_e.view(len(n_e.data), 1)


class HypernymVisual1(nn.Module):
    def __init__(self, visual_feature_dimension, embedding_dimension):
        super(HypernymVisual1, self).__init__()
        hidden_layer_unit_num = 1500
        # self.hidden = nn.Linear(visual_feature_dimension, hidden_layer_unit_num)
        self.embedding = nn.Linear(hidden_layer_unit_num, embedding_dimension)
        self.activate = nn.ReLU()

    def forward(self, vf, wf):
        # preserve values of vf positive
        vf = self.activate(vf)
        # hidden_out = self.hidden.forward(vf)
        # vf_embedding = self.embedding.forward(hidden_out)
        vf_embedding = self.embedding.forward(vf)
        sub = wf - vf_embedding
        act = self.activate.forward(sub)
        act_pow = act * act
        e = act_pow.sum(1)
        return e.view(len(e.data), 1)





class HypernymVisual2(nn.Module):
    def __init__(self, visual_feature_dimension, embedding_dimension):
        super(HypernymVisual2, self).__init__()
        hidden_layer_unit_num = 5000
        self.hidden1 = nn.Linear(visual_feature_dimension, hidden_layer_unit_num)
        self.activate1 = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_layer_unit_num, hidden_layer_unit_num)
        self.activate2 = nn.ReLU()
        self.embedding = nn.Linear(hidden_layer_unit_num, embedding_dimension)
        self.activate = nn.ReLU()

    def forward(self, vf, wf):
        hidden_out1 = self.hidden1.forward(vf)
        hidden_out1 = self.activate1.forward(hidden_out1)
        hidden_out2 = self.hidden2.forward(hidden_out1)
        hidden_out2 = self.activate2.forward(hidden_out2)
        vf_embedding = self.embedding.forward(hidden_out2)
        sub = wf - vf_embedding
        act = self.activate.forward(sub)
        act_pow = act * act
        e = act_pow.sum(1)
        return e.view(len(e.data), 1)