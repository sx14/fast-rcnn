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
        self.embedding = nn.Linear(visual_feature_dimension, embedding_dimension)
        # hidden_layer_unit_num = 1500
        # self.hidden = nn.Linear(visual_feature_dimension, hidden_layer_unit_num)
        # self.embedding = nn.Linear(hidden_layer_unit_num, embedding_dimension)
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


class HypernymVisual_cos(nn.Module):
    def __init__(self, visual_feature_dimension, embedding_dimension):
        super(HypernymVisual_cos, self).__init__()
        self.norm = nn.BatchNorm1d(4096)
        self.embedding = nn.Linear(visual_feature_dimension, embedding_dimension)
        self.activate = nn.ReLU()
        self.cos = nn.CosineSimilarity()

    def forward(self, vf, p_wfs, n_wfs):
        vf = self.norm.forward(vf)
        vf_embeddings = self.embedding.forward(vf)
        vf_embeddings = self.activate.forward(vf_embeddings)
        p_e = self.cos.forward(vf, p_wfs)
        n_wf_num = len(n_wfs)
        p_e_stack = torch.autograd.Variable(torch.zeros(n_wf_num * len(p_e))).cuda()
        n_e_stack = torch.autograd.Variable(torch.zeros(n_wf_num * len(p_e))).cuda()
        for v in range(0, len(vf_embeddings)):
            vf_embedding_rep = vf_embeddings[v].repeat(n_wf_num)
            n_e = self.cos.forward(vf_embedding_rep, n_wfs)
            n_e_stack[v * n_wf_num:(v + 1) * n_wf_num] = n_e[:]
            p_e_stack[v * n_wf_num:(v + 1) * n_wf_num] = p_e[v]
        # expect: n_e < p_e
        return n_e_stack.view(len(p_e_stack.data), 1), p_e_stack.view(len(p_e_stack.data), 1)


class HypernymVisual_acc(nn.Module):
    def __init__(self, visual_feature_dimension, embedding_dimension):
        super(HypernymVisual_acc, self).__init__()
        self.norm = nn.BatchNorm1d(4096)
        self.embedding = nn.Linear(visual_feature_dimension, embedding_dimension)
        self.activate = nn.ReLU()


    def forward(self, vf, p_wfs, n_wfs):
        vf = self.norm.forward(vf)
        vf_embeddings = self.embedding.forward(vf)
        p_sub = p_wfs - vf_embeddings
        p_act = self.activate.forward(p_sub)
        p_act_pow = p_act * p_act
        p_e = p_act_pow.sum(1)  # size: minibatch_size
        n_wf_num = len(n_wfs)
        p_e_stack = torch.autograd.Variable(torch.zeros(n_wf_num * len(p_e))).cuda()
        n_e_stack = torch.autograd.Variable(torch.zeros(n_wf_num * len(p_e))).cuda()
        for v in range(0, len(vf_embeddings)):
            n_sub = n_wfs - vf_embeddings[v]
            n_act = self.activate.forward(n_sub)
            n_act_pow = n_act * n_act
            n_e = n_act_pow.sum(1)
            n_e_stack[v*n_wf_num:(v+1)*n_wf_num] = n_e[:]
            p_e_stack[v*n_wf_num:(v+1)*n_wf_num] = p_e[v]
        return p_e_stack.view(len(p_e_stack.data), 1), n_e_stack.view(len(p_e_stack.data), 1)


class HypernymVisual_acc2(nn.Module):
    def __init__(self, visual_feature_dimension, embedding_dimension):
        super(HypernymVisual_acc2, self).__init__()
        hidden_layer_unit_num = 2000
        self.norm = nn.BatchNorm1d(4096)
        self.hidden = nn.Linear(visual_feature_dimension, hidden_layer_unit_num)
        self.embedding = nn.Linear(hidden_layer_unit_num, embedding_dimension)
        self.activate = nn.ReLU()

    def forward(self, vf, p_wfs, n_wfs):
        vf = self.norm.forward(vf)
        vf_hidden = self.hidden.forward(vf)
        vf_embeddings = self.embedding.forward(vf_hidden)
        p_sub = p_wfs - vf_embeddings
        p_act = self.activate.forward(p_sub)
        p_act_pow = p_act * p_act
        p_e = p_act_pow.sum(1)  # size: minibatch_size
        n_wf_num = len(n_wfs)
        p_e_stack = torch.autograd.Variable(torch.zeros(n_wf_num * len(p_e))).cuda()
        n_e_stack = torch.autograd.Variable(torch.zeros(n_wf_num * len(p_e))).cuda()
        for v in range(0, len(vf_embeddings)):
            n_sub = n_wfs - vf_embeddings[v]
            n_act = self.activate.forward(n_sub)
            n_act_pow = n_act * n_act
            n_e = n_act_pow.sum(1)
            n_e_stack[v*n_wf_num:(v+1)*n_wf_num] = n_e[:]
            p_e_stack[v*n_wf_num:(v+1)*n_wf_num] = p_e[v]
        return p_e_stack.view(len(p_e_stack.data), 1), n_e_stack.view(len(p_e_stack.data), 1)