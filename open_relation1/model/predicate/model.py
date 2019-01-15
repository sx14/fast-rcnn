import os
import torch
import numpy as np
from torch import nn
from open_relation1.train.train_config import hyper_params

pre_config = hyper_params['vrd']['predicate']
obj_config = hyper_params['vrd']['object']


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
    def __init__(self, vfeature_d, hidden_d, embedding_d):
        super(HypernymVisual_acc, self).__init__()
        self.hidden = nn.Linear(vfeature_d, hidden_d)
        self.embedding = nn.Linear(hidden_d, embedding_d)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.partial_order_similarity = PartialOrderSimilarity(2)

    def forward(self, vfs, p_lfs, n_lfs):
        vf_hidden = self.hidden.forward(vfs)
        vf_embeddings = self.embedding.forward(vf_hidden)
        p_scores = self.partial_order_similarity.forward(p_lfs, vf_embeddings)
        score_vec_len = len(n_lfs) + 1
        v_length = len(vfs)
        score_stack = torch.autograd.Variable(torch.zeros(v_length, score_vec_len)).cuda()
        for v in range(0, len(vf_embeddings)):
            n_scores = self.partial_order_similarity.forward(n_lfs, vf_embeddings[v])
            scores = torch.zeros(1+len(n_scores))
            scores[0] = p_scores[v]
            scores[1:] = n_scores
            score_stack[v] = scores
        return score_stack

    def forward1(self, vfs, pls, nls, label_vecs):
        vfs = self.activate(vfs)
        vfs = self.dropout(vfs)
        vf_hidden = self.hidden(vfs)

        vf_hidden = self.activate(vf_hidden)
        vf_hidden = self.dropout(vf_hidden)
        vf_embeddings = self.embedding(vf_hidden)

        p_scores = self.partial_order_similarity.forward(label_vecs[pls], vf_embeddings)

        score_vec_len = len(nls[0]) + 1
        v_length = len(vfs)
        score_stack = torch.autograd.Variable(torch.zeros(v_length, score_vec_len)).cuda()
        for v in range(0, v_length):
            n_scores = self.partial_order_similarity.forward(label_vecs[nls[v]], vf_embeddings[v])
            scores = torch.zeros(1+len(n_scores))
            scores[0] = p_scores[v]
            scores[1:] = n_scores
            score_stack[v] = scores

        return score_stack



    def forward2(self, vf, lfs):
        vf = self.activate(vf)
        vf_hidden = self.hidden.forward(vf)
        vf_hidden = self.activate(vf_hidden)
        vf_embedding = self.embedding.forward(vf_hidden)
        scores = self.partial_order_similarity.forward(lfs, vf_embedding)
        return scores

    def forward3(self, vf):
        vf = self.activate(vf)
        vf_hidden = self.hidden.forward(vf)
        vf_hidden = self.activate(vf_hidden)
        vf_embedding = self.embedding.forward(vf_hidden)
        return vf_embedding

class PredicateVisual_acc(nn.Module):
    def __init__(self):
        super(PredicateVisual_acc, self).__init__()

        # implicit
        self.obj_embedding = HypernymVisual_acc(obj_config['visual_d'],
                                                obj_config['hidden_d'],
                                                obj_config['embedding_d'])

        # loading obj embedding weights
        obj_embedding_weight_path = obj_config['latest_weight_path']
        if os.path.isfile(obj_embedding_weight_path):
            obj_weights = torch.load(obj_embedding_weight_path)
            self.obj_embedding.load_state_dict(obj_weights)
            print('Loading object embedding weights success.')
        else:
            print('No object embedding weights !!!')
            exit(-1)

        # freeze obj embedding
        obj_params = self.obj_embedding.parameters()
        for p in obj_params:
            p.requires_grad = False

        # predicate embedding level 1
        self.pre_embedding = HypernymVisual_acc(pre_config['visual_d'],
                                                pre_config['hidden_d'],
                                                pre_config['embedding_d'])
        # explicit
        # predicate embedding level 2
        input_d = obj_config['embedding_d'] * 2 + pre_config['embedding_d']
        hidden_d = input_d
        embedding_d = pre_config['embedding_d']
        self.hidden = nn.Linear(input_d, hidden_d)
        self.embedding = nn.Linear(hidden_d, embedding_d)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.partial_order_similarity = PartialOrderSimilarity(2)

    def forward1(self, vfs, pls, nls, label_vecs):
        sbj_vfs = vfs[:, :obj_config['visual_d']]
        pre_vfs = vfs
        obj_vfs = vfs[:, -obj_config['visual_d']:]

        sbj_embedding = self.obj_embedding.forward3(sbj_vfs)
        obj_embedding = self.obj_embedding.forward3(obj_vfs)
        pre_embedding0 = self.pre_embedding.forward3(pre_vfs)

        vfs = torch.cat([sbj_embedding, pre_embedding0, obj_embedding], 1)

        vfs = self.activate(vfs)
        vfs = self.dropout(vfs)
        vf_hidden = self.hidden(vfs)

        vf_hidden = self.activate(vf_hidden)
        vf_hidden = self.dropout(vf_hidden)
        vf_embeddings = self.embedding(vf_hidden)

        p_scores = self.partial_order_similarity.forward(label_vecs[pls], vf_embeddings)

        score_vec_len = len(nls[0]) + 1
        v_length = len(vfs)
        score_stack = torch.autograd.Variable(torch.zeros(v_length, score_vec_len)).cuda()
        for v in range(0, v_length):
            n_scores = self.partial_order_similarity.forward(label_vecs[nls[v]], vf_embeddings[v])
            scores = torch.zeros(1+len(n_scores))
            scores[0] = p_scores[v]
            scores[1:] = n_scores
            score_stack[v] = scores

        return score_stack

    def forward2(self, vfs, lfs):
        sbj_vfs = vfs[:, :obj_config['visual_d']]
        pre_vfs = vfs
        obj_vfs = vfs[:, -obj_config['visual_d']:]

        sbj_embedding = self.obj_embedding.forward3(sbj_vfs)
        obj_embedding = self.obj_embedding.forward3(obj_vfs)
        pre_embedding0 = self.pre_embedding.forward3(pre_vfs)

        vfs = torch.cat([sbj_embedding, pre_embedding0, obj_embedding], 1)

        vf = self.activate(vfs)
        vf_hidden = self.hidden.forward(vf)
        vf_hidden = self.activate(vf_hidden)
        vf_embedding = self.embedding.forward(vf_hidden)
        scores = self.partial_order_similarity.forward(lfs, vf_embedding)
        return scores

