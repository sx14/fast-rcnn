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
        self.embedding = nn.Linear(visual_feature_dimension, embedding_dimension)
        self.activate = nn.ReLU()
        self.partial_order_similarity = PartialOrderSimilarity(2)

    def forward(self, vf, p_lfs, n_lfs):
        vf_embeddings = self.embedding.forward(vf)
        p_scores = self.partial_order_similarity.forward(p_lfs, vf_embeddings)
        score_vec_len = len(n_lfs) + 1
        v_length = len(vf)
        score_stack = torch.autograd.Variable(torch.zeros(v_length, score_vec_len)).cuda()
        for v in range(0, len(vf_embeddings)):
            n_scores = self.partial_order_similarity.forward(n_lfs, vf_embeddings[v])
            scores = torch.zeros(1+len(n_scores))
            scores[0] = p_scores[v]
            scores[1:] = n_scores
            score_stack[v] = scores
        return score_stack


# if __name__ == '__main__':
#     s = PartialOrderSimilarity(2)
#     a = np.array([[0,0,0,0], [0,0,0,0]])
#     at = torch.from_numpy(np.array(a)).float()
#     b = np.array([[1,2,0,0], [1,2,0,0]])
#     bt = torch.from_numpy(np.array(b)).float()
#     sims = s.forward(at, bt)
#     m=1
#     vfs = torch.from_numpy(np.array([[1,2,0,0], [1,2,0,0]])).float()
#     plfs = torch.from_numpy(np.array([[5,6,0,0], [1,4,0,0]])).float()
#     nlfs = torch.from_numpy(np.array([[0,0,1,2], [0,0,2,3], [0,0,3,4]])).float()
#     n = HypernymVisual_acc(4, 4)
#     scores = n.forward(vfs, plfs, nlfs)
#     m = 1
