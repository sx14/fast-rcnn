import torch
from torch import nn


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
        p_e = self.cos.forward(vf_embeddings, p_wfs)
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
        score_stack = torch.autograd.Variable(torch.zeros(v_length, score_vec_len))
        for v in range(0, len(vf_embeddings)):
            n_scores = self.partial_order_similarity.forward(n_lfs, vf_embeddings[v])
            scores = torch.zeros(1+len(n_scores))
            scores[0] = p_scores[v]
            scores[1:] = n_scores
            score_stack[v] = scores
        return score_stack


# if __name__ == '__main__':
    # s = PartialOrderSimilarity(2)
    # a = np.array([[0,0,0,0], [0,0,0,0]])
    # at = torch.from_numpy(np.array(a)).float()
    # b = np.array([[1,2,0,0], [1,2,0,0]])
    # bt = torch.from_numpy(np.array(b)).float()
    # sims = s.forward(bt, at)
    # vfs = torch.from_numpy(np.array([[1,2,0,0], [1,2,0,0]])).float()
    # plfs = torch.from_numpy(np.array([[0,0,0,0], [0,0,0,0]])).float()
    # nlfs = torch.from_numpy(np.array([[0,0,1,2], [0,0,2,3], [0,0,3,4]])).float()
    # n = HypernymVisual_acc(4, 4)
    # scores = n.forward(vfs, plfs, nlfs)
