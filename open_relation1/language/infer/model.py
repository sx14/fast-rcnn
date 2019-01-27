import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import cosine_similarity as cos
from torch.nn.functional import pairwise_distance as euc
from torch.nn.functional import margin_ranking_loss as rank_loss


def order_sim(hypers, hypos):
    sub = hypers - hypos
    act = nn.functional.relu(sub)
    partial_order_dis = act.norm(p=2, dim=1)
    partial_order_sim = -partial_order_dis
    return partial_order_sim


class RelationEmbedding(nn.Module):
    def __init__(self, input_len, output_len):
        super(RelationEmbedding, self).__init__()
        self.hidden = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(input_len, output_len))
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(output_len, output_len))

    def forward(self, sbj_vec, obj_vec):
        sbj_obj_vec = torch.cat((sbj_vec, obj_vec), 1)
        hidden = self.hidden(sbj_obj_vec)
        embedding = self.output(hidden)
        return embedding


def relation_embedding_loss(sbj_vec1, pre_vec1, obj_vec1, pre_emb1,
                            sbj_vec2, pre_vec2, obj_vec2, pre_emb2):
    emb_sim = order_sim(pre_emb1, pre_emb2)
    sbj_sim = order_sim(sbj_vec1, sbj_vec2)
    obj_sim = order_sim(obj_vec1, obj_vec2)
    pre_sim = order_sim(pre_vec1, pre_vec2)
    target = emb_sim / (sbj_sim + pre_sim + obj_sim)
    return torch.var(target)


def order_rank_loss(pos_sim, neg_sim):
    y = Variable(torch.ones(len(pos_sim))).cuda()
    # expect: positive sample get higher score, min margin = 1
    loss = rank_loss(pos_sim, neg_sim, y, margin=1)
    return loss


def order_rank_eval(pos_vecs, neg_vecs, gt_vecs):
    pos_sim = order_sim(gt_vecs, pos_vecs)
    neg_sim = order_sim(gt_vecs, neg_vecs)
    diff = pos_sim - neg_sim
    true_count = np.where(diff > 0)[0].shape[0] * 1.0
    acc = true_count / pos_sim.shape[0]
    return acc, pos_sim, neg_sim


def order_rank_test(pred_vecs, gt_label_vecs):
    ranks = []
    for pred_vec in pred_vecs:
        emb_sim = order_sim(gt_label_vecs, pred_vec)
        ranked_inds = np.argsort(emb_sim).tolist()
        ranked_inds.reverse()
        ranks.append(ranked_inds)
    return ranks