import h5py
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


def order_sim(hypers, hypos):
    sub = hypers - hypos
    act = nn.functional.relu(sub)
    partial_order_dis = act.norm(p=2, dim=1)
    partial_order_sim = -partial_order_dis
    return partial_order_sim


def order_softmax_test(batch_scores, pos_neg_inds):
    loss_scores = Variable(torch.zeros(len(batch_scores), len(pos_neg_inds[0]))).float().cuda()
    for i in range(len(batch_scores)):
        loss_scores[i] = batch_scores[i, pos_neg_inds[i]]
    y = Variable(torch.zeros(len(batch_scores))).long().cuda()
    acc = 0.0
    for scores in loss_scores:
        p_score = scores[0]
        n_score_max = torch.max(scores[1:])
        if p_score > n_score_max:
            acc += 1
    acc = acc / len(batch_scores)
    return acc, loss_scores, y


class HypernymVisual(nn.Module):
    def __init__(self, vfeature_d, hidden_d, embedding_d, label_vec_path):
        super(HypernymVisual, self).__init__()
        self.hidden = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(vfeature_d, hidden_d)
        )

        self.embedding = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_d, embedding_d)
        )

        label_vec_file = h5py.File(label_vec_path, 'r')
        gt_label_vecs = np.array(label_vec_file['label_vec'])
        self._gt_label_vecs = Variable(torch.from_numpy(gt_label_vecs)).float().cuda()

    def forward(self, vfs):
        vf_hidden = self.hidden(vfs)
        vf_embeddings = self.embedding(vf_hidden)
        score_stack = Variable(torch.zeros(len(vf_embeddings), len(self._gt_label_vecs))).cuda()
        for i in range(len(vf_embeddings)):
            order_sims = order_sim(self._gt_label_vecs, vf_embeddings[i])
            score_stack[i] = order_sims
        return score_stack


