import torch
from torch import nn


class VisualFeatureEmbedding(nn.Module):
    def __init__(self, visual_feature_dimension, embedding_dimension):
        super(VisualFeatureEmbedding, self).__init__()
        self._embedding = nn.Linear(visual_feature_dimension, embedding_dimension)

    def forward(self, input):
        return self._embedding.forward(input)


class PartialOrderLoss(nn.Module):
    def __init__(self):
        super(PartialOrderLoss, self).__init__()
        self._hinge_loss = nn.HingeEmbeddingLoss()
        self._hinge_loss.zero_grad()

    def forward(self, pred_vec, word_vc, gt):
        # pred_vec: fc7 -> vec
        # word_vec: label -> vec
        # gt: 1(word_vec > pred_vec, positive)
        # gt: -1(word_vec < pred_vec, negative)
        sub = torch.add(pred_vec, -1, word_vc)
        act = nn.ReLU(sub)
        E = torch.FloatTensor()
        torch.addcmul(E, tensor1=act, tensor2=act, value=1)
        self._hinge_loss.zero_grad()
        return self._hinge_loss.forward(E, gt)
