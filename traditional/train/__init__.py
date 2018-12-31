# from torch import nn, autograd
# import torch
#
# m = nn.Sigmoid()
# loss = nn.BCELoss()
# input = autograd.Variable(torch.randn(3), requires_grad=True)
# target = autograd.Variable(torch.FloatTensor(3).random_(2))
# output = loss(m(input), target)
# output.backward()