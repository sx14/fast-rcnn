from torch.autograd import Variable
import torch
import pickle

# path = '/media/sunx/Data/dataset/voc2007/VOCdevkit/VOC2007/feature/fc7/002931.bin'

a = torch.FloatTensor([0,0,3])
zeros = torch.zeros(3)

r = torch.eq(a,zeros)
print(r)
