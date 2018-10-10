import torch
import numpy as np
a = np.array([2,3,1])
t = torch.max(torch.FloatTensor(a))
print(t)