# import torch
# import numpy as np
# from torch import nn
#
# a = nn.Linear(5, 5)
# params = a.parameters()
# for param in params:
#     print(param.data)
#     param.data[param.data < 0] = 0
#
# params = a.parameters()
# for param in params:
#     print(param.data)