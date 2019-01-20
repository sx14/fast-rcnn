from math import cos
import numpy as np

def ranking_score(n_item, s_max):
    ranks = np.arange(1, n_item+1).astype(np.float)

    s = (np.cos(ranks / n_item * np.pi) + 1) * (s_max * 1.0 / 2)
    return s



print(ranking_score(400, 1))