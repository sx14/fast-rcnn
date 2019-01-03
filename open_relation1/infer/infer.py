import numpy as np





def simple_infer(scores, org2path, label2index):
    # find the most precise predicted label in top 50 predictions
    org_label_inds = org2path.keys()
    ranked_inds = np.argsort(scores).tolist()
    ranked_inds.reverse()           # descending
    org_top_counter = 0
    iter = 0
    while ranked_inds[iter] not in org_label_inds and org_top_counter < 50:
        iter += 1






