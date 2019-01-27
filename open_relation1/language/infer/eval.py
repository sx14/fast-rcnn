import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from lang_dataset import LangDataset
from lang_config import train_params, lang_config
from model import RelationEmbedding
from model import order_rank_test as rank_test
from open_relation1.dataset.vrd.label_hier.pre_hier import prenet



# hyper params
epoch_num = 1
embedding_dim = train_params['embedding_dim']
batch_size = 1

rlt_path = lang_config['test']['raw_rlt_path']
test_set = LangDataset(rlt_path)
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# model
model = RelationEmbedding(embedding_dim*2, embedding_dim)
weight_path = train_params['model_save_path']
if os.path.isfile(weight_path):
    model.load_state_dict(torch.load(weight_path))
    print('Loading weights success.')
model.cuda()
model.eval()


batch_num = 0
acc_sum = 0
save_path = train_params['model_save_path']
best_path = train_params['best_model_path']

gt_vecs = test_set.get_gt_vecs().float().cuda()

for batch in test_dl:
    batch_num += 1
    sbj1, pre1, obj1, _, _, _, rlts = batch
    v_sbj1 = Variable(sbj1).float().cuda()
    v_pre1 = Variable(pre1).float().cuda()
    v_obj1 = Variable(obj1).float().cuda()
    with torch.no_grad():
        pre_emb1 = model(v_sbj1, v_obj1)

    batch_ranks = rank_test(pre_emb1, gt_vecs)

    for ranks in batch_ranks:
        gt_node = prenet.get_node_by_index(rlts[0][1])
        gt_label = gt_node.name()
        gt_hyper_inds = gt_node.trans_hyper_inds()
        print('===== GT: %s =====' % gt_label)
        for gt_h_ind in gt_hyper_inds:
            gt_h_node = prenet.get_node_by_index(gt_h_ind)
            print(gt_h_node.name())
        print('===== predicate =====')
        for pre_ind in ranks:
            pre_node = prenet.get_node_by_index(pre_ind)
            print(pre_node.name())


