import os
import torch
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from lang_dataset import LangDataset
from lang_config import train_params, data_config
from model import RelationEmbedding
# from model import relation_embedding_loss as loss_func
# from model import order_rank_loss as loss_func
from torch.nn.functional import cross_entropy as loss_func
from model import order_softmax_test as rank_test
# from model import order_rank_eval as rank_test



def eval(model, test_dl):
    model.eval()
    acc_sum = 0
    loss_sum = 0
    batch_num = 0
    for batch in test_dl:
        batch_num += 1
        sbj1, pre1, obj1, sbj2, pre2, obj2, _, pos_neg_inds = batch
        v_sbj1 = Variable(sbj1).float().cuda()
        v_pre1 = Variable(pre1).float().cuda()
        v_obj1 = Variable(obj1).float().cuda()
        v_sbj2 = Variable(sbj2).float().cuda()
        v_pre2 = Variable(pre2).float().cuda()
        v_obj2 = Variable(obj2).float().cuda()
        with torch.no_grad():
            pre_emb1 = model(v_sbj1, v_obj1)
        # pre_emb2 = model(v_sbj2, v_obj2)
        acc, score_stack, y = rank_test(pre_emb1, pos_neg_inds, test_set.get_gt_vecs())
        loss = loss_func(score_stack, y)
        # acc, _, _ = rank_test(pre_emb1, pre_emb2, v_pre1)
        acc_sum += acc
        loss_sum += loss
    avg_acc = acc_sum / batch_num
    avg_loss = loss_sum / batch_num
    model.train()
    return avg_acc, avg_loss


# training hyper params
epoch_num = train_params['epoch_num']
lr = train_params['lr']
embedding_dim = train_params['embedding_dim']
batch_size = train_params['batch_size']

# init dataset
rlt_path = data_config['train']['ext_rlt_path']
train_set = LangDataset(rlt_path)
train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)

rlt_path = data_config['test']['raw_rlt_path']
test_set = LangDataset(rlt_path)
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# model
save_model_path = train_params['save_model_path']
new_model_path = train_params['latest_model_path']
best_model_path = train_params['best_model_path']
model = RelationEmbedding(embedding_dim*2, embedding_dim)
if os.path.exists(new_model_path):
    model.load_state_dict(torch.load(new_model_path))
    print('Loading weights success.')
else:
    print('No pretrained weights.')
model.cuda()

# optimizer
optim = torch.optim.SGD(model.parameters(), lr=lr)
# weight_p, bias_p = [], []
# for name, p in model.named_parameters():
#     if 'bias' in name:
#         bias_p += [p]
#     else:
#         weight_p += [p]
# optim = torch.optim.SGD([{'params': weight_p, 'weight_decay': 1e-5},
#                          {'params': bias_p, 'weight_decay': 0}], lr=lr)

# training process record
sw = SummaryWriter()
batch_num = 0
best_acc = 0
for epoch in range(epoch_num):
    for batch in train_dl:
        batch_num += 1
        sbj1, pre1, obj1, sbj2, pre2, obj2, _, pos_neg_inds = batch
        v_sbj1 = Variable(sbj1).float().cuda()
        v_pre1 = Variable(pre1).float().cuda()
        v_obj1 = Variable(obj1).float().cuda()
        v_sbj2 = Variable(sbj2).float().cuda()
        v_pre2 = Variable(pre2).float().cuda()
        v_obj2 = Variable(obj2).float().cuda()

        pre_emb1 = model(v_sbj1, v_obj1)
        # pre_emb2 = model(v_sbj2, v_obj2)

        acc, score_stack, y = rank_test(pre_emb1, pos_neg_inds, train_set.get_gt_vecs())
        loss = loss_func(score_stack, y)
        sw.add_scalars('acc', {'train': acc}, batch_num)
        sw.add_scalars('loss', {'train': loss}, batch_num)

        # acc, pos_sim, neg_sim = rank_test(pre_emb1, pre_emb2, v_pre1)
        # loss = loss_func(pos_sim, neg_sim)

        print('Epoch %d | Batch %d | Loss %.2f | Acc: %.2f' % (epoch + 1, batch_num, loss.cpu().data, acc))

        optim.zero_grad()
        loss.backward()
        optim.step()

    print('\nevaluating ......')
    avg_acc, avg_loss = eval(model, test_dl)
    sw.add_scalars('acc', {'eval': avg_acc}, batch_num)
    sw.add_scalars('loss', {'eval': avg_loss}, batch_num)
    if avg_acc > best_acc:
        best_acc = avg_acc
        torch.save(model.state_dict(), best_model_path)
    print('>>>> Eval Acc: % .2f <<<<\n' % avg_acc)
    torch.save(model.state_dict(), save_model_path+str(epoch)+'.pkl')
    torch.save(model.state_dict(), new_model_path)
    train_set.update_pos_neg_pairs()


