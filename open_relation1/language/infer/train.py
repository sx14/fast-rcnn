import torch
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader
from lang_dataset import LangDataset
from lang_config import train_params, lang_config
from model import RelationEmbedding
# from model import relation_embedding_loss as loss_func
from model import order_rank_loss as loss_func
from model import order_rank_test as rank_test


def eval(model, test_dl):
    model.eval()
    acc_sum = 0
    batch_num = 0
    for batch in test_dl:
        batch_num += 1
        sbj1, pre1, obj1, sbj2, pre2, obj2 = batch
        v_sbj1 = Variable(sbj1).float().cuda()
        v_pre1 = Variable(pre1).float().cuda()
        v_obj1 = Variable(obj1).float().cuda()
        v_sbj2 = Variable(sbj2).float().cuda()
        v_pre2 = Variable(pre2).float().cuda()
        v_obj2 = Variable(obj2).float().cuda()

        pre_emb1 = model(v_sbj1, v_obj1)
        pre_emb2 = model(v_sbj2, v_obj2)

        acc, _, _ = rank_test(pre_emb1, pre_emb2, v_pre1)
        acc_sum += acc
    avg_acc = acc_sum / batch_num
    model.train()
    return avg_acc


# hyper params
epoch_num = train_params['epoch_num']
lr = train_params['lr']
embedding_dim = train_params['embedding_dim']
batch_size = train_params['batch_size']

rlt_path = lang_config['train']['rlt_save_path']
train_set = LangDataset(rlt_path)
train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)

rlt_path = lang_config['test']['rlt_save_path']
test_set = LangDataset(rlt_path)
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# model
model = RelationEmbedding(embedding_dim*2, embedding_dim)
model = model.cuda()

# optimizer
optim = SGD(model.parameters(), lr=lr)
batch_num = 0

best_acc = 0
save_path = train_params['model_save_path']
best_path = train_params['best_model_path']

for epoch in range(epoch_num):
    for batch in train_dl:
        batch_num += 1
        sbj1, pre1, obj1, sbj2, pre2, obj2 = batch
        v_sbj1 = Variable(sbj1).float().cuda()
        v_pre1 = Variable(pre1).float().cuda()
        v_obj1 = Variable(obj1).float().cuda()
        v_sbj2 = Variable(sbj2).float().cuda()
        v_pre2 = Variable(pre2).float().cuda()
        v_obj2 = Variable(obj2).float().cuda()

        pre_emb1 = model(v_sbj1, v_obj1)
        pre_emb2 = model(v_sbj2, v_obj2)
        acc, pos_sim, neg_sim = rank_test(pre_emb1, pre_emb2, v_pre1)
        loss = loss_func(pos_sim, neg_sim)

        print('Epoch %d | Batch %d | Loss %.2f | acc: %.2f' % (epoch + 1, batch_num + 1, loss.cpu().data, acc))

        optim.zero_grad()
        loss.backward()
        optim.step()

    print('\ntesting ......')
    avg_acc = eval(model, test_dl)
    if avg_acc > best_acc:
        best_acc = avg_acc
        torch.save(model.state_dict(), best_path)
    print('============ acc: % .2f' % avg_acc + ' ============\n')
    torch.save(model.state_dict(), save_path)
    train_set.update_pos_neg_pairs()


