import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.MyDataset import MyDataset
from model import model
from train_config import hyper_params


def train():
    config = hyper_params['visual genome']
    visual_feature_root = config['visual_feature_root']
    train_list_path = os.path.join(config['list_root'], 'train.txt')
    val_list_path = os.path.join(config['list_root'], 'val_small.txt')
    word_vec_path = config['word_vec_path']
    train_dataset = MyDataset(visual_feature_root, train_list_path, word_vec_path)
    val_dataset = MyDataset(visual_feature_root, val_list_path, word_vec_path)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'])
    # train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    net = model.HypernymVisual(config['visual_d'], config['embedding_d'])
    model_weights_path = config['weight_path']
    if os.path.isfile(model_weights_path):
        net.load_state_dict(torch.load(model_weights_path))
        print('Loading weights success.')
    net.cuda()
    print(net)
    params = net.parameters()
    optim = torch.optim.Adam(params=params, lr=0.0001)
    loss = torch.nn.HingeEmbeddingLoss()
    batch_counter = 0
    best_acc = 0
    for e in range(0, config['epoch']):
        for vf, wf, gt in train_dataloader:
            p,n = count_p_n(gt)
            batch_counter += 1
            batch_vf = torch.autograd.Variable(vf).cuda()
            batch_wf = torch.autograd.Variable(wf).cuda()
            batch_gt = torch.autograd.Variable(gt).cuda()
            E = net(batch_vf, batch_wf)
            corr = cal_acc(E.cpu().data, gt)
            t_acc = corr * 1.0 / vf.size()[0]
            l = loss(E, batch_gt)
            print('epoch: %d | batch: %d[%d/%d] | acc: %.2f | loss: %.2f' % (e, batch_counter, p, n, t_acc, l.cpu().data.numpy()))
            optim.zero_grad()
            l.backward()
            optim.step()
            if batch_counter % config['eval_freq'] == 0:
                best_threshold, e_acc = eval(val_dataset, net)
                print('eval acc: %.2f | best threshold: %.2f' % (e_acc, best_threshold))
                if e_acc > best_acc:
                    torch.save(net.state_dict(), model_weights_path)
                    print('Updating weights success.')
                    best_acc = e_acc


def cal_acc(E, gt):
    E = E.numpy()
    gt = gt.numpy()
    gt = (gt == np.ones(gt.size)) + 0  # 1/-1 -> 1/0
    sorted_indexes = np.argsort(E)
    sorted_E = E[sorted_indexes]
    sorted_gt = gt[sorted_indexes]
    tp = np.cumsum(sorted_gt, 0)
    inv_sorted_gt = (sorted_gt == np.zeros(sorted_gt.size)) + 0
    neg_sum = np.sum(inv_sorted_gt)
    fp = np.cumsum(inv_sorted_gt, 0)
    tn = fp * (-1) + neg_sum
    acc = (tp + tn) * 1.0 / gt.size
    best_acc_index = np.argmax(acc)
    return sorted_E[best_acc_index], acc[best_acc_index]


def eval(dataset, model):
    model.eval()
    val_dataloader = DataLoader(dataset, batch_size=dataset.__len__())
    acc_sum = 0
    best_threhold = 0
    for vf, wf, gt in val_dataloader:
        batch_vf = torch.autograd.Variable(vf)
        batch_wf = torch.autograd.Variable(wf)
        E = model(batch_vf, batch_wf)
        best_threhold, batch_acc = cal_acc(E.cpu().data, gt)
        acc_sum += batch_acc
    acc_sum = acc_sum / len(val_dataloader)
    return best_threhold, acc_sum


def count_p_n(gts):
    p = 0
    n = 0
    for gt in gts:
        if gt[0] > 0:
            p += 1
        else:
            n += 1
    return p, n

def t_acc():
    E = torch.FloatTensor([0.1,0.1,0.2,0.3,10])
    gt = torch.FloatTensor([1,1,0,1,0])
    cal_acc(E,gt)


if __name__ == '__main__':
    train()



