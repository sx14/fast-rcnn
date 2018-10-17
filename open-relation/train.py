import os
import shutil
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.MyDataset import MyDataset
from model import model
from train_config import hyper_params


def train():
    config = hyper_params['pascal']
    visual_feature_root = config['visual_feature_root']
    train_list_path = os.path.join(config['list_root'], 'train.txt')
    val_list_path = os.path.join(config['list_root'], 'val.txt')
    word_vec_path = config['word_vec_path']
    train_dataset = MyDataset(visual_feature_root, train_list_path, word_vec_path, config['batch_size'])
    val_dataset = MyDataset(visual_feature_root, val_list_path, word_vec_path, config['batch_size'])
    # train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'])
    # train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    net = model.HypernymVisual1(config['visual_d'], config['embedding_d'])
    latest_weights_path = config['latest_weight_path']
    best_weights_path = config['best_weight_path']
    if os.path.isfile(latest_weights_path):
        net.load_state_dict(torch.load(latest_weights_path))
        print('Loading weights success.')
    if os.path.isdir(config['log_root']):
        shutil.rmtree(config['log_root'])
        os.mkdir(config['log_root'])
    net.cuda()
    print(net)
    params = net.parameters()
    optim = torch.optim.Adam(params=params, lr=config['lr'])
    loss = torch.nn.HingeEmbeddingLoss()
    batch_counter = 0
    best_acc = 0
    training_loss = []
    training_acc = []
    for e in range(0, config['epoch']):
        # for vf, wf, gt in train_dataloader:
        train_dataset.init_package()
        while train_dataset.has_next_minibatch():
            vf, wf, gt = train_dataset.minibatch()
            p, n = count_p_n(gt)
            batch_counter += 1
            batch_vf = torch.autograd.Variable(vf).cuda()
            batch_wf = torch.autograd.Variable(wf).cuda()
            batch_gt = torch.autograd.Variable(gt).cuda()
            E = net(batch_vf, batch_wf)
            _, t_acc = cal_acc(E.cpu().data, gt)
            l = loss(E, batch_gt)
            l_raw = l.cpu().data.numpy().tolist()
            if batch_counter % config['print_freq'] == 0:
                info = 'epoch: %d | batch: %d[%d/%d] | acc: %.2f | loss: %.2f' % (e, batch_counter, p, n, t_acc, l_raw)
                print(info)
                log_path = config['log_path']
                with open(log_path, 'a') as log:
                    log.write(info+'\n')
                training_loss.append(l_raw)
                training_acc.append(t_acc)
            optim.zero_grad()
            l.backward()
            optim.step()
            if batch_counter % config['eval_freq'] == 0:
                loss_log_path = config['log_loss_path']
                save_log_data(loss_log_path, training_loss)
                training_loss = []
                acc_log_path = config['log_acc_path']
                save_log_data(acc_log_path, training_acc)
                training_acc = []
                best_threshold, e_acc = eval(val_dataset, net)
                info = 'eval acc: %.2f | best threshold: %.2f' % (e_acc, best_threshold)
                print(info)
                log_path = config['log_path']
                with open(log_path, 'a') as log:
                    log.write(info+'\n')
                torch.save(net.state_dict(), latest_weights_path)
                print('Updating weights success.')
                if e_acc > best_acc:
                    torch.save(net.state_dict(), best_weights_path)
                    print('Updating best weights success.')
                    best_acc = e_acc


def save_log_data(file_path, data):
    if not os.path.exists(file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(file_path, 'rb') as f:
            history_data = pickle.load(f)
        with open(file_path, 'wb') as f:
            history_data = history_data + data
            pickle.dump(history_data, f)


def cal_acc(E, gt):
    tmp_E = E.numpy()
    tmp_E = np.reshape(tmp_E, (tmp_E.size))
    tmp_gt = gt.numpy()
    tmp_gt = np.reshape(tmp_gt, (tmp_gt.size))
    tmp_gt = (tmp_gt == np.ones(tmp_gt.size)) + 0  # 1/-1 -> 1/0
    sorted_indexes = np.argsort(tmp_E)
    sorted_E = tmp_E[sorted_indexes]
    sorted_gt = tmp_gt[sorted_indexes]
    tp = np.cumsum(sorted_gt, 0)
    inv_sorted_gt = (sorted_gt == np.zeros(sorted_gt.size)) + 0
    neg_sum = np.sum(inv_sorted_gt)
    fp = np.cumsum(inv_sorted_gt, 0)
    tn = fp * (-1) + neg_sum
    acc = (tp + tn) * 1.0 / tmp_gt.size
    best_acc_index = np.argmax(acc)
    return sorted_E[best_acc_index], acc[best_acc_index]


def eval(dataset, model):
    model.eval()
    acc_sum = 0
    best_threhold = 0
    dataset.init_package()
    batch_sum = 0
    while dataset.has_next_minibatch():
        vf, wf, gt = dataset.minibatch()
    # val_dataloader = DataLoader(dataset, batch_size=dataset.__len__())
    # for vf, wf, gt in val_dataloader:
        batch_vf = torch.autograd.Variable(vf).cuda()
        batch_wf = torch.autograd.Variable(wf).cuda()
        E = model(batch_vf, batch_wf)
        best_threhold, batch_acc = cal_acc(E.cpu().data, gt)
        acc_sum += batch_acc
        batch_sum += 1
    acc_sum = acc_sum / batch_sum
    # acc_sum = acc_sum / len(val_dataloader)
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


if __name__ == '__main__':
    train()



