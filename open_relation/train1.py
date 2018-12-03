import os
import shutil
import pickle
import numpy as np
import torch
from dataset.MyDataset import MyDataset
from model import model
from train_config import hyper_params


def batch_adjust_lr(optimizer, org_lr, curr_batch, adjust_freq):
    lr = org_lr * (0.66 ** (curr_batch / adjust_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('==== adjust lr ====')
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    print('===================')


def epoch_adjust_lr(optimizer, org_lr, curr_epoch):
    lr = org_lr * (0.1 ** curr_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('==== adjust lr ====')
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    print('===================')


def train():
    config = hyper_params['pascal']
    visual_feature_root = config['visual_feature_root']
    train_list_path = os.path.join(config['list_root'], 'train.txt')
    val_list_path = os.path.join(config['list_root'], 'val_small.txt')
    word_vec_path = config['word_vec_path']
    label2path_path = config['label2path_path']
    train_dataset = MyDataset(visual_feature_root, train_list_path, word_vec_path, label2path_path, config['batch_size'])
    val_dataset = MyDataset(visual_feature_root, val_list_path, word_vec_path, label2path_path, config['batch_size'])
    net = model.HypernymVisual(config['visual_d'], config['embedding_d'])
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
    loss = torch.nn.MarginRankingLoss(margin=0.1, size_average=False)
    batch_counter = 0
    best_acc = 0
    optim_freq = 10
    training_loss = []
    training_acc = []
    for e in range(0, config['epoch']):
        epoch_adjust_lr(optim, config['lr'], e)
        train_dataset.init_package()
        loss_sum = 0
        wrong_sum = 0
        acc_sum = 0
        while train_dataset.has_next_minibatch():
            # if batch_counter % config['lr_adjust_freq'] == 0:
            #     batch_adjust_lr(optim, config['lr'], batch_counter, config['lr_adjust_freq'])
            vf, p_wfs, n_wfs, gts = train_dataset.minibatch2()
            batch_counter += 1
            batch_vf = torch.autograd.Variable(vf).cuda()
            batch_p_wfs = torch.autograd.Variable(p_wfs).cuda()
            batch_n_wfs = torch.autograd.Variable(n_wfs).cuda()
            gts = torch.autograd.Variable(gts).cuda()
            p_E, n_E = net(batch_vf, batch_p_wfs, batch_n_wfs)
            _, t_acc, t_wrong = cal_acc(p_E.cpu().data, n_E.cpu().data)
            # expect n_E > p_E
            l = loss(n_E, p_E, gts)
            l_raw = l.cpu().data.numpy().tolist()
            loss_sum += l_raw
            wrong_sum += t_wrong
            acc_sum += t_acc
            l.backward()
            if batch_counter % optim_freq == 0:
                optim.step()
                optim.zero_grad()
                info = 'epoch: %d | batch: %d | wrong: %d | loss: %.2f' % (e, batch_counter, wrong_sum/optim_freq, loss_sum/optim_freq)
                print(info)
                log_path = config['log_path']
                with open(log_path, 'a') as log:
                    log.write(info + '\n')
                training_loss.append(loss_sum/optim_freq)
                training_acc.append(wrong_sum/optim_freq)
                loss_sum = 0
                wrong_sum = 0
                acc_sum = 0
            if batch_counter % config['eval_freq'] == 0:
                loss_log_path = config['log_loss_path']
                save_log_data(loss_log_path, training_loss)
                training_loss = []
                acc_log_path = config['log_acc_path']
                save_log_data(acc_log_path, training_acc)
                training_acc = []
                best_threshold, e_acc, wrong = eval(val_dataset, net)
                info = 'eval wrong: %d | best threshold: %.2f' % (wrong, best_threshold)
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


def cal_acc(p_E, n_E):
    tmp_p_E = p_E.numpy()
    tmp_p_E = np.reshape(tmp_p_E, (tmp_p_E.size))
    tmp_n_E = n_E.numpy()
    tmp_n_E = np.reshape(tmp_n_E, (tmp_n_E.size))
    sub = tmp_n_E - tmp_p_E
    t = np.where(sub > 0)[0]
    wrong = len(sub) - len(t)
    acc = len(t) * 1.0 / len(sub)
    best_threshold = np.max(tmp_p_E)
    return best_threshold, acc, wrong


def eval(dataset, model):
    model.eval()
    acc_sum = 0
    threshold_sum = 0
    batch_sum = 0
    wrong_sum = 0
    dataset.init_package()
    while dataset.has_next_minibatch():
        vf, p_wf, n_wf, gt = dataset.minibatch2()
        batch_vf = torch.autograd.Variable(vf, volatile=True).cuda()
        batch_p_wf = torch.autograd.Variable(p_wf, volatile=True).cuda()
        batch_n_wf = torch.autograd.Variable(n_wf, volatile=True).cuda()
        p_E, n_E = model(batch_vf, batch_p_wf, batch_n_wf)
        batch_threshold, batch_acc, batch_wrong = cal_acc(p_E.cpu().data, n_E.cpu().data)
        print('wrong: '+str(batch_wrong))
        acc_sum += batch_acc
        wrong_sum += batch_wrong
        threshold_sum += batch_threshold
        batch_sum += 1
    avg_acc = acc_sum / batch_sum
    avg_threshold = threshold_sum / batch_sum
    avg_wrong = wrong_sum / batch_sum
    model.train()
    return avg_threshold, avg_acc, avg_wrong


if __name__ == '__main__':
    train()


