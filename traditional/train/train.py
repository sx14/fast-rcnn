import os
import numpy as np
import shutil
import pickle
import torch
from traditional.dataset.MyDataset import MyDataset
from traditional.model import model
from train_config import hyper_params


def train():
    dataset = 'vrd'

    # prepare data
    config = hyper_params[dataset]
    visual_feature_root = config['visual_feature_root']
    train_list_path = os.path.join(config['list_root'], 'train.txt')
    val_list_path = os.path.join(config['list_root'], 'small_val.txt')
    vg2path_path = config[dataset+'2path_path']
    train_dataset = MyDataset(visual_feature_root, train_list_path, vg2path_path, config['batch_size'])
    val_dataset = MyDataset(visual_feature_root, val_list_path, vg2path_path, config['batch_size'])

    # prepare training log
    if os.path.isdir(config['log_root']):
        shutil.rmtree(config['log_root'])
        os.mkdir(config['log_root'])

    # initialize model
    latest_weights_path = config['latest_weight_path']
    best_weights_path = config['best_weight_path']
    net = model.HypernymVisual_acc(config['visual_d'], config['class_num'])
    if os.path.isfile(best_weights_path):
        net.load_state_dict(torch.load(best_weights_path))
        print('Loading weights success.')
    net.cuda()
    print(net)

    # config training hyper params
    params = net.parameters()
    optim = torch.optim.SGD(params=params, lr=config['lr'])
    loss = torch.nn.BCELoss()               # multi-label cross entropy
    # loss = torch.nn.CrossEntropyLoss()    # single-label cross entropy


    # recorders
    batch_counter = 0
    best_acc = -1.0
    training_loss = []
    training_acc = []

    # train
    for e in range(0, config['epoch']):
        make_params_positive(params)
        train_dataset.init_package()
        while train_dataset.has_next_minibatch():
            batch_counter += 1
            # load a minibatch
            vf, lvs = train_dataset.minibatch_acc1()
            batch_vf = torch.autograd.Variable(vf).cuda()
            batch_lvs = torch.autograd.Variable(lvs).cuda()
            # forward
            score_vecs = net.forward(batch_vf)
            t_acc = cal_acc(score_vecs.cpu().data, batch_lvs.cpu().data)
            l = loss.forward(score_vecs, batch_lvs)
            l_raw = l.cpu().data.numpy().tolist()
            training_loss.append(l_raw)
            training_acc.append(t_acc)
            if batch_counter % config['print_freq'] == 0:
                print('epoch: %d | batch: %d | acc: %.2f | loss: %.2f' % (e, batch_counter, t_acc, l_raw))
                loss_log_path = config['log_loss_path']
                save_log_data(loss_log_path, training_loss)
                training_loss = []
                acc_log_path = config['log_acc_path']
                save_log_data(acc_log_path, training_acc)
                training_acc = []
                training_loss.append(l_raw)
            optim.zero_grad()
            l.backward()
            optim.step()
            if batch_counter % config['eval_freq'] == 0:
                make_params_positive(params)
                e_acc = eval(val_dataset, net)
                info = '======== eval acc: %.2f ========' % e_acc
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


def make_params_positive(params):
    for param in params:
        param.data[param.data < 0] = 0


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


def cal_acc(score_vecs, label_vecs):
    tp_counter = 0.0
    for i, score_vec in enumerate(score_vecs):
        label = label_vecs[i]
        score = score_vec[label]
        max_score = np.max(score_vec.numpy())
        if max_score == score:
            tp_counter += 1
    return tp_counter / len(score_vecs)


def eval(dataset, model):
    model.eval()
    acc_sum = 0.0
    batch_sum = 0
    dataset.init_package()
    with torch.no_grad():
        while dataset.has_next_minibatch():
            vfs, lfs = dataset.minibatch_acc()
            batch_vf = torch.autograd.Variable(vfs).cuda()
            batch_lfs = torch.autograd.Variable(lfs).cuda()
            scores = model(batch_vf)
            batch_acc = cal_acc(scores.cpu().data, batch_lfs.cpu().data)
            acc_sum += batch_acc
            batch_sum += 1
    avg_acc = acc_sum / batch_sum
    model.train()
    return avg_acc


if __name__ == '__main__':
    train()
