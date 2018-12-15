import os
import shutil
import pickle
import torch
from open_relation1.dataset.MyDataset import MyDataset
from open_relation1.model import model
from train_config import hyper_params


def train():
    # prepare data
    config = hyper_params['vg']
    visual_feature_root = config['visual_feature_root']
    train_list_path = os.path.join(config['list_root'], 'train.txt')
    val_list_path = os.path.join(config['list_root'], 'small_val.txt')
    label_vec_path = config['label_vec_path']
    vg2path_path = config['vg2path_path']
    train_dataset = MyDataset(visual_feature_root, train_list_path, label_vec_path, vg2path_path, config['batch_size'])
    val_dataset = MyDataset(visual_feature_root, val_list_path, label_vec_path, vg2path_path, config['batch_size'])

    # prepare training log
    if os.path.isdir(config['log_root']):
        shutil.rmtree(config['log_root'])
        os.mkdir(config['log_root'])

    # initialize model
    latest_weights_path = config['latest_weight_path']
    best_weights_path = config['best_weight_path']
    net = model.HypernymVisual_acc(config['visual_d'], config['embedding_d'])
    if os.path.isfile(latest_weights_path):
        net.load_state_dict(torch.load(latest_weights_path))
        print('Loading weights success.')
    net.cuda()
    print(net)

    # config training hyper params
    params = net.parameters()
    optim = torch.optim.Adam(params=params, lr=config['lr'])
    loss = torch.nn.CrossEntropyLoss(size_average=False)

    # recorders
    batch_counter = 0
    best_acc = -1.0
    training_loss = []
    training_acc = []
    for e in range(0, config['epoch']):
        train_dataset.init_package()
        while train_dataset.has_next_minibatch():
            vf, p_lfs, n_lfs = train_dataset.minibatch_acc()
            batch_counter += 1
            batch_vf = torch.autograd.Variable(vf).cuda()
            batch_p_wfs = torch.autograd.Variable(p_lfs).cuda()
            batch_n_wfs = torch.autograd.Variable(n_lfs).cuda()
            score_vecs = net(batch_vf, batch_p_wfs, batch_n_wfs)
            t_acc = cal_acc(score_vecs.cpu().data)
            gts = torch.zeros(len(score_vecs)).long()
            gts = torch.autograd.Variable(gts).cuda()
            l = loss.forward(score_vecs, gts)
            l_raw = l.cpu().data.numpy().tolist()
            if batch_counter % config['print_freq'] == 0:
                info = 'epoch: %d | batch: %d | acc: %.2f | loss: %.2f' % (e, batch_counter, t_acc, l_raw)
                print(info)
                log_path = config['log_path']
                with open(log_path, 'a') as log:
                    log.write(info+'\n')
                training_loss.append(l_raw)
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
                e_acc = eval(val_dataset, net)
                info = ' ======== eval acc: %.2f ========' % e_acc
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


def cal_acc(score_vecs):
    tp_counter = 0
    for score_vec in score_vecs:
        is_tp = True
        for i in range(1, len(score_vec)):
            if score_vec[i] > score_vec[0]:
                is_tp = False
                break
        if is_tp:
            tp_counter += 1
    acc = tp_counter / len(score_vecs)
    return acc


def eval(dataset, model):
    model.eval()
    acc_sum = 0
    batch_sum = 0
    dataset.init_package()
    with torch.no_grad():
        while dataset.has_next_minibatch():
            vf, p_wf, n_wf = dataset.minibatch_acc()
            batch_vf = torch.autograd.Variable(vf).cuda()
            batch_p_wf = torch.autograd.Variable(p_wf).cuda()
            batch_n_wf = torch.autograd.Variable(n_wf).cuda()
            scores = model(batch_vf, batch_p_wf, batch_n_wf)
            batch_acc = cal_acc(scores.cpu().data)
            acc_sum += batch_acc
            batch_sum += 1
    avg_acc = acc_sum / batch_sum
    model.train()
    return avg_acc


if __name__ == '__main__':
    train()
