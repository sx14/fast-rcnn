import os
import shutil
import pickle
import torch
from open_relation1.dataset.MyDataset import MyDataset
from open_relation1.model.predicate import model
from train_config import hyper_params


def train():
    dataset = 'vrd'
    target = 'predicate'
    # prepare data
    config = hyper_params[dataset][target]
    visual_feature_root = config['visual_feature_root']
    train_list_path = os.path.join(config['list_root'], 'train.txt')
    val_list_path = os.path.join(config['list_root'], 'small_val.txt')
    label_vec_path = config['label_vec_path']
    vg2path_path = config[dataset+'2path_path']
    vg2weight_path = config[dataset+'2weight_path']

    train_dataset = MyDataset(visual_feature_root, train_list_path, label_vec_path,
                              vg2path_path, vg2weight_path, config['batch_size'], config['negative_label_num'])
    val_dataset = MyDataset(visual_feature_root, val_list_path, label_vec_path,
                            vg2path_path, vg2weight_path, config['batch_size'], config['negative_label_num'])

    # prepare training log
    if os.path.isdir(config['log_root']):
        shutil.rmtree(config['log_root'])
        os.mkdir(config['log_root'])

    # init model
    latest_weights_path = config['latest_weight_path']
    best_weights_path = config['best_weight_path']
    net = model.PredicateVisual_acc()
    if os.path.isfile(latest_weights_path):
        net.load_state_dict(torch.load(latest_weights_path))
        print('Loading weights success.')
    net.cuda()
    net.train()
    print(net)

    # config training hyper params
    params = filter(lambda p: p.requires_grad, net.parameters())
    optim = torch.optim.SGD(params=params, lr=config['lr'])
    loss_func = torch.nn.CrossEntropyLoss(reduce=False)


    # recorders
    batch_counter = 0
    best_acc = -1.0
    training_loss = []
    training_acc = []

    # training
    for e in range(0, config['epoch']):
        train_dataset.init_package()
        while train_dataset.has_next_minibatch():
            batch_counter += 1

            # load a minibatch
            vfs, pls, nls, label_vecs, pws = train_dataset.minibatch_acc1(vf_d=config['visual_d'])

            # forward
            score_vecs = net.forward1(vfs, pls, nls, label_vecs)

            # cal training acc
            t_acc = cal_acc(score_vecs.cpu().data)
            gts = torch.zeros(len(score_vecs)).long()
            gts = torch.autograd.Variable(gts).cuda()

            # cal loss
            loss0 = loss_func.forward(score_vecs, gts)
            loss = torch.mean(loss0 * pws)

            l_raw = loss.cpu().data.numpy().tolist()
            training_loss.append(l_raw)
            training_acc.append(t_acc)
            if batch_counter % config['print_freq'] == 0:
                # logging
                loss_log_path = config['log_loss_path']
                save_log(loss_log_path, training_loss)
                training_loss = []
                acc_log_path = config['log_acc_path']
                save_log(acc_log_path, training_acc)
                training_acc = []
                training_loss.append(l_raw)
                print('epoch: %d | batch: %d | acc: %.2f | loss: %.2f' % (e, batch_counter, t_acc, l_raw))

            # backward propagate
            optim.zero_grad()
            loss.backward()
            optim.step()

            # evaluate
            if batch_counter % config['eval_freq'] == 0:
                # make_params_positive(params)
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
                    best_acc = e_acc
                    print('Updating best weights success.')


def make_params_positive(params):
    for param in params:
        param.data[param.data < 0] = 0


def save_log(file_path, data):
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
    tp_counter = 0.0
    for score_vec in score_vecs:
        is_tp = True
        for i in range(1, len(score_vec)):
            if score_vec[i] >= score_vec[0]:
                is_tp = False
                break
        if is_tp:
            tp_counter += 1
    acc = tp_counter / len(score_vecs)
    return acc


def eval(dataset, model):
    model.eval()
    acc_sum = 0.0
    batch_sum = 0
    dataset.init_package()
    with torch.no_grad():
        while dataset.has_next_minibatch():
            vfs, pls, nls, label_vecs, pws = dataset.minibatch_acc1()
            batch_vf = torch.autograd.Variable(vfs).cuda()
            label_vecs = torch.autograd.Variable(label_vecs).cuda()
            scores = model.forward1(batch_vf, pls, nls, label_vecs)
            batch_acc = cal_acc(scores.cpu().data)
            acc_sum += batch_acc
            batch_sum += 1
    avg_acc = acc_sum / batch_sum
    model.train()
    return avg_acc


if __name__ == '__main__':
    train()
