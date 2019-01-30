import os
import pickle
import torch
from tensorboardX import SummaryWriter
from open_relation.dataset.MyDataset import MyDataset
from open_relation.model.object.model import HypernymVisual, order_softmax_test
from train_config import hyper_params
from open_relation.dataset.vrd.label_hier.obj_hier import objnet as vrd_objnet
from open_relation.dataset.vrd.label_hier.pre_hier import prenet as vrd_prenet
from open_relation.dataset.vg.label_hier.obj_hier import objnet as vg_objnet
from open_relation.dataset.vg.label_hier.pre_hier import prenet as vg_prenet

labelnets = {
    'vrd': {'object': vrd_objnet, 'predicate': vrd_prenet},
    'vg': {'object': vg_objnet, 'predicate': vg_prenet},
}


def eval(dataset, model):
    model.eval()
    acc_sum = 0.0
    loss_sum = 0.0
    batch_sum = 0
    loss_func = torch.nn.CrossEntropyLoss(reduce=False)
    dataset.init_package()
    with torch.no_grad():
        while dataset.has_next_minibatch():
            vfs, pos_neg_inds, weights = dataset.minibatch()
            batch_vf = torch.autograd.Variable(vfs).cuda()
            all_scores = model(batch_vf)
            batch_acc, loss_scores, y = order_softmax_test(all_scores, pos_neg_inds)
            batch_loss = loss_func(loss_scores, y)
            batch_loss = torch.mean(batch_loss * weights)
            acc_sum += batch_acc
            loss_sum += batch_loss
            batch_sum += 1
    avg_acc = acc_sum / batch_sum
    avg_loss = loss_sum / batch_sum
    model.train()
    return avg_acc, avg_loss


""" ================  train ================ """

dataset = 'vrd'
target = 'object'

labelnet = labelnets[dataset][target]

# prepare data
config = hyper_params[dataset][target]

raw2path = labelnet.raw2path()
raw2weight_path = config['raw2weight_path']

visual_feat_root = config['visual_feature_root']
train_list_path = os.path.join(config['list_root'], 'train.txt')
train_dataset = MyDataset(visual_feat_root, train_list_path, raw2path, config['visual_d'],
                          raw2weight_path, labelnet.label_sum(), config['batch_size'], config['negative_label_num'])

val_list_path = os.path.join(config['list_root'], 'val.txt')
val_dataset = MyDataset(visual_feat_root, val_list_path, raw2path, config['visual_d'],
                        raw2weight_path, labelnet.label_sum(), config['batch_size'], config['negative_label_num'])

# init model
latest_weights_path = config['latest_weight_path']
best_weights_path = config['best_weight_path']
label_vec_path = config['label_vec_path']

net = HypernymVisual(config['visual_d'], config['hidden_d'],
                     config['embedding_d'], label_vec_path)
if os.path.isfile(latest_weights_path):
    net.load_state_dict(torch.load(latest_weights_path))
    print('Loading weights success.')
net.cuda()
net.train()
print(net)

# add L2 regularization
weight_p, bias_p = [], []
for name, p in net.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]

# optimizer
loss_func = torch.nn.CrossEntropyLoss(reduce=False)
optim = torch.optim.SGD([{'params': weight_p, 'weight_decay': 1e-5},
                         {'params': bias_p, 'weight_decay': 0}], lr=config['lr'])

# recorders
batch_counter = 0
best_acc = 0.0
sw = SummaryWriter()

# training
for e in range(0, config['epoch']):
    train_dataset.init_package()
    while train_dataset.has_next_minibatch():
        batch_counter += 1
        # load a minibatch
        vfs, pos_neg_inds, weights = train_dataset.minibatch()

        # forward
        all_scores = net(vfs)

        # cal acc, loss
        acc, loss_scores, y = order_softmax_test(all_scores, pos_neg_inds)
        loss = loss_func(loss_scores, y)
        loss = torch.mean(loss * weights)

        if batch_counter % config['print_freq'] == 0:
            # logging
            sw.add_scalars('acc', {'train': acc}, batch_counter)
            sw.add_scalars('loss', {'train': loss}, batch_counter)
            print('epoch: %d | batch: %d | acc: %.2f | loss: %.2f' % (e, batch_counter, acc, loss))

        # backward propagate
        optim.zero_grad()
        loss.backward()
        optim.step()

        # evaluate
        if batch_counter % config['eval_freq'] == 0:
            print('\nevaluating ......')
            e_acc, e_loss = eval(val_dataset, net)
            sw.add_scalars('acc', {'eval': e_acc}, batch_counter)
            sw.add_scalars('loss', {'eval': e_loss}, batch_counter)
            print('batch: %d >>> acc: %.2f  loss: %.2f' % (batch_counter, e_acc, e_loss))
            torch.save(net.state_dict(), latest_weights_path)
            print('Updating weights success.')
            if e_acc > best_acc:
                torch.save(net.state_dict(), best_weights_path)
                best_acc = e_acc
                print('Updating best weights success.\n')
