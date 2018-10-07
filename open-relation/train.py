import os
import torch
from torch.utils.data import DataLoader
from dataset.MyDataset import MyDataset
from model import model
from train_config import hyper_params


def train():
    params = hyper_params['visual genome']
    visual_feature_root = params['visual_feature_root']
    train_list_path = os.path.join(params['list_root'], 'train.txt')
    val_list_path = os.path.join(params['list_root'], 'val.txt')
    word_vec_path = params['word_vec_path']
    train_dataset = MyDataset(visual_feature_root, train_list_path, word_vec_path)
    val_dataset = MyDataset(visual_feature_root, val_list_path, word_vec_path)
    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    net = model.HypernymVisual(params['visual_d'], params['embedding_d'])
    model_weights_path = 'model/vs_weights.pkl'
    if os.path.isfile(model_weights_path):
        net.load_state_dict(torch.load(model_weights_path))
        print('Loading weights success.')
    net.cuda()
    print(net)
    params = net.parameters()
    optim = torch.optim.Adam(params=params, lr=0.0001)
    loss = torch.nn.HingeEmbeddingLoss()
    batch_counter = 0
    for e in range(0, params['epoch']):
        for vf, wf, gt in train_dataloader:
            batch_counter += 1
            batch_vf = torch.autograd.Variable(vf).cuda()
            batch_wf = torch.autograd.Variable(wf).cuda()
            batch_gt = torch.autograd.Variable(gt).cuda()
            E = net(batch_vf, batch_wf)
            corr = correct(E, batch_gt)
            acc = corr * 1.0 / vf.size()[0]
            l = loss(E, batch_gt)
            print('epoch: %d | batch: %d | acc: %.2f | loss: %.2f' % (e, batch_counter, acc, l.cpu().data.numpy()))
            optim.zero_grad()
            l.backward()
            optim.step()
            torch.save(net.state_dict(), 'model/weights.pkl')
            if batch_counter % params['eval_freq'] == 0:
                torch.save(net.state_dict(), model_weights_path)
                print('Saving weights success.')
                # acc = eval(val_dataset, net)
                # print('epoch: %d | batch: %d | acc: %.2f' % (e, batch_counter, acc))


def correct(E, gt):
    pred = torch.eq(E.cpu().data, torch.zeros(E.data.size())).int()
    pred = pred.numpy()
    pred = pred * 2
    pred = pred - 1
    r = torch.eq(torch.from_numpy(pred).float(), gt.cpu().data)
    return torch.sum(r)


def eval(dataset, model):
    val_dataloader = DataLoader(dataset, batch_size=hyper_params['batch_size'])
    acc = 0
    for vf, wf, gt in val_dataloader:
        batch_vf = torch.autograd.Variable(vf)
        batch_wf = torch.autograd.Variable(wf)
        batch_gt = torch.autograd.Variable(gt)
        E = model(batch_vf, batch_wf)
        corr = correct(E, batch_gt)
        acc += corr
    return acc * 1.0 / val_dataloader.__len__()


if __name__ == '__main__':
    train()
