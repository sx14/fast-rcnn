import os
import torch
from torch.utils.data import DataLoader
from dataset.pascaltest.PascalDataset import PascalDataset
from model import model
import train_config

hyper_params = {
    "visual_d": 4096,
    "embedding_d": 300,
    "epoch": 30,
    "batch_size": 200,
    "eval_freq": 1,
    "visual_feature_root": train_config.VISUAL_FEATURE_ROOT,
    "list_root": train_config.LIST_ROOT,
    "word_vec_path": "wordnet-embedding/dataset/word_vec_wn.h5"
}


def train():
    visual_feature_root = hyper_params["visual_feature_root"]
    train_list_path = os.path.join(hyper_params["list_root"], "train.txt")
    val_list_path = os.path.join(hyper_params["list_root"], "val.txt")
    word_vec_path = hyper_params["word_vec_path"]
    train_dataset = PascalDataset(visual_feature_root, train_list_path, word_vec_path)
    val_dataset = PascalDataset(visual_feature_root, val_list_path, word_vec_path)
    train_dataloader = DataLoader(train_dataset, batch_size=hyper_params["batch_size"], shuffle=True)
    net = model.HypernymVisual(hyper_params["visual_d"], hyper_params["embedding_d"])
    net.cuda()
    print(net)
    params = net.parameters()
    optim = torch.optim.Adam(params=params, lr=0.0001)
    loss = torch.nn.HingeEmbeddingLoss()
    batch_counter = 0
    for e in range(0, hyper_params["epoch"]):
        for vf, wf, gt in train_dataloader:
            batch_counter += 1
            batch_vf = torch.autograd.Variable(vf).cuda()
            batch_wf = torch.autograd.Variable(wf).cuda()
            batch_gt = torch.autograd.Variable(gt).cuda()
            E = net(batch_vf, batch_wf)
            corr = correct(E, batch_gt)
            acc = corr * 1.0 / vf.size()[0]
            l = loss(E, batch_gt)
            print("epoch: %d | batch: %d | acc: %.2f | loss: %.2f" % (e, batch_counter, acc, l.cpu().data.numpy()))
            optim.zero_grad()
            l.backward()
            optim.step()
            # if batch_counter % hyper_params["eval_freq"] == 0:
            #     acc = eval(val_dataset, net)
            #     print("epoch: %d | batch: %d | acc: %.2f | loss: %.2f" % (e, batch_counter, acc))


def correct(E, gt):
    pred = torch.eq(E.cpu().data, torch.zeros(E.data.size())).int()
    pred = pred.numpy()
    pred = pred * 2
    pred = pred - 1
    r = torch.eq(torch.from_numpy(pred).float(), gt.cpu().data)
    return torch.sum(r)


def eval(dataset, model):
    val_dataloader = DataLoader(dataset, batch_size=hyper_params["batch_size"])
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
