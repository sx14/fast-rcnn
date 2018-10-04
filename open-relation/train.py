import torch
from torch.utils.data import Dataset, DataLoader
from dataset.pascal_test.PascalDataset import PascalDataset
from model import model


hyper_params = {
    "visual_d": 4096,
    "embedding_d": 300,
    "epoch": 30,
    "batch_size": 200
}


def train():
    train_visual_feature_root = ''
    train_list_path = ''
    dataset = PascalDataset(train_visual_feature_root, train_list_path)
    dataloader = DataLoader(dataset, batch_size=hyper_params["batch_size"], shuffle=True)
    net = model.VisualFeatureEmbedding(hyper_params["visual_d"], hyper_params["embedding_d"])
    params = net.parameters()
    optim = torch.optim.Adam(params=params, lr=0.01)
    loss = model.PartialOrderLoss()
    for e in range(0, hyper_params["epoch"]):
        for visual_features, labels in dataloader:
            

if __name__ == '__main__':
    train()