from torch.utils.data import Dataset
import os
import pickle


class PascalDataset(Dataset):
    def __init__(self, raw_data_root, list_path):
        self._raw_data_root = raw_data_root
        self._index = []
        self._labels = []
        with open(list_path, 'r') as list_file:
            list = list_file.read().splitlines()
        for item in list:
            item_info = item.split(' ')
            feature_file = item_info[0]
            item_id = item_info[1]
            item_label = item_info[2]
            # package id, offset
            self._labels.append(item_label)
            self._index.append([feature_file, item_id])
        pass

    def __getitem__(self, index):
        feature_path = os.path.join(self._raw_data_root, self._index[index][0])
        with open(feature_path) as feature_file:
            features = pickle.load(feature_file)
        item_data = features[self._index[index][1]]
        item_label = self._labels[index]
        return {'feature': item_data, 'label': item_label}

    def __len__(self):
        return len(self._index)
