from torch.utils.data import Dataset
import os
import pickle


class MyDataset(Dataset):
    def __init__(self, raw_data_root, list_path, package_capacity):
        self._package_capacity = package_capacity
        self._raw_data_root = raw_data_root
        self._index = []
        self._labels = []
        with open(list_path, 'r') as list_file:
            list = list_file.readlines()
        for item in list:
            item_info = item.split(' ')
            item_pos = item_info[0]
            item_label = item_info[1]
            # package id, offset
            self._labels.append(item_label)
            self._index.append(item_pos.split('/'))
        pass

    def __getitem__(self, index):
        package_path = os.path.join(self._raw_data_root, self._index[index][0])
        with open(package_path) as data_package:
            data = pickle.load(data_package)
        item_data = data[self._index[index][0]]
        item_label = self._labels[index]
        return {'feature': item_data, 'label': item_label}

    def __len__(self):
        return len(self._index)
