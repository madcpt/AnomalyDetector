import pandas as pd
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class dataset(Dataset):
    def __init__(self):
        self.data_file_path = 'A1Benchmark'
        self.sliding_window_size = 60
        self.values, self.labels = self.load_data()

    @staticmethod
    def slide_window(values, labels, window_size):
        assert len(values) == len(labels)
        Xs, Ys = [], []
        for i in range(len(values) - window_size + 1):
            Xs.append(values[i: i + window_size])
            Ys.append(any(labels[i: i + window_size]))
        return Xs, Ys

    def load_data(self):
        all_data = [[], []]
        for i in range(1, 68):
            pathname = os.path.join(self.data_file_path, "real_" + str(i) + ".csv")
            data = pd.read_csv(pathname)
            values, labels = data['value'].tolist(), data['is_anomaly'].tolist()
            values_max, values_min = max(values), min(values)
            values = [(value - values_min) / (values_max - values_min) for value in values]
            Xs, Ys = self.slide_window(values, labels, self.sliding_window_size)
            all_data[0] += Xs
            all_data[1] += Ys
        return torch.tensor(all_data[0]), torch.tensor(all_data[1]).long()

    def __getitem__(self, index):
        return self.values[index], self.labels[index]

    def __len__(self):
        return self.labels.size(0)


def get_dataloader(batch_size=64):
    data = dataset()
    train_size = int(len(data) * 0.9)
    test_size = len(data) - train_size
    train_set, test_set = random_split(data, [train_size, test_size])

    print("Train Size: ", len(train_set), " Negative Rate: ", 1 - sum(train_set.dataset.labels[np.array(train_set.indices)]).item()/len(train_set))
    print("Test Size: ", len(test_set), " Negative Rate: ", 1 - sum(test_set.dataset.labels[np.array(test_set.indices)]).item()/len(test_set))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader


if __name__ == '__main__':
    train, test = get_dataloader(64)
    for data in train:
        print(data[0].shape)  # [batch, 60]
        print(data[1].shape)  # [batch]
        exit()
