import pandas as pd
import os
from random import random, randrange

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset


class dataset(Dataset):
    def __init__(self):
        self.data_file_path = 'A1Benchmark'
        self.sliding_window_size = 64
        self.values, self.labels = self.load_data()

    def amplify(self, X, Y, rate):
        if random() < rate:
            index = randrange(0, self.sliding_window_size)
            X[index] += (sum(X[index + 1:]) + np.mean(X)) * (1 + np.var(X)) * np.random.normal(0, 1)
            Y = True
        return X, Y

    @staticmethod
    def slide_window(values, labels, window_size):
        assert len(values) == len(labels)
        Xs, Ys = [], []
        for i in range(len(values) - window_size + 1):
            # X, Y = self.amplify(values[i: i + window_size], any(labels[i: i + window_size]), 0.4)
            X, Y = values[i: i + window_size], any(labels[i: i + window_size])
            Xs.append(X)
            Ys.append(Y)
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


class Amplify(dataset):
    def __init__(self, train, rate=0.4):
        # super().__init__()
        self.values = train[0]
        self.labels = train[1]
        for i in range(train[0].size(0)):
            X = train[0][i].numpy()
            if random() < rate:
                index = randrange(0, len(X))
                self.values[i][index] += (sum(X[index + 1:]) + np.mean(X)) * (1 + np.var(X)) * np.random.normal(0, 1)
                self.labels[i] = True


def get_dataloader(batch_size=64):
    data = dataset()
    train_size = int(len(data) * 0.9)
    test_size = len(data) - train_size
    train_set, test_set = random_split(data, [train_size, test_size])
    train_set = Amplify(train_set.dataset[np.array(train_set.indices)], 0.45)

    print("Train Size: ", len(train_set), " Negative Rate: ",
          1 - sum(train_set.labels).item() / len(train_set))
    print("Test Size: ", len(test_set), " Negative Rate: ",
          1 - sum(test_set.dataset.labels[np.array(test_set.indices)]).item() / len(test_set))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader


if __name__ == '__main__':
    train, test = get_dataloader(64)
    for data in train:
        print(data[0].shape)  # [batch, 60]
        print(data[1].shape)  # [batch]
        exit()
