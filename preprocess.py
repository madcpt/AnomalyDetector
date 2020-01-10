import pandas as pd
import os
from random import random, randrange
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset


class dataset(Dataset):
    def __init__(self, normalize=True):
        self.data_file_path = 'A1Benchmark'
        self.sliding_window_size = 64
        self.values, self.labels = self.load_data(normalize=normalize)

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

    def load_data(self, normalize=True):
        all_data = [[], []]
        total_points, anomaly_points = 0, 0
        for i in range(1, 68):
            pathname = os.path.join(self.data_file_path, "real_" + str(i) + ".csv")
            data = pd.read_csv(pathname)
            values, labels = data['value'].tolist(), data['is_anomaly'].tolist()
            values_max, values_min = max(values), min(values)
            if normalize:
                values = [(value - values_min) / (values_max - values_min) for value in values]
            Xs, Ys = self.slide_window(values, labels, self.sliding_window_size)
            # print(values)
            # print(labels)
            # exit()
            total_points += len(values)
            anomaly_points += sum(labels)
            all_data[0] += Xs
            all_data[1] += Ys
        print(total_points, anomaly_points)
        return torch.tensor(all_data[0]).float(), torch.tensor(all_data[1]).long()

    def __getitem__(self, index):
        return self.values[index], self.labels[index]

    def __len__(self):
        return self.labels.size(0)


# noinspection PyMissingConstructor
class Amplify(dataset):
    def __init__(self, train, rate=0.4, use_sr=False):
        self.values = train[0].float()
        self.labels = train[1]
        for i in range(len(self)):
            X = train[0][i].numpy()
            if random() < rate:
                index = randrange(0, len(X))
                self.values[i][index] += (sum(X[index + 1:]) + np.mean(X)) * (1 + np.var(X)) * np.random.normal(0, 1)
                self.labels[i] = True
        if use_sr:
            print('Using SR, get yourself a cup of coffee.')
            from SR.silency import Silency
            amp_window_size = 4  # less than period
            series_window_size = 4  # (maybe) as same as period
            score_window_size = 16  # a number enough larger than period
            spec = Silency(amp_window_size, series_window_size, score_window_size)
            for i in range(len(self)):
                self.values[i] = torch.from_numpy(spec.generate_anomaly_score(self.values[i].numpy())).float()


# noinspection PyMissingConstructor
class GANTrainData(dataset):
    def __init__(self, train, use_sr=False):
        value, label = train
        index = label.eq(0).nonzero().squeeze()
        self.values = value[index].float()
        self.labels = label[index]

        if use_sr:
            print('Using SR, get yourself a cup of coffee.')
            from SR.silency import Silency
            amp_window_size = 4  # less than period
            series_window_size = 4  # (maybe) as same as period
            score_window_size = 16  # a number enough larger than period
            spec = Silency(amp_window_size, series_window_size, score_window_size)
            for i in range(len(self)):
                self.values[i] = torch.from_numpy(spec.generate_anomaly_score(self.values[i].numpy())).float()


# noinspection PyMissingConstructor
class GANTestData(dataset):
    def __init__(self, test, contaminate_rate=0.2, use_sr=False):
        value, label = test
        normal_index = label.eq(0).nonzero().squeeze()
        abnormal_index = label.eq(1).nonzero().squeeze()
        abnormal_cnt = min(int(normal_index.size(0) / (1 - contaminate_rate) * contaminate_rate),
                           abnormal_index.size(0))
        normal_cnt = int(abnormal_cnt / contaminate_rate) - abnormal_cnt
        abnormal_index = abnormal_index[torch.randperm(abnormal_index.size(0))[:abnormal_cnt]]
        normal_index = normal_index[torch.randperm(normal_index.size(0))[:normal_cnt]]
        all_index = torch.cat([abnormal_index, normal_index])
        all_index = all_index[torch.randperm(all_index.size(0))]
        self.values = value[all_index]
        self.labels = label[all_index]

        if use_sr:
            print('Using SR, get yourself a cup of coffee.')
            from SR.silency import Silency
            amp_window_size = 4  # less than period
            series_window_size = 4  # (maybe) as same as period
            score_window_size = 16  # a number enough larger than period
            spec = Silency(amp_window_size, series_window_size, score_window_size)
            for i in range(len(self)):
                self.values[i] = torch.from_numpy(spec.generate_anomaly_score(self.values[i].numpy())).float()


def show_stat(train_set, test_set):
    print("Total Stat: ", sum(train_set.labels).item() + sum(test_set.labels).item(), len(train_set) + len(test_set),
          "Negative Rate: ",
          1 - (sum(train_set.labels).item() - sum(test_set.labels).item()) / (len(train_set) + len(test_set)))
    print("Train Size: ", len(train_set), " Negative Rate: ", 1 - sum(train_set.labels).item() / len(train_set))
    print("Test Size: ", len(test_set), " Negative Rate: ", 1 - sum(test_set.labels).item() / len(test_set))


def get_dataloader(batch_size=64, rate=0.4, split=0.9, use_sr=False, normalize=True):
    data = dataset(normalize)
    train_size = int(len(data) * split)
    test_size = len(data) - train_size
    train_set, test_set = random_split(data, [train_size, test_size])
    train_set = Amplify(train_set.dataset[np.array(train_set.indices)], rate=rate, use_sr=use_sr)
    test_set = Amplify(test_set.dataset[np.array(test_set.indices)], rate=rate, use_sr=use_sr)
    # TODO
    # with open('data/test.pl', 'wb') as f:
    #     pickle.dump(test_set, f)
    # with open('data/test.pl', 'rb') as f:
    #     test_set = pickle.load(f)
    show_stat(train_set, test_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader


def get_gan_data(batch_size=64, contaminate_rate=0.2, split=0.8, use_sr=False, normalize=True):
    data = dataset(normalize)
    train_size = int(len(data) * split)
    test_size = len(data) - train_size
    train_set, test_set = random_split(data, [train_size, test_size])
    train_set = GANTrainData(train_set.dataset[np.array(train_set.indices)], use_sr=use_sr)
    test_set = GANTestData(test_set.dataset[np.array(test_set.indices)], contaminate_rate=contaminate_rate,
                           use_sr=use_sr)
    show_stat(train_set, test_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader


if __name__ == '__main__':
    train, test = get_gan_data(batch_size=512, contaminate_rate=0.5, split=0.9, use_sr=False, normalize=True)
    # train, test = get_dataloader(batch_size=512, rate=0.4, split=0.9, use_sr=False, normalize=True)
    # for data in train:
    #     print(data[0].shape)  # [batch, 60]
    #     print(data[1].shape)  # [batch]
    #     exit()
