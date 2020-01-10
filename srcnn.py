import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from configuration import clstm_config
from preprocess import get_dataloader


# import sranodec as anom


class SRCNN(nn.Module):
    def __init__(self, device):
        super(SRCNN, self).__init__()
        self.device = device
        self.bidirectional = True
        self.n_layers = 2
        self.rnn_hidden = 32
        # less than period
        # self.amp_window_size=24
        # (maybe) as same as period
        # self.series_window_size=24
        # a number enough larger than period
        # self.score_window_size=64
        # self.spec = anom.Silency(self.amp_window_size, self.series_window_size, self.score_window_size)
        # self.sr_layer = self.spec.generate_anomaly_score

        self.dropout = nn.Dropout(0.2).to(device)
        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=64, kernel_size=64, stride=1, padding=64),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  ).to(device)
        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=64, stride=1),
                                  # nn.MaxPool1d(kernel_size=2, stride=2),
                                  ).to(device)

        self.classifier = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 2)).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, inputs):
        # inputs [batch, len, 1]
        # inputs = inputs.squeeze(dim=-1)
        # print(inputs.shape)
        # inputs = inputs + 0.00001* torch.from_numpy(np.ones((inputs.shape[0],inputs.shape[1]))).cuda().float()
        # sr = []
        # for input in inputs:
        #     sr.append(torch.tensor(self.sr_layer(input.cpu().numpy())).unsqueeze(dim=0).to(self.device))
        # inputs = torch.cat(sr, dim=0)
        # inputs = inputs.unsqueeze(dim=-1).float()
        # inputs [batch, len, 1]
        batch_size, len, _ = inputs.shape
        inputs = inputs.reshape((batch_size, 1, len))
        # print(inputs.shape)
        inputs = self.cnn1(inputs)
        # print(inputs.shape)
        inputs = self.cnn2(inputs)
        # print(inputs.shape)
        inputs = inputs.squeeze()
        # exit()

        pred = self.classifier(inputs)
        # print(pred.shape)
        pred = torch.sigmoid(pred).squeeze(dim=-1)
        return pred

    def run_epoch(self, dataset, train=True, draw=False):
        if train:
            self.train()
        else:
            self.eval()
        l, tp, tn, fp, fn = 0, 0.01, 0.01, 0.01, 0.01
        for x, y in dataset:
            x, y = x.to(self.device), y.to(self.device)
            out = self.forward(x.unsqueeze(dim=-1))
            loss = self.criterion(out, y)
            if train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            pred = out.detach().argmax(dim=-1)
            l += loss.item()
            tp += ((pred == y) & (pred == 1)).sum().item()
            tn += ((pred == y) & (pred != 1)).sum().item()
            fp += ((pred != y) & (pred == 1)).sum().item()
            fn += ((pred != y) & (pred != 1)).sum().item()
            fp_ = (pred != y) & (pred == 1)
            if draw:
                for i, ti in enumerate(fp_):
                    if ti == 1:
                        print(x[i])
                        print(y[i])
                        exit()
        cnt = tp + tn + fp + fn
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return "loss=%.4f, acc=%.4f, precision=%.4f, recall=%.4f, F1=%.4f" % (
            l / cnt, (tp + tn) / cnt, precision, recall, f1)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(device)

    config = clstm_config()
    train, test = get_dataloader(config.batch_size, rate=0.1, split=0.7, use_sr=False, normalize=True)

    model = SRCNN(device)
    model = model
    print(model)

    for epoch in range(100):
        print('epoch %d:' % epoch, end=' ')
        # train
        model.run_epoch(train, True)
        # test
        print('test: ', model.run_epoch(test, False, epoch==10))
