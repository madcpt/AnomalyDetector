import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from configuration import clstm_config
from preprocess import get_dataloader


def loss_f(output, target):
    result = 10000 * torch.log(output) * target + (1 - torch.log(output)) * (1 - target)
    return result.sum()


class BaseLSTM(nn.Module):
    def __init__(self, device):
        super(BaseLSTM, self).__init__()
        self.device = device
        self.bidirectional = True
        self.n_layers = 2
        self.rnn_hidden = 32
        self.dropout = nn.Dropout(0.2).to(device)
        self.classifier = nn.Sequential(nn.Linear(60, 30), nn.Tanh(), nn.Linear(30, 2)).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_init_state(self, inputs):
        batch_size = inputs.size(0)
        layers = 2 * self.n_layers if self.bidirectional else self.n_layers
        h0 = torch.zeros((layers, batch_size, self.rnn_hidden)).to(self.device)
        c0 = torch.zeros((layers, batch_size, self.rnn_hidden)).to(self.device)
        return h0, c0

    def forward(self, inputs):
        # inputs [batch, len, 1]
        # h0, c0 = self.get_init_state(inputs)
        # print(inputs.shape, h0.shape, c0.shape)
        # inputs = self.dropout(inputs)
        # output, (hn, cn) = self.lstm(inputs)
        # output, (hn, cn) = self.lstm(inputs, (h0, c0))
        # print(output.shape, hn.shape, cn.shape)
        # if self.bidirectional:
        #     hidden = torch.cat([hn[-2], hn[-1]], dim=-1)
        # else:
        #     hidden = hn[-1]
        pred = self.classifier(inputs)
        # print(pred.shape)
        pred = torch.sigmoid(pred).squeeze(dim=-1)
        return pred

    def run_epoch(self, dataset, train=True):
        if train:
            self.train()
        else:
            self.eval()
        l, hit, cnt = 0, 0, 0
        for x, y in dataset:
            x, y = x.to(self.device), y.to(self.device)
            out = self.forward(x)
            loss = self.criterion(out, y)
            if train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            pred = out.detach().argmax(dim=-1)
            l += loss.item()
            hit += (pred == y).sum().item()
            cnt += y.size(0)
        return "loss=%.4f, acc=%.4f" % (l/cnt, hit/cnt)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(device)

    config = clstm_config()
    train, test = get_dataloader(config.batch_size)

    model = BaseLSTM(device)
    model = model
    print(model)

    print('test: ', model.run_epoch(test, False))
    for epoch in range(100):
        print('epoch %d:' % epoch)
        # train
        print('train: ', model.run_epoch(train, True))
        # test
        print('test: ', model.run_epoch(test, False))
