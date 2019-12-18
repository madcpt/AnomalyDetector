import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from configuration import clstm_config
from DataLoader import DataLoader


class BaseLSTM(nn.Module):
    def __init__(self):
        super(BaseLSTM, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        print(self.device)
        self.bidirectional = True
        self.n_layers = 2
        self.rnn_hidden = 64
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
        self.max1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
        self.max2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.lstm = nn.LSTM(1, self.rnn_hidden, self.n_layers, bidirectional=self.bidirectional, batch_first=True)
        self.classifier = nn.Linear(2 * self.rnn_hidden if self.bidirectional else self.rnn_hidden, 2)
        self.criterion = nn.CrossEntropyLoss()

    def get_init_state(self, inputs):
        batch_size = inputs.size(0)
        layers = 2 * self.n_layers if self.bidirectional else self.n_layers
        h0 = torch.zeros((layers, batch_size, self.rnn_hidden)).to(self.device)
        c0 = torch.zeros((layers, batch_size, self.rnn_hidden)).to(self.device)
        return h0, c0

    def forward(self, inputs):
        # inputs [batch, len, 1]
        batch_size, len, _ = inputs.shape
        inputs = inputs.reshape((batch_size, 1, len))
        inputs = self.max1(self.conv1(inputs))
        inputs = self.max2(self.conv2(inputs))
        inputs = inputs.reshape((batch_size, -1, 1))

        h0, c0 = self.get_init_state(inputs)
        # print(inputs.shape, h0.shape, c0.shape)
        inputs = self.dropout(inputs)
        # output, (hn, cn) = self.lstm(inputs)
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        # print(output.shape, hn.shape, cn.shape)
        if self.bidirectional:
            hidden = torch.cat([hn[-2], hn[-1]], dim=-1)
        else:
            hidden = hn[-1]
        pred = self.classifier(hidden)
        # print(pred.shape)
        return pred

    def eval_epoch(self, test_data):
        model.eval()
        x, y = torch.from_numpy(test_data[0]).to(self.device).float(), torch.from_numpy(test_data[1]).to(
            self.device).long()
        output = self.forward(x)
        pred = output.argmax(dim=-1)
        try:
            assert pred.shape == y.shape
        except AssertionError:
            print(pred.shape, y.shape)
            exit()
        return (pred == y).sum().item() / pred.size(0)

    def train_batch(self, optimizer, x, y):
        model.train()
        x, y = torch.from_numpy(x).to(self.device).float(), torch.from_numpy(y).to(self.device).long()
        output = self.forward(x)
        # print(output.shape)
        # print(y.shape)
        loss = self.criterion(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pred = output.argmax(dim=-1)
        # print(pred.shape)
        target = y.long()
        hit = (pred == target).sum().item()
        cnt = pred.size(0)
        return loss.item(), hit, cnt


if __name__ == '__main__':
    config = clstm_config()
    dataloader = DataLoader(config)
    test = dataloader.get_test_data()

    model = BaseLSTM()
    model = model.to(model.device)
    print(model)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    test_data = dataloader.get_test_data()
    print('test: acc=%.5f' % (model.eval_epoch(test_data)))
    for epoch in range(100):
        # train
        l, hit, cnt = 0.0, 0, 0
        for j in tqdm(range(dataloader.num_batches)):
            data = dataloader.next_batch()
            l_j, hit_j, cnt_j = model.train_batch(optim, *data)
            l += l_j
            hit += hit_j
            cnt += cnt_j
        print('train: loss=%.5f, acc=%.5f' % (l / cnt, hit / cnt))
        # test
        test_data = dataloader.get_test_data()
        print('test: acc=%.5f' % (model.eval_epoch(test_data)))
