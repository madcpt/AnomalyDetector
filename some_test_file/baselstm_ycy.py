import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from configuration import clstm_config
from preprocess import get_dataloader


class BaseLSTM(nn.Module):
    def __init__(self, device):
        super(BaseLSTM, self).__init__()
        self.device = device
        self.bidirectional = True
        self.n_layers = 2
        self.rnn_hidden = 32
        self.dropout = nn.Dropout(0.2).to(device)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=1),
        ).to(device)
        self.lstm = nn.LSTM(1, self.rnn_hidden, self.n_layers, bidirectional=self.bidirectional, batch_first=True).to(
            device)
        self.classifier = nn.Sequential(nn.Linear(2 * self.rnn_hidden if self.bidirectional else self.rnn_hidden, 80),
                                        nn.ReLU(),
                                        nn.Linear(80, 100),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(100, 1)).to(device)


    def get_init_state(self, inputs):
        batch_size = inputs.size(0)
        layers = 2 * self.n_layers if self.bidirectional else self.n_layers
        h0 = torch.zeros((layers, batch_size, self.rnn_hidden)).to(self.device)
        c0 = torch.zeros((layers, batch_size, self.rnn_hidden)).to(self.device)
        return h0, c0

    def forward(self, inputs):
        # inputs [batch, len, 1]
        batch_size, len, _ = inputs.shape
        # inputs = inputs.reshape((batch_size, 1, len))
        # inputs = self.cnn(inputs)
        # inputs = inputs.reshape((batch_size, -1, 1))

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
        pred = torch.sigmoid(pred).squeeze(dim=-1)
        return pred, output#batch*len*(hid_dim*2)
