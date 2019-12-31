import argparse
import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration import clstm_config
from preprocess import get_dataloader
from utils.evaluate import *


class discriminator(nn.Module):

    # cnn + linear
    # dcgan
    def __init__(self):
        super(discriminator, self).__init__()

        # SR CNN 指定的网络参数？
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
        )
        self.linear = nn.Sequential(
            nn.Linear(128, 24),
            nn.Tanh(),
            nn.Linear(24, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.main(x)
        # return x
        x = x.squeeze()
        return self.linear(x)


class Generator(nn.Module):
    def __init__(self):
        # 不对称，待修改
        super(Generator, self).__init__()
        self.Main = nn.Sequential(
            nn.ConvTranspose1d(1, 64, kernel_size=4, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()

        )

    def forward(self, x):
        return self.Main(x)


if __name__ == "__main__":
    MAX_EPOCH = 100
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # x = torch.randn(64,1,64)
    # y = torch.randint(0,2,(64,))
    # model = discriminator()
    # print(model(x).shape)
    # noise = torch.randn(64,1,4)
    netG = Generator().to(device)
    netD = discriminator().to(device)
    # print(netG(noise).shape)
    criterion = nn.BCELoss(reduction='sum')
    beta1 = 0.5
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(beta1, 0.999))
    config = clstm_config()
    train, test = get_dataloader(batch_size=512, rate=0, split=0.9, use_sr=False, normalize=True)

    for epoch in range(MAX_EPOCH):
        print(f'====== epoch {epoch} ======')
        netD.train()
        netG.train()
        # ----------------------------- train
        lossD, lossG = 0, 0
        for x, y in train:
            batch_size = x.size(0)
            # -------- train D
            x, y = x.to(device), y.to(device).float()
            x = x.unsqueeze(dim=1)  # x: [b, 1, 64]
            output = netD(x).squeeze()  # output: [b]

            #     # 正确错误是一半一半哈？
            errD_real = criterion(output, y)
            errD_real.backward()
            lossD += errD_real.item()

            noise = torch.randn(batch_size, 1, 4, device=device)
            fake = netG(noise)  # fake: x

            label = torch.randint(0, 1, (batch_size,), device=device).float()
            errD_fake = criterion(netD(fake.detach()).view(-1), label)
            errD_fake.backward()
            optimizerD.step()
            netD.zero_grad()
            lossD += errD_fake.item()

            # -------- train G
            output = netD(fake).view(-1)
            label = torch.randint(1, 2, (batch_size,), device=device).float()
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
            netG.zero_grad()
            lossG += errG.item()
            # 统计
            # brea
        netD.eval()
        netG.eval()
        print(f"Generator loss: {lossD} ; Discriminator loss: {lossG} .")
        with torch.no_grad():
            outs = []
            for x, y in test:
                x, y = x.to(device).unsqueeze(dim=1), y.to(device)
                output = netD(x).view(-1)
            #     pred = (output >= 0.5).long()
            #     outs.append([pred, y])
            # print(f'test acc: {calculate_acc(outs)}')
            # print(f'test f1 score: {calculate_f1score(outs)}')
                outs.append([output, y])
            print(f'test acc: {evaluate_acc(outs)}')
            print(f'test f1 score: {evaluate_f1score_threshold(outs)}')
