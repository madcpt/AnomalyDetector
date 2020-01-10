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
from preprocess import get_gan_data
from utils.evaluate import *
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
            nn.ConvTranspose1d(1, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU(True),
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


def l1_loss(input, target):
    return torch.mean(torch.abs(input - target))

def l1_loss_lyt(input, target):
    input = input.view(input.size(0), 1)
    return torch.abs(input - target)

def l2_loss(input, target):
    return torch.mean(torch.pow((input - target), 2))


class netG(nn.Module):
    def __init__(self):
        super(netG, self).__init__()
        self.encoder1 = discriminator()
        self.decoder = Generator()
        self.encoder2 = discriminator()

    def forward(self, x):
        latent_i = self.encoder1(x)
        # print(latent_i.size())
        latent_i = latent_i.unsqueeze(dim=1)
        gen_imag = self.decoder(latent_i)
        # print(gen_imag.size())
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o


class GANomaly(nn.Module):
    def __init__(self, device):
        super(GANomaly, self).__init__()
        self.device = device
        self.netD = discriminator().to(device)
        self.netG = netG().to(device)
        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()
        self.lr = 0.00001
        self.l1 = l1_loss_lyt
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr)
        self.w_adv = 1
        self.w_con = 1
        self.w_enc = 1

    def forward_g(self, input):
        self.real_label = torch.ones(input.size()[0], 1, device=self.device)
        self.fake_label = torch.zeros(input.size()[0], 1, device=self.device)
        # To DO 这里可以把discriminator 和 netD 分开 提取高维的latent space feature vector
        self.input = input
        self.fake, self.latent_i, self.latent_o = self.netG(input)
        self.score = self.l1(self.latent_i, self.latent_o)

    def forward_d(self, input):
        self.feat_real = self.netD(input)
        self.pred_real = self.feat_real
        self.feat_fake = self.netD(self.fake.detach())
        self.pred_fake = self.feat_fake

    def backward_g(self, input):
        self.err_g_adv = self.l_adv(self.netD(input)[1], self.netD(self.fake)[1])
        self.err_g_con = self.l_con(self.fake, input)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv * self.w_adv + \
                     self.err_g_con * self.w_con + \
                     self.err_g_enc * self.w_enc
        self.err_g.backward(retain_graph=True)

    def backward_d(self, input):
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()

    def optimize_params(self, input, is_train=True):
        # Forward-pass
        self.forward_g(input)
        self.forward_d(input)

        if is_train:
            # Backward-pass
            # netg
            self.optimizerG.zero_grad()
            self.backward_g(input)
            self.optimizerG.step()

            # netd
            self.optimizerD.zero_grad()
            self.backward_d(input)
            self.optimizerD.step()


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(device)

    # Train the network
    # x = torch.rand(512, 1, 64, device=device)
    network = GANomaly(device)
    # network.optimize_params(x)

    # Test the network with discriminator
    train, test = get_gan_data(batch_size=512, contaminate_rate=0.4, split=0.8, use_sr=False, normalize=True)

    network.eval()
    X, Y = [], []
    for x, y in test:
        x = x.unsqueeze(dim=1).float().to(device)
        network.optimize_params(x, False)
        X.append(network.score.cpu().detach())
        Y.append(y)
    X = torch.cat(X, dim=0).squeeze()
    Y = torch.cat(Y, dim=0)
    # print(X.shape)
    # print(Y.shape)
    f1 = normalize_and_get_f1_score(Y, X)
    print(f1)

    for epoch in range(100):
        network.train()
        for x, y in train:
            x = x.unsqueeze(dim=1).float().to(device)
            network.optimize_params(x)
        # print('train: ', network.score)

        network.eval()
        X, Y = [], []
        for x, y in test:
            x = x.unsqueeze(dim=1).float().to(device)
            network.optimize_params(x, False)
            X.append(network.score.cpu().detach())
            Y.append(y)
        X = torch.cat(X, dim=0).squeeze()
        Y = torch.cat(Y, dim=0)
        # print(X.shape)
        # print(Y.shape)
        f1 = normalize_and_get_f1_score(Y, X)
        # f1s = evaluate_f1score_threshold
        print(f1)
