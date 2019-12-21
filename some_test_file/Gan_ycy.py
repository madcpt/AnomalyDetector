import argparse
import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseGan import discriminator, Generator
from preprocess import get_dataloader
from configuration import clstm_config
from utils.evaluate import *

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
    criterion = nn.BCELoss()
    beta1 = 0.5
    optimizerD = optim.Adam(netD.parameters(), lr=1e-5, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-5, betas=(beta1, 0.999))
    config = clstm_config()
    train, test = get_dataloader(batch_size=512, rate=0.4, split=0.9, use_sr=True, normalize=True)

    for epoch in range(MAX_EPOCH):
        print(f'====== epoch {epoch} ======')
        netD.train()
        netG.train()
        # ----------------------------- train
        lossD, lossG = 0, 0
        for x, y in train:
            # -------- train D
            x, y = x.to(device), y.to(device).float()
            # print(x.shape)
            x = x.unsqueeze(dim=1)
            # print(x.shape)
            # print(x.shape)
            output = netD(x).view(-1)
            # print(output.shape)
            # print(output.shape)

            #     # 正确错误是一半一半哈？
            errD_real = criterion(output, y)
            errD_real.backward()
            lossD += errD_real.item()

            batch_size = 64
            noise = torch.randn(batch_size, 1, 4, device=device)
            fake = netG(noise)
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
            # break
        netD.eval()
        netG.eval()
        print(f"Generator loss: {lossD} ; Discriminator loss: {lossG} .")
        with torch.no_grad():
            outs = []
            for x, y in test:
                x, y = x.to(device).unsqueeze(dim=1), y.to(device).float()
                output = netD(x)

                outs.append([output, y])
            print(f'test acc: {calculate_acc(outs)}')
            print(f'test f1 score: {calculate_f1score(outs)}')