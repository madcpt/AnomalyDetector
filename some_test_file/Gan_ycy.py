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
from some_test_file.baselstm_ycy import BaseLSTM as Classifier
from preprocess import get_dataloader
from configuration import clstm_config
from utils.evaluate import *

if __name__ == "__main__":
    MAX_EPOCH = 300
    PRE_EPOCH = 10
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
    netC = Classifier(device)
    netD = discriminator().to(device)
    # print(netG(noise).shape)

    loss_op = nn.BCELoss(reduction='sum')
    beta1 = 0.5

    optimizerC = torch.optim.Adam(netC.parameters(), lr=1e-4)
    optimizerD = optim.Adam(netD.parameters(), lr=1e-5, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-5, betas=(beta1, 0.999))
    config = clstm_config()
    test,train = get_dataloader(batch_size=512, rate=0.4, split=0.9, use_sr=True, normalize=True)

    ######## Pretrain Classifier ########
    for epoch in range(PRE_EPOCH):
        print(f'====== pretrain epoch {epoch} ======')
        netC.train()
        lossC = 0
        for x, y in train:
            optimizerC.zero_grad()
            x, y = x.to(device), y.float().to(device)
            output, _ = netC(x.unsqueeze(dim=-1))
            loss = loss_op(output, y)
            loss.backward()
            optimizerC.step()
            lossC += loss.item()

        netC.eval()
        print(f"Pretrained classifier loss: {lossC} .")
        with torch.no_grad():
            outs = []
            for x, y in test:
                x, y = x.to(device), y.to(device)
                output, _ = netC(x.unsqueeze(dim=-1))
                pred = (output >= 0.5).long()
                outs.append([pred, y])
            print(f'test acc: {calculate_acc(outs)}')
            print(f'test f1 score: {calculate_f1score(outs)}')

    input('Enter anything to continue training GAN ...')

    for epoch in range(MAX_EPOCH):
        print(f'====== epoch {epoch} ======')
        netC.train()
        netD.train()
        netG.train()
        # ----------------------------- train
        lossC, lossD, lossG = 0, 0, 0
        for x, y in train:
            batch_size = x.size()[0]
            optimizerC.zero_grad()
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            x, y = x.to(device), y.to(device).float()
            # -------- train D
            real_x, real_y = x.unsqueeze(dim=1), torch.ones(batch_size).float().to(device)

            noise = torch.randn(batch_size, 1, 4, device=device)
            fake_x = netG(noise)

            fake_y = torch.zeros(batch_size).float().to(device)

            shuffle_ind = np.arange(0, 2 * batch_size)
            np.random.shuffle(shuffle_ind)
            shuffle_ind = torch.tensor(shuffle_ind, device=device)
            D_x = torch.cat([real_x, fake_x.detach()], dim=0).index_select(dim=0, index=shuffle_ind)
            D_y = torch.cat([real_y, fake_y], dim=0).index_select(dim=0, index=shuffle_ind)
            D_output = netD(D_x).view(-1)
            errD = loss_op(D_output, D_y)
            errD.backward()
            optimizerD.step()
            netD.zero_grad()
            lossD += errD.item()

            # -------- train C
            output_r, _ = netC(x.unsqueeze(dim=-1))
            errC_real = loss_op(output_r.view(-1), y)
            errC_real.backward()
            output_f, _ = netC(fake_x.detach().transpose(1,2))
            errC_fake = loss_op(output_f.view(-1), real_y)
            errC_fake = errC_fake / (MAX_EPOCH-epoch+1)
            errC_fake.backward()
            optimizerC.step()
            netC.zero_grad()
            lossC += errC_fake.item()
            lossC += errC_real.item()

            # -------- train G
            output_gd = netD(fake_x).view(-1)
            errGd = loss_op(output_gd, real_y)
            output_gc,_ = netC(fake_x.transpose(1,2))
            errGc = loss_op(output_gc.view(-1), real_y)
            errG = errGc * errGd
            errG.backward()
            optimizerG.step()
            netG.zero_grad()
            lossG += errG.item()
            # 统计
            # break
        netC.eval()
        netD.eval()
        netG.eval()
        print(f"Generator loss: {lossD} ; Discriminator loss: {lossG} ; Classifier loss: {lossC}.")
        with torch.no_grad():
            outs = []
            for x, y in test:
                x, y = x.to(device), y.to(device)
                output, _ = netC(x.unsqueeze(dim=-1))
                pred = (output >= 0.5).long()
                outs.append([pred, y])
            print(f'test acc: {calculate_acc(outs)}')
            print(f'test f1 score: {calculate_f1score(outs)}')
