import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from configuration import clstm_config
from preprocess import get_dataloader
from utils.evaluate import *
import torch.autograd as autograd   

class discriminator(nn.Module):

    # cnn + linear
    # dcgan
    def __init__(self):
        super(discriminator,self).__init__()

        # SR CNN
        self.main= nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=64, kernel_size = 4, stride=2,padding=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.LeakyReLU(0.2, inplace = True),
                #nn.BatchNorm1d(64),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size= 4, stride=2,padding=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.LeakyReLU(0.2, inplace = True),
                #nn.BatchNorm1d(128),
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 4, stride=1,padding=0),
         ) 
        self.linear = nn.Sequential(        
                nn.Linear(128 ,24),
                nn.Tanh(),
                nn.Linear(24,1),
                #nn.Sigmoid(),
                )

    def forward(self,x):
        x = self.main(x)
        #return x
        x = x.squeeze()
        return self.linear(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.Main = nn.Sequential(
            nn.ConvTranspose1d(1,64 , kernel_size = 4 , stride = 2 ,padding = 3, bias =False ),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64,32, kernel_size = 4 , stride =2 ,padding = 1,bias =False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.ConvTranspose1d(32,16, kernel_size = 4 , stride =2 ,padding = 1,bias =False),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.ConvTranspose1d(16,8, kernel_size = 4 , stride =2 ,padding = 1,bias =False),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.ConvTranspose1d(8,1, kernel_size = 4 , stride =2 ,padding = 1,bias =False),
            nn.Tanh()

        )
    
    def forward(self,x):
        return self.Main(x)


def cal_gradient_penalty(netD,real_data,fake_data,batchsize,device):
    
    LAMBDA = 10
    # print("Real_data shape ",real_data.size())
    # print("Fake_data shape ",fake_data.size())
    alpha = torch.rand(real_data.size())
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device) 

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    #print("penaly input shape",interpolates.shape)
    disc_interpolates = netD(interpolates)
    #
    #print("penaly output shape",disc_interpolates.shape)

    #disc_interpolates.backward(torch.ones(disc_interpolates.size()).to(device))

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]      

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    
    return gradient_penalty





if __name__ == "__main__":
    MAX_EPOCH = 100
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    netG = Generator().to(device)
    netD = discriminator().to(device)
    optimizerD = optim.RMSprop(netD.parameters(),lr = 1e-5 )
    optimizerG = optim.RMSprop(netG.parameters(),lr = 1e-5 )
    config = clstm_config()
    train, test = get_dataloader(config.batch_size)

    #----------------------------- train
    for epoch in range(100):
        print(f'====== epoch {epoch} ======')
        netD.train()
        netG.train()
        # ----------------------------- train
        #lossD, lossG = 0, 0

        for x,y  in train:
        # -------- train D

            #print('input shape == >',x.shape)
            for parm in netD.parameters():
                parm.data.clamp_(-0.01,0.01)

            x, y = x.to(device), y.to(device).float()
            x = x.unsqueeze(dim=1)
            one=torch.ones(x.shape[0],1).to(device)
            mone=-1*one.to(device)
            output = netD(x)
            output.backward(one)
         

            batch_size = x.shape[0]
            noise = torch.randn(batch_size,1,4,device = device)
            fake = netG(noise) 
            output1 = netD(fake.detach())
            output1.backward(mone)
            netD.zero_grad()


            #train with penalty
            penalty = cal_gradient_penalty(netD,x,fake.detach(),batch_size,device)
            penalty.backward()
            optimizerD.step()
            netD.zero_grad()

            # -------- train G

            output = netD(fake.detach())
            output.backward(one)
            optimizerG.step()
            netG.zero_grad()
 
        netD.eval()
        netG.eval()
       # print(f"Generator loss: {lossD} ; Discriminator loss: {lossG} .")
        if epoch >=20:
            with torch.no_grad():
                outs = []
                for x, y in test:
                    x, y = x.to(device).unsqueeze(dim=1), y.to(device)
                    output = netD(x).view(-1)
                    y_eval = (y==1).long()
                    outs.append([output, y_eval])
                print(f'test acc: {evaluate_acc(outs)}')
                print(f'test f1 score: {evaluate_f1score_threshold(outs,bg=0,ed=0.6,step=0.05)}')
