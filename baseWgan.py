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

class discriminator(nn.Module):

    # cnn + linear
    # dcgan
    def __init__(self):
        super(discriminator,self).__init__()

        # SR CNN 指定的网络参数？
        self.main= nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=64, kernel_size = 4, stride=2,padding=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.LeakyReLU(0.2, inplace = True),
                nn.BatchNorm1d(64),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size= 4, stride=2,padding=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.LeakyReLU(0.2, inplace = True),
                nn.BatchNorm1d(128),
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 4, stride=1,padding=0),
         ) 
        self.linear = nn.Sequential(        
                nn.Linear(128 ,24),
                nn.Tanh(),
                nn.Linear(24,1),
             #modify 1   nn.Sigmoid(),
                )

    def forward(self,x):
        x = self.main(x)
        #return x
        x = x.squeeze()
        return self.linear(x)

class Generator(nn.Module):
    def __init__(self):
        #不对称，待修改
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







if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    #x = torch.randn(64,1,64)
    #y = torch.randint(0,2,(64,))
    # model = discriminator()
    # print(model(x).shape)
    #noise = torch.randn(64,1,4)
    netG = Generator().to(device)
    netD = discriminator().to(device)
    #print(netG(noise).shape)
    
    #modify 2
    #criterion = nn.BCELoss()
    #beta1 = 0.5

    #modify 3
    optimizerD = optim.RMSprop(netD.parameters(),lr = 1e-5 )
    optimizerG = optim.RMSprop(netG.parameters(),lr = 1e-5 )
    config = clstm_config()
    train, test = get_dataloader(config.batch_size)



 
     
    #----------------------------- train
    for x,y  in train:
        # -------- train D
        for parm in netD.parameters():
            parm.data.clamp_(-0.01,0.01)



        x, y = x.to(device), y.to(device).float()
        #print(x.shape)
        x = x.unsqueeze(dim=1)

        one=torch.ones(x.shape[0],1).to(device)
        mone=-1*one.to(device)
        #print(x.shape)
        #print(x.shape)
        output = netD(x)
        output.backward(one)
        #print(output.shape)
        #print(output.shape)

    #     # 正确错误是一半一半哈？
        #errD_real = criterion(output,y)
        #errD_real.backward()
        
        batch_size = x.shape[0]
        noise = torch.randn(batch_size,1,4,device = device)
        #print(type(noise),noise.shape)
        fake = netG(noise) 
        #label = torch.randint(0,1,(batch_size,),device = device).float()
        output1 = netD(fake.detach())
        output1.backward(mone)
        #errD_fake = criterion(netD(fake.detach()).view(-1),label)
        #errD_fake.backward()
        optimizerD.step()
        netD.zero_grad()

        # -------- train G
        # It is said that we should try G more @ycy
      
        output = netD(fake.detach())
        #label = torch.randint(1,2,(batch_size,),device = device).float()
        #errG = criterion(output,label)
        #errG.backward()
        output.backward(one)
        optimizerG.step()
        netG.zero_grad()
    # 统计
        #print("running")
        #break
    
