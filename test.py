import torch
from torch import nn

net1 = nn.Conv1d(in_channels=60, out_channels=120, kernel_size=64, stride=1)
net2 = nn.MaxPool1d(kernel_size=2, stride=2)

input = torch.randn((512, 60, 64))

print(input.shape)

out1 = net1(input)
print(out1.shape)

out2 = net2(out1)
print(out2.shape)
