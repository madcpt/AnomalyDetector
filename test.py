import torch
from torch import nn

# criterion = torch.nn.BCEWithLogitsLoss()
#
# output = torch.tensor([[1.0], [0.0]])
#
# target = torch.tensor([[1.0], [0]])
#
# print(criterion(output, target))

# m = nn.Conv1d(16, 60, 1, stride=2)
m = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
m = nn.MaxPool1d(3, stride=2)

input = torch.randn(1, 1, 10)
output = m(input)


print(m)
print(input.shape)
print(output.shape)