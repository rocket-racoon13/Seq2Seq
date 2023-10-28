import torch
import torch.nn as nn

a = torch.randn(2, 3, 5)
b = nn.Softmax(dim=-1)
print(torch.LongTensor(2))