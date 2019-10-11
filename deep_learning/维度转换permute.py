import torch
import numpy as np

a = torch.rand(4, 3, 28, 28)
a1 = a.transpose(1, 3)
print(a1.shape)
b = torch.rand(4, 3, 28, 32)
b1 = b.transpose(1, 3)
print(b1.shape)
print('b:', b.shape)
c = b.transpose(1, 3).transpose(1, 2)
print('c:', c.shape)
d = b.permute(0, 2, 3, 1)
print('d:', d.shape)
