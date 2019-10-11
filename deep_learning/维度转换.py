import torch
import numpy as np

a = torch.rand(4, 1, 28, 28)
print(a.shape)
b = a.view(4, 28*28)
a.reshape(4, 28*28)
print(a.shape)
print(b.shape)
print(a.shape)
c = a.unsqueeze(0)
print(c.shape)
a = torch.tensor([1.2, 2.3])
b = a.unsqueeze(-1)
print(b)


a = torch.randn(1, 2)
print(a.shape)
print(a)
b = torch.randn(2, 3)

c = b.t()
print(c)
