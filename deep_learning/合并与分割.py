import torch
import numpy as np

a = torch.rand(4, 32, 8)
b = torch.rand(4, 32, 8)
c = torch.cat([a, b], dim = 0)
print(c.shape)

a1 = torch.rand(4, 3, 32, 32)
a2 = torch.rand(5, 3, 32, 32)
a2 = torch.rand(4, 1, 32, 32)
d = torch.cat([a1, a2], dim=1)
print(d.shape)
a1 = torch.rand(4, 3, 16, 32)
a2 = torch.rand(4, 3, 16, 32)
b = torch.cat([a1, a2], dim=2)
print(b.shape)
print(torch.stack([a1, a2], dim=2).shape)
