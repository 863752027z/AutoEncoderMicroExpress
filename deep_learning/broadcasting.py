import numpy as np
import torch

a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)
c = torch.cat([a, b], dim=0)
print(c.shape)

a1 = torch.rand(4, 3, 32, 32)
a2 = torch.rand(5, 3, 32, 32)
c = torch.cat([a1, a2], dim=0)
print(c.shape)
a = torch.rand(32, 8)
b = torch.rand(32, 8)
c = torch.stack([a, b], dim=0)
print(c.shape)