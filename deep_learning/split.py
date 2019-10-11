import torch

a = torch.rand(32, 8)
b = torch.rand(32, 8)
c = torch.stack([a, b], dim=0)
print(c.shape)

aa, bb = c.split([1,1], dim=0)
print(aa.shape)
print(bb.shape)

aa, bb = c.split(1, dim=0)
print(aa.shape)
print(bb.shape)