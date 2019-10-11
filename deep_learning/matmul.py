import torch

a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
c = torch.matmul(a, b)
print(c.shape)

#broadcasting
b = torch.rand(4, 1, 64, 32)
d = torch.matmul(a, b)
print(d.shape)



a = torch.exp(torch.ones(2, 2))
print(a)

a = torch.tensor(3.14)
print(a.floor())
print(a.ceil())
print(a.trunc())
print(a.frac())

a = torch.tensor(3.49)
print(a.round())
a = torch.tensor(3.5)
print(a.round())
