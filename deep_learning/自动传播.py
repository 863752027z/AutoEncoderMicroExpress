import torch
import numpy as np


a = torch.randn(2, 3)
print(isinstance(a, torch.FloatTensor))
b = torch.tensor([1.1])
c = torch.tensor([1.1, 2.2])
data = np.ones(2)
d = torch.from_numpy(data)
e = torch.randn(2, 3)
f = torch.randn(1, 2, 3)
print(f.shape)
g = torch.randn(2, 3, 28, 28)
print(g)
print(a.shape)

a = np.array([2, 3.3])
b = torch.from_numpy(a)
print(b)
c = torch.empty(2, 3)
print(c)
d = torch.rand(3, 3)
print(d)
e = torch.rand_like(d)
print(e)
torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1))
print(torch.full([10], 0))
print(torch.full([2,3], 7))
print(torch.full([], 7))

torch.randperm(10) #生成10个随机的索引
a = torch.rand(2, 3)
b = torch.rand(2, 2)
idx = torch.randperm(2)
print(a)
print(b)
print(idx)
print(a[idx])













