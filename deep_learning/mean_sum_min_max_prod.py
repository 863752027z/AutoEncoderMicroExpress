import torch

a = torch.arange(8).view(2, 4).float()
print(a)
print(a.min())
print(a.max())
print(a.mean())
print(a.prod())
print(a.sum())
print(a.argmax())
print(a.argmin())