import torch

a = torch.full([4], 1/4.)
print(a)
b = a*torch.log2(a)
print(b)
c = -(a*torch.log2(a)).sum()
print(c)
a = torch.tensor([0.1, 0.1, 0.1, 0.7])
b = -(a*torch.log2(a)).sum()
print(b)
a = torch.tensor([0.001, 0.001, 0.001, 0.999])
b = -(a*torch.log2(a)).sum()
print(b)