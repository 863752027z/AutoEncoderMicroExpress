import torch

a = torch.randn(4, 10)
print(a)
print(a.max(dim=1))
print(a.argmax(dim=1))
a.max(dim=1, keepdim=True)
a.argmax(dim=1, keepdim=True)

print(a.topk(3, dim=1))
print(a.topk(3, dim=1, largest=False))
print(a.kthvalue(8, dim=1))
print(a>0)
print(torch.eq(a, a))