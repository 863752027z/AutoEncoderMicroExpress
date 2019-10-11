import torch

prob = torch.randn(4, 10)
idx = prob.topk(dim=1, k=3)
print(idx)
print(idx[0])
print(idx[1])
idx =idx[1]

temp = torch.arange(10)
print(temp)
label = temp + 100
print(label)

print(idx.long())
g = torch.gather(label.expand(4, 10), dim=1, index=idx.long())
print(g)