import torch
x = torch.randn(3, requires_grad=True)
y = x*2
z = y*y*3
print(z)