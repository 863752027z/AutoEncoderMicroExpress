import torch
x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5, 5, requires_grad=True)
print(x)
print(y)
z = x**2 + y**2
z.backward(torch.zeros_like(x))
print(x.grad)