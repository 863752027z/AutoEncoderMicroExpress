import numpy as np
import cv2
import torch
from torch import autograd
import torchvision

print(torch.cuda.is_available())
x = torch.tensor(2.0)
a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
c = torch.tensor(3.0, requires_grad=True)

y = a**2 *x + b * x + c
print('before:', a.grad, b.grad, c.grad)
grads = autograd.grad(y, [a, b, c])
print('after:', grads[0], grads[1], grads[2])


model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = torch.nn.Linear(512, 100)
optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
print('end')
















































