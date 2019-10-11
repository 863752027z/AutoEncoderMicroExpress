import torch

ran = torch.rand(2, 3)
grad = ran*15
print(ran)
print(grad)
max = grad.max()
print(max)
median = grad.median()
print(median)
grad.clamp(10)
