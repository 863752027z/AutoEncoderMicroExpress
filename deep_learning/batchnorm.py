import torch
import torch.nn as nn
import numpy


x = torch.rand(100, 16, 784)
layer = nn.BatchNorm1d(16)
out = layer(x)
print(layer.running_mean)
print(layer.running_var)


