import numpy as np
import torch


x = torch.rand(3, 4)
mask = x.ge(0.5)
print(mask)
