import torch
import numpy as np
import matplotlib.pyplot as plt

x = torch.linspace(-10, 10, 60)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
plt.ylim((0, 1))
sigmoid = torch.sigmoid(x)
plt.plot(x.numpy(), sigmoid.numpy())
plt.show()