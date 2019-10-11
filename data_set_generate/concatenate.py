import numpy as np

A = np.arange(12).reshape(3, 2, 2)
print(A)

A = A.transpose((1, 2, 0))
print(A)
print(A.shape)