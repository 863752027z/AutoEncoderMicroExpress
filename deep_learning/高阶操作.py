import torch

cond = torch.tensor([[0.6769, 0.7271], [0.8884, 0.4136]])
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])
c = torch.where(cond > 0.5, a, b)
print(c)
