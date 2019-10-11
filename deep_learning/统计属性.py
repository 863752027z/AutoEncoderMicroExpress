import torch

a = torch.full([8], 1)
b = a.view(2, 4)
c = a.view(2, 2, 2)
print(a)
print(b)
print(c)

#计算1范数
a1 = a.norm(1)
b1 = b.norm(1)
c1 = c.norm(1)

#计算2范数
a2 = a.norm(2)
b2 = b.norm(2)
c2 = c.norm(2)


