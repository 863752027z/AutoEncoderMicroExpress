import torch
epoch = 100
num_epochs = 100
loss = 0.1

print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss))

x = torch.arange(512).view(512, 1, 1)
y = torch.arange(512).view(512, 1, 1)
print(x.shape[1])
x = x.view(x.shape[0], x.shape[1])
print(x.shape)