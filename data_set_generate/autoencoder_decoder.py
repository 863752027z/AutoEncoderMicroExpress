import cv2
import os
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


dataset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=img_transform)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


for i, data in enumerate(train_loader):
   print(i)
exit(0)

for epoch in range(num_epochs):
    for idx, (data, label) in enumerate(train_loader):
        img = data
        img = Variable(img).cuda()
        # =========forward===========
        output = model(img)
        print(output.shape)
        temp_img = output[0][0]
        print(temp_img.shape)
        temp_img = temp_img.cpu().detach().numpy()
        cv2.imshow('img', temp_img)

        cv2.waitKey(0)
        loss = criterion(output, img)
        # =========backward=========
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ==============log=========
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))
        '''
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './dc_img/image_{}.png'.format(epoch))
        '''
    # torch.save(model.state_dict(), './conv_autoencoder.pth')
