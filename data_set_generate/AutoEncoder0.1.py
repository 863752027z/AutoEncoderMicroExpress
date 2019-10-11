import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


def printGPU():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(0))


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


printGPU()
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=img_transform)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda:0')
model = autoencoder().to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for idx, (data, label) in enumerate(train_loader):
        img = data
        img = img.to(device)
        # =========forward===========
        output = model(img)
        #print(temp_img.shape)

        if epoch > 3:
            temp_img = output[0][0]
            input = img[0][0].cpu().detach().numpy()
            temp_img = temp_img.cpu().detach().numpy()
            temp_img = np.concatenate([input, temp_img])
            cv2.imshow('img', temp_img)
            cv2.waitKey(300)
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
