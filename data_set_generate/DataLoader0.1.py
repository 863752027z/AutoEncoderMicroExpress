import cv2
import numpy as np
import torch
import torch.utils.data as Data
from torchvision import transforms, models, datasets

BATCH_SIZE = 12
file_path = 'F:/ZLW_generate/s15/'
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
dataset = datasets.ImageFolder(file_path, transform=data_transform)
train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(dataset)
#print(train_loader)

epochs = 10

for i, (data, label) in enumerate(train_loader):
    #print(i)
    #print(data[0])
    #print(i, data[0].shape)
    print(i, data.shape, label)






for i in range(2000):
    print(dataset[i][0].size())
    print('label:', dataset[i][1])
    img = dataset[i][0].numpy()
    img = img.transpose((1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(img)
    cv2.imshow('img', img)
    cv2.waitKey(30)

print(dataset.class_to_idx)
