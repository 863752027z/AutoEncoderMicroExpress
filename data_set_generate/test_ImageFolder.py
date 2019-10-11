import cv2
import torch
from torchvision import datasets, transforms
import torch.utils.data as Data


file_path = 'F:/SAMM_FACE_CUT/SAMM'

#, transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
data_transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.ImageFolder(file_path, transform=data_transform)
print(dataset)
print(len(dataset))

'''
for i in range(len(dataset)):
    print(str(i) + ' label:', dataset[i][1])
    img = dataset[i][0].numpy()
    img = img.transpose((1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('img', img)
    cv2.waitKey(1)
'''

print(dataset.class_to_idx)

#train_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)
