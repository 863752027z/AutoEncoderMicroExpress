import torch
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
from torchvision import datasets, models, transforms

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
file_path = 'F:/ZLW_generate/s15/15_0101disgustingteeth/'
dataset = datasets.ImageFolder(file_path, transform=data_transform)
train_loader = Data.DataLoader()