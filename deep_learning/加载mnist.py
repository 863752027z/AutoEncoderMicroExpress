import torchvision.datasets as datasets
trainset = datasets.MNIST(root='F:/ZLW/ZLW_lab/deep_learning/MNIST',
                          train=True,
                          download=True,
                          transform=None)