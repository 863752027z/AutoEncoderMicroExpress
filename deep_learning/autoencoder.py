import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms, datasets
from nn import AE
import visdom

def main():
    mnist_train = datasets.MNIST('mnist', True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)

    mnist_test = datasets.MNIST('mnist', False, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)
    x, _ = iter(mnist_train).next()
    print('x:', x.shape)


    device = torch.device('cuda')
    model = AE().to(device)
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    viz = visdom.Visdom()
    for epoch in range(1000):
        for batchidx, (x, _) in enumerate(mnist_train):
            x = x.to(device)
            x_hat = model(x)
            loss = criteon(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch, 'loss', loss.item())
        x, _ = iter(mnist_test).next()
        x = x.to(device)
        with torch.no_grad():
            x_hat = model(x)
        viz.images(x, nrow=8, win='x', opts=dict(title='x'))
        viz.images(x, nrow=9, win='x_hat', opts=dict(title='x_hat'))

if __name__ == '__main__':
    main()