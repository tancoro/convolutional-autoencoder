import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # encoder
        self.conv1 = nn.Sequential(nn.ZeroPad2d((1,0,1,0)),
                              nn.Conv2d(3, 32, kernel_size=5, stride=2),
                              nn.BatchNorm2d(32),
                              nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1,1,1,1)),
                              nn.Conv2d(32, 64, kernel_size=5, stride=2),
                              nn.BatchNorm2d(64),
                              nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
                              nn.BatchNorm2d(128),
                              nn.ReLU())
        # self.fc1 = nn.Conv2d(128, 10, kernel_size=3)

        # decoder
        # self.fc2 = nn.Sequential(nn.ConvTranspose2d(10, 128, kernel_size=3),
        #                    nn.ReLU())
        self.conv3d = nn.Sequential(
                              nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
                              nn.BatchNorm2d(64), nn.ReLU())
        self.conv2d = nn.Sequential(
                              nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
                              nn.BatchNorm2d(32), nn.ReLU())
        self.conv1d = nn.Sequential(
                              nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2),
                              nn.BatchNorm2d(3))


    def forward(self, x):
        x = self.conv1(x)
        # print(f'conv1: {x.size()}')
        x = self.conv2(x)
        # print(f'conv2: {x.size()}')
        encoded = self.conv3(x)
        # print(f'conv3: {encoded.size()}')

        decoded = self.conv3d(encoded)
        # print(f'conv3d: {decoded.size()}')
        decoded = self.conv2d(decoded)[:, :, 1:-1, 1:-1]
        # print(f'conv2d: {decoded.size()}')
        decoded = self.conv1d(decoded)[:, :, :-1, :-1]
        # print(f'conv1d: {decoded.size()}')
        decoded = nn.Sigmoid()(decoded)
        return encoded, decoded

