import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # encoder
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(3, 32, kernel_size=5, stride=2), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0), nn.ReLU())

        # decoder
        self.conv3d = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU())
        self.conv2d = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2), nn.ReLU())
        self.conv1d = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2))

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def decode(self, x):
        x = self.conv3d(x)
        x = self.conv2d(x)[:, :, 1:-1, 1:-1]
        x = self.conv1d(x)[:, :, :-1, :-1]
        x = nn.Sigmoid()(x)
        return x

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.autoencoder = Autoencoder()
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Conv2d(128, 10, kernel_size=3)

    def forward(self, x):
        encoded = self.autoencoder.encode(x)
        cx = self.dropout2(encoded)
        cx = self.fc1(cx)
        cx = cx.view(cx.size(0), -1)
        cx = F.softmax(cx, dim=1)
        cx = cx.clamp(min=1e-8)

        decoded = self.autoencoder.decode(encoded)
        return cx, decoded
