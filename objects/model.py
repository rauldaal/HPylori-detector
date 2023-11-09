import torch
from torch import nn

class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        self.capa1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
        self.capa2 = nn.MaxPool2d(2, stride=2, return_indices=True)  # b, 16, 5, 5
        self.capa3 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
        self.capa4 = nn.MaxPool2d(2, stride=1, return_indices=True)  # b, 8, 2, 2

        self.capa5 = nn.MaxUnpool2d(2, stride=1)  # b, 8, 3, 3
        self.capa6 = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1)
        self.capa7 = nn.MaxUnpool2d(2, stride= 2)
        self.capa8 = nn.ConvTranspose2d(16, 1, 3, stride=3, padding=1)

    def encoder(self, x):
      x = self.capa1(x)
      x = torch.relu(x)
      x, self.indices1 = self.capa2(x)
      x = self.capa3(x)
      x = torch.relu(x)
      x, self.indices2 = self.capa4(x)
      return x

    def decoder(self, x):
      x = self.capa5(x, self.indices2)
      x = self.capa6(x)
      x = torch.relu(x)
      x = self.capa7(x, self.indices1)
      x = self.capa8(x)
      x = torch.sigmoid(x)
      return x
