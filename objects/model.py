import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 32x32
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # 16x16
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32
            nn.Sigmoid()  # Sigmoid values between (0, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x