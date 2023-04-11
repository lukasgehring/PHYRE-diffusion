import torch
import torch.nn.functional as F
from torch import nn


# Convolutional Variational Autoencoder
class ConvVAE(nn.Module):

    def __init__(self, kernel_size=4, init_channels=8, image_channels=3, latent_dim=64):
        super(ConvVAE, self).__init__()

        self.kernel_size = kernel_size
        self.init_channels = init_channels
        self.image_channels = image_channels
        self.latent_dim = latent_dim

        # 3 conv-layer and 2 dense-layer for encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=self.init_channels, kernel_size=self.kernel_size,
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=self.init_channels, out_channels=self.init_channels * 2, kernel_size=self.kernel_size,
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=self.init_channels * 2, out_channels=self.init_channels * 4, kernel_size=self.kernel_size,
            stride=2, padding=1
        )

        self.encFC1 = nn.Linear(2048, self.latent_dim)
        self.encFC2 = nn.Linear(2048, self.latent_dim)

        # 1 dense-layer and 3 conv-layer for decoder
        self.decFC1 = nn.Linear(self.latent_dim, 2048)

        self.dec1 = nn.ConvTranspose2d(
            in_channels=self.init_channels * 4, out_channels=self.init_channels * 2, kernel_size=self.kernel_size,
            stride=2, padding=1
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=self.init_channels * 2, out_channels=self.init_channels, kernel_size=self.kernel_size,
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=self.init_channels, out_channels=image_channels, kernel_size=self.kernel_size,
            stride=2, padding=1
        )

    def encoder(self, x):
        # Input is an image of size 64x64 and output are mu and logVarience to generate z
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))

        x = x.view(-1, 2048)

        mu = self.encFC1(x)
        logVar = self.encFC2(x)

        return mu, logVar

    def reparameterize(self, mu, logVar):
        # Sample mu + std * eps to generate z
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        # Input is the latent representation z and output is the reconstruction of size 64x64
        x = F.relu(self.decFC1(z))

        x = x.view(-1, 32, 8, 8)

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))

        return x

    def forward(self, x):
        mu, logVar = self.encoder(x)

        z = self.reparameterize(mu, logVar)

        reconstruction = self.decoder(z)

        return reconstruction, mu, logVar
