import torch.nn as nn
from torchvision import models
import torch

class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )


    def forward(self, x):
        x = self.model(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            # 3x28x28 -> 8x14x14
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            # 8x14x14 -> 16x7x7
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            # 16x7x7 -> 32x4x4
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU())

        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (32, 4, 4)),
            # 32x4x4 -> 16x7x7
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            # 16x7x7 -> 8x14x14
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            # 8x14x14 -> 3x28x28
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar