import torch.nn as nn
from torchvision import models

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