import torch
from torch import nn
from .aib import AIB2d


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, input_channels: int = 3, use_aib: bool = True):
        super().__init__()
        self.use_aib = use_aib
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            AIB2d(64) if use_aib else nn.Identity(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            AIB2d(128) if use_aib else nn.Identity(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.conv1(x)
        x = self.conv2(x)
        feat = self.conv3(x).flatten(1)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits
