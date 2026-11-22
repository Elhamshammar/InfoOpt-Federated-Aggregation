"""ResNet18SmallImage — the Stage-4 second backbone.

A SMALL-IMAGE ResNet-18 variant (NOT an unmodified ImageNet-stem
ResNet-18):
  - residual block counts [2, 2, 2, 2] at standard ResNet-18 widths
    [64, 128, 256, 512];
  - 3x3, stride-1, padding-1 input stem (no 7x7 stride-2 stem);
  - NO initial max-pooling;
  - BatchNorm throughout;
  - adaptive global average pooling;
  - dataset-specific linear classification head exposed as
    `model.classifier` (so `model.classifier.in_features` works and the
    existing signal code is unchanged);
  - randomly initialized; NO pretrained weights or downloads.

Interface contract (shared with SmallCNN):
  logits = model(x)
  logits, feat = model(x, return_features=True)
"""
import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class ResNet18SmallImage(nn.Module):
    def __init__(self, num_classes: int = 10, input_channels: int = 3,
                 use_aib: bool = False):
        super().__init__()
        if use_aib:
            raise ValueError("Stage-4 requires AIB disabled; "
                             "ResNet18SmallImage has no AIB modules")
        self.use_aib = False
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, 3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        feat = self.pool(out).flatten(1)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits
