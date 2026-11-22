import torch
from torch import nn


class AIB2d(nn.Module):
    """Lightweight Adaptive Information Bottleneck gate.

    This module is intentionally simple for a first reproducible version:
    it learns channel-wise affine modulation. The information-theoretic
    regularization/scoring is handled at the client/aggregation level.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x + self.beta
