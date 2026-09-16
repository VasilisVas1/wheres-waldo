"""A small CNN for the stage-1 "is this 64x64 crop Waldo?" classifier.

Kept deliberately small: with only 886 base patches (19 source scenes), a
large network would overfit almost immediately. Four conv blocks with
batchnorm + dropout, global average pooling instead of a big flatten+FC head
(fewer parameters, less to overfit), one logit output for BCEWithLogitsLoss.
"""

from __future__ import annotations

import torch
from torch import nn


class PatchClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 16),
            self._conv_block(16, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x).squeeze(-1)  # raw logits, shape (B,)
