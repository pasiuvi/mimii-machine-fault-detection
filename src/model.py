"""
Question 2a – the ANN: an ImageNet-pretrained EfficientNet-B0 adapted to log-mel
spectrograms, with a small head that also receives the machine TYPE (pump/fan).

    log-mel [B, 1, 128, T]
      -> stem conv 3x3, stride 2   (1 input channel: pretrained RGB filters summed)
      -> 7 MBConv stages (16 blocks, depthwise-separable convs, squeeze-excitation, SiLU)
      -> 1x1 conv to 1280 channels, BatchNorm, SiLU
      -> global average pooling over frequency AND time  -> 1280-d clip embedding
      -> concat one-hot machine type (2)                  -> 1282
      -> Linear 1282->128, ReLU, Dropout 0.3
      -> Linear 128->1  = one logit  (sigmoid -> fault score in [0, 1])

Loss: BCEWithLogitsLoss(pos_weight = negatives/positives on the training set),
so a missed fault costs about 4x more than a false alarm during training,
matching the cost asymmetry in Question 1. The alarm threshold is chosen later
on the validation set at a 5 % false-alarm rate (not 0.5).
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src import config


class FaultCNN(nn.Module):
    def __init__(self, pretrained: bool = True, n_types: int = 2,
                 hidden: int = config.HEAD_HIDDEN, dropout: float = config.DROPOUT):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1
                net = efficientnet_b0(weights=weights)
            except Exception as e:  # no internet -> still build the same architecture
                warnings.warn(f"Could not download ImageNet weights ({e.__class__.__name__}); "
                              "building with random weights.")
                weights, net = None, efficientnet_b0(weights=None)
        else:
            net = efficientnet_b0(weights=None)
        self.pretrained = weights is not None

        # Adapt the stem from 3 (RGB) to 1 (spectrogram) input channel.
        # Summing the RGB filters keeps the pretrained edge/texture detectors.
        old = net.features[0][0]
        stem = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                         stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            stem.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        net.features[0][0] = stem

        self.features = net.features          # 9 modules: stem, 7 MBConv stages, final 1x1 conv
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.n_types = n_types
        self.head = nn.Sequential(
            nn.Linear(1280 + n_types, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)          # [B, 1280]

    def forward(self, x: torch.Tensor, mtype: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.embed(x), F.one_hot(mtype, self.n_types).float()], dim=1)
        return self.head(z).squeeze(1)                          # logit, [B]

    # ---- fine-tuning depth (used in the Q3b study) ----
    def set_trainable_stages(self, n_last: int) -> None:
        """Train only the last `n_last` of the 9 feature modules (+ the head).
        n_last=0 -> head only; n_last=9 -> full fine-tuning."""
        n_total = len(self.features)
        for i, block in enumerate(self.features):
            trainable = i >= n_total - n_last
            for p in block.parameters():
                p.requires_grad = trainable


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def make_loss(pos_weight: float) -> nn.Module:
    """Weighted binary cross-entropy on the logit (numerically stable)."""
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
