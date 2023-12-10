import math

import torch
import torch.nn as nn
from torch import Tensor

from studiosr.models.common import BaseModule


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.rcb(x)

        x = torch.add(x, identity)

        return x


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample_block(x)

        return x


class SRResNet(BaseModule):
    def __init__(
        self,
        scale: int = 4,
        n_colors: int = 3,
        img_range: float = 1.0,
        channels: int = 64,
        num_rcb: int = 16,
    ) -> None:
        super().__init__()

        self.img_range = img_range
        if n_colors == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        # Low frequency information extraction layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_colors, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # High frequency information extraction block
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # High-frequency information linear fusion layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        # zoom block
        upsampling = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        else:
            raise NotImplementedError(f"scale `{scale}` is not supported.")
        self.upsampling = nn.Sequential(*upsampling)

        # reconstruction block
        self.conv3 = nn.Conv2d(channels, n_colors, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)

        x = x / self.img_range + self.mean
        return x
