import math

import torch
import torch.nn as nn

from studiosr.models.common import BaseModule, Normalizer, conv2d


class VDSR(BaseModule):
    def __init__(
        self,
        scale: int = 4,
        n_colors: int = 3,
        channels: int = 64,
        img_range: float = 1.0,
        n_layers: int = 18,
    ) -> None:
        super().__init__()

        self.img_range = img_range
        self.normalizer = Normalizer(img_range=img_range)

        self.upsample = nn.Upsample(scale_factor=scale, mode="bicubic")
        layers = [conv2d(n_colors, channels, 3), nn.ReLU(True)]
        for _ in range(n_layers):
            layers.extend([conv2d(channels, channels, 3), nn.ReLU(True)])
        layers.append(conv2d(channels, n_colors, 3))
        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                stddev = math.sqrt(2 / (m.kernel_size[0] * m.kernel_size[1] * m.in_channels))
                nn.init.normal_(m.weight.data, 0.0, stddev)
                nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer.normalize(x)

        u = self.upsample(x)
        x = self.layers(u)
        x = x + u

        x = self.normalizer.unnormalize(x)
        return x
