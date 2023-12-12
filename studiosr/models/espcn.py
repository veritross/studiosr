import math

import torch
from torch import nn

from studiosr.models.common import BaseModule


class ESPCN(BaseModule):
    def __init__(
        self,
        scale: int = 4,
        n_colors: int = 3,
        channels: int = 64,
        img_range: float = 1.0,
    ) -> None:
        super().__init__()
        hidden_channels = channels // 2
        out_channels = int(n_colors * (scale**2))

        self.img_range = img_range
        if n_colors == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        # Feature mapping
        self.feature_maps = nn.Sequential(
            nn.Conv2d(n_colors, channels, (5, 5), (1, 1), (2, 2)),
            nn.Tanh(),
            nn.Conv2d(channels, hidden_channels, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),
        )

        # Sub-pixel convolution layer
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(scale),
        )

        # Initial model weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data, 0.0, 0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(
                        module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel()))
                    )
                    nn.init.zeros_(module.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.feature_maps(x)
        x = self.sub_pixel(x)

        x = x / self.img_range + self.mean
        return x
