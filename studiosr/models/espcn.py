import math

import torch
from torch import nn

from studiosr.models.common import Model, Normalizer


class ESPCN(Model):
    def __init__(
        self,
        scale: int = 4,
        n_colors: int = 3,
        img_range: float = 1.0,
        channels: int = 64,
    ) -> None:
        super().__init__()
        hidden_channels = channels // 2
        out_channels = int(n_colors * (scale**2))

        self.img_range = img_range
        self.normalizer = Normalizer(img_range=img_range)

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
        x = self.normalizer.normalize(x)

        x = self.feature_maps(x)
        x = self.sub_pixel(x)

        x = self.normalizer.unnormalize(x)
        return x
