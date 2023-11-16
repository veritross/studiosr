import torch
from torch import nn

from .base import BaseModule


class SRCNN(BaseModule):
    def __init__(
        self,
        scale: int = 4,
        n_colors: int = 3,
        img_range: float = 1.0,
        residual: bool = False,
    ) -> None:
        super(SRCNN, self).__init__()

        self.img_range = img_range
        self.residual = residual
        if n_colors == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.upsample = nn.Upsample(scale_factor=scale, mode="bicubic")
        self.layers = nn.Sequential(
            nn.Conv2d(n_colors, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_colors, kernel_size=5, padding=5 // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        u = self.upsample(x)
        x = self.layers(u)
        if self.residual:
            x = x + u

        x = x / self.img_range + self.mean
        return x
