import torch
from torch import nn

from studiosr.models.common import Model, Normalizer


class SRCNN(Model):
    """Super-Resolution Convolutional Neural Network (SRCNN).

    Args:
        - scale (int): Upsampling scale factor. Default is 4.
        - n_colors (int): Number of input image channels. Default is 3.
        - img_range (float): Range of pixel values in the input image. Default is 1.0.
        - residual (bool): Flag indicating whether to use residual learning. Default is False.

    Attributes:
        - img_range (float): Range of pixel values in the input image.
        - residual (bool): Flag indicating whether to use residual learning.
        - mean (torch.Tensor): Mean value used for input normalization.
        - upsample (nn.Upsample): Upsampling layer using bicubic interpolation.
        - layers (nn.Sequential): Sequential layers defining the SRCNN architecture.

    Methods:
        - forward(x: torch.Tensor) -> torch.Tensor: Forward pass of the SRCNN model.

    Note:
        The SRCNN model architecture upsamples the input image using bicubic interpolation,
        performs patch extraction and representation (Conv2d / ReLU) for extracting the spatial features.
        It translates extraced features using defined non-linear mapping (Conv2d / ReLU),
        then reconstructs them to produce final high-resolution image (Conv2d).
        Refer to the http://google.github.io/styleguide/pyguide.html.
    """

    def __init__(
        self,
        scale: int = 4,
        n_colors: int = 3,
        img_range: float = 1.0,
        residual: bool = False,
    ) -> None:
        super().__init__()

        self.img_range = img_range
        self.residual = residual
        self.normalizer = Normalizer(img_range=img_range)

        self.upsample = nn.Upsample(scale_factor=scale, mode="bicubic")
        self.layers = nn.Sequential(
            nn.Conv2d(n_colors, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_colors, kernel_size=5, padding=5 // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer.normalize(x)

        u = self.upsample(x)
        x = self.layers(u)
        if self.residual:
            x = x + u

        x = self.normalizer.unnormalize(x)
        return x
