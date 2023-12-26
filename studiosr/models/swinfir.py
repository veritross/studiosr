from typing import Dict, List

import torch
import torch.nn as nn

from studiosr.models.swinir import SwinIR


class FourierUnit(nn.Module):
    def __init__(self, embed_dim: int, fft_norm: str = "ortho") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.conv_layer = torch.nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.fft_norm = fft_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        fft_dim = (-2, -1)
        fft_x = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)  # (batch, c, h, w/2+1)
        fft_x = torch.cat((fft_x.real, fft_x.imag), dim=1)  # (batch, c*2, h, w/2+1)

        fft_x = self.conv_layer(fft_x)  # (batch, c*2, h, w/2+1)
        fft_x = self.lrelu(fft_x)

        fft_x_re, fft_x_im = fft_x.split(self.embed_dim, dim=1)  # (batch, c, h, w/2+1)
        fft_x = torch.complex(fft_x_re, fft_x_im)

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(fft_x, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)  # (batch, c, h, w)

        return output


class SpectralTransform(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.conv_before_fft = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.fu = FourierUnit(embed_dim // 2)

        self.conv_after_fft = torch.nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_before_fft(x)
        output = self.fu(x)
        output = self.conv_after_fft(output + x)
        return output


class SpatialB(nn.Module):
    def __init__(self, embed_dim: int, red: int = 1) -> None:
        super(SpatialB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // red, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim // red, embed_dim, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        return out + x


class SFB(nn.Module):
    def __init__(self, embed_dim: int, red: int = 1) -> None:
        super(SFB, self).__init__()
        self.S = SpatialB(embed_dim, red)
        self.F = SpectralTransform(embed_dim)
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.S(x)
        f = self.F(x)
        out = torch.cat([s, f], dim=1)
        out = self.fusion(out)
        return out


class SwinFIR(SwinIR):
    def __init__(
        self,
        scale: int = 4,
        n_colors: int = 3,
        img_range: float = 1.0,
        embed_dim: int = 180,
        depths: List[int] = [6, 6, 6, 6, 6, 6],
        num_heads: List[int] = [6, 6, 6, 6, 6, 6],
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        upsampler: str = "pixelshuffle",
    ) -> None:
        super().__init__(
            scale=scale,
            n_colors=n_colors,
            img_range=img_range,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            upsampler=upsampler,
            resi_connection=SFB,
        )
        self.conv_after_body = SFB(embed_dim)

    def get_training_config(self) -> Dict:
        training_config = dict(
            batch_size=32,
            learning_rate=0.0002,
            beta1=0.9,
            beta2=0.99,
            weight_decay=0.0,
            max_iters=500000,
            gamma=0.5,
            milestones=[250000, 400000, 450000, 475000],
            bfloat16=False,
        )
        return training_config
