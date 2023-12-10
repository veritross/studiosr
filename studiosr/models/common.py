import math
from typing import List

import numpy as np
import torch
import torch.nn as nn


class BaseModule(nn.Module):
    @torch.no_grad()
    def inference(self, image: np.ndarray) -> np.ndarray:
        scale = 255.0 if self.img_range == 1.0 else 1.0
        device = next(self.parameters()).get_device()
        device = torch.device("cpu") if device < 0 else device
        x = image.transpose(2, 0, 1).astype(np.float32) / scale
        x = torch.from_numpy(x).unsqueeze(0)
        x = x.to(device)
        output = self.forward(x) * scale
        y = output.squeeze().cpu().numpy().round().clip(0, 255)
        return y.astype(np.uint8).transpose(1, 2, 0)


def conv2d(in_channels: int, out_channels: int, kernel_size: int) -> nn.Module:
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2))


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        img_range: float,
        rgb_mean: List[float] = [0.4488, 0.4371, 0.4040],
        rgb_std: List[float] = [1.0, 1.0, 1.0],
        sign: int = -1,
    ) -> None:
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * img_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, scale: int, n_feats: int, num_out_ch: int = None) -> None:
        m = []
        if num_out_ch is not None:
            m.append(conv2d(n_feats, (scale**2) * num_out_ch, 3))
            m.append(nn.PixelShuffle(scale))
        elif (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv2d(n_feats, 4 * n_feats, 3))
                m.append(nn.PixelShuffle(2))
        else:
            m.append(conv2d(n_feats, (scale**2) * n_feats, 3))
            m.append(nn.PixelShuffle(scale))
        super().__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, n_feats: int, kernel_size: int, res_scale: float = 1.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            conv2d(n_feats, n_feats, kernel_size),
            nn.ReLU(True),
            conv2d(n_feats, n_feats, kernel_size),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ChannelAttention(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
