import math
import os
from typing import Dict, List

import torch
import torch.nn as nn

from .base import BaseModule, download_weights


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
    def __init__(self, scale: int, n_feats: int) -> None:
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv2d(n_feats, 4 * n_feats, 3))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(conv2d(n_feats, 9 * n_feats, 3))
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
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


class EDSR(BaseModule):
    def __init__(
        self,
        scale: int = 4,
        n_colors: int = 3,
        img_range: float = 1.0,
        n_resblocks: int = 32,
        n_feats: int = 256,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.img_range = img_range
        self.sub_mean = MeanShift(img_range)
        self.add_mean = MeanShift(img_range, sign=1)

        kernel_size = 3
        self.head = nn.Sequential(conv2d(n_colors, n_feats, kernel_size))
        m_body = [ResBlock(n_feats, kernel_size, res_scale) for _ in range(n_resblocks)]
        self.body = nn.Sequential(*m_body, conv2d(n_feats, n_feats, kernel_size))
        self.tail = nn.Sequential(Upsampler(scale, n_feats), conv2d(n_feats, n_colors, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def get_training_config(self) -> Dict:
        training_config = dict(
            batch_size=16,
            learning_rate=0.0001,
            beta1=0.9,
            beta2=0.99,
            weight_decay=0.0,
            max_iters=1000000,
            gamma=0.5,
            milestones=[200000, 400000, 600000, 800000],
        )
        return training_config

    @classmethod
    def from_pretrained(cls, scale: int = 4) -> nn.Module:
        url = {
            "r16f64x2.pth": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt",
            "r16f64x3.pth": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt",
            "r16f64x4.pth": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt",
            "r32f256x2.pth": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt",
            "r32f256x3.pth": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt",
            "r32f256x4.pth": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt",
        }
        model = EDSR(scale=scale, img_range=255.0)
        file_name = f"r32f256x{scale}.pth"
        model_dir = "pretrained"
        os.makedirs(model_dir, exist_ok=True)
        link = url[file_name]
        path = os.path.join(model_dir, file_name)
        download_weights(link, path)
        pretrained = torch.load(path)
        model.load_state_dict(pretrained, strict=False)
        return model
