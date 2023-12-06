from typing import Dict

import torch
import torch.nn as nn

from .base import BaseModule
from .edsr import MeanShift, Upsampler, conv2d


class CALayer(nn.Module):
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


class RCAB(nn.Module):
    def __init__(self, n_feat: int, kernel_size: int, reduction: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            conv2d(n_feat, n_feat, kernel_size),
            nn.ReLU(True),
            conv2d(n_feat, n_feat, kernel_size),
            CALayer(n_feat, reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, n_feat: int, kernel_size: int, reduction: int, n_resblocks: int) -> None:
        super().__init__()
        m_body = [RCAB(n_feat, kernel_size, reduction) for _ in range(n_resblocks)]
        self.body = nn.Sequential(*m_body, conv2d(n_feat, n_feat, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res += x
        return res


class RCAN(BaseModule):
    def __init__(
        self,
        scale: int = 4,
        n_colors: int = 3,
        img_range: float = 1.0,
        n_resgroups: int = 10,
        n_resblocks: int = 20,
        n_feats: int = 64,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.img_range = img_range
        self.sub_mean = MeanShift(img_range)
        self.add_mean = MeanShift(img_range, sign=1)

        kernel_size = 3
        self.head = nn.Sequential(conv2d(n_colors, n_feats, kernel_size))
        m_body = [ResidualGroup(n_feats, kernel_size, reduction, n_resblocks) for _ in range(n_resgroups)]
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
        import os
        import zipfile

        import gdown

        pretrained_dir = "pretrained"
        rcan_dir = "models_ECCV2018RCAN"
        rcan_path = os.path.join(pretrained_dir, rcan_dir)
        if not os.path.exists(rcan_path):
            os.makedirs(pretrained_dir, exist_ok=True)
            id = "10bEK-NxVtOS9-XSeyOZyaRmxUTX3iIRa"
            path = os.path.join(pretrained_dir, "RCAN.zip")
            gdown.download(id=id, output=path, quiet=False)
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(pretrained_dir)
            os.remove(path)
        model_path = os.path.join(rcan_path, f"RCAN_BIX{scale}.pt")
        model = RCAN(scale=scale, img_range=255.0)
        model.load_state_dict(torch.load(model_path), True)
        return model
