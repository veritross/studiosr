import os
from typing import Dict

import gdown
import torch
import torch.nn as nn

from studiosr.models.common import MeanShift, Model, ResBlock, Upsampler, conv2d
from studiosr.utils import download


class EDSR(Model):
    def __init__(
        self,
        scale: int = 4,
        n_colors: int = 3,
        img_range: float = 1.0,
        n_feats: int = 256,
        n_resblocks: int = 32,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.n_colors = n_colors
        self.img_range = img_range
        self.n_feats = n_feats
        self.n_resblocks = n_resblocks
        self.res_scale = res_scale

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

    def get_model_config(self) -> Dict:
        config = super().get_model_config()
        config.update(
            dict(
                scale=self.scale,
                n_colors=self.n_colors,
                img_range=self.img_range,
                n_feats=self.n_feats,
                n_resblocks=self.n_resblocks,
                res_scale=self.res_scale,
            )
        )
        return config

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
    def from_pretrained(cls, scale: int = 4, dataset: str = "DIV2K") -> "EDSR":
        assert scale in [2, 3, 4]
        assert dataset in ["DIV2K", "DF2K"]

        model_dir = "pretrained"
        os.makedirs(model_dir, exist_ok=True)
        if dataset == "DIV2K":
            url = {
                "r32f256x2.pth": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt",
                "r32f256x3.pth": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt",
                "r32f256x4.pth": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt",
            }
            model = EDSR(scale=scale, img_range=255.0)
            file_name = f"r32f256x{scale}.pth"

            link = url[file_name]
            path = os.path.join(model_dir, file_name)
            if not os.path.exists(path):
                download(link, path)
        elif dataset == "DF2K":
            file_ids = {
                2: "1XEqY_nkUMdIid4lM9zAW99rYDx5eftBT",
                3: "1H1yFCFK14Z0DWAZHCtGXcWS6377fbkJE",
                4: "1TeH67rKNSR3dXs56aLqsA-UvLL3TZL-g",
            }
            model = EDSR(scale=scale)
            file_name = f"EDSRx{scale}.pth"
            file_id = file_ids[scale]
            path = os.path.join(model_dir, file_name)
            if not os.path.exists(path):
                gdown.download(id=file_id, output=path, quiet=False)

        pretrained = torch.load(path, map_location="cpu")
        model.load_state_dict(pretrained, strict=False)
        return model
