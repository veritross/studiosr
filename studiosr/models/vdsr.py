import math
import os
from typing import Dict

import gdown
import torch
import torch.nn as nn

from studiosr.models.common import Model, Normalizer, conv2d


class VDSR(Model):
    def __init__(
        self,
        scale: int = 4,
        n_colors: int = 3,
        img_range: float = 1.0,
        channels: int = 64,
        n_layers: int = 18,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.n_colors = n_colors
        self.img_range = img_range
        self.channels = channels
        self.n_layers = n_layers

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

    def get_model_config(self) -> Dict:
        config = super().get_model_config()
        config.update(
            dict(
                channels=self.channels,
                n_layers=self.n_layers,
            )
        )
        return config

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
        )
        return training_config

    @classmethod
    def from_pretrained(cls, scale: int = 4) -> "VDSR":
        assert scale in [2, 3, 4]
        file_ids = {
            2: "1eQnGseT3SqQirB5ueAFfsClhLlpeoUOX",
            3: "1wXOnLFf7rWglzzVMzYSVrb5Po79vUUq3",
            4: "1Q5DKy7oAQbgGqxI-unxPy9X3GcHwZokC",
        }
        model = VDSR(scale=scale)
        file_name = f"VDSRx{scale}.pth"
        model_dir = "pretrained"
        os.makedirs(model_dir, exist_ok=True)
        file_id = file_ids[scale]
        path = os.path.join(model_dir, file_name)
        if not os.path.exists(path):
            gdown.download(id=file_id, output=path, quiet=False)
        pretrained = torch.load(path, map_location="cpu")
        model.load_state_dict(pretrained)
        return model
