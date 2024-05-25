import os
from typing import Dict

import gdown
import torch
import torch.nn as nn

from studiosr.models.common import MeanShift, Model, Upsampler, conv2d
from studiosr.models.rcan import ResidualGroup


class LAM_Module(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma * out + x
        out = out.view(m_batchsize, -1, height, width)
        return out


class CSAM_Module(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.chanel_in = in_dim
        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))

        out = self.gamma * out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x


class HAN(Model):
    def __init__(
        self,
        scale: int = 4,
        n_colors: int = 3,
        img_range: float = 1.0,
        n_feats: int = 64,
        n_resblocks: int = 20,
        n_resgroups: int = 10,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.img_range = img_range
        self.scale = scale
        self.n_colors = n_colors
        self.img_range = img_range
        self.n_feats = n_feats
        self.n_resblocks = n_resblocks
        self.n_resgroups = n_resgroups
        self.reduction = reduction

        self.sub_mean = MeanShift(img_range)
        self.add_mean = MeanShift(img_range, sign=1)

        kernel_size = 3
        self.head = nn.Sequential(conv2d(n_colors, n_feats, kernel_size))
        m_body = [ResidualGroup(n_feats, kernel_size, reduction, n_resblocks) for _ in range(n_resgroups)]
        self.body = nn.Sequential(*m_body, conv2d(n_feats, n_feats, kernel_size))
        self.tail = nn.Sequential(Upsampler(scale, n_feats), conv2d(n_feats, n_colors, kernel_size))

        self.csa = CSAM_Module(n_feats)
        self.la = LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats * 11, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        for name, midlayer in self.body._modules.items():
            res = midlayer(res)
            if name == "0":
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1), res1], 1)
        out1 = res
        res = self.la(res1)
        out2 = self.last_conv(res)

        out1 = self.csa(out1)
        out = torch.cat([out1, out2], 1)
        res = self.last(out)

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
                n_resgroups=self.n_resgroups,
                reduction=self.reduction,
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
    def from_pretrained(cls, scale: int = 4) -> "HAN":

        file_ids = {
            2: "12NhWDksOXiVnGw-Zbv6Y20J2DnRRKkJ2",
            3: "1bcos3CfYZ-qfSszxEnPohJaUFgVihOB_",
            4: "1f86ez0hgFLwe9hjhQogHpkACtYgfqrRi",
            8: "1Z5mYsASGKfn77ze25EjS8sUNAJ-KjzpO",
        }
        model = HAN(scale=scale, img_range=255.0)
        file_name = f"HAN_BIX{scale}.pt"
        model_dir = "pretrained"
        os.makedirs(model_dir, exist_ok=True)
        file_id = file_ids[scale]
        path = os.path.join(model_dir, file_name)
        if not os.path.exists(path):
            gdown.download(id=file_id, output=path, quiet=False)
        pretrained = torch.load(path, map_location="cpu")
        model.load_state_dict(pretrained, strict=False)
        return model
