import os
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn

from studiosr.models.common import Model
from studiosr.utils import download


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation, groups=groups
    )


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == "batch":
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == "instance":
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError("normalization layer [{:s}] is not found".format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == "reflect":
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == "replicate":
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError("padding layer [{:s}] is not implemented".format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(
    in_nc,
    out_nc,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != "zero" else None
    padding = padding if pad_type == "zero" else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError("activation layer [{:s}] is not found".format(act_type))
    return layer


def mean_channels(F):
    assert F.dim() == 4
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert F.dim() == 4
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super().__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation("lrelu", neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.cca = CCALayer(self.distilled_channels * 4)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


def load_state_dict(path):
    state_dict = torch.load(path)
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if "module" in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit


class IMDN(Model):
    def __init__(
        self,
        scale: int = 4,
        n_colors: int = 3,
        img_range: float = 1.0,
        n_feats: int = 64,
        n_modules: int = 6,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.n_colors = n_colors
        self.img_range = img_range
        self.n_feats = n_feats
        self.n_modules = n_modules

        self.fea_conv = conv_layer(n_colors, n_feats, kernel_size=3)
        self.IMDB1 = IMDModule(in_channels=n_feats)
        self.IMDB2 = IMDModule(in_channels=n_feats)
        self.IMDB3 = IMDModule(in_channels=n_feats)
        self.IMDB4 = IMDModule(in_channels=n_feats)
        self.IMDB5 = IMDModule(in_channels=n_feats)
        self.IMDB6 = IMDModule(in_channels=n_feats)
        self.c = conv_block(n_feats * n_modules, n_feats, kernel_size=1, act_type="lrelu")
        self.LR_conv = conv_layer(n_feats, n_feats, kernel_size=3)
        self.upsampler = pixelshuffle_block(n_feats, n_colors, upscale_factor=scale)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output

    def get_model_config(self) -> Dict:
        config = super().get_model_config()
        config.update(
            dict(
                scale=self.scale,
                n_colors=self.n_colors,
                img_range=self.img_range,
                n_feats=self.n_feats,
                n_modules=self.n_modules,
            )
        )
        return config

    @classmethod
    def from_pretrained(cls, scale: int = 4) -> "IMDN":
        assert scale in [2, 3, 4]
        config = {"scale": scale}
        model = IMDN(**config)
        file_name = f"IMDN_x{scale}.pth"
        model_url = "https://github.com/Zheng222/IMDN/raw/master/checkpoints/"
        model_dir = "pretrained"
        os.makedirs(model_dir, exist_ok=True)
        link = model_url + file_name
        path = os.path.join(model_dir, file_name)
        if not os.path.exists(path):
            download(link, path)
        pretrained = load_state_dict(path)
        model.load_state_dict(pretrained)
        return model
