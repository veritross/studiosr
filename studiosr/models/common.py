import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def diverge_images(image: np.ndarray) -> List[np.ndarray]:
    transformed_images = []
    for i in range(4):
        rotated = np.rot90(image, k=i, axes=[0, 1])
        flipped = np.fliplr(rotated)
        transformed_images.extend([rotated, flipped])
    return transformed_images


def converge_images(images: List[np.ndarray]) -> np.ndarray:
    transformed_images = []
    for i, image in enumerate(images):
        image = np.fliplr(image) if i & 1 else image
        image = np.rot90(image, k=i // 2, axes=[1, 0])
        transformed_images.append(image)
    image = np.mean(transformed_images, axis=0)
    image = image.round().clip(0, 255).astype(np.uint8)
    return image


class BaseModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img_range = 1.0

    @torch.no_grad()
    def inference(self, image: np.ndarray) -> np.ndarray:
        self.eval()
        scale = 255.0 if self.img_range == 1.0 else 1.0
        device = next(self.parameters()).get_device()
        device = torch.device("cpu") if device < 0 else device
        x = image.transpose(2, 0, 1).astype(np.float32) / scale
        x = torch.from_numpy(x).unsqueeze(0)
        x = x.to(device)
        output = self.forward(x) * scale
        output = output.squeeze().cpu().numpy().transpose(1, 2, 0)
        return output.round().clip(0, 255).astype(np.uint8)

    @torch.no_grad()
    def inference_with_self_ensemble(self, image: np.ndarray) -> np.ndarray:
        self.eval()
        scale = 255.0 if self.img_range == 1.0 else 1.0
        device = next(self.parameters()).get_device()
        device = torch.device("cpu") if device < 0 else device
        images = diverge_images(image)
        outputs = []
        for image in images:
            x = image.transpose(2, 0, 1).astype(np.float32) / scale
            x = torch.from_numpy(x).unsqueeze(0)
            x = x.to(device)
            output = self.forward(x) * scale
            outputs.append(output.squeeze().cpu().numpy().transpose(1, 2, 0))
        image = converge_images(outputs)
        return image

    def get_training_config(self) -> Dict:
        training_config = dict()
        return training_config

    @classmethod
    def from_pretrained(cls, scale: int = 4) -> "BaseModule":
        model = cls(scale=scale)
        return model


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
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int = 96, norm_layer: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = None if norm_layer is None else norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim: int = 96) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: List[int]) -> torch.Tensor:
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x


class Normalizer(nn.Module):
    def __init__(self, img_range: float = 1.0, img_mean: List[float] = [0.4488, 0.4371, 0.4040]) -> None:
        super().__init__()
        self.img_range = img_range
        self.img_mean = torch.Tensor(img_mean).view(1, 3, 1, 1)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self.img_mean = self.img_mean.type_as(x)
        return x / self.img_range - self.img_mean

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.img_mean) * self.img_range


def window_partition(x: torch.Tensor, window_size: List[int]) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: List[int], H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def calculate_mask(x_size: List[int], window_size: int, shift_size: int) -> torch.Tensor:
    H, W = x_size
    img_mask = torch.zeros((1, H, W, 1))
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def check_image_size(x: torch.Tensor, window_size: int) -> torch.Tensor:
    _, _, h, w = x.size()
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
    return x
