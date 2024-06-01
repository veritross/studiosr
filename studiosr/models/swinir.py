import os
from itertools import repeat
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_

from studiosr.models.common import (
    Mlp,
    Model,
    Normalizer,
    Upsampler,
    calculate_mask,
    check_image_size,
    window_partition,
    window_reverse,
)
from studiosr.utils import download


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int = 96, norm_layer: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = None if norm_layer is None else norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv_scale = head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.qkv_scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=tuple(repeat(self.window_size, 2)),
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W, C = x.shape
        x_size = (H, W)

        shortcut = x
        x = self.norm1(x)

        # cyclic
        shifted_x = torch.roll(x, (-self.shift_size, -self.shift_size), (1, 2)) if self.shift_size > 0 else x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, mask=calculate_mask(x_size, self.window_size, self.shift_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        x = torch.roll(shifted_x, (self.shift_size, self.shift_size), (1, 2)) if self.shift_size > 0 else shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class RSTB(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        resi_connection: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.residual_group = BasicLayer(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )
        self.conv = resi_connection(dim) if resi_connection else nn.Conv2d(dim, dim, 3, 1, 1)
        self.patch_embed = lambda x: torch.permute(x, (0, 2, 3, 1))
        self.patch_unembed = lambda x: torch.permute(x, (0, 3, 1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x)))) + x


def check_image_size_for_eval(x: torch.Tensor, window_size: int) -> torch.Tensor:
    _, _, h, w = x.size()
    h_pad = (h // window_size + 1) * window_size - h
    w_pad = (w // window_size + 1) * window_size - w
    x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, : h + h_pad, :]
    x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, : w + w_pad]
    return x


class SwinIR(Model):
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
        resi_connection: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.n_colors = n_colors
        self.img_range = img_range
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.upsampler = upsampler

        self.normalizer = Normalizer(img_range=img_range)
        self.conv_first = nn.Conv2d(n_colors, embed_dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        self.patch_unembed = lambda x: torch.permute(x, (0, 3, 1, 2))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = RSTB(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],  # no impact on SR results
                resi_connection=resi_connection,
            )
            self.layers.append(layer)
        self.norm = nn.LayerNorm(embed_dim)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        if self.upsampler == "pixelshuffle":
            # for classical SR
            num_feat = 64
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = Upsampler(scale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, n_colors, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = Upsampler(scale, embed_dim, n_colors)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]

        preprocess = check_image_size if self.training else check_image_size_for_eval
        x = preprocess(x, self.window_size)

        x = self.normalizer.normalize(x)

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.upsample(x)

        x = self.normalizer.unnormalize(x)
        return x[:, :, : H * self.scale, : W * self.scale]

    def get_model_config(self) -> Dict:
        config = super().get_model_config()
        config.update(
            dict(
                embed_dim=self.embed_dim,
                depths=self.depths,
                num_heads=self.num_heads,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                drop_path_rate=self.drop_path_rate,
                upsampler=self.upsampler,
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
    def from_pretrained(
        cls,
        scale: int = 4,
        light: bool = False,
        dataset: str = "DF2K",
        pretrained: bool = True,
    ) -> "SwinIR":
        assert scale in [2, 3, 4, 8]
        assert dataset in ["DIV2K", "DF2K"]

        config = {"scale": scale}
        img_size = 64 if dataset == "DF2K" else 48
        task = "001_classicalSR"
        label = "M"
        if light:
            config["depths"] = [6, 6, 6, 6]
            config["embed_dim"] = 60
            config["num_heads"] = [6, 6, 6, 6]
            config["upsampler"] = "pixelshuffledirect"
            task = "002_lightweightSR"
            dataset = "DIV2K"
            img_size = 64
            label = "S"

        model = SwinIR(**config)

        if pretrained:
            file_name = f"{task}_{dataset}_s{img_size}w8_SwinIR-{label}_x{scale}.pth"
            model_url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
            model_dir = "pretrained"
            os.makedirs(model_dir, exist_ok=True)
            link = model_url + file_name
            path = os.path.join(model_dir, file_name)
            if not os.path.exists(path):
                download(link, path)
            pretrained = torch.load(path, map_location="cpu")
            params_key = "params"
            params = pretrained[params_key] if params_key in pretrained else pretrained
            model.load_state_dict(params, strict=False)

        return model
