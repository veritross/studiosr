import math
from typing import List, Optional

import torch
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import einsum, nn
from torch.nn.functional import pad

from studiosr.models.common import Model, Normalizer

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


# helper classes


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, window_height=None, window_width=None):
        if window_height is None or window_width is None:
            return self.fn(self.norm(x)) + x
        else:
            return self.fn(self.norm(x), window_height, window_width) + x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# MBConv


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1"),
        )

    def forward(self, x):
        return x * self.gate(x)


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0.0 or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


def MBConv(dim_in, dim_out, *, downsample, expansion_rate=4, shrinkage_rate=0.25, dropout=0.0):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out),
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


class Attention(nn.Module):
    def __init__(self, dim, dim_head=32, dropout=0.0, window_size=7):
        super().__init__()
        assert (dim % dim_head) == 0, "dimension should be divisible by dimension per head"

        self.heads = dim // dim_head
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))

        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        grid = rearrange(grid, "c i j -> (i j) c")
        rel_pos = rearrange(grid, "i ... -> i 1 ...") - rearrange(grid, "j ... -> 1 j ...")
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        _, height, width, window_height, window_width, _, _, h = *x.shape, x.device, self.heads

        # flatten

        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, "b n (h d ) -> b h n d", h=h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, "i j h -> h i j")

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h (w1 w2) d -> b w1 w2 (h d)", w1=window_height, w2=window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, "(b x y) ... -> b x y ...", x=height, y=width)


class Adaptive_Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.0,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, "dimension should be divisible by dimension per head"

        self.dim = dim
        self.dim_head = dim_head
        self.heads = dim // dim_head
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))

        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))

        # relative positional bias
        # self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

    def forward(self, x, window_height, window_width):
        _, h, w, w1, w2, _, device = *x.shape, x.device

        x = self.norm(x)

        # flatten

        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, "b n (h d ) -> b h n d", h=self.heads), (q, k, v))

        # scale

        q = q * self.scale

        # sim
        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add positional bias

        pos_height = torch.arange(window_height).to(device)
        pos_width = torch.arange(window_width).to(device)
        grid = torch.stack(torch.meshgrid(pos_height, pos_width, indexing="ij")).to(device)
        grid = rearrange(grid, "c i j -> (i j) c")
        rel_pos = (rearrange(grid, "i ... -> i 1 ...") - rearrange(grid, "j ... -> 1 j ...")).to(device)
        rel_pos += torch.tensor([window_height - 1, window_width - 1]).to(device)
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_width - 1, 1]).to(device)).sum(dim=-1).to(device)
        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

        rel_pos_bias = nn.Embedding((2 * window_height - 1) * (2 * window_width - 1), self.heads).to(device)
        rel_pos_bias = rel_pos_bias(rel_pos_indices)
        rel_pos_bias = rearrange(rel_pos_bias, "i j h -> h i j")

        sim = sim + rel_pos_bias
        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h (w1 w2) d -> b w1 w2 (h d)", w1=w1, w2=w2)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, "(b x y) ... -> b x y ...", x=h, y=w)


class BlockAttention(nn.Module):
    def __init__(self, layer_dim, dim_head, dropout):
        super().__init__()
        self.attention = PreNormResidual(
            layer_dim, Adaptive_Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout)
        )
        self.feedforward = PreNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout))
        self.rearrange2 = Rearrange("b x y w1 w2 d -> b d (x w1) (y w2)")

    def forward(self, x, window_height, window_width):
        x = Rearrange("b d (x w1) (y w2) -> b x y w1 w2 d", w1=window_height, w2=window_width)(x)
        x = self.attention(x, window_height, window_width)
        x = self.feedforward(x)
        x = self.rearrange2(x)
        return x


class GridAttention(nn.Module):
    def __init__(self, layer_dim, dim_head, dropout, is_rev=False, is_VT=False):
        super().__init__()
        self.attention = PreNormResidual(
            layer_dim, Adaptive_Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout)
        )
        self.feedforward = PreNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout))
        self.is_rev = is_rev
        self.is_VT = is_VT
        if not is_VT:
            self.rearrange2 = Rearrange("b x y w1 w2 d -> b d (w1 x) (w2 y)")

    def forward(self, x, window_height, window_width):
        x = Rearrange("b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=window_height, w2=window_width)(x)
        x = self.attention(x, window_height, window_width)
        x = self.feedforward(x)
        if not self.is_VT:
            x = self.rearrange2(x)
        return x


def block_att(layer_dim, dim_head, dropout, window_size):
    net = nn.Sequential(
        Rearrange("b d (x w1) (y w2) -> b x y w1 w2 d", w1=window_size, w2=window_size),  # block-like attention
        PreNormResidual(
            layer_dim, Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=window_size)
        ),
        PreNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout)),
        Rearrange("b x y w1 w2 d -> b d (x w1) (y w2)"),
    )
    return net


def grid_att(layer_dim, dim_head, dropout, window_size, is_rev=False, is_VT=False):
    net = []
    if not is_rev:
        net.append(
            Rearrange("b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=window_size, w2=window_size)
        )  # grid-like attention
    net.append(
        PreNormResidual(
            layer_dim, Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=window_size)
        )
    )
    net.append(PreNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout)))

    if not is_VT:
        net.append(Rearrange("b x y w1 w2 d -> b d (w1 x) (w2 y)"))

    return nn.Sequential(*net)


class HFFB(nn.Sequential):
    def __init__(
        self,
        dim_in,
        hidden_dim,
        dim_out,
    ):
        m = []
        m.append(nn.Conv2d(dim_in, hidden_dim, 1))
        m.append(nn.Conv2d(hidden_dim, dim_out, 3, padding=1))
        super(HFFB, self).__init__(*m)


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"scale {scale} is not supported. " "Supported scales: 2^n and 3.")
        super(Upsample, self).__init__(*m)


class MaxSR(Model):
    def __init__(
        self,
        scale: int = 4,
        n_colors: int = 3,
        img_range: float = 1.0,
        adaptive: bool = True,
        dim: int = 128,
        dim_head: int = 32,
        depth: List[int] = [4, 4, 4, 4],
        window_size: int = 8,
        mbconv_expansion_rate: float = 4,
        mbconv_shrinkage_rate: float = 0.25,
        dropout: float = 0.1,
    ):
        super().__init__()

        # variables
        self.adaptive = adaptive
        num_stages = len(depth)
        self.img_range = img_range
        self.normalizer = Normalizer(img_range=img_range)

        # convolutional stem

        self.conv_stem_first = nn.Conv2d(n_colors, dim, 3, stride=1, padding=1)
        self.conv_stem_second = nn.Conv2d(dim, dim, 3, padding=1)

        # MaxViT Block
        block = self.Ada_MaxViT_Block if self.adaptive else self.MaxViT_Block
        self.stages = []
        for d in depth:
            self.stages.append(
                block(dim_head, dim, dim, d, mbconv_expansion_rate, mbconv_shrinkage_rate, dropout, window_size)
            )
        self.stages = nn.ModuleList(self.stages)

        # HFFB
        self.HFFB = HFFB(dim_in=dim * num_stages, hidden_dim=dim, dim_out=dim)  # 3*upscale*upscale

        # Upsampling
        self.upscale = scale
        self.window_size = window_size
        self.Upsample = Upsample(scale=scale, num_feat=dim)
        self.conv_last = nn.Conv2d(dim, n_colors, 3, 1, 1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        if not self.adaptive:
            x = self.check_image_size(x)

        x = self.normalizer.normalize(x)

        # Convolutional stem
        Fm1 = self.conv_stem_first(x)
        x = self.conv_stem_second(Fm1)

        F = []

        # MaxViT Block
        for stage in self.stages:
            for ind, layers in enumerate(stage):
                if ind % 3 != 0 and self.adaptive:
                    batch, channel, height, width = x.shape
                    window_height, window_width = self.calculate_window_size(height, width)
                    x = pad(x, (0, (window_width * window_width) - width, 0, (window_height * window_height) - height))
                    x = layers(x, window_height, window_width)
                else:
                    x = layers(x)
            F.append(x)

        F_cat = torch.cat(F, dim=1)

        # HFFB
        if self.adaptive:
            F_cat = F_cat[:, :, :H, :W]

        x = self.HFFB(F_cat) + Fm1

        # Upsampling
        x = self.Upsample(x)
        x = self.conv_last(x)

        x = self.normalizer.unnormalize(x)

        if not self.adaptive:
            x = x[:, :, : H * self.upscale, : W * self.upscale]

        return x

    def calculate_window_size(self, height, width):
        return math.ceil(math.sqrt(height)), math.ceil(math.sqrt(width))

    def Ada_MaxViT_Block(
        self,
        dim_head,
        layer_dim_in,
        layer_dim,
        layer_depth,
        mbconv_expansion_rate,
        mbconv_shrinkage_rate,
        dropout,
        window_size,
    ):
        layers = nn.ModuleList([])
        for stage_ind in range(layer_depth):
            is_first = stage_ind == 0
            stage_dim_in = layer_dim_in if is_first else layer_dim

            mbconv = MBConv(
                stage_dim_in,
                layer_dim,
                downsample=0,
                expansion_rate=mbconv_expansion_rate,
                shrinkage_rate=mbconv_shrinkage_rate,
            )
            block_attention = BlockAttention(
                layer_dim=layer_dim,
                dim_head=dim_head,
                dropout=dropout,
            )
            grid_attention = GridAttention(
                layer_dim=layer_dim,
                dim_head=dim_head,
                dropout=dropout,
            )
            layers.extend([mbconv, block_attention, grid_attention])

        return layers

    def MaxViT_Block(
        self,
        dim_head,
        layer_dim_in,
        layer_dim,
        layer_depth,
        mbconv_expansion_rate,
        mbconv_shrinkage_rate,
        dropout,
        window_size,
    ):
        layers = nn.ModuleList([])
        for stage_ind in range(layer_depth):
            is_first = stage_ind == 0
            stage_dim_in = layer_dim_in if is_first else layer_dim

            block = nn.Sequential(
                MBConv(
                    stage_dim_in,
                    layer_dim,
                    downsample=0,
                    expansion_rate=mbconv_expansion_rate,
                    shrinkage_rate=mbconv_shrinkage_rate,
                ),
                block_att(layer_dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=window_size),
                grid_att(layer_dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=window_size),
            )
            layers.append(block)

        return layers

    @classmethod
    def from_pretrained(
        cls,
        scale: int = 4,
        light: bool = True,
        adaptive: bool = False,
        ckpt_path: Optional[str] = None,
    ):
        config = {
            "scale": scale,
            "n_colors": 3,
            "adaptive": adaptive,
            "dim": 128,
            "dim_head": 32,
            "depth": [4, 4, 4, 4],
            "window_size": 8,
            "mbconv_expansion_rate": 4,
            "mbconv_shrinkage_rate": 0.25,
            "dropout": 0.1,
        }
        if light:
            config["dim"] = 48
            config["dim_head"] = 12
            config["depth"] = [2, 2, 2, 2]
        model = MaxSR(**config)

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt)

        return model
