import copy
import math
import warnings
from typing import OrderedDict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .presnet import ConvNormLayer


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=nn.SiLU(inplace=True))
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=nn.SiLU(inplace=True))
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=nn.SiLU(inplace=True))
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_2d(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (t * cos) + (rotate_half(t) * sin)


class Mixed2DRoPE(nn.Module):
    def __init__(self, head_dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2‑D RoPE"
        self.head_dim = head_dim
        self.num_heads = num_heads
        d_half = head_dim // 2
        mag = 1.0 / (theta ** (torch.arange(0, d_half, 2).float() / head_dim))
        fx, fy = [], []
        for _ in range(num_heads):
            angle = torch.rand(1) * math.tau if rotate else torch.zeros(1)
            fx.append(torch.cat([mag * torch.cos(angle), mag * torch.cos(angle + math.pi / 2)], dim=-1))
            fy.append(torch.cat([mag * torch.sin(angle), mag * torch.sin(angle + math.pi / 2)], dim=-1))
        # The random per-head rotations are part of the model, so they must be saved with the checkpoint
        self.register_buffer("freq_x", torch.stack(fx, 0))
        self.register_buffer("freq_y", torch.stack(fy, 0))

        self._cache: dict[tuple[int, int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)
        self._cache.clear()
        for name in ("freq_x", "freq_y"):
            if prefix + name in missing_keys:
                missing_keys.remove(prefix + name)
                warnings.warn(f"{prefix + name} missing from checkpoint (saved before RoPE frequencies were "
                              f"persisted); using freshly sampled frequencies that differ from training")

    @torch.no_grad()
    def _get_cos_sin(self, h: int, w: int, device: torch.device, dtype: torch.dtype):
        key = (h, w, device, dtype)
        if key in self._cache:
            return self._cache[key]

        t_x = torch.arange(w, device=device).repeat_interleave(h).to(dtype)
        t_y = torch.arange(h, device=device).repeat(w).to(dtype)

        angle = ((t_x.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * self.freq_x.unsqueeze(1)) +
                 (t_y.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * self.freq_y.unsqueeze(1)))
        cos_half, sin_half = angle.cos(), angle.sin()

        cos = torch.repeat_interleave(cos_half, 2, dim=-1).to(dtype)
        sin = torch.repeat_interleave(sin_half, 2, dim=-1).to(dtype)

        self._cache[key] = (cos, sin)
        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor, h: int, w: int):
        cos, sin = self._get_cos_sin(h, w, q.device, q.dtype)
        return apply_rotary_2d(q, cos, sin), apply_rotary_2d(k, cos, sin)


class RotaryMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, rope: Mixed2DRoPE):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = rope

    def forward(self, x: torch.Tensor, *, h: int, w: int, attn_mask: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        q, k = self.rope(q, k, h, w)

        q = q.reshape(B * self.num_heads, L, self.head_dim)
        k = k.reshape(B * self.num_heads, L, self.head_dim)
        v = v.reshape(B * self.num_heads, L, self.head_dim)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=float(self.dropout.p))
        attn_out = attn_out.view(B, self.num_heads, L, self.head_dim).permute(0, 2, 1, 3).reshape(B, L, self.embed_dim)
        return self.o_proj(attn_out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 rope: Mixed2DRoPE | None = None):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = RotaryMultiheadAttention(d_model, nhead, dropout, rope)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, src: torch.Tensor, *, h: int, w: int, src_mask=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        src = self.self_attn(src, h=h, w=w, attn_mask=src_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int, norm: Optional[nn.LayerNorm] = None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src: torch.Tensor, *, h: int, w: int, src_mask=None):
        for layer in self.layers:
            src = layer(src, h=h, w=w, src_mask=src_mask)
        return self.norm(src) if self.norm else src


class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=(128, 256, 512),
                 feat_strides=(8, 16, 32),
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 use_encoder_idx=(2,),
                 num_encoder_layers=1,
                 expansion=0.5,
                 depth_mult=1.0):
        super().__init__()
        self.in_channels = list(in_channels)
        self.feat_strides = list(feat_strides)
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = list(use_encoder_idx)
        self.num_encoder_layers = num_encoder_layers
        self.out_channels = [hidden_dim] * len(in_channels)
        self.out_strides = self.feat_strides

        self.input_proj = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(c, hidden_dim, 1, bias=False)),
                ("bn", nn.BatchNorm2d(hidden_dim))
            ]))
            for c in in_channels
        ])

        rope = Mixed2DRoPE(hidden_dim // nhead, nhead)
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, rope=rope)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in self.use_encoder_idx
        ])

        self.lateral_convs = nn.ModuleList([
            ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=nn.SiLU(inplace=True))
            for _ in range(len(in_channels) - 1)
        ])
        self.fpn_blocks = nn.ModuleList([
            CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), expansion)
            for _ in range(len(in_channels) - 1)
        ])
        self.downsample_convs = nn.ModuleList([
            ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=nn.SiLU(inplace=True))
            for _ in range(len(in_channels) - 1)
        ])
        self.pan_blocks = nn.ModuleList([
            CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), expansion)
            for _ in range(len(in_channels) - 1)
        ])

    def forward(self, feats: list[torch.Tensor]):
        assert len(feats) == len(self.in_channels)

        proj_feats = [proj(feat) for proj, feat in zip(self.input_proj, feats)]

        for enc_i, feat_i in enumerate(self.use_encoder_idx):
            B, C, H, W = proj_feats[feat_i].shape
            src = proj_feats[feat_i].flatten(2).transpose(1, 2)
            src = self.encoder[enc_i](src, h=H, w=W)
            proj_feats[feat_i] = src.transpose(1, 2).view(B, C, H, W).contiguous()

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            high = inner_outs[0]
            low = proj_feats[idx - 1]
            high = self.lateral_convs[len(self.in_channels) - 1 - idx](high)
            inner_outs[0] = high
            inner_out = F.interpolate(high, scale_factor=2.0, mode="nearest")
            inner_out = torch.cat([inner_out, low], dim=1)
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](inner_out)
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            low = outs[-1]
            high = inner_outs[idx + 1]
            low = self.downsample_convs[idx](low)
            out = torch.cat([low, high], dim=1)
            outs.append(self.pan_blocks[idx](out))

        return outs
