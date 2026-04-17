import math
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Time Embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        # t: (B,) float in [0,1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )
        args = t[:, None] * freqs[None]  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


class TimeEmbedMLP(nn.Module):
    def __init__(self, sin_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(sin_dim),
            nn.Linear(sin_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.net(t)  # (B, out_dim)


# ---------------------------------------------------------------------------
# FiLM Conditioning
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    def __init__(self, embed_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 2 * channels)

    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        # x: (B,C,H,W), e: (B, embed_dim)
        gamma, beta = self.proj(e).chunk(2, dim=-1)  # (B,C) each
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return gamma * x + beta


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_embed_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = FiLMLayer(time_embed_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.film(h, t_emb)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        seq = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        seq = self.norm(seq)
        out, _ = self.attn(seq, seq, seq)
        return (seq + out).permute(0, 2, 1).view(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.up(x))


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    ~1.5M parameter UNet for MNIST flow matching.
    Input: (B,1,28,28) + t:(B,). Output: (B,1,28,28) vector field.
    """

    def __init__(self, base_ch: int = 32, time_embed_dim: int = 256):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4]  # [32, 64, 128]
        te = time_embed_dim

        self.time_embed = TimeEmbedMLP(128, te)

        # Encoder
        self.stem = nn.Conv2d(1, ch[0], 3, padding=1)
        self.down1 = nn.ModuleList([ResBlock(ch[0], ch[0], te), ResBlock(ch[0], ch[0], te)])
        self.ds1 = Downsample(ch[0])
        self.down2 = nn.ModuleList([ResBlock(ch[0], ch[1], te), ResBlock(ch[1], ch[1], te)])
        self.ds2 = Downsample(ch[1])
        self.down3 = nn.ModuleList([ResBlock(ch[1], ch[2], te), ResBlock(ch[2], ch[2], te)])

        # Bottleneck
        self.mid1 = ResBlock(ch[2], ch[2], te)
        self.mid_attn = SelfAttention2D(ch[2], num_heads=4)
        self.mid2 = ResBlock(ch[2], ch[2], te)

        # Decoder — skip connections concatenate encoder outputs
        self.up3 = Upsample(ch[2], ch[1])
        self.dec3 = nn.ModuleList([ResBlock(ch[1] + ch[1], ch[1], te), ResBlock(ch[1], ch[1], te)])
        self.up2 = Upsample(ch[1], ch[0])
        self.dec2 = nn.ModuleList([ResBlock(ch[0] + ch[0], ch[0], te), ResBlock(ch[0], ch[0], te)])

        # Output head
        self.out_norm = nn.GroupNorm(8, ch[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        te = self.time_embed(t)

        # Encoder
        h = self.stem(x)
        for blk in self.down1:
            h = blk(h, te)
        skip1 = h
        h = self.ds1(h)
        for blk in self.down2:
            h = blk(h, te)
        skip2 = h
        h = self.ds2(h)
        for blk in self.down3:
            h = blk(h, te)

        # Bottleneck
        h = self.mid1(h, te)
        h = self.mid_attn(h)
        h = self.mid2(h, te)

        # Decoder
        h = self.up3(h)
        h = torch.cat([h, skip2], dim=1)
        for blk in self.dec3:
            h = blk(h, te)
        h = self.up2(h)
        h = torch.cat([h, skip1], dim=1)
        for blk in self.dec2:
            h = blk(h, te)

        return self.out_conv(self.out_act(self.out_norm(h)))
