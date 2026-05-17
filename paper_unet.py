import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freq = torch.exp(
            torch.arange(half, device=device, dtype=t.dtype) * (-math.log(10000) / (half - 1))
        )
        emb = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.q = nn.Conv1d(ch, ch, 1)
        self.k = nn.Conv1d(ch, ch, 1)
        self.v = nn.Conv1d(ch, ch, 1)
        self.proj = nn.Conv1d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x).reshape(b, c, h * w)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / math.sqrt(c), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))
        out = self.proj(out).reshape(b, c, h, w)
        return out + x_in


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, ch: int = 128, time_dim: int = 512):
        super().__init__()

        self.time = SinusoidalPosEmb(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(3, ch, 3, padding=1)

        self.down1a = ResBlock(ch, ch, time_dim)
        self.down1b = ResBlock(ch, ch, time_dim)
        self.downsample1 = Downsample(ch)

        self.down2a = ResBlock(ch, ch * 2, time_dim)
        self.down2attn = AttnBlock(ch * 2)
        self.down2b = ResBlock(ch * 2, ch * 2, time_dim)
        self.downsample2 = Downsample(ch * 2)

        self.mid1 = ResBlock(ch * 2, ch * 4, time_dim)
        self.midattn = AttnBlock(ch * 4)
        self.mid2 = ResBlock(ch * 4, ch * 2, time_dim)

        self.upsample2 = Upsample(ch * 2)
        self.up2a = ResBlock(ch * 4, ch * 2, time_dim)
        self.up2attn = AttnBlock(ch * 2)
        self.up2b = ResBlock(ch * 2, ch * 2, time_dim)

        self.upsample1 = Upsample(ch * 2)
        self.up1a = ResBlock(ch * 3, ch, time_dim)
        self.up1b = ResBlock(ch, ch, time_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(self.time(t))
        x = self.in_conv(x)

        h1 = self.down1a(x, emb)
        h1 = self.down1b(h1, emb)
        h = self.downsample1(h1)

        h2 = self.down2a(h, emb)
        h2 = self.down2attn(h2)
        h2 = self.down2b(h2, emb)
        h = self.downsample2(h2)

        h = self.mid1(h, emb)
        h = self.midattn(h)
        h = self.mid2(h, emb)

        h = self.upsample2(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up2a(h, emb)
        h = self.up2attn(h)
        h = self.up2b(h, emb)

        h = self.upsample1(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up1a(h, emb)
        h = self.up1b(h, emb)

        return self.out(h)