"""
generator.py  –  Person B
U-Net-style conditional generator.

Input : concat(cloudy [4ch], sar [2ch])  →  6 channels total
        (verified against real data: cloudy=[4,256,256], sar=[2,256,256])
Output: reconstructed clean optical      →  4 channels (R, G, B, NIR)

Architecture: standard encoder-decoder with skip connections.
Dropout in decoder bottleneck enables MC-Dropout inference (Person C).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _norm(num_ch: int) -> nn.Module:
    """
    GroupNorm instead of BatchNorm.
    Stable at any batch size (normalises within each sample, not across batch).
    num_groups=8 works for all channel counts we use (32, 64, 128, 256, 512, 1024).
    Falls back to fewer groups if num_ch < 8.
    """
    num_groups = min(8, num_ch)
    # GroupNorm requires num_ch % num_groups == 0
    while num_ch % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_ch)


class ConvBlock(nn.Module):
    """Conv → GroupNorm → LeakyReLU × 2"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool → ConvBlock"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool  = nn.MaxPool2d(2)
        self.conv  = ConvBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Bilinear upsample → concat(skip) → ConvBlock"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Crop skip if spatial sizes mismatch (edge-case for non-pow-2 inputs)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class CloudRemovalGenerator(nn.Module):
    """
    U-Net generator for cloud removal.

    Parameters
    ----------
    in_channels  : int   cloudy_ch + sar_ch  (default 9 = 8+1)
    out_channels : int   clean_ch             (default 4)
    base_filters : int   feature map count at first level (default 64)
    dropout      : float spatial dropout in bottleneck (enables MC-Dropout)
    """

    def __init__(
        self,
        in_channels: int  = 6,
        out_channels: int = 4,
        base_filters: int = 64,
        dropout: float    = 0.3,
    ):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)            # 256 → 256
        self.enc2 = DownBlock(f,     f*2)                # 256 → 128
        self.enc3 = DownBlock(f*2,   f*4)                # 128 →  64
        self.enc4 = DownBlock(f*4,   f*8)                #  64 →  32

        # Bottleneck (dropout here powers MC-Dropout confidence proxy)
        self.bottleneck = DownBlock(f*8, f*16, dropout=dropout)  # 32 → 16

        # Decoder
        self.dec4 = UpBlock(f*16, f*8,  f*8,  dropout=dropout)  # 16 →  32
        self.dec3 = UpBlock(f*8,  f*4,  f*4)                    #  32 →  64
        self.dec2 = UpBlock(f*4,  f*2,  f*2)                    #  64 → 128
        self.dec1 = UpBlock(f*2,  f,    f)                      # 128 → 256

        # Output head: 1×1 conv + Sigmoid (keep output in [0,1])
        self.head = nn.Sequential(
            nn.Conv2d(f, out_channels, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def forward(self, cloudy: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cloudy : [B, 4, 256, 256]
        sar    : [B, 2, 256, 256]

        Returns
        -------
        reconstructed : [B, 4, 256, 256]  values in [0, 1]
        """
        x = torch.cat([cloudy, sar], dim=1)   # [B, 6, 256, 256]

        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)

        # Decode with skip connections
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.head(d1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Stub (Day-1 fallback for Person C to integrate against immediately)
# ---------------------------------------------------------------------------

class StubGenerator(nn.Module):
    """
    Returns a blurred copy of the first 4 cloudy bands as a dummy reconstruction.
    Zero trainable parameters — exists only so Person C's pipeline
    can be wired up from hour 1.
    """

    def forward(self, cloudy: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        blurred = F.avg_pool2d(cloudy[:, :4], kernel_size=15, stride=1, padding=7)
        return torch.clamp(blurred, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gen = CloudRemovalGenerator()
    total = sum(p.numel() for p in gen.parameters())
    print(f"Generator params: {total:,}")

    cloudy = torch.rand(2, 4, 256, 256)
    sar    = torch.rand(2, 2, 256, 256)
    out    = gen(cloudy, sar)
    print(f"Output shape: {out.shape}  min={out.min():.3f}  max={out.max():.3f}")
    assert out.shape == (2, 4, 256, 256), "Shape mismatch!"
    print("Generator OK ✓")
