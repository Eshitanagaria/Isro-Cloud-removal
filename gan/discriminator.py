"""
discriminator.py  –  Person B
PatchGAN discriminator: classifies 70×70 overlapping patches as real/fake.

Input : concat(reconstructed_or_real [4ch], cloudy [4ch], sar [2ch]) → 10 channels
        (conditioning on the input modalities so the discriminator judges
         whether the reconstruction is *consistent* with what it was given)
Output: [B, 1, N, N] patch-level real/fake logits  (N depends on architecture depth)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class DiscConvBlock(nn.Module):
    """Conv → InstanceNorm → LeakyReLU"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not norm)
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# PatchGAN Discriminator
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator (standard from pix2pix).

    Parameters
    ----------
    in_channels : int
        real/fake channels + conditional channels.
        Default = 13  (4 clean + 8 cloudy + 1 SAR)
    base_filters : int
        Feature map count at first conv layer. Default = 64.
    n_layers : int
        Number of strided conv layers. Default = 3 → ~70px receptive field.
    """

    def __init__(
        self,
        in_channels:  int = 10,
        base_filters: int = 64,
        n_layers:     int = 3,
    ):
        super().__init__()
        f = base_filters

        # First layer: no norm
        layers: list[nn.Module] = [DiscConvBlock(in_channels, f, stride=2, norm=False)]

        # Intermediate strided layers
        in_f, out_f = f, f
        for i in range(1, n_layers):
            in_f  = out_f
            out_f = min(f * (2 ** i), f * 8)
            layers.append(DiscConvBlock(in_f, out_f, stride=2, norm=True))

        # Final strided layer (stride=1 to keep spatial resolution)
        in_f  = out_f
        out_f = min(f * (2 ** n_layers), f * 8)
        layers.append(DiscConvBlock(in_f, out_f, stride=1, norm=True))

        # Output: 1-channel logit map
        layers.append(
            nn.Conv2d(out_f, 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def forward(
        self,
        reconstructed: torch.Tensor,   # [B, 4, 256, 256]  predicted or real clean
        cloudy:        torch.Tensor,   # [B, 4, 256, 256]
        sar:           torch.Tensor,   # [B, 2, 256, 256]
    ) -> torch.Tensor:
        """Returns patch logits [B, 1, N, N]. No sigmoid – use LSGAN or BCEWithLogits."""
        x = torch.cat([reconstructed, cloudy, sar], dim=1)  # [B, 10, 256, 256]
        return self.model(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    disc = PatchGANDiscriminator()
    total = sum(p.numel() for p in disc.parameters())
    print(f"Discriminator params: {total:,}")

    recon  = torch.rand(2, 4, 256, 256)
    cloudy = torch.rand(2, 4, 256, 256)
    sar    = torch.rand(2, 2, 256, 256)
    out = disc(recon, cloudy, sar)
    print(f"Output shape: {out.shape}")   # e.g. [2, 1, 30, 30]
    print("Discriminator OK ✓")
