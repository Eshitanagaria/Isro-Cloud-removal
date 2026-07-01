"""
losses.py  –  Person B  (core GAN losses)
─────────────────────────────────────────────────────────────────────────────
Contains the base GAN losses used during training.

Person C will add the spectral-consistency loss (NDVI/NDWI/band-ratio) by
dropping in   losses/spectral_loss.py   and passing it to train.py via the
`extra_gen_loss_fn` hook — no edits to this file needed.

Loss strategy: LSGAN (Least-Squares GAN)
  · More stable than vanilla GAN
  · Targets: real=1, fake=0  for discriminator
             real=1           for generator (fools discriminator)
  · Combined generator loss = λ_adv * L_adv  +  λ_pix * L_pixel  [+ λ_spec * L_spectral]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LSGAN adversarial losses
# ---------------------------------------------------------------------------

class LSGANLoss(nn.Module):
    """
    Least-Squares GAN loss  (Mao et al. 2017).
    Uses MSE with target tensors of all-ones (real) or all-zeros (fake).
    """

    def __init__(self):
        super().__init__()

    def discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_D = 0.5 * E[(D(real) - 1)²]  +  0.5 * E[(D(fake))²]
        """
        loss_real = F.mse_loss(real_logits, torch.ones_like(real_logits))
        loss_fake = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
        return 0.5 * (loss_real + loss_fake)

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """
        L_G_adv = 0.5 * E[(D(G(x)) - 1)²]
        """
        return 0.5 * F.mse_loss(fake_logits, torch.ones_like(fake_logits))


# ---------------------------------------------------------------------------
# Pixel-level reconstruction loss
# ---------------------------------------------------------------------------

class PixelLoss(nn.Module):
    """
    L1 pixel loss masked to the cloudy region.

    Masked loss focuses learning on reconstructing exactly the hidden pixels;
    the clear pixels act as an anchor but contribute less signal.

    Parameters
    ----------
    mask_weight  : float   extra weight on masked (cloudy) pixels  (default 2.0)
    """

    def __init__(self, mask_weight: float = 2.0):
        super().__init__()
        self.mask_weight = mask_weight

    def forward(
        self,
        predicted: torch.Tensor,   # [B, 4, H, W]
        target:    torch.Tensor,   # [B, 4, H, W]
        mask:      torch.Tensor,   # [B, 1, H, W]  1=cloud, 0=clear
    ) -> torch.Tensor:
        """Weighted L1: masked pixels carry extra weight."""
        abs_err = (predicted - target).abs()                  # [B, 4, H, W]

        # Build per-pixel weight map: masked pixels get mask_weight, others get 1
        weight = 1.0 + (self.mask_weight - 1.0) * mask       # [B, 1, H, W]
        weighted = abs_err * weight                           # broadcast over channels

        return weighted.mean()


# ---------------------------------------------------------------------------
# Perceptual / Feature-matching loss  (optional, used in train.py)
# ---------------------------------------------------------------------------

class FeatureMatchingLoss(nn.Module):
    """
    Discriminator feature-matching loss (Salimans et al.).
    Penalizes the L1 distance between intermediate discriminator features
    for real and fake inputs, improving training stability.

    Usage: call discriminator in 'feature' mode (returns list of activations).
    The PatchGANDiscriminator in this repo doesn't expose features directly,
    so this class is provided as an optional upgrade path.
    """

    def forward(
        self,
        real_features: list[torch.Tensor],
        fake_features: list[torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, requires_grad=True)
        for real_f, fake_f in zip(real_features, fake_features):
            loss = loss + F.l1_loss(fake_f, real_f.detach())
        return loss / max(len(real_features), 1)


# ---------------------------------------------------------------------------
# Combined generator loss container
# ---------------------------------------------------------------------------

class GeneratorLoss(nn.Module):
    """
    Wraps all generator loss terms into a single callable.

    Parameters
    ----------
    lambda_adv   : float  weight for adversarial loss   (default 1.0)
    lambda_pixel : float  weight for L1 pixel loss       (default 100.0)
    lambda_spec  : float  weight for Person C's spectral loss  (default 10.0)
    """

    def __init__(
        self,
        lambda_adv:   float = 1.0,
        lambda_pixel: float = 100.0,
        lambda_spec:  float = 10.0,
    ):
        super().__init__()
        self.lambda_adv   = lambda_adv
        self.lambda_pixel = lambda_pixel
        self.lambda_spec  = lambda_spec

        self.adv_loss   = LSGANLoss()
        self.pixel_loss = PixelLoss(mask_weight=2.0)

    def forward(
        self,
        fake_logits:         torch.Tensor,
        predicted:           torch.Tensor,
        target:              torch.Tensor,
        mask:                torch.Tensor,
        spectral_loss_fn=None,   # Person C's callable: (predicted, target) → scalar
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        total_loss : scalar tensor
        breakdown  : dict of individual loss values (for logging)
        """
        l_adv   = self.adv_loss.generator_loss(fake_logits)
        l_pixel = self.pixel_loss(predicted, target, mask)

        total = self.lambda_adv * l_adv + self.lambda_pixel * l_pixel

        breakdown = {
            "G_adv":   l_adv.item(),
            "G_pixel": l_pixel.item(),
        }

        # ── Person C hook ──────────────────────────────────────────────────
        if spectral_loss_fn is not None:
            l_spec = spectral_loss_fn(predicted, target)
            total  = total + self.lambda_spec * l_spec
            breakdown["G_spectral"] = l_spec.item()
        # ───────────────────────────────────────────────────────────────────

        breakdown["G_total"] = total.item()
        return total, breakdown
