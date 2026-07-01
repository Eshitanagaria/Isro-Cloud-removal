"""
train.py  –  Person B
Full GAN training loop.

Usage (from project root):
    python gan/train.py --data_dir data/synthetic --epochs 50

Checkpoints are saved to  checkpoints/  every N epochs.
Person C's spectral loss is plugged in via --spectral_loss flag once available.
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gan.generator     import CloudRemovalGenerator, StubGenerator
from gan.discriminator import PatchGANDiscriminator
from gan.losses        import LSGANLoss, GeneratorLoss
from gan.dataset       import build_dataloaders

# Optional: Person C's spectral loss  (imported lazily so training works
#           even before /losses/ exists)
def _try_import_spectral_loss():
    try:
        from losses.spectral_loss import SpectralConsistencyLoss
        print("[train] ✓ Person C's SpectralConsistencyLoss loaded.")
        return SpectralConsistencyLoss()
    except ImportError:
        print("[train] ⚠  losses/spectral_loss.py not found — training without spectral loss.")
        return None


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def ssim_approx(pred: torch.Tensor, target: torch.Tensor, C1=0.01**2, C2=0.03**2) -> float:
    """Fast spatial SSIM approximation (single-batch, no sliding window)."""
    mu1, mu2 = pred.mean(), target.mean()
    s1  = pred.std()
    s2  = target.std()
    s12 = ((pred - mu1) * (target - mu2)).mean()
    num   = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    denom = (mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)
    return (num / denom).item()


def sam_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Spectral Angle Mapper — matches Person A's eval/evaluate.py exactly.
    Lower is better (0 = perfect). Returned in degrees.
    """
    # [B, C, H, W] -> [B*H*W, C]
    B, C, H, W = pred.shape
    p = pred.permute(0, 2, 3, 1).reshape(-1, C)
    t = target.permute(0, 2, 3, 1).reshape(-1, C)
    dot   = (p * t).sum(dim=1)
    p_norm = p.norm(dim=1)
    t_norm = t.norm(dim=1)
    cos_angle = (dot / (p_norm * t_norm + eps)).clamp(-1, 1)
    return torch.acos(cos_angle).mean().item() * (180.0 / 3.14159265)


# ---------------------------------------------------------------------------
# Train / Validate one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    gen, disc, opt_g, opt_d,
    gen_loss_fn, disc_loss_fn,
    loader, device,
    spectral_loss_fn=None,
    use_stub: bool = False,
    warmup: bool = False,       # if True: pixel-only, no discriminator
) -> dict:
    gen.train()
    if not warmup:
        disc.train()

    totals = {"G_adv": 0, "G_pixel": 0, "G_spectral": 0, "G_total": 0, "D_total": 0}
    n = 0

    for batch in loader:
        cloudy = batch["cloudy"].to(device)   # [B, 8, H, W]
        sar    = batch["sar"].to(device)      # [B, 1, H, W]
        mask   = batch["mask"].to(device)     # [B, 1, H, W]
        clean  = batch["clean"].to(device)    # [B, 4, H, W]
        B = cloudy.size(0)

        # ── Generate ────────────────────────────────────────────────────
        fake_clean = gen(cloudy, sar)         # [B, 4, H, W]

        if warmup:
            # Warmup: pixel loss only — no discriminator, no adversarial pressure
            # This lets the generator learn basic reconstruction before GAN kicks in
            opt_g.zero_grad()
            l_pixel = gen_loss_fn.pixel_loss(fake_clean, clean, mask)
            g_loss  = gen_loss_fn.lambda_pixel * l_pixel
            if spectral_loss_fn is not None:
                l_spec = spectral_loss_fn(fake_clean, clean)
                g_loss = g_loss + gen_loss_fn.lambda_spec * l_spec
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
            opt_g.step()
            breakdown = {"G_pixel": l_pixel.item(), "G_total": g_loss.item()}
            for k, v in breakdown.items():
                totals[k] = totals.get(k, 0) + v * B
            n += B
            continue

        # ── Discriminator step ──────────────────────────────────────────
        opt_d.zero_grad()

        real_logits = disc(clean,      cloudy, sar)
        fake_logits = disc(fake_clean.detach(), cloudy, sar)
        d_loss = disc_loss_fn.discriminator_loss(real_logits, fake_logits)

        d_loss.backward()
        # Gradient clipping — prevents NaN explosions in early training
        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
        opt_d.step()

        # ── Generator step ──────────────────────────────────────────────
        opt_g.zero_grad()

        fake_logits_for_g = disc(fake_clean, cloudy, sar)
        g_loss, breakdown = gen_loss_fn(
            fake_logits=fake_logits_for_g,
            predicted=fake_clean,
            target=clean,
            mask=mask,
            spectral_loss_fn=spectral_loss_fn,
        )

        g_loss.backward()
        # Gradient clipping — prevents NaN explosions in early training
        torch.nn.utils.clip_grad_norm_(gen.parameters(),  max_norm=1.0)
        opt_g.step()

        # Accumulate
        for k, v in breakdown.items():
            totals[k] = totals.get(k, 0) + v * B
        totals["D_total"] += d_loss.item() * B
        n += B

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(gen, loader, device) -> dict:
    gen.eval()
    total_psnr, total_ssim, total_sam, n = 0.0, 0.0, 0.0, 0

    for batch in loader:
        cloudy = batch["cloudy"].to(device)
        sar    = batch["sar"].to(device)
        clean  = batch["clean"].to(device)
        B = cloudy.size(0)

        pred = gen(cloudy, sar)
        total_psnr += psnr(pred, clean) * B
        total_ssim += ssim_approx(pred, clean) * B
        total_sam  += sam_metric(pred, clean) * B
        n += B

    return {
        "val_psnr": total_psnr / n,
        "val_ssim": total_ssim / n,
        "val_sam":  total_sam  / n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Person B — GAN training loop")
    p.add_argument("--data_dir",      default="data/synthetic",
                   help="Path to data/synthetic/ folder")
    p.add_argument("--checkpoint_dir", default="checkpoints",
                   help="Where to save model checkpoints")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr_g",          type=float, default=1e-4,
                   help="Generator learning rate")
    p.add_argument("--lr_d",          type=float, default=5e-5,
                   help="Discriminator learning rate")
    p.add_argument("--lambda_adv",    type=float, default=1.0)
    p.add_argument("--lambda_pixel",  type=float, default=100.0)
    p.add_argument("--lambda_spec",   type=float, default=10.0,
                   help="Weight for Person C's spectral loss (0 = disabled)")
    p.add_argument("--base_filters",  type=int,   default=64)
    p.add_argument("--dropout",       type=float, default=0.3,
                   help="Dropout in generator bottleneck (powers MC-Dropout for Person C)")
    p.add_argument("--save_every",    type=int,   default=5,
                   help="Save checkpoint every N epochs")
    p.add_argument("--warmup_epochs", type=int,   default=3,
                   help="Epochs of pixel-only pre-training before GAN adversarial loss")
    p.add_argument("--num_workers",   type=int,   default=0)
    p.add_argument("--stub",          action="store_true",
                   help="Use StubGenerator (Day-1 fallback for Person C)")
    p.add_argument("--spectral_loss", action="store_true",
                   help="Attempt to load Person C's spectral loss from losses/")
    p.add_argument("--resume",        default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Reproducibility ─────────────────────────────────────────────────
    torch.manual_seed(args.seed)

    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        synthetic_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Models ──────────────────────────────────────────────────────────
    if args.stub:
        print("[train] ⚠  Using StubGenerator (Day-1 fallback). No training will occur.")
        gen = StubGenerator().to(device)
        disc = PatchGANDiscriminator().to(device)
    else:
        gen  = CloudRemovalGenerator(
            base_filters=args.base_filters,
            dropout=args.dropout,
        ).to(device)
        disc = PatchGANDiscriminator().to(device)

    gen_params  = sum(p.numel() for p in gen.parameters())
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"[train] Generator:     {gen_params:,} params")
    print(f"[train] Discriminator: {disc_params:,} params")

    # ── Losses ──────────────────────────────────────────────────────────
    gen_loss_fn  = GeneratorLoss(
        lambda_adv=args.lambda_adv,
        lambda_pixel=args.lambda_pixel,
        lambda_spec=args.lambda_spec,
    )
    disc_loss_fn = LSGANLoss()

    # ── Person C's spectral loss hook ───────────────────────────────────
    spectral_loss_fn = None
    if args.spectral_loss:
        sl = _try_import_spectral_loss()
        if sl is not None:
            sl = sl.to(device)
            spectral_loss_fn = sl

    # ── Optimizers ──────────────────────────────────────────────────────
    opt_g = Adam(gen.parameters(),  lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    sched_g = CosineAnnealingLR(opt_g, T_max=args.epochs, eta_min=args.lr_g * 0.01)
    sched_d = CosineAnnealingLR(opt_d, T_max=args.epochs, eta_min=args.lr_d * 0.01)

    # ── Resume ──────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        gen.load_state_dict(ckpt["gen"])
        disc.load_state_dict(ckpt["disc"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[train] Resumed from epoch {start_epoch}")

    # ── Checkpoint dir ──────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ───────────────────────────────────────────────────
    log = []
    best_psnr = -float("inf")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        is_warmup = (epoch < args.warmup_epochs)
        if is_warmup and epoch == 0:
            print(f"[train] Warmup phase: epochs 0-{args.warmup_epochs-1} use pixel loss only (no discriminator)")
        if not is_warmup and epoch == args.warmup_epochs:
            print(f"[train] GAN phase started at epoch {epoch}")

        train_metrics = train_one_epoch(
            gen, disc, opt_g, opt_d,
            gen_loss_fn, disc_loss_fn,
            train_loader, device,
            spectral_loss_fn=spectral_loss_fn,
            warmup=is_warmup,
        )
        val_metrics = validate(gen, val_loader, device)

        sched_g.step()
        sched_d.step()

        elapsed = time.time() - t0

        # ── Logging ─────────────────────────────────────────────────────
        row = {"epoch": epoch, "time_s": round(elapsed, 1), **train_metrics, **val_metrics}
        log.append(row)

        print(
            f"[Epoch {epoch+1:03d}/{args.epochs}]  "
            f"G={train_metrics.get('G_total', 0):.4f}  "
            f"D={train_metrics.get('D_total', 0):.4f}  "
            f"PSNR={val_metrics['val_psnr']:.2f}dB  "
            f"SSIM={val_metrics['val_ssim']:.4f}  "
            f"SAM={val_metrics['val_sam']:.2f}deg  "
            f"({elapsed:.1f}s)"
        )

        # ── Save checkpoints ────────────────────────────────────────────
        def save_ckpt(tag: str):
            state = {
                "epoch":    epoch,
                "gen":      gen.state_dict(),
                "disc":     disc.state_dict(),
                "opt_g":    opt_g.state_dict(),
                "opt_d":    opt_d.state_dict(),
                "val_psnr": val_metrics["val_psnr"],
                "args":     vars(args),
            }
            path = ckpt_dir / f"ckpt_{tag}.pt"
            torch.save(state, path)
            print(f"  [ckpt] Saved -> {path}")

        if (epoch + 1) % args.save_every == 0:
            save_ckpt(f"epoch{epoch+1:03d}")

        if val_metrics["val_psnr"] > best_psnr:
            best_psnr = val_metrics["val_psnr"]
            save_ckpt("best")

    # ── Save final checkpoint + training log ────────────────────────────
    save_ckpt("final")
    log_path = ckpt_dir / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n[train] Done. Log saved -> {log_path}")
    print(f"[train] Best val PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
