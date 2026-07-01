"""
ablation.py  –  Person B  (Checkpoint 3 deliverable)
─────────────────────────────────────────────────────────────────────────────
Ablation study: does dropping temporal or SAR conditioning hurt output quality?

Evaluates three model variants on the validation set using saved checkpoints:

  Variant 1  (full)         : cloudy [4ch] + SAR [2ch]          → 6ch input
  Variant 2  (no-SAR)       : cloudy [4ch] only                  → 4ch input
  Variant 3  (no-cloudy)    : SAR    [2ch] only                  → 2ch input

For each variant:
  - Trains a small model (or loads an existing ckpt) on the synthetic data
  - Evaluates PSNR, SSIM, MAE on the validation split
  - Saves a side-by-side visual comparison strip

Outputs a concise 1-pager text report to:   ablation_report.txt
And visual comparison grid to:              ablation_comparison.png

Usage:
    # Quick eval-only mode (requires pre-trained checkpoints):
    python gan/ablation.py --mode eval --data_dir data/synthetic

    # Full train+eval mode (trains each variant from scratch — slow):
    python gan/ablation.py --mode train --data_dir data/synthetic --epochs 20

    # Eval against the already-trained full model checkpoint:
    python gan/ablation.py --mode eval --full_ckpt checkpoints/ckpt_best.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gan.dataset       import build_dataloaders
from gan.generator     import CloudRemovalGenerator
from gan.discriminator import PatchGANDiscriminator
from gan.losses        import LSGANLoss, GeneratorLoss


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS = {
    "full":       {"cloudy_ch": 4, "sar_ch": 2, "label": "Full (cloudy + SAR)"},
    "no_sar":     {"cloudy_ch": 4, "sar_ch": 0, "label": "No-SAR  (cloudy only)"},
    "no_cloudy":  {"cloudy_ch": 0, "sar_ch": 2, "label": "No-Cloudy (SAR only)"},
}


# ---------------------------------------------------------------------------
# Masked forward helpers
# ---------------------------------------------------------------------------

def _make_input(batch: dict, cloudy_ch: int, sar_ch: int, device: torch.device):
    """Build the generator input tensor for a given variant."""
    parts = []
    if cloudy_ch > 0:
        parts.append(batch["cloudy"].to(device)[:, :cloudy_ch])
    if sar_ch > 0:
        parts.append(batch["sar"].to(device)[:, :sar_ch])
    return torch.cat(parts, dim=1)


def _build_model(cloudy_ch: int, sar_ch: int, device: torch.device) -> CloudRemovalGenerator:
    in_ch = cloudy_ch + sar_ch
    model = CloudRemovalGenerator(in_channels=in_ch, out_channels=4, base_filters=64, dropout=0.3)
    return model.to(device)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def ssim_approx(pred: torch.Tensor, target: torch.Tensor) -> float:
    C1, C2 = 0.01**2, 0.03**2
    mu1, mu2 = pred.mean(), target.mean()
    s1, s2   = pred.std(), target.std()
    s12 = ((pred - mu1) * (target - mu2)).mean()
    num   = (2*mu1*mu2 + C1) * (2*s12 + C2)
    denom = (mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)
    return (num / denom).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred - target).abs().mean().item()


# ---------------------------------------------------------------------------
# Train one variant
# ---------------------------------------------------------------------------

def train_variant(
    name: str, cloudy_ch: int, sar_ch: int,
    train_loader, val_loader, device,
    epochs: int, save_dir: Path,
) -> Path:
    """Train a single ablation variant and return path to best checkpoint."""
    print(f"\n{'='*60}")
    print(f"  Training variant: {VARIANTS[name]['label']}")
    print(f"{'='*60}")

    model = _build_model(cloudy_ch, sar_ch, device)
    disc  = PatchGANDiscriminator(
        in_channels=4 + cloudy_ch + sar_ch   # clean + conditioning
    ).to(device)

    gen_loss_fn  = GeneratorLoss(lambda_adv=1.0, lambda_pixel=100.0)
    disc_loss_fn = LSGANLoss()

    opt_g = Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = Adam(disc.parameters(),  lr=1e-4, betas=(0.5, 0.999))

    sched_g = CosineAnnealingLR(opt_g, T_max=epochs, eta_min=2e-6)
    sched_d = CosineAnnealingLR(opt_d, T_max=epochs, eta_min=1e-6)

    best_psnr = -float("inf")
    best_path = save_dir / f"ablation_{name}_best.pt"

    for epoch in range(epochs):
        model.train(); disc.train()

        for batch in train_loader:
            x     = _make_input(batch, cloudy_ch, sar_ch, device)
            clean = batch["clean"].to(device)
            mask  = batch["mask"].to(device)

            fake = model.forward_ablation(x)   # see subclass below

            # Discriminator
            opt_d.zero_grad()
            # Build conditioning for discriminator (use available modalities)
            cond_cloudy = batch["cloudy"].to(device)[:, :cloudy_ch] if cloudy_ch > 0 \
                          else torch.zeros(clean.shape[0], 4, 256, 256, device=device)
            cond_sar    = batch["sar"].to(device)[:, :sar_ch] if sar_ch > 0 \
                          else torch.zeros(clean.shape[0], 2, 256, 256, device=device)

            rl = disc(clean,        cond_cloudy, cond_sar)
            fl = disc(fake.detach(), cond_cloudy, cond_sar)
            d_loss = disc_loss_fn.discriminator_loss(rl, fl)
            d_loss.backward(); opt_d.step()

            # Generator
            opt_g.zero_grad()
            fl2 = disc(fake, cond_cloudy, cond_sar)
            g_loss, _ = gen_loss_fn(fl2, fake, clean, mask)
            g_loss.backward(); opt_g.step()

        sched_g.step(); sched_d.step()

        # Validation
        val_psnr = evaluate_variant(name, cloudy_ch, sar_ch, model, val_loader, device)["psnr"]
        print(f"  Epoch {epoch+1:02d}/{epochs}  val_PSNR={val_psnr:.2f} dB")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({"gen": model.state_dict()}, best_path)

    print(f"  Best PSNR: {best_psnr:.2f} dB  ->  {best_path}")
    return best_path


# ---------------------------------------------------------------------------
# Evaluate one variant (Zero-Shot using Full Model)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_variant(
    name: str, cloudy_ch: int, sar_ch: int,
    model_or_path,         # CloudRemovalGenerator or Path/str to full ckpt
    val_loader,
    device: torch.device,
) -> dict:
    """Return dict with psnr, ssim, mae averaged over validation set."""
    
    # We ALWAYS load the FULL model (in_channels=6) for zero-shot ablation
    if isinstance(model_or_path, (str, Path)):
        model = _build_model(4, 2, device)  # Always build full model
        ckpt  = torch.load(model_or_path, map_location=device)
        model.load_state_dict(ckpt.get("gen", ckpt))
    else:
        model = model_or_path

    model.eval()
    total = {"psnr": 0.0, "ssim": 0.0, "mae": 0.0}
    n = 0

    for batch in val_loader:
        # Build 6-channel input
        cloudy = batch["cloudy"].to(device)
        sar    = batch["sar"].to(device)
        
        # Zero out missing modalities for ablation
        if name == "no_cloudy":
            cloudy = torch.zeros_like(cloudy)
        elif name == "no_sar":
            sar = torch.zeros_like(sar)
            
        clean = batch["clean"].to(device)
        B     = clean.shape[0]

        pred = model(cloudy, sar)

        total["psnr"] += psnr(pred,  clean) * B
        total["ssim"] += ssim_approx(pred, clean) * B
        total["mae"]  += mae(pred,   clean) * B
        n += B

    return {k: v / n for k, v in total.items()}


def _ablation_forward(model: CloudRemovalGenerator, x: torch.Tensor) -> torch.Tensor:
    # Deprecated: model.forward_ablation(x) is used instead
    return model.forward_ablation(x)



# ---------------------------------------------------------------------------
# Visual comparison strip
# ---------------------------------------------------------------------------

def save_comparison_strip(
    results: dict,          # variant_name -> {"pred": np.array [4,H,W], "clean": np.array}
    save_path: Path,
):
    """Save a PNG showing cloudy | SAR | pred_full | pred_nosar | pred_nocloudy | clean."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  [ablation] matplotlib not installed — skipping visual strip.")
        return

    n_variants = len(results)
    fig, axes = plt.subplots(1, n_variants + 1, figsize=(4 * (n_variants + 1), 4))
    fig.suptitle("Ablation: Cloud Removal GAN", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.axis("off")

    variant_names = list(results.keys())
    for col, vname in enumerate(variant_names):
        data  = results[vname]
        pred  = data["pred"][:3].transpose(1, 2, 0)   # RGB [H,W,3]
        pred  = np.clip(pred, 0, 1)
        label = VARIANTS[vname]["label"]
        m     = data["metrics"]

        axes[col].imshow(pred)
        axes[col].set_title(
            f"{label}\nPSNR={m['psnr']:.1f}dB  SSIM={m['ssim']:.3f}",
            fontsize=8,
        )

    # Ground truth in last column
    clean = list(results.values())[0]["clean"][:3].transpose(1, 2, 0)
    clean = np.clip(clean, 0, 1)
    axes[-1].imshow(clean)
    axes[-1].set_title("Ground Truth (clean)", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [ablation] Comparison strip saved -> {save_path}")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(all_metrics: dict, save_path: Path, mode: str):
    lines = [
        "=" * 60,
        "  ABLATION REPORT  —  Person B",
        "  Cloud Removal GAN: SAR vs Optical Conditioning Study",
        "=" * 60,
        f"  Mode: {mode}",
        f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "  Question: Does dropping SAR or cloudy conditioning hurt quality?",
        "",
        "-" * 60,
        f"  {'Variant':<30} {'PSNR (dB)':>10} {'SSIM':>8} {'MAE':>8}",
        "-" * 60,
    ]

    best_psnr_variant = max(all_metrics, key=lambda k: all_metrics[k]["psnr"])

    for name, m in all_metrics.items():
        label  = VARIANTS[name]["label"]
        marker = " <-- BEST" if name == best_psnr_variant else ""
        lines.append(
            f"  {label:<30} {m['psnr']:>10.2f} {m['ssim']:>8.4f} {m['mae']:>8.4f}{marker}"
        )

    lines += [
        "-" * 60,
        "",
        "  FINDINGS:",
    ]

    full_psnr   = all_metrics.get("full",      {}).get("psnr", 0)
    nosar_psnr  = all_metrics.get("no_sar",    {}).get("psnr", 0)
    nocld_psnr  = all_metrics.get("no_cloudy", {}).get("psnr", 0)

    sar_drop  = full_psnr - nosar_psnr
    cld_drop  = full_psnr - nocld_psnr

    lines.append(f"  Removing SAR    costs {sar_drop:+.2f} dB PSNR")
    lines.append(f"  Removing cloudy costs {cld_drop:+.2f} dB PSNR")
    lines.append("")

    if sar_drop > 1.0:
        lines.append("  -> SAR conditioning provides meaningful signal (>1dB gain).")
    elif sar_drop > 0.3:
        lines.append("  -> SAR conditioning provides modest benefit (~0.3-1dB).")
    else:
        lines.append("  -> SAR conditioning provides minimal benefit in this dataset.")

    if cld_drop > 1.0:
        lines.append("  -> Optical (cloudy) conditioning is strongly beneficial.")
    else:
        lines.append("  -> SAR alone achieves comparable performance to optical.")

    lines += [
        "",
        "  FALLBACK RECOMMENDATION:",
        "  If full 3-channel conditioning does not converge in time,",
        "  the variant with higher PSNR between no_sar/no_cloudy is the",
        "  recommended fallback (keep the stronger modality).",
        "",
        "=" * 60,
    ]

    report = "\n".join(lines)
    save_path.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n  Report saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Person B — Ablation Study")
    p.add_argument("--data_dir",   default="data/synthetic")
    p.add_argument("--mode",       choices=["train", "eval"], default="eval",
                   help="'train' trains each variant; 'eval' uses existing checkpoints")
    p.add_argument("--epochs",     type=int, default=20,
                   help="Epochs per variant (train mode only)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--full_ckpt",  default="checkpoints/ckpt_best.pt",
                   help="Checkpoint for the full variant (eval mode)")
    p.add_argument("--out_dir",    default="ablation_results",
                   help="Output directory for checkpoints, report, visuals")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Which validation sample index to use for visual comparison")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablation] Device: {device}")
    print(f"[ablation] Mode:   {args.mode}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        synthetic_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=False,      # no augmentation for ablation fairness
    )

    all_metrics = {}
    visual_data = {}

    # ── Get one validation sample for visual comparison ─────────────────
    val_batches = list(val_loader)
    sample_batch_idx  = args.sample_idx // args.batch_size
    sample_within_idx = args.sample_idx  % args.batch_size
    if sample_batch_idx >= len(val_batches):
        sample_batch_idx = 0; sample_within_idx = 0
    ref_batch = val_batches[sample_batch_idx]
    ref_clean = ref_batch["clean"][sample_within_idx].numpy()   # [4,256,256]

    # ── Per-variant loop ────────────────────────────────────────────────
    for name, cfg in VARIANTS.items():
        cloudy_ch = cfg["cloudy_ch"]
        sar_ch    = cfg["sar_ch"]
        label     = cfg["label"]

        print(f"\n[ablation] Variant: {label}  (in_channels={cloudy_ch+sar_ch})")

        if args.mode == "train":
            ckpt_path = train_variant(
                name, cloudy_ch, sar_ch,
                train_loader, val_loader, device,
                epochs=args.epochs, save_dir=out_dir,
            )
            model_or_path = ckpt_path

        else:
            # eval mode: Zero-shot ablation using the FULL checkpoint for ALL variants.
            ckpt = args.full_ckpt
            if not os.path.isfile(ckpt):
                print(f"  WARNING: full checkpoint '{ckpt}' not found.")
            model_or_path = ckpt

        metrics = evaluate_variant(
            name, cloudy_ch, sar_ch,
            model_or_path, val_loader, device,
        )
        all_metrics[name] = metrics
        print(f"  PSNR={metrics['psnr']:.2f} dB  SSIM={metrics['ssim']:.4f}  MAE={metrics['mae']:.4f}")

        # Visual: get prediction for one reference patch
        if isinstance(model_or_path, (str, Path)) and os.path.isfile(model_or_path):
            model_eval = _build_model(4, 2, device)  # always full model
            ckpt = torch.load(model_or_path, map_location=device)
            model_eval.load_state_dict(ckpt.get("gen", ckpt))
        else:
            model_eval = _build_model(4, 2, device)

        model_eval.eval()
        with torch.no_grad():
            # Build 6-channel input
            c_ref = ref_batch["cloudy"][sample_within_idx:sample_within_idx+1].to(device)
            s_ref = ref_batch["sar"][sample_within_idx:sample_within_idx+1].to(device)
            
            # Zero out missing modalities
            if name == "no_cloudy":
                c_ref = torch.zeros_like(c_ref)
            elif name == "no_sar":
                s_ref = torch.zeros_like(s_ref)
                
            pred = model_eval(c_ref, s_ref).squeeze(0).cpu().numpy()

        visual_data[name] = {"pred": pred, "clean": ref_clean, "metrics": metrics}

    # ── Report + visuals ────────────────────────────────────────────────
    write_report(all_metrics, out_dir / "ablation_report.txt", args.mode)

    save_comparison_strip(visual_data, out_dir / "ablation_comparison.png")

    # Save raw metrics as JSON
    metrics_path = out_dir / "ablation_metrics.json"
    metrics_path.write_text(
        json.dumps({"mode": args.mode, "metrics": all_metrics}, indent=2),
        encoding="utf-8",
    )
    print(f"[ablation] Raw metrics -> {metrics_path}")
    print("[ablation] Done.")


if __name__ == "__main__":
    main()
