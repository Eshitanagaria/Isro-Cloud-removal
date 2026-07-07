"""
predict.py  
──────────────────────────────────────────────────────────────────────────────
LOCKED INFERENCE CONTRACT  

    predict(cloudy, sar, temporal=None) → reconstructed

Parameters
──────────
cloudy     : np.ndarray | torch.Tensor  shape [4, 256, 256]  float32  in [0,1]
sar        : np.ndarray | torch.Tensor  shape [2, 256, 256]  float32  in [0,1]
             (VV + VH channels, normalised from dB with SAR_MIN=-50, SAR_MAX=25)
temporal   : np.ndarray | torch.Tensor  shape [4, 256, 256]  float32  in [0,1]
             Optional temporal composite. Blended with GAN output in clear-sky
             regions if provided. If None, runs SAR+cloudy-only mode (fallback).

Returns
───────
reconstructed : np.ndarray  shape [4, 256, 256]  float32  in [0,1]
                Clean optical reconstruction  (R, G, B, NIR)

──────────────────────────────────────────────────────────────────────────────

"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

# ── local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gan.generator import CloudRemovalGenerator, StubGenerator


# ---------------------------------------------------------------------------
# Module-level singleton — loaded once and reused
# ---------------------------------------------------------------------------
_model:  Optional[CloudRemovalGenerator] = None
_device: Optional[torch.device]          = None
_ckpt_path: Optional[str]                = None   # track which ckpt is loaded


def _load_model(
    checkpoint_path: Optional[str],
    base_filters: int = 64,
    dropout: float    = 0.3,
    device: Optional[torch.device] = None,
) -> tuple[CloudRemovalGenerator, torch.device]:
    """Internal: instantiate and load weights."""
    global _model, _device, _ckpt_path

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 6 input channels: 4 cloudy + 2 SAR (VV+VH)
    model = CloudRemovalGenerator(
        in_channels=6,
        out_channels=4,
        base_filters=base_filters,
        dropout=dropout,
    ).to(device)

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("gen", ckpt)   # support both raw and wrapped checkpoints
        model.load_state_dict(state)
        print(f"[predict] Loaded checkpoint: {checkpoint_path}")
    else:
        if checkpoint_path is not None:
            print(f"[predict] WARNING: Checkpoint not found: {checkpoint_path}. Using random weights.")
        else:
            print("[predict] WARNING: No checkpoint supplied. Using random weights (stub mode).")

    model.eval()
    _model     = model
    _device    = device
    _ckpt_path = checkpoint_path
    return model, device


# ---------------------------------------------------------------------------
# Public inference contract
# ---------------------------------------------------------------------------

def predict(
    cloudy:         Union[np.ndarray, torch.Tensor],
    sar:            Union[np.ndarray, torch.Tensor],
    temporal:       Optional[Union[np.ndarray, torch.Tensor]] = None,
    checkpoint_path: Optional[str]                            = None,
    base_filters:   int   = 64,
    dropout:        float = 0.3,
    device:         Optional[torch.device] = None,
) -> np.ndarray:
    """
    Reconstruct a cloud-free optical patch from cloudy + SAR inputs.

    Parameters
    ----------
    cloudy          : [8, 256, 256]  float32, values in [0, 1]
    sar             : [1, 256, 256]  float32, values in [0, 1]
    temporal        : [4, 256, 256]  float32, values in [0, 1]  (optional)
                      When provided, blended with the GAN output in clear-sky
                      regions to sharpen spectral accuracy.
    checkpoint_path : path to  checkpoints/ckpt_best.pt  (or any .pt file)
    base_filters    : must match what was used during training (default 64)
    dropout         : must match training value (keep for MC-Dropout compat)
    device          : override torch device

    Returns
    -------
    np.ndarray  [4, 256, 256]  float32, values in [0, 1]
    """
    global _model, _device, _ckpt_path

    # ── Load / reload model ─────────────────────────────────────────────
    if _model is None or checkpoint_path != _ckpt_path:
        _load_model(checkpoint_path, base_filters=base_filters,
                    dropout=dropout, device=device)

    dev = _device

    # ── Input normalisation + tensor conversion ──────────────────────────
    def _to_tensor(x, expected_ch: int) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            x = x.unsqueeze(0)   # add batch dim
        assert x.shape[1] == expected_ch, \
            f"Expected {expected_ch} channels, got {x.shape[1]}"
        return x.to(dev)

    cloudy_t = _to_tensor(cloudy, 4)    # [1, 4, 256, 256]
    sar_t    = _to_tensor(sar,    2)    # [1, 2, 256, 256]

    # ── Inference ───────────────────────────────────────────────────────
    with torch.no_grad():
        recon = _model(cloudy_t, sar_t)    # [1, 4, 256, 256]

    # ── Optional: blend with temporal composite in clear regions ─────────
    # Temporal blending: where the cloud mask indicates clear sky (mask==0),
    # use a weighted average of recon and temporal to preserve spectral
    # fidelity; where mask==1 (cloud), trust the GAN output fully.
    if temporal is not None:
        temporal_t = _to_tensor(temporal, 4)   # [1, 4, 256, 256]
        # Derive soft clear-sky weight from first cloudy channel
        # (lower intensity in cloudy image suggests cloud presence)
        cloud_weight = (cloudy_t[:, 0:1] < 0.8).float()    # crude proxy; 1=cloud region
        recon = cloud_weight * recon + (1.0 - cloud_weight) * (0.5 * recon + 0.5 * temporal_t)

    # ── Output ──────────────────────────────────────────────────────────
    out = recon.squeeze(0).clamp(0.0, 1.0).cpu().numpy()   # [4, 256, 256]
    return out


# ---------------------------------------------------------------------------
# Batch predict (convenience for Person A's eval harness)
# ---------------------------------------------------------------------------

def predict_batch(
    cloudy_batch:   Union[np.ndarray, torch.Tensor],
    sar_batch:      Union[np.ndarray, torch.Tensor],
    temporal_batch: Optional[Union[np.ndarray, torch.Tensor]] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Batch version of predict().

    Parameters
    ----------
    cloudy_batch  : [B, 8, 256, 256]
    sar_batch     : [B, 1, 256, 256]
    temporal_batch: [B, 4, 256, 256]  optional

    Returns
    -------
    np.ndarray  [B, 4, 256, 256]
    """
    global _model, _device, _ckpt_path

    if _model is None or checkpoint_path != _ckpt_path:
        _load_model(checkpoint_path, device=device)

    dev = _device

    def _prep(x, ch):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        assert x.ndim == 4 and x.shape[1] == ch
        return x.to(dev)

    cloudy_t = _prep(cloudy_batch, 4)
    sar_t    = _prep(sar_batch, 2)

    with torch.no_grad():
        recon = _model(cloudy_t, sar_t)

    if temporal_batch is not None:
        temporal_t   = _prep(temporal_batch, 4)
        cloud_weight = (cloudy_t[:, 0:1] < 0.8).float()
        recon = cloud_weight * recon + (1.0 - cloud_weight) * (0.5 * recon + 0.5 * temporal_t)

    return recon.clamp(0.0, 1.0).cpu().numpy()


# ---------------------------------------------------------------------------
# Quick sanity-check  (run: python gan/predict.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== predict.py sanity check ===")
    cloudy   = np.random.rand(4, 256, 256).astype(np.float32)
    sar      = np.random.rand(2, 256, 256).astype(np.float32)
    temporal = np.random.rand(4, 256, 256).astype(np.float32)

    # No checkpoint → random weights (stub behaviour)
    out = predict(cloudy, sar, temporal=None, checkpoint_path=None)
    print(f"  Without temporal: {out.shape}  min={out.min():.3f}  max={out.max():.3f}")

    out_t = predict(cloudy, sar, temporal=temporal, checkpoint_path=None)
    print(f"  With temporal:    {out_t.shape}  min={out_t.min():.3f}  max={out_t.max():.3f}")

    print("predict.py OK ✓")
