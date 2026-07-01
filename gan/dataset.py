"""
dataset.py  –  Person A / B shared DataLoader
I/O contract (frozen, verified against real .npy shapes 2026-07-01):
  patch size  : 256 × 256
  file format : .npy  (prototype); GeoTIFF for geo products
  bands       : clean   → 4ch  [4,256,256]  float32  [0,1]  (R,G,B,NIR)
                cloudy  → 4ch  [4,256,256]  float64→float32  [0,1]
                sar     → 2ch  [2,256,256]  float32  dB scale (norm to [0,1])
                mask    → 1ch  [256,256]→[1,256,256]  float64→float32  [0,1]

  SAR normalisation: linear min-max clamp  (SAR_MIN=-50 dB, SAR_MAX=25 dB)
"""

# SAR value range in dB — clip then map to [0, 1]
SAR_MIN_DB: float = -50.0
SAR_MAX_DB: float =  25.0

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SyntheticPatchDataset(Dataset):
    """
    Loads aligned (cloudy, sar, mask, clean) patch tuples from
    data/synthetic/{cloudy,sar,mask,clean}/.

    Returns
    -------
    dict with keys:
        'cloudy' : FloatTensor [4, 256, 256]  in [0, 1]
        'sar'    : FloatTensor [2, 256, 256]  in [0, 1]  (normalised from dB)
        'mask'   : FloatTensor [1, 256, 256]  in [0, 1]
        'clean'  : FloatTensor [4, 256, 256]  in [0, 1]
        'idx'    : int
    """

    def __init__(self, root_dir: str, augment: bool = False):
        """
        Parameters
        ----------
        root_dir : str
            Path to the data/synthetic directory containing
            clean/, cloudy/, mask/, sar/ sub-folders.
        augment  : bool
            If True, apply random horizontal / vertical flips.
        """
        self.root_dir = root_dir
        self.augment = augment

        # Collect sorted file lists
        self.clean_files  = sorted(glob.glob(os.path.join(root_dir, "clean",  "*.npy")))
        self.cloudy_files = sorted(glob.glob(os.path.join(root_dir, "cloudy", "*.npy")))
        self.sar_files    = sorted(glob.glob(os.path.join(root_dir, "sar",    "*.npy")))
        self.mask_files   = sorted(glob.glob(os.path.join(root_dir, "mask",   "*.npy")))

        assert len(self.clean_files) > 0, f"No clean patches found in {root_dir}/clean"
        n = len(self.clean_files)
        assert len(self.cloudy_files) == n, "Mismatch: cloudy vs clean count"
        assert len(self.sar_files)    == n, "Mismatch: sar vs clean count"
        assert len(self.mask_files)   == n, "Mismatch: mask vs clean count"

        print(f"[Dataset] Found {n} patch pairs in '{root_dir}'")

    def __len__(self) -> int:
        return len(self.clean_files)

    def __getitem__(self, idx: int) -> dict:
        # Load numpy arrays and cast to float32
        clean  = np.nan_to_num(np.load(self.clean_files[idx]), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)   # [4, 256, 256]  already [0,1]
        cloudy = np.nan_to_num(np.load(self.cloudy_files[idx]), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)  # [4, 256, 256]  already [0,1]
        sar    = np.load(self.sar_files[idx]).astype(np.float32)     # [2, 256, 256]  dB scale
        mask   = np.load(self.mask_files[idx]).astype(np.float32)    # [256, 256]     [0,1]

        # Ensure channel-first [C, H, W]
        # All arrays except mask are already [C,H,W]; mask needs an axis
        if mask.ndim == 2:
            mask = mask[np.newaxis]   # [1, 256, 256]

        # Normalise SAR from dB to [0, 1] via linear min-max clamp
        sar = np.clip(sar, SAR_MIN_DB, SAR_MAX_DB)
        sar = (sar - SAR_MIN_DB) / (SAR_MAX_DB - SAR_MIN_DB)

        # Clamp optical / mask to [0, 1]
        clean  = np.clip(clean,  0.0, 1.0)
        cloudy = np.clip(cloudy, 0.0, 1.0)
        mask   = np.clip(mask,   0.0, 1.0)

        # Optional augmentation (same random flip applied to all)
        if self.augment:
            clean, cloudy, sar, mask = self._augment(clean, cloudy, sar, mask)

        return {
            "cloudy": torch.from_numpy(cloudy),
            "sar":    torch.from_numpy(sar),
            "mask":   torch.from_numpy(mask),
            "clean":  torch.from_numpy(clean),
            "idx":    idx,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _augment(*arrays):
        """Apply the same random flip to all arrays."""
        if np.random.rand() > 0.5:   # horizontal flip
            arrays = tuple(np.flip(a, axis=-1).copy() for a in arrays)
        if np.random.rand() > 0.5:   # vertical flip
            arrays = tuple(np.flip(a, axis=-2).copy() for a in arrays)
        return arrays


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_dataloaders(
    synthetic_dir: str,
    batch_size: int = 8,
    val_fraction: float = 0.15,
    num_workers: int = 0,
    augment_train: bool = True,
    seed: int = 42,
):
    """
    Build train / val DataLoaders from the synthetic patch directory.

    Parameters
    ----------
    synthetic_dir : str
        Path to data/synthetic/ (contains clean/, cloudy/, sar/, mask/).
    batch_size    : int
    val_fraction  : float   fraction of dataset used for validation
    num_workers   : int     DataLoader worker processes (0 = main thread)
    augment_train : bool    apply random flips to training set
    seed          : int     for reproducible split

    Returns
    -------
    (train_loader, val_loader)
    """
    full_ds = SyntheticPatchDataset(synthetic_dir, augment=augment_train)

    n_val   = max(1, int(len(full_ds) * val_fraction))
    n_train = len(full_ds) - n_val

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    # Disable augmentation for validation subset
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[DataLoader] train={n_train} | val={n_val} | batch={batch_size}")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick sanity-check (run this file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    synthetic_dir = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic"
    train_loader, val_loader = build_dataloaders(synthetic_dir, batch_size=4)

    batch = next(iter(train_loader))
    print("\n=== Batch shapes ===")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:8s}: {tuple(v.shape)}  min={v.min():.3f}  max={v.max():.3f}")
    print("Dataset OK ✓")
