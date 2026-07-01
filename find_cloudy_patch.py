import numpy as np
import glob

for f in sorted(glob.glob("data/synthetic/mask/patch_*.npy")):
    mask = np.load(f)
    cloud_pct = mask.mean()
    print(f"{f}: {cloud_pct:.2f}")