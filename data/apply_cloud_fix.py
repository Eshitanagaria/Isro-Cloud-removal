import numpy as np

def apply_cloud(patch, mask, cloud_brightness=0.9):
    patch_norm = patch.astype(np.float32)
    max_val = np.percentile(patch_norm, 99)
    if max_val < 1e-6:
        return None, None
    patch_scaled = np.clip(patch_norm / max_val, 0, 1)
    if not np.isfinite(patch_scaled).all():
        return None, None
    cloud_layer = np.ones_like(patch_scaled) * cloud_brightness
    mask_3d = mask[np.newaxis, :, :]
    cloudy = patch_scaled * (1 - mask_3d) + cloud_layer * mask_3d
    return cloudy, patch_scaled
