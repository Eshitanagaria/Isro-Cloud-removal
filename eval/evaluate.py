
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def spectral_angle_mapper(pred, target, eps=1e-8):
    pred_flat = pred.reshape(pred.shape[0], -1).T
    target_flat = target.reshape(target.shape[0], -1).T
    dot = np.sum(pred_flat * target_flat, axis=1)
    pred_norm = np.linalg.norm(pred_flat, axis=1)
    target_norm = np.linalg.norm(target_flat, axis=1)
    cos_angle = dot / (pred_norm * target_norm + eps)
    cos_angle = np.clip(cos_angle, -1, 1)
    angles = np.arccos(cos_angle)
    return np.degrees(np.mean(angles))

def evaluate(pred, target):
    pred_hwc = np.transpose(pred, (1, 2, 0))
    target_hwc = np.transpose(target, (1, 2, 0))
    ssim_val = ssim(target_hwc, pred_hwc, data_range=1.0, channel_axis=2)
    psnr_val = psnr(target_hwc, pred_hwc, data_range=1.0)
    sam_val = spectral_angle_mapper(pred, target)
    return {
        "SSIM": round(float(ssim_val), 4),
        "PSNR_dB": round(float(psnr_val), 2),
        "SAM_deg": round(float(sam_val), 2)
    }
