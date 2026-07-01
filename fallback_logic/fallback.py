import numpy as np


def cloud_percentage(mask):
    """
    mask: numpy array, values 0-1, same as Person A's mask/patch_XXXX.npy
    Returns the fraction of the patch covered by cloud (0.0 - 1.0)
    """
    return float(np.mean(mask))


def reconstruct_with_fallback(cloudy, sar, temporal, mask, predict_fn, threshold=0.80):
    """
    cloudy, sar, temporal: input tensors/arrays for the model
    mask: cloud mask for this patch (from Person A's data)
    predict_fn: B's model call, e.g. predict_fn(cloudy, sar, temporal) -> reconstructed
    threshold: cloud % above which we switch to SAR-heavy mode
    """
    cloud_pct = cloud_percentage(mask)

    if cloud_pct > threshold:
        # SAR-heavy: zero out the cloudy optical input, lean on SAR + temporal
        recon = predict_fn(cloudy * 0.0, sar, temporal)
        mode = "SAR-heavy (cloud > 80%)"
    else:
        # Normal fused reconstruction
        recon = predict_fn(cloudy, sar, temporal)
        mode = "Normal fusion"

    return recon, cloud_pct, mode

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gan.predict import predict as b_predict

    cloudy = np.random.rand(4, 256, 256).astype(np.float32)
    sar = np.random.rand(2, 256, 256).astype(np.float32)
    temporal = np.random.rand(4, 256, 256).astype(np.float32)

    # Wrap her predict() so it matches our (cloudy, sar, temporal) signature
    def real_predict(cloudy, sar, temporal):
        return b_predict(cloudy, sar, temporal=temporal, checkpoint_path=None)

    # Test case 1: heavy cloud (90%)
    heavy_mask = np.ones((256, 256)) * 0.9
    recon, pct, mode = reconstruct_with_fallback(cloudy, sar, temporal, heavy_mask, real_predict)
    print(f"Heavy cloud test -> cloud%: {pct:.2f}, mode: {mode}, output shape: {recon.shape}")

    # Test case 2: light cloud (20%)
    light_mask = np.ones((256, 256)) * 0.2
    recon, pct, mode = reconstruct_with_fallback(cloudy, sar, temporal, light_mask, real_predict)
    print(f"Light cloud test -> cloud%: {pct:.2f}, mode: {mode}, output shape: {recon.shape}")