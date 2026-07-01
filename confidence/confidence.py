import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def confidence_map(cloudy, sar, temporal, predict_fn, n_passes=8, noise_std=0.05):
    outputs = []
    for _ in range(n_passes):
        noisy_cloudy = cloudy + np.random.normal(0, noise_std, cloudy.shape).astype(np.float32)
        noisy_cloudy = np.clip(noisy_cloudy, 0.0, 1.0)
        out = predict_fn(noisy_cloudy, sar, temporal)
        outputs.append(out)

    outputs = np.stack(outputs, axis=0)
    mean_pred = outputs.mean(axis=0)
    variance = outputs.var(axis=0).mean(axis=0)   # raw per-pixel variance, [256, 256]

    # Relative normalization: stretch variance to its own observed range,
    # so confidence reflects RELATIVE uncertainty across this image,
    # rather than being squashed by an arbitrary fixed scale.
    v_min, v_max = variance.min(), variance.max()
    if v_max - v_min < 1e-12:
        # Degenerate case: truly zero variance everywhere (rare, e.g. flat input)
        confidence = np.ones_like(variance)
    else:
        norm_variance = (variance - v_min) / (v_max - v_min)   # now spans [0, 1]
        confidence = 1.0 - norm_variance                        # invert: low variance = high confidence

    return mean_pred, confidence
def save_confidence_heatmap(confidence, output_path="confidence/confidence_map.png"):
    """
    Saves the confidence map as a color heatmap image.
    Yellow/bright = high confidence, dark/purple = low confidence.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.imshow(confidence, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Confidence (0 = low, 1 = high)")
    plt.title("Confidence Estimation Map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[confidence] Saved heatmap to {output_path}")


if __name__ == "__main__":
    from gan.predict import predict as b_predict

    cloudy = np.load("data/synthetic/cloudy/patch_0143.npy").astype(np.float32)
    sar = np.load("data/synthetic/sar/patch_0143.npy").astype(np.float32)
    temporal = None

    def real_predict(cloudy, sar, temporal):
        return b_predict(cloudy, sar, temporal=temporal, checkpoint_path="checkpoints/ckpt_best.pt")

    mean_pred, conf = confidence_map(cloudy, sar, temporal, real_predict, noise_std=0.1)
    save_confidence_heatmap(conf)
    print(f"Mean prediction shape: {mean_pred.shape}")
    print(f"Confidence map shape: {conf.shape}")
    print(f"Confidence range: min={conf.min():.3f}, max={conf.max():.3f}, mean={conf.mean():.3f}")