import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gan.predict import predict as b_predict
from fallback_logic.fallback import reconstruct_with_fallback
from confidence.confidence import confidence_map


def real_predict(cloudy, sar, temporal):
    return b_predict(cloudy, sar, temporal=temporal, checkpoint_path="checkpoints/ckpt_best.pt")

def run_pipeline(cloudy, sar, temporal, mask):
    """
    Full end-to-end: cloudy input -> fallback-aware reconstruction
    -> confidence map -> final packaged output.
    """
    # Step 1: reconstruct with fallback logic (handles >80% cloud case)
    recon, cloud_pct, mode = reconstruct_with_fallback(
        cloudy, sar, temporal, mask, real_predict
    )

    # Step 2: confidence map (variance across perturbed passes)
    mean_pred, confidence = confidence_map(cloudy, sar, temporal, real_predict)

    return {
        "reconstructed": recon,
        "cloud_percentage": cloud_pct,
        "mode": mode,
        "confidence_map": confidence,
    }

if __name__ == "__main__":
    print("=== Full pipeline test (REAL data + REAL checkpoint) ===")

    cloudy = np.load("data/synthetic/cloudy/patch_0143.npy").astype(np.float32)
    sar = np.load("data/synthetic/sar/patch_0143.npy").astype(np.float32)
    mask = np.load("data/synthetic/mask/patch_0143.npy").astype(np.float32)

    # temporal is optional — we don't have a real one yet, so skip it
    temporal = None

    CHECKPOINT_PATH = "checkpoints/ckpt_best.pt"

    def real_predict(cloudy, sar, temporal):
        return b_predict(cloudy, sar, temporal=temporal, checkpoint_path=CHECKPOINT_PATH)

    result = run_pipeline(cloudy, sar, temporal, mask)

    print(f"Cloud %: {result['cloud_percentage']:.2f}")
    print(f"Mode: {result['mode']}")
    print(f"Reconstructed shape: {result['reconstructed'].shape}")
    print(f"Confidence map shape: {result['confidence_map'].shape}")
    print(f"Confidence mean: {result['confidence_map'].mean():.3f}")
    print("Pipeline OK ✓")