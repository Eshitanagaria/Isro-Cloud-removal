
# =============================================================
# Person A — LULC Validation Script
# ISRO Antariksh Hackathon 2025
# Runs RandomForest on spectral indices, compares accuracy
# on clean vs cloudy vs reconstructed patches
# =============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE = '/content/drive/MyDrive/isro_hackathon/data/synthetic'
N_PATCHES = 126

def compute_indices(patch):
    R, G, NIR = patch[0], patch[1], patch[3]
    eps = 1e-8
    NDVI = (NIR - R) / (NIR + R + eps)
    NDWI = (G - NIR) / (G + NIR + eps)
    BAND_RATIO = R / (NIR + eps)
    return np.stack([NDVI, NDWI, BAND_RATIO], axis=0)

def auto_label(indices):
    labels = np.full(indices[0].shape, 2, dtype=np.int32)
    labels[indices[0] > 0.3] = 0
    labels[indices[1] > 0.1] = 1
    return labels

def build_dataset(indices_list, label_list):
    X = np.vstack([i.reshape(3, -1).T for i in indices_list])
    y = np.concatenate([l.reshape(-1) for l in label_list])
    return X, y

clean_idx, cloudy_idx, labels = [], [], []
for i in range(N_PATCHES):
    s = f"{i:04d}"
    c  = np.load(f"{BASE}/clean/patch_{s}.npy")
    cl = np.load(f"{BASE}/cloudy/patch_{s}.npy")
    ci = compute_indices(c)
    clean_idx.append(ci)
    cloudy_idx.append(compute_indices(cl))
    labels.append(auto_label(ci))

X_clean,  y = build_dataset(clean_idx,  labels)
X_cloudy, _ = build_dataset(cloudy_idx, labels)

X_tr, X_te, y_tr, y_te = train_test_split(X_clean, y, test_size=0.2, random_state=42)
_, X_te_cloudy, _, _   = train_test_split(X_cloudy, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

acc_clean  = accuracy_score(y_te, rf.predict(X_te))
acc_cloudy = accuracy_score(y_te, rf.predict(X_te_cloudy))

print(f"Clean  : {acc_clean*100:.2f}%")
print(f"Cloudy : {acc_cloudy*100:.2f}%")
print(f"Delta  : {(acc_clean-acc_cloudy)*100:.2f}% degradation from cloud")

results = pd.DataFrame({
    "Source": ["Clean (ground truth)", "Cloudy (degraded)", "Reconstructed (TBD)"],
    "LULC Accuracy (%)": [round(acc_clean*100,2), round(acc_cloudy*100,2), "TBD"]
})
results.to_csv("lulc_results.csv", index=False)
print(results.to_string(index=False))
