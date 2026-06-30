
# =============================================================
# Person A — Data Pipeline Script
# ISRO Antariksh Hackathon 2025
# Pulls Sentinel-2 + Sentinel-1 tiles from GEE, generates
# synthetic cloud pairs, saves .npy patches to Google Drive
# =============================================================

# REQUIREMENTS:
# pip install earthengine-api geemap rasterio scipy scikit-image

import ee, geemap, rasterio, numpy as np, os
from scipy.ndimage import gaussian_filter

# --- CONFIG (edit these) ---
OUTPUT_BASE = '/content/drive/MyDrive/isro_hackathon/data/synthetic'
PATCH_SIZE  = 256
CRS         = 'EPSG:32646'

# --- INIT ---
ee.Authenticate()
ee.Initialize(project='YOUR_PROJECT_ID')
patch_counter = 0

# --- CLOUD MASK GENERATOR ---
def generate_cloud_mask(size=256, num_blobs=None, seed=None):
    rng = np.random.default_rng(seed)
    if num_blobs is None:
        num_blobs = rng.integers(1, 4)
    mask = np.zeros((size, size), dtype=np.float32)
    for _ in range(num_blobs):
        cx, cy = rng.integers(0, size, size=2)
        radius = rng.integers(size // 8, size // 3)
        yy, xx = np.ogrid[:size, :size]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        blob = np.clip(1 - dist / radius, 0, 1)
        mask = np.maximum(mask, blob)
    mask = gaussian_filter(mask, sigma=size // 20)
    return np.clip(mask, 0, 1)

# --- CLOUD APPLICATOR ---
def apply_cloud(patch, mask, cloud_brightness=0.9):
    patch_norm = patch.astype(np.float32)
    max_val = np.percentile(patch_norm, 99)
    patch_scaled = np.clip(patch_norm / max_val, 0, 1)
    cloud_layer = np.ones_like(patch_scaled) * cloud_brightness
    mask_3d = mask[np.newaxis, :, :]
    cloudy = patch_scaled * (1 - mask_3d) + cloud_layer * mask_3d
    return cloudy, patch_scaled

# --- TILE PROCESSOR ---
def process_tile(aoi_coords, date_range, tile_name, cloud_thresh=10):
    global patch_counter
    aoi = ee.Geometry.Rectangle(aoi_coords)

    s2_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi).filterDate(*date_range)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
        .sort("CLOUDY_PIXEL_PERCENTAGE"))
    if s2_col.size().getInfo() == 0:
        print(f"[{tile_name}] No S2 images found, skipping."); return 0

    s1_col = (ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi).filterDate(*date_range)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    if s1_col.size().getInfo() == 0:
        print(f"[{tile_name}] No S1 images found, skipping."); return 0

    s2_img = s2_col.first().select(["B4","B3","B2","B8"])
    s1_img = s1_col.first().select(["VV","VH"])

    s2_path = f"/content/data/{tile_name}_s2.tif"
    s1_path = f"/content/data/{tile_name}_s1.tif"
    os.makedirs("/content/data", exist_ok=True)

    geemap.ee_export_image(s2_img, filename=s2_path, scale=10, region=aoi, crs=CRS, file_per_band=False)
    geemap.ee_export_image(s1_img, filename=s1_path, scale=10, region=aoi, crs=CRS, file_per_band=False)

    with rasterio.open(s2_path) as src: s2_arr = src.read()
    with rasterio.open(s1_path) as src: s1_arr = src.read()

    if s2_arr.shape[1:] != s1_arr.shape[1:]:
        print(f"[{tile_name}] Shape mismatch, skipping."); return 0

    for sub in ["clean","cloudy","mask","sar"]:
        os.makedirs(f"{OUTPUT_BASE}/{sub}", exist_ok=True)

    saved = 0
    _, H, W = s2_arr.shape
    for top in range(0, H - PATCH_SIZE + 1, PATCH_SIZE):
        for left in range(0, W - PATCH_SIZE + 1, PATCH_SIZE):
            s2_p = s2_arr[:, top:top+PATCH_SIZE, left:left+PATCH_SIZE]
            s1_p = s1_arr[:, top:top+PATCH_SIZE, left:left+PATCH_SIZE]
            mask = generate_cloud_mask(size=PATCH_SIZE, seed=patch_counter)
            cloudy, clean = apply_cloud(s2_p, mask)
            i = f"{patch_counter:04d}"
            np.save(f"{OUTPUT_BASE}/clean/patch_{i}.npy", clean)
            np.save(f"{OUTPUT_BASE}/cloudy/patch_{i}.npy", cloudy)
            np.save(f"{OUTPUT_BASE}/mask/patch_{i}.npy", mask)
            np.save(f"{OUTPUT_BASE}/sar/patch_{i}.npy", s1_p.astype(np.float32))
            patch_counter += 1; saved += 1

    print(f"[{tile_name}] Saved {saved} patches. Total: {patch_counter}")
    return saved

# --- RUN ---
tiles = [
    {"aoi_coords": [91.70,26.10,91.78,26.18], "tile_name": "guwahati_1",  "date_range": ("2023-11-01","2024-04-30")},
    {"aoi_coords": [91.85,25.55,91.93,25.63], "tile_name": "shillong_1",  "date_range": ("2023-11-01","2024-04-30")},
    {"aoi_coords": [93.93,27.06,94.01,27.14], "tile_name": "itanagar_1",  "date_range": ("2023-11-01","2024-04-30")},
    {"aoi_coords": [93.92,24.78,94.00,24.86], "tile_name": "imphal_1",    "date_range": ("2023-11-01","2024-04-30")},
    {"aoi_coords": [91.70,26.10,91.78,26.18], "tile_name": "guwahati_2",  "date_range": ("2024-05-01","2024-10-31")},
    {"aoi_coords": [91.85,25.55,91.93,25.63], "tile_name": "shillong_2",  "date_range": ("2024-05-01","2024-10-31")},
    {"aoi_coords": [93.93,27.06,94.01,27.14], "tile_name": "itanagar_2",  "date_range": ("2024-05-01","2024-10-31")},
    {"aoi_coords": [93.92,24.78,94.00,24.86], "tile_name": "imphal_2",    "date_range": ("2024-05-01","2024-10-31")},
    {"aoi_coords": [91.27,26.13,91.35,26.21], "tile_name": "nalbari_1",   "date_range": ("2023-11-01","2024-04-30")},
    {"aoi_coords": [92.79,26.58,92.87,26.66], "tile_name": "tezpur_1",    "date_range": ("2023-11-01","2024-04-30")},
    {"aoi_coords": [92.94,24.32,93.02,24.40], "tile_name": "aizawl_1",    "date_range": ("2023-11-01","2024-04-30")},
    {"aoi_coords": [91.36,23.84,91.44,23.92], "tile_name": "agartala_1",  "date_range": ("2023-11-01","2024-04-30")},
    {"aoi_coords": [91.36,25.57,91.44,25.65], "tile_name": "nongpoh_1",   "date_range": ("2023-11-01","2024-04-30")},
]

total = sum(process_tile(**t) for t in tiles)
print(f"DONE. Total patches: {total}")
