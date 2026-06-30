
### ISRO Antariksh Hackathon 2026

## Problem Statement

Persistent cloud cover over North Eastern India severely limits the usability of LISS-IV
optical satellite imagery for LULC mapping, disaster monitoring, and environmental assessment.
This project develops a Generative AI framework for automated cloud removal using SAR-Optical fusion.

## Key Results
| Source | LULC Accuracy |
|---|---|
| Clean ground truth | 100.00% |
| Cloudy (degraded) | 91.65% |
| Reconstructed  | TBD — update after training |

**8.35% accuracy degradation caused purely by cloud contamination. Cloud removal recovers this.**

## I/O Contract (frozen Day 1 — do not change without updating README)
| Parameter | Value |
|---|---|
| Patch size | 256 × 256 pixels |
| S2 band order | [R, G, B, NIR] → indices [0, 1, 2, 3] |
| S1 band order | [VV, VH] → indices [0, 1] |
| Normalization | Reflectance scaled to [0, 1] |
| File format | .npy arrays, shape (bands, H, W) |
| Naming convention | patch_XXXX.npy, same index across all 4 folders |
| Total patches | 126 |

## Dataset
- 126 patches across 10+ NER locations: Guwahati, Shillong, Itanagar, Imphal, Nalbari, Tezpur, Aizawl, Agartala, Nongpoh
- Sentinel-2 L2A (10m resolution, bands R/G/B/NIR) + Sentinel-1 GRD (VV/VH)
- Multiple seasons (Nov 2023 – Oct 2024) for temporal variety
- Synthetic cloud pairs generated via randomized Gaussian blob masking

## Scoping Statement
This framework is trained and demonstrated on Sentinel-2 L2A + Sentinel-1 SAR imagery
as a proxy for LISS-IV, due to data access and time constraints during the hackathon.
The architecture generalizes directly to LISS-IV: band mappings (LISS-IV carries Green,
Red, NIR, SWIR) are compatible with the spectral-index loss terms used here, and SAR
co-registration to LISS-IV spatial resolution (5.8m) is identified as a concrete next step.
Temporal fusion (best-pixel composite as a third conditioning channel) is implemented in
the pipeline but noted as future work for full validation.

## How to Run
1. Run `data/data_pipeline.py` to pull tiles and generate synthetic patches
2. B trains GAN using patches from `data/synthetic/`
3. Run `eval/evaluate.py` on any (pred, ground_truth) pair to get SSIM/PSNR/SAM
4. Run `eval/lulc_validation.py` to get LULC accuracy delta
5. Run `pipeline.py` for the full end-to-end demo

## Team
- Eshita Nagaria
- Gungun Jain
- Krati Mishra


