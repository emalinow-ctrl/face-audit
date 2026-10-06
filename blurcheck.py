"""Blur / censorship quality metrics for face crops.

All classic CV, no ML, no network. Metrics run on the face crop at NATIVE
video resolution — never on the downscaled detection frame.

What the metrics actually measure (validated against compressed 1080p footage;
naive intuition fails here, so read carefully):
  - edge_density  : Canny edge energy. Gaussian/defocus blur destroys edges, so
                    blurred faces score low; sharp AND pixelated faces score high.
  - flat_frac + laplacian_var : mosaic/pixelation censorship is large perfectly
                    flat blocks separated by razor boundaries => very high flat
                    fraction AND high laplacian variance. Heavy makeup on a sharp
                    face can also spike laplacian variance, but its gradients are
                    everywhere (low flat fraction), which tells the two apart.
  - solid mask check: near-zero std => black bar / sticker cover => censored.

Decision order matters: solid mask -> pixelated -> blurred -> sharp.
"""
from __future__ import annotations

import cv2
import numpy as np


def laplacian_var(roi_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())


def highfreq_ratio(roi_gray: np.ndarray) -> float:
    h, w = roi_gray.shape
    if h < 8 or w < 8:
        return 0.0
    F = np.fft.fftshift(np.fft.fft2(roi_gray.astype(np.float32)))
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    power = np.abs(F) ** 2
    total = power.sum()
    if total <= 0:
        return 0.0
    hf = power[dist > min(h, w) * 0.25].sum()
    return float(hf / total)


def edge_density(roi_gray: np.ndarray) -> float:
    edges = cv2.Canny(roi_gray, 50, 150)
    return float(edges.mean() / 255.0)


def flat_frac(roi_gray: np.ndarray, eps: float = 6.0) -> float:
    """Fraction of pixels with near-zero gradient magnitude.

    Mosaic/pixelated images are mostly perfectly flat blocks => very high.
    Sharp faces (even with heavy makeup) and blurred faces score much lower.
    """
    gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float((mag < eps).mean())


def assess_face(roi_gray: np.ndarray, cfg: dict) -> dict:
    """Return {'censored': bool, 'metrics': {...}, 'reason': str}."""
    std = float(roi_gray.std())
    if std < cfg["solid_mask_std"]:
        return {
            "censored": True,
            "reason": "solid_mask",
            "metrics": {"std": std, "lap_var": 0.0, "hf_ratio": 0.0, "edge_den": 0.0},
        }
    metrics = {
        "std": std,
        "lap_var": laplacian_var(roi_gray),
        "hf_ratio": highfreq_ratio(roi_gray),
        "edge_den": edge_density(roi_gray),
        "flat_frac": flat_frac(roi_gray),
    }
    # 1) mosaic / pixelation: flat blocks + razor block boundaries
    if (metrics["flat_frac"] > cfg["pixel_flat_threshold"]
            and metrics["lap_var"] > cfg["pixel_lap_threshold"]):
        return {"censored": True, "reason": "pixelated", "metrics": metrics}
    # 2) gaussian blur / defocus: edge structure destroyed
    if metrics["edge_den"] < cfg["edge_threshold"]:
        return {"censored": True, "reason": "blurred", "metrics": metrics}
    return {"censored": False, "reason": "sharp", "metrics": metrics}
