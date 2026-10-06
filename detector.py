"""Local face detection via InsightFace (SCRFD, ONNX Runtime).

No network calls after the one-time model download (~300 MB into ~/.insightface).

Each detection also gets a `face_confidence` score in [0, 1]: how confident the
pipeline is that the box actually contains a face. It combines the detector's
own score with 5-point landmark geometry plausibility (eye distance vs. face
width, landmarks inside the box, eyes-above-mouth ordering). Once enough human
review labels exist, `learn.py tune` fits a tiny logistic regression on those
features and the detector uses the learned mapping instead of the heuristic.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

# Feature order shared with learn.py — do not reorder without retraining.
FACE_FEATURES = ["det_score", "eye_ratio", "kps_inside", "order_ok"]


def _pick_providers(use_gpu: bool) -> list[str]:
    """Only request CUDA when this onnxruntime build actually has it.

    Requesting CUDAExecutionProvider from a CPU-only build makes onnxruntime
    emit a UserWarning on every session creation (harmless, but noisy).
    """
    available = ort.get_available_providers()
    if use_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def kps_features(kps, bbox: tuple) -> dict | None:
    """Geometry plausibility features from 5-point landmarks."""
    if kps is None:
        return None
    kps = np.asarray(kps, dtype=float)
    if kps.shape != (5, 2):
        return None
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    eye_dist = float(np.linalg.norm(kps[0] - kps[1]))
    inside = sum(1 for px, py in kps if x1 <= px <= x2 and y1 <= py <= y2) / 5.0
    eyes_y = (kps[0][1] + kps[1][1]) / 2.0
    mouth_y = (kps[3][1] + kps[4][1]) / 2.0
    return {
        "eye_ratio": eye_dist / w,
        "kps_inside": inside,
        "order_ok": 1.0 if mouth_y > eyes_y else 0.0,
    }


def heuristic_face_confidence(det_score: float, feats: dict | None) -> float:
    """Fallback confidence before any human labels exist."""
    if feats is None:
        return float(det_score * 0.7)
    geo = 1.0
    if not (0.12 <= feats["eye_ratio"] <= 0.7):
        geo *= 0.5
    if feats["kps_inside"] < 1.0:
        geo *= 0.6
    if feats["order_ok"] < 1.0:
        geo *= 0.7
    return float(min(1.0, det_score * geo))


class FaceDetector:
    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        use_gpu: bool = True,
        learned_params: str | Path | None = None,
    ):
        providers = _pick_providers(use_gpu)
        self.app = FaceAnalysis(name=model_pack, providers=providers)
        # ctx_id=0 selects the first GPU when CUDA provider is active; harmless on CPU.
        self.app.prepare(ctx_id=0, det_size=tuple(det_size), det_thresh=det_thresh)
        # Which provider actually got picked (useful for the startup log line).
        self.active_providers = self.app.models["detection"].session.get_providers()

        # Learned face-confidence mapping from human review labels (see learn.py).
        self.face_model: dict | None = None
        if learned_params:
            p = Path(learned_params)
            if p.exists():
                data = json.loads(p.read_text()).get("face_conf") or {}
                if data.get("n", 0) >= 50:
                    self.face_model = data

    def _learned_confidence(self, det_score: float, feats: dict | None) -> float | None:
        if self.face_model is None or feats is None:
            return None
        m = self.face_model
        x = np.array([det_score] + [feats[k] for k in FACE_FEATURES[1:]], dtype=float)
        mean = np.array(m["mean"], dtype=float)
        std = np.array(m["std"], dtype=float)
        w = np.array(m["weights"], dtype=float)
        z = float(((x - mean) / np.maximum(std, 1e-6)) @ w[:-1] + w[-1])
        return float(1.0 / (1.0 + math.exp(-z)))

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """Return [{'bbox', 'score', 'face_confidence', 'kps_feats'}, ...]."""
        out = []
        for f in self.app.get(frame_bgr):
            x1, y1, x2, y2 = f.bbox.astype(int)
            bbox = (int(x1), int(y1), int(x2), int(y2))
            score = float(f.det_score)
            feats = kps_features(getattr(f, "kps", None), bbox)
            learned = self._learned_confidence(score, feats)
            conf = learned if learned is not None else heuristic_face_confidence(score, feats)
            out.append({
                "bbox": bbox,
                "score": score,
                "face_confidence": conf,
                "kps_feats": feats,
            })
        return out
