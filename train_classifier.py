#!/usr/bin/env python3
"""v2 upgrade path: train a tiny blurred-vs-sharp face classifier, fully offline.

You supply a folder of SHARP face crops (e.g. cropped from FFHQ / WIDER FACE
with this project's detector — the real footage is never needed for training).
The 'censored' class is synthesized with three augmentation families that cover
real-world censorship styles:

  - gaussian blur   (kernel 15..51)
  - mosaic pixelation (downscale to 6..24 px, nearest-neighbor upscale)
  - black bar / full mask (random band or whole-face fill)

Trains MobileNetV3-Small (2 classes), exports ONNX, ready for onnxruntime.

    pip install torch torchvision   # CPU wheels are fine
    python train_classifier.py --faces /data/face_crops --out models/blur_classifier.onnx --epochs 12

Integration sketch for blurcheck.py:

    import onnxruntime as ort
    sess = ort.InferenceSession("models/blur_classifier.onnx",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    # preprocess crop -> 128x128 RGB float32 normalized like ImageNet
    prob_blurred = softmax(sess.run(None, {"input": blob})[0])[0, 1]
    censored = prob_blurred > 0.5
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("train_classifier")

IMG = 128  # model input size


# ---------------------------------------------------------------- augmentations
def _rand_odd(lo: int, hi: int) -> int:
    v = random.randint(lo, hi)
    return v if v % 2 == 1 else v + 1


def aug_gaussian(img: np.ndarray) -> np.ndarray:
    k = _rand_odd(15, 51)
    return cv2.GaussianBlur(img, (k, k), 0)


def aug_pixelate(img: np.ndarray) -> np.ndarray:
    s = random.randint(6, 24)
    small = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)


def aug_blackbar(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    mode = random.random()
    if mode < 0.5:  # eye bar
        y0 = random.randint(int(h * 0.25), int(h * 0.45))
        y1 = y0 + random.randint(int(h * 0.12), int(h * 0.25))
        out[y0:y1, :] = 0
    elif mode < 0.8:  # horizontal band anywhere
        y0 = random.randint(0, int(h * 0.6))
        out[y0:y0 + random.randint(int(h * 0.2), int(h * 0.5)), :] = 0
    else:  # full mask
        out[:] = 0
    # slight noise so "solid" isn't trivially zero-std in every case
    if random.random() < 0.3:
        noise = np.random.randint(0, 8, out.shape, dtype=np.uint8)
        out = cv2.add(out, noise)
    return out


AUGMENTERS = [aug_gaussian, aug_pixelate, aug_blackbar]


# ---------------------------------------------------------------- dataset
def load_faces(faces_dir: Path) -> list[np.ndarray]:
    imgs = []
    for p in sorted(faces_dir.rglob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        imgs.append(cv2.resize(img, (IMG, IMG)))
    if len(imgs) < 100:
        raise SystemExit(f"only {len(imgs)} face crops found — need at least ~100 (more is better)")
    log.info("loaded %d sharp face crops", len(imgs))
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.models import mobilenet_v3_small

    sharp = load_faces(args.faces)
    rng = random.Random(0)
    rng.shuffle(sharp)
    n_val = max(10, int(len(sharp) * args.val_frac))
    val_sharp, train_sharp = sharp[:n_val], sharp[n_val:]

    def preprocess(img: np.ndarray) -> np.ndarray:
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        return ((x - mean) / std).transpose(2, 0, 1)

    def make_split(crops: list[np.ndarray]) -> TensorDataset:
        xs, ys = [], []
        for c in crops:
            xs.append(preprocess(c)); ys.append(0)                       # sharp
            aug = random.choice(AUGMENTERS)(c)
            xs.append(preprocess(aug)); ys.append(1)                     # censored
        return TensorDataset(torch.from_numpy(np.stack(xs)),
                             torch.tensor(ys, dtype=torch.long))

    train_ds, val_ds = make_split(train_sharp), make_split(val_sharp)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            tot += len(xb)
        model.eval()
        vtot, vcorrect = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vcorrect += (model(xb).argmax(1) == yb).sum().item()
                vtot += len(xb)
        log.info("epoch %2d  train acc %.3f  loss %.4f  val acc %.3f",
                 epoch, correct / tot, loss_sum / tot, vcorrect / vtot)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, IMG, IMG, device=device)
    torch.onnx.export(model.eval(), dummy, str(args.out),
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=14)
    log.info("exported %s", args.out)


if __name__ == "__main__":
    main()
