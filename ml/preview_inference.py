from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


W, H = 512, 288
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-8)


def preprocess(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis = image.copy()
    layer = np.zeros_like(vis)
    layer[mask == 1] = (0, 255, 0)
    layer[mask == 2] = (0, 0, 255)
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.48
    return (vis * (1 - alpha) + layer * alpha).astype(np.uint8)


def largest_ring_center(mask: np.ndarray) -> tuple[float, float] | None:
    ring = (mask == 2).astype(np.uint8) * 255
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(ring, 8)
    best = None
    best_area = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best = tuple(float(v) for v in centroids[i])
    return best if best_area >= 6 else None


def fit_local_guide(mask: np.ndarray, center: tuple[float, float], radius: int = 54) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Debug geometry: fit up to two guide components near the ring.

    This is intentionally conservative. The Android implementation will add
    temporal tracking and calibrated table bounds after the trained model is ready.
    """
    cx, cy = center
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    roi = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
    guide = ((mask == 1) & roi).astype(np.uint8) * 255
    guide = cv2.morphologyEx(guide, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(guide, 8)
    fits = []
    comps = sorted(range(1, n), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    for i in comps[:5]:
        ys, xs = np.where(labels == i)
        if len(xs) < 10:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)]
        proj = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
        p1 = (int(round(x0 + vx * proj.min())), int(round(y0 + vy * proj.min())))
        p2 = (int(round(x0 + vx * proj.max())), int(round(y0 + vy * proj.max())))
        fits.append((p1, p2))
        if len(fits) == 2:
            break
    return fits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot read {args.image}")

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0].name
    logits = session.run(None, {inp: preprocess(image)})[0]
    probs = softmax(logits, 1)[0]
    mask = probs.argmax(0).astype(np.uint8)

    vis = overlay_mask(image, mask)
    center = largest_ring_center(mask)
    if center is not None:
        sx = image.shape[1] / W
        sy = image.shape[0] / H
        cv2.drawMarker(vis, (int(center[0] * sx), int(center[1] * sy)), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        for p1, p2 in fit_local_guide(mask, center):
            a = (int(p1[0] * sx), int(p1[1] * sy))
            b = (int(p2[0] * sx), int(p2[1] * sy))
            cv2.line(vis, a, b, (255, 255, 0), 3, cv2.LINE_AA)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), vis)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
