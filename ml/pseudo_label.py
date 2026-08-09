from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def norm(v):
    d = math.hypot(v[0], v[1])
    return (v[0] / d, v[1] / d) if d > 1e-6 else (0.0, 0.0)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def white_fraction(mask, cx, cy, radius):
    h, w = mask.shape
    x1 = max(0, int(cx - radius)); x2 = min(w, int(cx + radius + 1))
    y1 = max(0, int(cy - radius)); y2 = min(h, int(cy + radius + 1))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((mask[y1:y2, x1:x2] > 0).mean())


def pseudo_label(image: np.ndarray):
    """Return a conservative 3-class mask or None.

    This is only a bootstrap tool. It deliberately rejects ambiguous frames.
    Human-reviewed real masks remain the source of truth for final fine-tuning.
    """
    h, w = image.shape[:2]
    x = int(w * 0.1787); y = int(h * 0.2080)
    right = int(w * 0.8215); bottom = int(h * 0.9410)
    roi = image[y:bottom, x:right]
    rh, rw = roi.shape[:2]
    if rw < 250 or rh < 150:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 150), (179, 85, 255))
    border = max(4, int(rw * 0.008))
    cv2.rectangle(white, (0, 0), (rw - 1, rh - 1), 0, border)

    lines = cv2.HoughLinesP(
        white, 1, np.pi / 360.0,
        threshold=18,
        minLineLength=max(35, int(rw * 0.045)),
        maxLineGap=12,
    )
    if lines is None:
        return None

    segs = []
    for raw in lines[:, 0]:
        p1 = (float(raw[0]), float(raw[1]))
        p2 = (float(raw[2]), float(raw[3]))
        length = dist(p1, p2)
        if length > rw * 0.04:
            segs.append((p1, p2, length))
    segs.sort(key=lambda s: s[2], reverse=True)
    segs = segs[:50]

    best = None
    best_score = -1e18
    for seg in segs[:15]:
        if seg[2] < rw * 0.18:
            continue
        for junction, other_end in ((seg[0], seg[1]), (seg[1], seg[0])):
            local = white_fraction(white, junction[0], junction[1], max(12, int(rw * 0.022)))
            if local < 0.04:
                continue

            search = max(10, int(rw * 0.025))
            sample_r = max(5, int(rw * 0.008))
            solid = 0.0
            for yy in range(int(junction[1] - search), int(junction[1] + search) + 1, 4):
                for xx in range(int(junction[0] - search), int(junction[0] + search) + 1, 4):
                    solid = max(solid, white_fraction(white, xx, yy, sample_r))
            # A filled white cue ball is rejected; the contact ring is hollow.
            if solid > 0.82:
                continue

            score = seg[2] + local * rw * 0.30 - solid * rw * 0.20
            if score > best_score:
                best_score = score
                best = (seg, junction, other_end)

    if best is None:
        return None

    incoming_seg, junction, other_end = best
    incoming = norm((junction[0] - other_end[0], junction[1] - other_end[1]))
    branches = []
    for seg in segs:
        if seg is incoming_seg or seg[2] > rw * 0.14 or seg[2] < rw * 0.012:
            continue
        d1 = dist(seg[0], junction); d2 = dist(seg[1], junction)
        if min(d1, d2) > rw * 0.04:
            continue
        near, far = (seg[0], seg[1]) if d1 <= d2 else (seg[1], seg[0])
        outward = norm((far[0] - junction[0], far[1] - junction[1]))
        if dot(outward, incoming) < -0.35:
            continue
        if dist(far, junction) < rw * 0.015:
            continue
        quality = seg[2] - min(d1, d2) * 0.30
        branches.append((quality, near, far, outward))

    branches.sort(key=lambda b: b[0], reverse=True)
    kept = []
    for branch in branches:
        duplicate = False
        for old in kept:
            cosine = max(-1.0, min(1.0, dot(branch[3], old[3])))
            angle = math.degrees(math.acos(cosine))
            if angle < 8.0:
                duplicate = True
                break
        if not duplicate:
            kept.append(branch)
        if len(kept) == 2:
            break

    if not kept:
        return None

    result = np.zeros((h, w), np.uint8)
    thickness = max(3, int(rw * 0.004))

    def gp(p):
        return (int(round(p[0] + x)), int(round(p[1] + y)))

    cv2.line(result, gp(other_end), gp(junction), 1, thickness, cv2.LINE_8)
    for _, near, far, _ in kept:
        cv2.line(result, gp(near), gp(far), 1, thickness, cv2.LINE_8)
    ring_r = max(10, int(rw * 0.016))
    cv2.circle(result, gp(junction), ring_r, 2, thickness, cv2.LINE_8)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    out_images = args.output / "images"
    out_masks = args.output / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    accepted = 0
    rejected = 0
    for image_path in sorted(args.images.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        mask = pseudo_label(image)
        if mask is None:
            rejected += 1
            continue
        cv2.imwrite(str(out_images / f"{image_path.stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        cv2.imwrite(str(out_masks / f"{image_path.stem}.png"), mask)
        accepted += 1

    print(f"accepted={accepted} rejected={rejected} output={args.output}")


if __name__ == "__main__":
    main()
