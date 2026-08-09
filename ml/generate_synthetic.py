from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np


W, H = 1280, 720


def rand_color(rng: random.Random) -> tuple[int, int, int]:
    palette = [
        (20, 70, 220), (40, 180, 255), (30, 190, 80), (170, 60, 180),
        (210, 100, 45), (30, 30, 30), (210, 210, 210), (70, 90, 170),
    ]
    return rng.choice(palette)


def unit(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)], np.float32)


def draw_ball(img: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int], rng: random.Random) -> None:
    cv2.circle(img, center, radius + 3, (15, 15, 15), -1, cv2.LINE_AA)
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)
    if rng.random() < 0.55:
        cv2.ellipse(img, center, (radius, max(3, radius // 3)), rng.uniform(-35, 35), 0, 360, (245, 245, 245), -1, cv2.LINE_AA)
    if rng.random() < 0.65:
        rr = max(4, radius // 3)
        cv2.circle(img, center, rr, (245, 245, 245), -1, cv2.LINE_AA)
        n = str(rng.randint(1, 15))
        cv2.putText(img, n, (center[0] - rr // 2, center[1] + rr // 2), cv2.FONT_HERSHEY_SIMPLEX, rr / 18.0, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.circle(img, (center[0] - radius // 3, center[1] - radius // 3), max(2, radius // 5), (255, 255, 255), -1, cv2.LINE_AA)


def make_sample(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    # BGR felt with mild spatial/noise variation.
    felt = rng.choice([(190, 125, 25), (170, 110, 18), (70, 135, 35), (150, 95, 25)])
    img = np.empty((H, W, 3), np.uint8)
    img[:] = felt
    noise = rng.uniform(3, 10)
    img = np.clip(img.astype(np.float32) + np.random.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    mask = np.zeros((H, W), np.uint8)

    # Rails/pockets and UI-like hard negatives.
    rail = 34
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (45, 55, 80), rail)
    for p in [(rail, rail), (W // 2, rail), (W - rail, rail), (rail, H - rail), (W // 2, H - rail), (W - rail, H - rail)]:
        cv2.circle(img, p, 32, (5, 5, 5), -1, cv2.LINE_AA)

    for _ in range(rng.randint(8, 16)):
        x = rng.randint(90, W - 90)
        y = rng.randint(85, H - 85)
        draw_ball(img, (x, y), rng.randint(15, 20), rand_color(rng), rng)

    # Random cue-like line as a strong negative.
    if rng.random() < 0.8:
        a = (rng.randint(-150, W + 150), rng.randint(-100, H + 100))
        ang = rng.uniform(0, math.pi)
        b = (int(a[0] + math.cos(ang) * rng.randint(260, 600)), int(a[1] + math.sin(ang) * rng.randint(260, 600)))
        cv2.line(img, a, b, (35, 80, 130), rng.randint(7, 12), cv2.LINE_AA)
        cv2.line(img, a, b, (80, 145, 210), rng.randint(2, 4), cv2.LINE_AA)

    # Native guide geometry.
    jx = rng.randint(240, W - 220)
    jy = rng.randint(150, H - 150)
    incoming_angle = rng.uniform(-math.pi, math.pi)
    incoming_dir = unit(incoming_angle)
    incoming_len = rng.randint(220, 540)
    start = np.array([jx, jy], np.float32) - incoming_dir * incoming_len
    junction = np.array([jx, jy], np.float32)

    guide_thickness = rng.randint(3, 5)
    outline_thickness = guide_thickness + rng.randint(2, 4)

    def draw_native_segment(a: np.ndarray, b: np.ndarray) -> None:
        pa = tuple(np.round(a).astype(int))
        pb = tuple(np.round(b).astype(int))
        cv2.line(img, pa, pb, (12, 12, 12), outline_thickness, cv2.LINE_AA)
        cv2.line(img, pa, pb, (245, 245, 245), guide_thickness, cv2.LINE_AA)
        cv2.line(mask, pa, pb, 1, guide_thickness + 1, cv2.LINE_8)

    draw_native_segment(start, junction)

    # Contact ring: black-ish outline + white ring, transparent center.
    ring_r = rng.randint(14, 20)
    c = (jx, jy)
    cv2.circle(img, c, ring_r + 4, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.circle(img, c, ring_r, (245, 245, 245), 3, cv2.LINE_AA)
    cv2.circle(mask, c, ring_r, 2, 4, cv2.LINE_8)

    # One or two short branches. One may be almost straight through.
    branch_count = 1 if rng.random() < 0.45 else 2
    base_angles = []
    if rng.random() < 0.45:
        base_angles.append(incoming_angle)
    else:
        base_angles.append(incoming_angle + rng.uniform(-1.25, 1.25))
    if branch_count == 2:
        second = incoming_angle + rng.choice([-1, 1]) * rng.uniform(0.30, 1.25)
        if abs((second - base_angles[0] + math.pi) % (2 * math.pi) - math.pi) < 0.22:
            second += 0.35
        base_angles.append(second)

    for angle in base_angles:
        d = unit(angle)
        gap = rng.uniform(ring_r * 0.45, ring_r * 0.95)
        length = rng.uniform(30, 75)
        a = junction + d * gap
        b = junction + d * (gap + length)
        draw_native_segment(a, b)

    # Add white/bright hard negatives near the ring.
    for _ in range(rng.randint(2, 6)):
        x = rng.randint(max(5, jx - 110), min(W - 5, jx + 110))
        y = rng.randint(max(5, jy - 110), min(H - 5, jy + 110))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), rng.randint(2, 7), (rng.randint(185, 255),) * 3, -1, cv2.LINE_AA)
        else:
            cv2.line(img, (x, y), (x + rng.randint(-20, 20), y + rng.randint(-20, 20)), (220, 220, 220), rng.randint(1, 3), cv2.LINE_AA)

    # Photometric perturbation.
    if rng.random() < 0.65:
        alpha = rng.uniform(0.82, 1.18)
        beta = rng.uniform(-14, 14)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if rng.random() < 0.25:
        img = cv2.GaussianBlur(img, (5, 5), rng.uniform(0.2, 1.0))
    return img, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    images = args.output / "images"
    masks = args.output / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    for i in range(args.count):
        img, mask = make_sample(rng)
        name = f"synthetic_{i:06d}"
        cv2.imwrite(str(images / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(masks / f"{name}.png"), mask)
        if (i + 1) % 100 == 0:
            print(f"generated {i+1}/{args.count}")

    print(f"done output={args.output}")


if __name__ == "__main__":
    main()
