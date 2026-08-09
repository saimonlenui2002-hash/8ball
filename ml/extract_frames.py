from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def dhash(gray: np.ndarray, size: int = 9) -> np.ndarray:
    small = cv2.resize(gray, (size, size - 1), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).reshape(-1)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("output")
    ap.add_argument("--fps", type=float, default=2.0, help="target extraction rate")
    ap.add_argument("--dedupe-threshold", type=int, default=6)
    ap.add_argument("--prefix", default="frame")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / max(args.fps, 0.1)))
    idx = 0
    saved = 0
    last_hash = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current = dhash(gray)
        if last_hash is not None and hamming(last_hash, current) <= args.dedupe_threshold:
            idx += 1
            continue

        path = out / f"{args.prefix}_{saved:06d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
        last_hash = current
        saved += 1
        idx += 1

    cap.release()
    print(f"saved={saved} output={out}")


if __name__ == "__main__":
    main()
