from __future__ import annotations

import argparse
import random
from pathlib import Path


VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images = sorted(p for p in (args.data / "images").iterdir() if p.suffix.lower() in VALID_EXT)
    valid = []
    for p in images:
        mask = args.data / "masks" / f"{p.stem}.png"
        if mask.exists():
            valid.append(p.name)
    if len(valid) < 3:
        raise SystemExit("Need at least 3 annotated images")

    rnd = random.Random(args.seed)
    rnd.shuffle(valid)
    n = len(valid)
    n_test = max(1, round(n * args.test))
    n_val = max(1, round(n * args.val))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise SystemExit("Split leaves no training samples")

    splits = {
        "train": valid[:n_train],
        "val": valid[n_train:n_train + n_val],
        "test": valid[n_train + n_val:],
    }
    out = args.data / "splits"
    out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        (out / f"{name}.txt").write_text("\n".join(items) + "\n", encoding="utf-8")
        print(name, len(items))


if __name__ == "__main__":
    main()
