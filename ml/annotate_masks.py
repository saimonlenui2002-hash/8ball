from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {0: "background", 1: "native_guide", 2: "contact_ring"}
PALETTE = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
}


class Annotator:
    def __init__(self, images: list[Path], masks_dir: Path):
        self.images = images
        self.masks_dir = masks_dir
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.cls = 1
        self.brush = 5
        self.painting = False
        self.image = None
        self.mask = None
        self._load()

    def _mask_path(self) -> Path:
        return self.masks_dir / f"{self.images[self.index].stem}.png"

    def _load(self) -> None:
        self.image = cv2.imread(str(self.images[self.index]), cv2.IMREAD_COLOR)
        if self.image is None:
            raise RuntimeError(f"Cannot read {self.images[self.index]}")
        mp = self._mask_path()
        if mp.exists():
            self.mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if self.mask.shape[:2] != self.image.shape[:2]:
                self.mask = np.zeros(self.image.shape[:2], np.uint8)
        else:
            self.mask = np.zeros(self.image.shape[:2], np.uint8)

    def _save(self) -> None:
        cv2.imwrite(str(self._mask_path()), self.mask)

    def _overlay(self) -> np.ndarray:
        vis = self.image.copy()
        color = np.zeros_like(vis)
        for cls, bgr in PALETTE.items():
            if cls == 0:
                continue
            color[self.mask == cls] = bgr
        alpha = (self.mask > 0).astype(np.float32)[..., None] * 0.45
        vis = (vis * (1 - alpha) + color * alpha).astype(np.uint8)
        title = f"{self.index+1}/{len(self.images)}  class={self.cls}:{CLASS_NAMES[self.cls]}  brush={self.brush}"
        cv2.putText(vis, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "1=guide 2=ring 0=erase  [ ] brush  S save  N/P next/prev  Q quit", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    def mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.painting = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.painting = False
        if self.painting or event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.mask, (x, y), self.brush, int(self.cls), -1, cv2.LINE_8)

    def run(self) -> None:
        cv2.namedWindow("annotate", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("annotate", self.mouse)
        while True:
            cv2.imshow("annotate", self._overlay())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                self._save()
                break
            if key in (ord("0"), ord("1"), ord("2")):
                self.cls = int(chr(key))
            elif key == ord("["):
                self.brush = max(1, self.brush - 1)
            elif key == ord("]"):
                self.brush = min(40, self.brush + 1)
            elif key == ord("s"):
                self._save()
            elif key == ord("n"):
                self._save()
                self.index = min(len(self.images) - 1, self.index + 1)
                self._load()
            elif key == ord("p"):
                self._save()
                self.index = max(0, self.index - 1)
                self._load()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("masks")
    args = ap.parse_args()
    image_dir = Path(args.images)
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    if not images:
        raise SystemExit("No images found")
    Annotator(images, Path(args.masks)).run()


if __name__ == "__main__":
    main()
