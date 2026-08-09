from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import build_model


INPUT_W = 512
INPUT_H = 288
NUM_CLASSES = 3


class GuideDataset(Dataset):
    def __init__(self, root: Path, names: list[str], train: bool):
        self.root = root
        self.names = names
        self.train = train

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx].strip()
        image_path = self.root / "images" / name
        mask_path = self.root / "masks" / f"{Path(name).stem}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Missing image/mask for {name}")

        if self.train:
            image, mask = augment(image, mask)

        image = cv2.resize(image, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)
        mask = np.clip(mask, 0, NUM_CLASSES - 1).astype(np.int64)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()
        return image, mask


def augment(image: np.ndarray, mask: np.ndarray):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    if random.random() < 0.75:
        alpha = random.uniform(0.82, 1.18)
        beta = random.uniform(-18, 18)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if random.random() < 0.25:
        k = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (k, k), 0)

    if random.random() < 0.45:
        h, w = image.shape[:2]
        angle = random.uniform(-2.5, 2.5)
        scale = random.uniform(0.97, 1.03)
        tx = random.uniform(-0.015, 0.015) * w
        ty = random.uniform(-0.015, 0.015) * h
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        m[:, 2] += (tx, ty)
        image = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image, mask


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    losses = []
    for cls in range(1, NUM_CLASSES):
        p = probs[:, cls]
        t = one_hot[:, cls]
        inter = (p * t).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        losses.append(1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean())
    return torch.stack(losses).mean()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    inter = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    union = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    loss_sum = 0.0
    n = 0
    weights = torch.tensor([0.15, 1.0, 2.0], device=device)
    with torch.no_grad():
        for image, mask in loader:
            image = image.to(device)
            mask = mask.to(device)
            logits = model(image)
            loss = F.cross_entropy(logits, mask, weight=weights) + 0.65 * dice_loss(logits, mask)
            loss_sum += float(loss.item())
            n += 1
            pred = logits.argmax(1)
            for cls in range(NUM_CLASSES):
                p = pred == cls
                t = mask == cls
                inter[cls] += (p & t).sum().cpu()
                union[cls] += (p | t).sum().cpu()
    iou = inter / torch.clamp(union, min=1)
    return {
        "loss": loss_sum / max(n, 1),
        "iou_background": float(iou[0]),
        "iou_guide": float(iou[1]),
        "iou_ring": float(iou[2]),
        "miou_fg": float(iou[1:].mean()),
    }


def read_split(root: Path, name: str) -> list[str]:
    path = root / "splits" / f"{name}.txt"
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("ml/runs"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_names = read_split(args.data, "train")
    val_names = read_split(args.data, "val")
    if not train_names or not val_names:
        raise SystemExit("train.txt and val.txt must be non-empty")

    train_ds = GuideDataset(args.data, train_names, train=True)
    val_ds = GuideDataset(args.data, val_names, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    weights = torch.tensor([0.15, 1.0, 2.0], device=device)

    args.out.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for image, mask in bar:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(image)
            loss = F.cross_entropy(logits, mask, weight=weights) + 0.65 * dice_loss(logits, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item())
            count += 1
            bar.set_postfix(loss=f"{running/max(count,1):.4f}")
        sched.step()

        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(count, 1)
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

        state = {
            "model": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "input_size": [INPUT_W, INPUT_H],
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(state, args.out / "last.pt")
        if metrics["miou_fg"] > best_score:
            best_score = metrics["miou_fg"]
            torch.save(state, args.out / "best.pt")

        (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"done best_miou_fg={best_score:.4f} checkpoint={args.out/'best.pt'}")


if __name__ == "__main__":
    main()
