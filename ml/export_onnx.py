from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch

from model import build_model


INPUT_W = 512
INPUT_H = 288


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    num_classes = int(state.get("num_classes", 3))
    model = build_model(num_classes)
    model.load_state_dict(state["model"])
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W, dtype=torch.float32)
    # Use the stable TorchScript exporter so this pipeline works across the
    # supported PyTorch 2.x range without requiring the newer onnxscript stack.
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )

    model_onnx = onnx.load(str(args.output))
    onnx.checker.check_model(model_onnx)
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"exported={args.output} size_mb={size_mb:.2f}")


if __name__ == "__main__":
    main()
