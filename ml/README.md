# ML pipeline for native 8 Ball Pool guideline

This directory contains the training/export pipeline for the next-generation detector.

## Goal

The Android app should stop guessing trajectories from generic Hough lines. A small semantic-segmentation model will identify only the native aiming graphics already drawn by 8 Ball Pool. Geometry remains deterministic.

### Model classes

The model predicts three classes per pixel:

0. `background`
1. `native_guide` — incoming guide plus short outgoing branches
2. `contact_ring` — the native collision/contact marker

We intentionally do **not** ask the network to classify cue ball, object balls, UI, or physics. After segmentation, deterministic geometry:

1. fits the incoming native line;
2. finds the contact-ring center;
3. removes an incoming-line corridor around the ring;
4. clusters the remaining short guide pixels into one or two outgoing branches;
5. fits each branch with PCA/`fitLine`;
6. starts our extension at the outer end of the native branch;
7. intersects with calibrated rail bounds and optionally reflects the ray.

This hybrid approach keeps ML narrow and makes final line angles mathematical rather than learned.

## Dataset layout

```text
ml/data/
  images/
    frame_000001.jpg
    ...
  masks/
    frame_000001.png
    ...
  splits/
    train.txt
    val.txt
    test.txt
```

Masks are single-channel PNG files with values `0`, `1`, `2` matching the classes above.

Do not commit third-party videos or user-provided private captures to the public repository. Keep raw source videos outside the repository and commit only code/metadata that you have rights to publish.

## Recommended data mix

- public gameplay/reference material for visual diversity;
- official Miniclip screenshots for current UI reference where licensing/usage permits;
- a small target-device validation set from the real device, kept private;
- hard negatives: loading screens, menus, ball numbers, highlights, pocket rims, cue graphics.

## Commands

Extract diverse frames:

```bash
python ml/extract_frames.py input.mp4 ml/data/images --fps 2 --dedupe-threshold 6
```

Train:

```bash
pip install -r ml/requirements.txt
python ml/train_segmentation.py --data ml/data --epochs 60 --batch-size 12
```

Export the best checkpoint to ONNX:

```bash
python ml/export_onnx.py --checkpoint ml/runs/best.pt --output ml/models/native_guide_seg.onnx
```

Preview inference + geometry:

```bash
python ml/preview_inference.py --model ml/models/native_guide_seg.onnx --image sample.jpg --output preview.jpg
```

## Mobile target

Default network input is `512x288`. The Tiny U-Net architecture is intentionally small and uses depthwise-separable blocks. The exported ONNX model is intended to run through OpenCV DNN on Android, avoiding a second heavy inference runtime.

The app should only render a trajectory if:

- the contact ring is detected with sufficient confidence;
- the incoming guide reaches the ring;
- at least one outgoing branch is geometrically connected to the ring;
- the branch direction is stable across several frames.

When confidence is low, the correct behavior is to render nothing rather than guess.
