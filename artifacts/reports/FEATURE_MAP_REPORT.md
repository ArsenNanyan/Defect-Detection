# Feature Maps & Receptive Fields (Step 3)

Forward hooks on the scratch **custom CNN** and **ResNet-18**, visualizing 12 normalized
channels per layer on the same thin-scratch and small-pit test images, plus the analytical
receptive-field (RF) growth. Reproduce (PYTHONPATH=src):

```bash
python scripts/feature_lab.py
```

Figures: `artifacts/reports/figures/featmaps/<net>_<layer>_<scratch|pit>.png`.
RF data: `artifacts/reports/receptive_fields.json`.

## Receptive field — the headline numbers (224×224 input)

| Network | Deepest tap | Final RF (px) | Final feature map | Each deep neuron sees |
|---|---|---|---|---|
| Custom CNN | conv3 | **18** | 56×56 (stride product 4) | a tiny 18×18 patch — local texture only |
| ResNet-18 | layer4 | **435** | 7×7 (stride product 32) | **more than the whole 224 image** (global) |

Custom CNN RF growth: conv1 **3** → +maxpool **4** → conv2 **8** → +maxpool **10** → conv3 **18**.
ResNet-18 RF growth: conv1 7 → maxpool 11 → layer1 43 → layer2 99 → layer3 211 → layer4 **435**.

RF recurrence used: `RF_out = RF_in + (k−1)·jump_in`, `jump_out = jump_in · stride`.

## Why stride and pooling lose spatial detail (the 1-page point)

Every stride-2 convolution or 2×2 max-pool **halves the spatial grid** and doubles the
"jump" (the input-pixel spacing between adjacent feature-map cells). Two effects compound:

1. **Resolution collapses.** ResNet-18 takes 224×224 down to **7×7** at layer4 (jump 32):
   one deep cell summarizes a 32-pixel stride of the input. A **thin scratch a few pixels
   wide** can fall between sampling positions or be averaged into its background by pooling,
   so its exact location is no longer recoverable from deep features — only the *fact* that
   a scratch-like texture is present survives. This is exactly why a classifier (which only
   needs "what") thrives on deep global features while **localization needs the earlier,
   higher-resolution layers** — and why detectors (YOLO) and U-Net keep multi-scale / skip
   connections instead of only using the deepest map.
2. **RF outgrows the object.** By layer4 the RF (435) exceeds the image, so a deep neuron's
   response mixes the defect with all of its surroundings — great for context and class
   identity, poor for pinpointing a small pit.

The custom CNN sits at the opposite extreme: with RF 18 and only two pools, its deepest
units still see fine local texture at 56×56. That is *why a 94k-param CNN classifies
NEU-DET so well* (macro-F1 0.953 on test) — the six classes are largely distinguishable by
**local texture statistics**, which a small RF captures. It is also why that same CNN would
be far weaker at tasks needing global shape reasoning.

## Scratch vs pit, CNN vs ResNet (what the channels show)

- **Shallow layers** (`custom_cnn_conv1`, `resnet18_conv1`) fire on **oriented edges and
  contrast** — on the scratch image several channels light up along the linear groove; on
  the pit image they ring the small circular pits. These are generic, image-statistics
  filters, nearly task-independent.
- **Deep layers** (`custom_cnn_conv3`, `resnet18_layer4`) are **abstract and low-resolution**.
  ResNet's 7×7 layer4 maps are blobby class-evidence detectors — you can no longer read the
  scratch's path from them. The custom CNN's 56×56 conv3 still preserves a coarse trace of
  the scratch line, visibly more localized than ResNet's layer4.
- **Thin scratches vs small pits**: thin scratches excite a *few* strongly oriented shallow
  channels (sparse, directional); small pits excite *many* blob/edge channels more
  isotropically. Both are far clearer in shallow maps than deep ones — the practical
  argument for multi-scale features in the detection and segmentation tracks.

## Interview one-liners
- Receptive field and spatial resolution trade off: downsampling buys larger RF (more
  context, cheaper compute) at the cost of where-information.
- A classification score can say *what* defect is present but not *where* — deep features
  have thrown the location away. That is the whole reason this project also trains a detector.
