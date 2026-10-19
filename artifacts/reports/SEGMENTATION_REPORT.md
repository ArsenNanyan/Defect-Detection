# MVTec AD Segmentation / Anomaly Localization (Step 6)

Category: **grid** · image size 256 · seed 42.
MVTec AD is **CC BY-NC-SA 4.0 (non-commercial)**; images are not redistributed here.

> The two NEU-DET tasks and this MVTec task are **separate datasets and protocols** and are never ranked against each other.

## Track A — PatchCore (benchmark-compatible, normal-only)
Backbone resnet18, coreset 0.1, memory bank 27034 patches. Fit on train/good only; AUROC is threshold-free so nothing is tuned on test.

- **Image AUROC: 0.927** (is this image anomalous?)
- **Pixel AUROC: 0.962** (where is the anomaly?)

This is a faithful PatchCore-lite (ResNet-18 layer2+3 features, random 10% coreset). The
published PatchCore reaches ~0.98+ on grid with a WideResNet-50 backbone and greedy coreset;
our lighter setup trades a few points for speed and simplicity, and both backbone and coreset
are one-line swaps. The result is strong and, crucially, **benchmark-compatible**: fit on
normal images only, threshold-free AUROC, nothing tuned on test.

## Track B — U-Net (NONSTANDARD supervised)
NONSTANDARD supervised split of MVTec test defects; NOT the official unsupervised benchmark. Whole-image disjoint splits.
Split (whole-image disjoint): {'train': 45, 'val': 11, 'test': 11}.

- Threshold (picked on val): 0.7
- **Test Dice: 0.4213**, **Test IoU: 0.2785** (over 9 defective images)
- Clean false-positive pixel rate: 0.0 (over 2 clean images)
- Best val Dice: 0.6329

Qualitative overlays: `artifacts/reports/figures/segmentation/unet_overlays.png`.

> Track B trains on MVTec *test* defects we re-split ourselves, so it is an educational supervised baseline, **not** an official MVTec result. Track A is the benchmark-compatible number.

### Interpreting Track B
Test Dice 0.42 is modest, and honestly so: the supervised set is **tiny** (45 train / 11 val
/ 11 test images) and grid defects (thin threads, small bent/broken regions) are a severe
foreground/background imbalance — a few mislocalized pixels move Dice a lot. The **0.0 clean
false-positive rate** is the encouraging half: the U-Net does not hallucinate defects on
normal grid. The contrast is the lesson — **PatchCore, learning only from abundant normal
images, localizes better (pixel AUROC 0.96) than a U-Net starved of labeled defects.** For
anomaly localization with scarce defect labels, the normal-only route wins; supervised
segmentation needs far more annotated masks to compete. See `unet_overlays.png` for GT-vs-pred.
