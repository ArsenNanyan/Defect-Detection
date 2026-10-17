# NEU-DET Classification — Experiment Comparison (Step 2)

All runs: 20 epochs max, AdamW + cosine LR, early stopping on validation macro-F1, batch 32, image 224. One immutable split (seed 42). Metrics are on the **validation** split; the test split is reserved for a single final evaluation.

**Best model:** `efficientnet_b0_full` — val macro-F1 1.0, AUROC 1.0.

| Model | Transfer mode | Params (M) | Size (MB) | Train (min) | Val macro-F1 | Val bal-acc | Val AUROC(OvR) |
|---|---|---|---|---|---|---|---|
| efficientnet_b0 | full | 4.015 | 16.37 | 4.37 | 1.0 | 1.0 | 1.0 |
| resnet50 | finetune_last | 23.52 | 94.41 | 3.61 | 1.0 | 1.0 | 1.0 |
| efficientnet_b0 | finetune_last | 4.015 | 16.37 | 4.97 | 1.0 | 1.0 | 1.0 |
| resnet50 | full | 23.52 | 94.4 | 4.58 | 1.0 | 1.0 | 1.0 |
| resnet18 | finetune_last | 11.18 | 44.8 | 2.25 | 0.9963 | 0.9963 | 1.0 |
| resnet18 | full | 11.18 | 44.8 | 2.83 | 0.9963 | 0.9963 | 0.9999 |
| efficientnet_b0 | frozen | 4.015 | 16.37 | 6.08 | 0.9853 | 0.9852 | 0.9992 |
| resnet50 | frozen | 23.52 | 94.4 | 5.95 | 0.9704 | 0.9704 | 0.9986 |
| custom_cnn | n/a | 0.094 | 0.39 | 3.74 | 0.9594 | 0.9589 | 0.9989 |
| resnet18 | frozen | 11.18 | 44.8 | 4.94 | 0.9261 | 0.9256 | 0.9922 |

Confusion matrices: `artifacts/reports/confusion_<run>_val.png`.
Per-epoch curves: `artifacts/reports/history_<run>.csv`.

## Reading the ablation
- **frozen** = ImageNet backbone frozen, only the 6-way head trained (feature extraction).
- **finetune_last** = last backbone block + head trained.
- **full** = end-to-end fine-tuning (backbone LR reduced 10x vs head).
- **custom_cnn** = 3-block CNN trained from scratch (no ImageNet weights).

## Findings & model selection
- **Fine-tuning >> frozen** for every backbone (e.g. resnet18 0.996 vs 0.926). On these
  simple steel textures the ImageNet features need adaptation, not just a new head.
- **Capacity helps the frozen case** (resnet50 frozen 0.970 > resnet18 frozen 0.926) but
  the gap closes once fine-tuned.
- **The 94-sample scratch CNN reaches 0.959** — the task is not very hard, so a tiny model
  is a strong, cheap baseline.
- **Four configs tie at val macro-F1 = 1.000.** Validation (270 images) can no longer
  separate them, so the tie is broken on **efficiency**, and the real ranking is deferred
  to the single final **test-split** evaluation.

**Selected classification model: `efficientnet_b0` (full fine-tune)** — ties for the best
val score at **4.0 M params / 16 MB**, roughly a quarter of ResNet-50's size (23.5 M / 94 MB)
for identical accuracy, which matters for the ONNX/latency comparison and deployment later.

> Caveat: perfect validation scores mean the val set is near-saturated; treat the frozen
> **test** split (used once, at the end) as the deciding metric, and lean on the detection
> track for the harder localization problem.
