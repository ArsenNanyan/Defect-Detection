"""STEP 2 (sweep) — run the full classification experiment grid and build the report.

Experiments
-----------
  custom_cnn                      (trained fully; no transfer mode)
  resnet18        x {frozen, finetune_last, full}
  resnet50        x {frozen, finetune_last, full}
  efficientnet_b0 x {frozen, finetune_last, full}

For each run: train (20 epochs default), keep the best-val-macro-F1 checkpoint, then
evaluate that checkpoint on the val split (confusion matrix + OvR AUROC). Results are
merged into:
    artifacts/reports/classification_comparison.csv
    artifacts/reports/CLASSIFICATION_REPORT.md

The frozen test split is NOT touched here (used once at the very end).

Usage:
    set PYTHONPATH=src   &&   python scripts/run_cls_sweep.py --epochs 20
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

import pandas as pd
import yaml

# allow `import inspection...` when run as a plain script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspection.eval_cls import run_eval  # noqa: E402
from inspection.train_cls import train  # noqa: E402

GRID = [
    ("custom_cnn", None),
    ("resnet18", "frozen"),
    ("resnet18", "finetune_last"),
    ("resnet18", "full"),
    ("resnet50", "frozen"),
    ("resnet50", "finetune_last"),
    ("resnet50", "full"),
    ("efficientnet_b0", "frozen"),
    ("efficientnet_b0", "finetune_last"),
    ("efficientnet_b0", "full"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/classification.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    if args.epochs is not None:
        base["train"]["epochs"] = args.epochs
    base["train"]["num_workers"] = args.num_workers

    rep = Path("artifacts/reports")
    rep.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for model, mode in GRID:
        cfg = copy.deepcopy(base)
        cfg["model"] = model
        cfg["transfer_mode"] = mode or "full"
        tag = f"{model}" + (f"/{mode}" if mode else "")
        print(f"\n{'=' * 70}\n>>> TRAIN {tag}\n{'=' * 70}", flush=True)
        try:
            tr = train(cfg)
            ev = run_eval(tr["checkpoint"], "val", cfg["data"].get("raw_root",
                          "data/raw/neu_det"), cfg["data"]["splits_dir"], rep)
            rows.append(dict(
                run=tr["run"], model=model, transfer_mode=(mode or "n/a"),
                params_M=tr["params_M"], ckpt_size_MB=tr["ckpt_size_MB"],
                train_minutes=tr["train_minutes"],
                val_macro_f1=ev["macro_f1"], val_balanced_acc=ev["balanced_accuracy"],
                val_auroc_ovr=ev["auroc_ovr_macro"], device=tr["device"],
            ))
        except Exception:  # noqa: BLE001
            print(f"!!! RUN FAILED: {tag}\n{traceback.format_exc()}", flush=True)
            rows.append(dict(run=tag, model=model, transfer_mode=(mode or "n/a"),
                             params_M=None, ckpt_size_MB=None, train_minutes=None,
                             val_macro_f1=None, val_balanced_acc=None,
                             val_auroc_ovr=None, device="error"))

    df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False,
                                        na_position="last")
    df.to_csv(rep / "classification_comparison.csv", index=False)
    _write_markdown(df, rep / "CLASSIFICATION_REPORT.md", base["train"]["epochs"])
    print("\n=== SWEEP COMPLETE ===")
    print(df.to_string(index=False))


def _write_markdown(df: pd.DataFrame, path: Path, epochs: int) -> None:
    best = df.dropna(subset=["val_macro_f1"]).head(1)
    best_line = (
        f"**Best model:** `{best.iloc[0]['run']}` — val macro-F1 "
        f"{best.iloc[0]['val_macro_f1']}, AUROC {best.iloc[0]['val_auroc_ovr']}."
        if not best.empty else "No successful runs."
    )
    lines = [
        "# NEU-DET Classification — Experiment Comparison (Step 2)",
        "",
        f"All runs: {epochs} epochs max, AdamW + cosine LR, early stopping on validation "
        "macro-F1, batch 32, image 224. One immutable split (seed 42). Metrics are on the "
        "**validation** split; the test split is reserved for a single final evaluation.",
        "",
        best_line,
        "",
        "| Model | Transfer mode | Params (M) | Size (MB) | Train (min) | Val macro-F1 | Val bal-acc | Val AUROC(OvR) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {r['transfer_mode']} | {r['params_M']} | "
            f"{r['ckpt_size_MB']} | {r['train_minutes']} | {r['val_macro_f1']} | "
            f"{r['val_balanced_acc']} | {r['val_auroc_ovr']} |"
        )
    lines += [
        "",
        "Confusion matrices: `artifacts/reports/confusion_<run>_val.png`.",
        "Per-epoch curves: `artifacts/reports/history_<run>.csv`.",
        "",
        "## Reading the ablation",
        "- **frozen** = ImageNet backbone frozen, only the 6-way head trained (feature extraction).",
        "- **finetune_last** = last backbone block + head trained.",
        "- **full** = end-to-end fine-tuning (backbone LR reduced 10x vs head).",
        "- **custom_cnn** = 3-block CNN trained from scratch (no ImageNet weights).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # also dump raw json for programmatic use
    (path.parent / "classification_comparison.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
