"""STEP 3 — feature-map & receptive-field lab.

* Registers forward hooks on shallow + deep conv layers of the scratch custom CNN and
  of ResNet-18, and visualizes 12 normalized channels per layer on the same images
  (a thin-scratch image and a small-pit image).
* Computes the theoretical receptive field (RF) layer-by-layer for both networks.

Outputs:
    artifacts/reports/figures/featmaps/<net>_<layer>_<image>.png
    artifacts/reports/receptive_fields.json

Usage (PYTHONPATH=src):
    python scripts/feature_lab.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inspection.models import build_model  # noqa: E402
from inspection.transforms import build_transforms  # noqa: E402

CKPT = Path("artifacts/checkpoints")
FIG = Path("artifacts/reports/figures/featmaps")
RAW = Path("data/raw/neu_det")


# ---------- receptive field ----------
def receptive_field(layers: list[tuple[str, int, int, int]]) -> list[dict]:
    """layers: list of (name, kernel, stride, padding). Returns RF + jump per layer."""
    rf, jump = 1, 1
    rows = []
    for name, k, s, _p in layers:
        rf = rf + (k - 1) * jump
        jump = jump * s
        rows.append({"layer": name, "kernel": k, "stride": s,
                     "receptive_field": rf, "jump(stride_product)": jump})
    return rows


CUSTOM_CNN_LAYERS = [
    ("conv1(3x3)", 3, 1, 1), ("maxpool1(2x2,s2)", 2, 2, 0),
    ("conv2(3x3)", 3, 1, 1), ("maxpool2(2x2,s2)", 2, 2, 0),
    ("conv3(3x3)", 3, 1, 1),
]
# ResNet-18 down to end of layer4 (basic blocks = two 3x3; stride-2 at each stage start)
RESNET18_LAYERS = [
    ("conv1(7x7,s2)", 7, 2, 3), ("maxpool(3x3,s2)", 3, 2, 1),
    ("layer1.b1(3x3)", 3, 1, 1), ("layer1.b1(3x3)", 3, 1, 1),
    ("layer1.b2(3x3)", 3, 1, 1), ("layer1.b2(3x3)", 3, 1, 1),
    ("layer2.b1(3x3,s2)", 3, 2, 1), ("layer2.b1(3x3)", 3, 1, 1),
    ("layer2.b2(3x3)", 3, 1, 1), ("layer2.b2(3x3)", 3, 1, 1),
    ("layer3.b1(3x3,s2)", 3, 2, 1), ("layer3.b1(3x3)", 3, 1, 1),
    ("layer3.b2(3x3)", 3, 1, 1), ("layer3.b2(3x3)", 3, 1, 1),
    ("layer4.b1(3x3,s2)", 3, 2, 1), ("layer4.b1(3x3)", 3, 1, 1),
    ("layer4.b2(3x3)", 3, 1, 1), ("layer4.b2(3x3)", 3, 1, 1),
]


# ---------- feature maps ----------
def _save_grid(fmap: torch.Tensor, path: Path, title: str, n: int = 12) -> None:
    """fmap: (C,H,W). Plot first n channels, each min-max normalized."""
    c = min(n, fmap.shape[0])
    cols = 4
    rows = (c + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.ravel()
    for i in range(c):
        ch = fmap[i]
        lo, hi = ch.min(), ch.max()
        ch = (ch - lo) / (hi - lo + 1e-6)
        axes[i].imshow(ch.cpu().numpy(), cmap="viridis")
        axes[i].set_title(f"ch{i}", fontsize=7)
        axes[i].axis("off")
    for i in range(c, len(axes)):
        axes[i].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def hook_and_plot(model, taps: dict, x: torch.Tensor, net_name: str, img_tag: str) -> None:
    acts: dict[str, torch.Tensor] = {}
    handles = []
    for tap_name, module in taps.items():
        handles.append(module.register_forward_hook(
            lambda _m, _i, o, k=tap_name: acts.__setitem__(k, o.detach()[0])))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    for tap_name, fmap in acts.items():
        _save_grid(fmap, FIG / f"{net_name}_{tap_name}_{img_tag}.png",
                   f"{net_name} · {tap_name} · {img_tag}  ({tuple(fmap.shape)})")


def _load_image(rel: str, tf) -> torch.Tensor:
    from PIL import Image

    return tf(Image.open(RAW / rel).convert("RGB")).unsqueeze(0)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    _, eval_tf = build_transforms(224, pretrained=True)

    import pandas as pd

    val = pd.read_csv("data/splits/test.csv")
    scratch_rel = val[val["label"] == "scratches"].iloc[0]["image_path"]
    pit_rel = val[val["label"] == "pitted_surface"].iloc[0]["image_path"]
    images = {"scratch": scratch_rel, "pit": pit_rel}

    # custom CNN
    cnn, _ = build_model("custom_cnn", 6)
    ck = torch.load(CKPT / "custom_cnn_best.pt", map_location="cpu", weights_only=False)
    cnn.load_state_dict(ck["model_state"])
    cnn_taps = {"conv1": cnn[2], "conv2": cnn[6], "conv3": cnn[10]}  # post-ReLU

    # ResNet-18
    rn, _ = build_model("resnet18", 6, "full")
    ck2 = torch.load(CKPT / "resnet18_full_best.pt", map_location="cpu", weights_only=False)
    rn.load_state_dict(ck2["model_state"])
    rn_taps = {"conv1": rn.conv1, "layer1": rn.layer1, "layer2": rn.layer2,
               "layer4": rn.layer4}

    for tag, rel in images.items():
        x = _load_image(rel, eval_tf)
        hook_and_plot(cnn, cnn_taps, x, "custom_cnn", tag)
        hook_and_plot(rn, rn_taps, x, "resnet18", tag)

    rf = {
        "input_size": 224,
        "custom_cnn": receptive_field(CUSTOM_CNN_LAYERS),
        "resnet18_to_layer4": receptive_field(RESNET18_LAYERS),
    }
    Path("artifacts/reports/receptive_fields.json").write_text(
        json.dumps(rf, indent=2), encoding="utf-8")
    print("custom_cnn final RF:", rf["custom_cnn"][-1]["receptive_field"])
    print("resnet18 final RF  :", rf["resnet18_to_layer4"][-1]["receptive_field"])
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
