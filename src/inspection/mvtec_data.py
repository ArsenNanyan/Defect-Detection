"""MVTec AD data access for the segmentation/anomaly track (Step 6).

Expected category layout (verify after download):
    <root>/<category>/train/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/<defect_type>/*.png
    <root>/<category>/ground_truth/<defect_type>/*_mask.png

Two consumers:
* Track A (PatchCore): normal-only fit on train/good; evaluate on all of test/ with
  image labels (good=0, defect=1) and pixel GT masks.
* Track B (U-Net): a SELF-CONSTRUCTED, explicitly non-standard supervised split of the
  defective test images (+ optional good images with all-zero masks). Because MVTec's
  official masks live under test/, Track B re-splits them into train/val/test itself and
  never trains on an image later evaluated as its own test.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class Item:
    image_path: Path
    mask_path: Path | None   # None => normal image (all-zero mask)
    defect_type: str         # "good" for normal
    label: int               # 0 normal, 1 anomalous


def _is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS and p.is_file()


def category_root(root: str | Path, category: str) -> Path:
    r = Path(root) / category
    if not (r / "train" / "good").exists():
        raise FileNotFoundError(
            f"{r/'train'/'good'} not found. Download MVTec AD category '{category}' "
            f"into {Path(root)/category} (see README segmentation section)."
        )
    return r


def list_train_good(root: str | Path, category: str) -> list[Path]:
    r = category_root(root, category)
    return sorted(p for p in (r / "train" / "good").iterdir() if _is_img(p))


def list_test_items(root: str | Path, category: str) -> list[Item]:
    """All test images with their GT mask (good images => mask None)."""
    r = category_root(root, category)
    gt = r / "ground_truth"
    items: list[Item] = []
    for sub in sorted((r / "test").iterdir()):
        if not sub.is_dir():
            continue
        defect = sub.name
        for img in sorted(p for p in sub.iterdir() if _is_img(p)):
            if defect == "good":
                items.append(Item(img, None, "good", 0))
            else:
                # MVTec masks are named <stem>_mask.png under ground_truth/<defect>/
                mp = gt / defect / f"{img.stem}_mask{img.suffix}"
                if not mp.exists():
                    cand = list((gt / defect).glob(f"{img.stem}_mask.*"))
                    mp = cand[0] if cand else None
                items.append(Item(img, mp, defect, 1))
    return items


def _load_image(path: Path, size: int) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


def _load_mask(path: Path | None, size: int) -> torch.Tensor:
    if path is None:
        return torch.zeros(1, size, size)
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    arr = (np.asarray(m, dtype=np.float32) > 127).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


class MVTecAnomalyDataset(Dataset):
    """For PatchCore eval and generic iteration: returns (image, mask, label)."""

    def __init__(self, items: list[Item], image_size: int = 256) -> None:
        self.items = items
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        img = _load_image(it.image_path, self.image_size)
        mask = _load_mask(it.mask_path, self.image_size)
        return img, mask, it.label


class MVTecTrainGood(Dataset):
    """train/good images only (for PatchCore memory-bank fitting)."""

    def __init__(self, paths: list[Path], image_size: int = 256) -> None:
        self.paths = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return _load_image(self.paths[i], self.image_size)


def supervised_unet_split(root: str | Path, category: str, seed: int,
                          val_frac: float, test_frac: float,
                          include_good_frac: float) -> dict[str, list[Item]]:
    """Build a NON-STANDARD supervised split of defective test images (+ some good).

    Returns {"train": [...], "val": [...], "test": [...]}; whole-image disjoint.
    """
    items = list_test_items(root, category)
    defects = [it for it in items if it.label == 1 and it.mask_path is not None]
    goods = [it for it in items if it.label == 0]
    rng = random.Random(seed)

    def split(lst: list[Item]) -> tuple[list[Item], list[Item], list[Item]]:
        lst = lst[:]
        rng.shuffle(lst)
        n = len(lst)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        return lst[n_test + n_val:], lst[n_test:n_test + n_val], lst[:n_test]

    d_tr, d_va, d_te = split(defects)
    n_good = int(round(len(goods) * include_good_frac))
    g_tr, g_va, g_te = split(goods[:n_good])
    return {"train": d_tr + g_tr, "val": d_va + g_va, "test": d_te + g_te}


class MVTecSegDataset(Dataset):
    """Supervised segmentation: returns (image, mask) for U-Net training/eval."""

    def __init__(self, items: list[Item], image_size: int = 256, augment: bool = False):
        self.items = items
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        img = _load_image(it.image_path, self.image_size)
        mask = _load_mask(it.mask_path, self.image_size)
        if self.augment and random.random() < 0.5:
            img = torch.flip(img, dims=[2])
            mask = torch.flip(mask, dims=[2])
        return img, mask
