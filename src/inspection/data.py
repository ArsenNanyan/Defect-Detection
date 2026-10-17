"""NEU-DET classification dataset backed by the frozen split manifests.

Reads data/splits/{train,val,test}.csv (one immutable split for all experiments) and
resolves image paths relative to the raw dataset root. Returns (image_tensor, label_idx).
Single-label = the authoritative `label` column (class subfolder).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from . import CLASS_NAMES, CLASS_TO_IDX


class NEUClassificationDataset(Dataset):
    def __init__(
        self,
        split: str,
        raw_root: str | Path = "data/raw/neu_det",
        splits_dir: str | Path = "data/splits",
        transform=None,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.transform = transform
        csv = Path(splits_dir) / f"{split}.csv"
        if not csv.exists():
            raise FileNotFoundError(
                f"{csv} not found. Run scripts/make_splits.py first."
            )
        self.df = pd.read_csv(csv)
        if "label" not in self.df.columns:
            raise ValueError(f"{csv} has no 'label' column; regenerate splits.")
        self.labels = self.df["label"].map(CLASS_TO_IDX)
        if self.labels.isna().any():
            bad = self.df.loc[self.labels.isna(), "label"].unique()
            raise ValueError(f"Unknown label(s) in {csv}: {bad}")
        self.labels = self.labels.astype(int).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.raw_root / row["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]

    @property
    def class_names(self) -> list[str]:
        return CLASS_NAMES

    def class_counts(self) -> dict[str, int]:
        return self.df["label"].value_counts().to_dict()
