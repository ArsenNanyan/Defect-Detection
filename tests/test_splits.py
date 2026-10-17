"""Data tests for the frozen NEU-DET split (guide STEP/section 16).

No duplicate content or group across splits; labels legal; manifests consistent.
Run: pytest -q   (from repo root)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

SPLITS = Path("data/splits")
CLASSES = {"crazing", "inclusion", "patches", "pitted_surface",
           "rolled-in_scale", "scratches"}


@pytest.fixture(scope="module")
def frames():
    if not (SPLITS / "train.csv").exists():
        pytest.skip("splits not generated; run scripts/make_splits.py")
    return {s: pd.read_csv(SPLITS / f"{s}.csv") for s in ("train", "val", "test")}


def test_all_splits_present(frames):
    for s in ("train", "val", "test"):
        assert len(frames[s]) > 0


def test_labels_legal(frames):
    for df in frames.values():
        assert set(df["label"]).issubset(CLASSES)


def test_no_sha256_across_splits(frames):
    seen: dict[str, str] = {}
    for s, df in frames.items():
        for h in df["sha256"]:
            assert h not in seen or seen[h] == s, f"sha256 {h[:8]} spans splits"
            seen[h] = s


def test_no_group_across_splits(frames):
    seen: dict[int, str] = {}
    for s, df in frames.items():
        for g in df["group_id"]:
            assert g not in seen or seen[g] == s, f"group_id {g} spans splits"
            seen[g] = s


def test_total_count_is_1800(frames):
    assert sum(len(df) for df in frames.values()) == 1800


def test_every_image_has_annotation(frames):
    for df in frames.values():
        assert df["annotation_path"].astype(bool).all()
