"""STEP 1 (part B) — build ONE immutable, leakage-audited split for NEU-DET.

Policy
------
* Pool train+validation, then stratified 70/15/15 by image-level class, seed 42.
* image-level label = authoritative class folder (cross-checked against XML in audit).
* Leakage prevention:
    - group_id = near-duplicate cluster (connected components over dhash Hamming
      distance <= DHASH_THRESHOLD). Whole groups go to one split so duplicates
      never straddle train/val/test.
    - hard assert: no sha256 and no group_id appears in more than one split.
* Output is the single split reused by ALL classification and detection experiments.

Manifest columns: image_path, annotation_path, class_names, group_id, sha256, split

Usage
-----
    python scripts/make_splits.py --raw data/raw/neu_det --out data/splits --seed 42
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from dataset_common import (
    dhash,
    hamming_hex,
    iter_samples,
    parse_annotation,
    sha256_of,
)

DHASH_THRESHOLD = 4   # Hamming distance over 64-bit dhash to call two images near-dup
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _cluster_near_duplicates(stems: list[str], hashes: dict[str, str]) -> dict[str, int]:
    """Union-Find over dhash Hamming distance; returns stem -> group_id."""
    parent = {s: s for s in stems}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # O(n^2) over 1800 items (~1.6M comparisons) — fine for a one-off audit.
    for i in range(len(stems)):
        for j in range(i + 1, len(stems)):
            if hamming_hex(hashes[stems[i]], hashes[stems[j]]) <= DHASH_THRESHOLD:
                union(stems[i], stems[j])

    roots = {}
    gid = {}
    next_id = 0
    for s in stems:
        r = find(s)
        if r not in roots:
            roots[r] = next_id
            next_id += 1
        gid[s] = roots[r]
    return gid


def _stratified_group_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign splits per class, keeping whole groups together, hitting target fracs."""
    import random

    rng = random.Random(seed)
    assignment: dict[int, str] = {}

    for cls, sub in df.groupby("folder_class"):
        groups = sub.groupby("group_id")
        # order groups by descending size for better balance, then shuffle ties
        gsizes = [(gid, len(g)) for gid, g in groups]
        rng.shuffle(gsizes)
        gsizes.sort(key=lambda t: t[1], reverse=True)
        total = len(sub)
        target = {k: v * total for k, v in SPLIT_FRACS.items()}
        current = defaultdict(int)
        for gid, size in gsizes:
            # assign to the split that is furthest below its target
            deficit = {k: target[k] - current[k] for k in SPLIT_FRACS}
            pick = max(deficit, key=deficit.get)
            assignment[gid] = pick
            current[pick] += size
    df = df.copy()
    df["split"] = df["group_id"].map(assignment)
    return df


def build(raw: Path, out: Path, seed: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    samples = iter_samples(raw)

    recs = []
    hashes: dict[str, str] = {}
    for s in samples:
        parse_annotation(s)
        stem = s.image_path.stem
        dh = dhash(s.image_path)
        hashes[stem] = dh
        recs.append(
            dict(
                stem=stem,
                image_path=str(s.image_path.relative_to(raw)).replace("\\", "/"),
                annotation_path=(
                    str(s.annotation_path.relative_to(raw)).replace("\\", "/")
                    if s.annotation_path
                    else ""
                ),
                folder_class=s.folder_class,
                label=s.folder_class,  # authoritative single-label (class subfolder)
                class_names=";".join(sorted(set(s.xml_classes)) or [s.folder_class]),
                multi_class=len(set(s.xml_classes)) > 1,
                sha256=sha256_of(s.image_path),
                dhash=dh,
            )
        )
    df = pd.DataFrame(recs)

    gid = _cluster_near_duplicates(df["stem"].tolist(), hashes)
    df["group_id"] = df["stem"].map(gid)

    df = _stratified_group_split(df, seed)

    # ---- leakage asserts ----
    for col in ("sha256", "group_id"):
        spanning = df.groupby(col)["split"].nunique()
        bad = spanning[spanning > 1]
        assert bad.empty, f"{col} spans multiple splits: {bad.index.tolist()[:5]}"

    manifest_cols = [
        "image_path",
        "annotation_path",
        "label",
        "class_names",
        "multi_class",
        "group_id",
        "sha256",
        "split",
    ]
    for split in ("train", "val", "test"):
        part = df[df["split"] == split][manifest_cols].sort_values("image_path")
        part.to_csv(out / f"{split}.csv", index=False)

    # combined manifest + metadata
    df[manifest_cols].sort_values(["split", "image_path"]).to_csv(
        out / "all.csv", index=False
    )
    meta = dict(
        seed=seed,
        dhash_threshold=DHASH_THRESHOLD,
        split_fractions=SPLIT_FRACS,
        totals={s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
        per_class={
            cls: {s: int(((df["folder_class"] == cls) & (df["split"] == s)).sum())
                  for s in ("train", "val", "test")}
            for cls in sorted(df["folder_class"].unique())
        },
        n_near_duplicate_groups=int(df["group_id"].nunique()),
        n_multi_image_groups=int((df.groupby("group_id").size() > 1).sum()),
    )
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/neu_det"))
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    meta = build(args.raw, args.out, args.seed)
    print(json.dumps(meta, indent=2))
    print(f"\nWrote manifests to {args.out}")


if __name__ == "__main__":
    main()
