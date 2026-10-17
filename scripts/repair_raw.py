"""STEP 1 (part A.1) — documented, idempotent repairs to the raw NEU-DET release.

The community NEU-DET mirror ships with one misfiled annotation:
  * image  train/images/crazing/crazing_240.jpg  has no annotation in train/
  * file   validation/annotations/crazing_240.xml  references crazing_240.jpg but
           there is no crazing_240 image under validation/

The XML clearly belongs to the train image (its <filename> is crazing_240.jpg and
no validation image exists), so we relocate it to train/annotations/. This is the
only mutation we make to the raw data; it is logged here and in the README so the
pipeline is reproducible from the original archive.

The exact-duplicate pair (patches_101.jpg == patches_105.jpg, identical bytes) is
NOT deleted — it is left in place and kept together in one split by make_splits.py's
group clustering, and recorded in the audit report.

Usage:
    python scripts/repair_raw.py --raw data/raw/neu_det
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def repair(raw: Path) -> list[str]:
    actions: list[str] = []
    orphan = raw / "validation" / "annotations" / "crazing_240.xml"
    target = raw / "train" / "annotations" / "crazing_240.xml"
    img = raw / "train" / "images" / "crazing" / "crazing_240.jpg"

    if target.exists():
        actions.append(f"skip: {target} already exists")
    elif orphan.exists() and img.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(orphan), str(target))
        actions.append(f"moved: {orphan} -> {target}")
    else:
        actions.append("skip: orphan XML or train image not found (already repaired?)")
    return actions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/neu_det"))
    args = ap.parse_args()
    for line in repair(args.raw):
        print(line)


if __name__ == "__main__":
    main()
