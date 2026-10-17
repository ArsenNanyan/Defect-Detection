"""STEP 1 (part A) — audit NEU-DET images and annotations.

Checks performed
----------------
* every image resolves to an XML and vice-versa (orphans both directions)
* declared XML size matches real pixel size; depth sanity
* corrupt / unreadable images
* box validity: zero-area, out-of-bounds, very small, class name legal
* folder-class vs XML-class agreement; multi-class images
* duplicate file content (sha256) and near-duplicates (dhash)
* class counts and image-dimension distribution

Outputs
-------
    <out>/audit_report.csv     one row per image with all findings
    <out>/audit_summary.json   aggregate counts and problem lists
    <out>/figures/samples_<class>.png   >=6 boxed examples per class

Usage
-----
    python scripts/audit_data.py --raw data/raw/neu_det --out artifacts/reports
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from dataset_common import (  # noqa: E402
    CLASS_NAMES,
    dhash,
    iter_samples,
    load_image_size,
    parse_annotation,
    sha256_of,
)

MIN_BOX_SIDE = 3  # px; boxes thinner than this on any side are flagged "very small"


def audit(raw: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    samples = iter_samples(raw)
    rows = []
    problems: dict[str, list[str]] = defaultdict(list)
    sha_map: dict[str, list[str]] = defaultdict(list)
    dhash_map: dict[str, list[str]] = defaultdict(list)

    # collect annotation files to detect XML-without-image orphans
    xml_on_disk = {
        p.stem: p
        for split in ("train", "validation")
        for p in (raw / split / "annotations").glob("*.xml")
    }
    matched_xml_stems: set[str] = set()

    for s in samples:
        parse_annotation(s)
        real = load_image_size(s)
        rel = str(s.image_path.relative_to(raw))
        if s.annotation_path is not None:
            matched_xml_stems.add(s.annotation_path.stem)

        size_mismatch = False
        if real and s.width and s.height and (real != (s.width, s.height)):
            size_mismatch = True
            s.notes.append(f"size_mismatch:xml={s.width}x{s.height},real={real[0]}x{real[1]}")

        # box checks
        n_bad = n_small = n_oob = 0
        W, H = (real or (s.width, s.height) or (None, None))
        for b in s.boxes:
            if b.area <= 0:
                n_bad += 1
            if (b.xmax - b.xmin) < MIN_BOX_SIDE or (b.ymax - b.ymin) < MIN_BOX_SIDE:
                n_small += 1
            if W and H and (b.xmin < 0 or b.ymin < 0 or b.xmax > W or b.ymax > H):
                n_oob += 1
            if b.name not in CLASS_NAMES:
                s.notes.append(f"illegal_class:{b.name}")

        xml_classes = set(s.xml_classes)
        multi_class = len(xml_classes) > 1
        class_disagree = bool(xml_classes) and (s.folder_class not in xml_classes)

        sha = sha256_of(s.image_path)
        dh = dhash(s.image_path)
        sha_map[sha].append(rel)
        dhash_map[dh].append(rel)

        for n in s.notes:
            problems[n.split(":")[0]].append(rel)
        if n_bad:
            problems["zero_area_box"].append(rel)
        if n_small:
            problems["very_small_box"].append(rel)
        if n_oob:
            problems["out_of_bounds_box"].append(rel)
        if multi_class:
            problems["multi_class_image"].append(rel)
        if class_disagree:
            problems["folder_xml_class_disagree"].append(rel)

        rows.append(
            dict(
                image_path=rel,
                annotation_path=(
                    str(s.annotation_path.relative_to(raw)) if s.annotation_path else ""
                ),
                orig_split=s.orig_split,
                folder_class=s.folder_class,
                xml_classes=";".join(sorted(xml_classes)),
                n_boxes=len(s.boxes),
                width=W,
                height=H,
                depth=s.depth,
                corrupt=s.corrupt,
                size_mismatch=size_mismatch,
                zero_area_boxes=n_bad,
                very_small_boxes=n_small,
                out_of_bounds_boxes=n_oob,
                multi_class=multi_class,
                class_disagree=class_disagree,
                sha256=sha,
                dhash=dh,
                notes=";".join(s.notes),
            )
        )

    # XML without any matching image
    xml_orphans = sorted(set(xml_on_disk) - matched_xml_stems)
    for stem in xml_orphans:
        problems["xml_without_image"].append(str(xml_on_disk[stem].relative_to(raw)))

    df = pd.DataFrame(rows)
    df.to_csv(out / "audit_report.csv", index=False)

    dup_content = {k: v for k, v in sha_map.items() if len(v) > 1}
    near_dups = {k: v for k, v in dhash_map.items() if len(v) > 1}

    summary = dict(
        n_images=len(df),
        n_xml_files=len(xml_on_disk),
        images_missing_xml=problems.get("missing_xml", []),
        xml_without_image=problems.get("xml_without_image", []),
        corrupt_images=problems.get("corrupt_image", []),
        size_mismatches=int(df["size_mismatch"].sum()),
        zero_area_boxes=problems.get("zero_area_box", []),
        very_small_boxes=len(problems.get("very_small_box", [])),
        out_of_bounds_boxes=problems.get("out_of_bounds_box", []),
        multi_class_images=problems.get("multi_class_image", []),
        folder_xml_class_disagree=problems.get("folder_xml_class_disagree", []),
        class_counts_by_folder=dict(Counter(df["folder_class"])),
        orig_split_counts=dict(Counter(df["orig_split"])),
        image_dimension_counts={
            f"{w}x{h}": int(c)
            for (w, h), c in Counter(zip(df["width"], df["height"], strict=False)).items()
        },
        duplicate_content_groups={k: v for k, v in dup_content.items()},
        near_duplicate_groups_count=len(near_dups),
        boxes_per_image={
            "min": int(df["n_boxes"].min()),
            "max": int(df["n_boxes"].max()),
            "mean": round(float(df["n_boxes"].mean()), 3),
        },
    )
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_samples(df, raw, out / "figures")
    return summary


def _plot_samples(df: pd.DataFrame, raw: Path, fig_dir: Path, per_class: int = 6) -> None:
    for cls in CLASS_NAMES:
        sub = df[df["folder_class"] == cls].head(per_class)
        if sub.empty:
            continue
        cols = 3
        rows = (len(sub) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
        for ax, (_, r) in zip(axes, sub.iterrows(), strict=False):
            img = Image.open(raw / r["image_path"]).convert("RGB")
            ax.imshow(img)
            ann = raw / r["annotation_path"] if r["annotation_path"] else None
            if ann and ann.exists():
                import xml.etree.ElementTree as ET

                for obj in ET.parse(ann).getroot().findall("object"):
                    bb = obj.find("bndbox")
                    x0, y0 = int(bb.findtext("xmin")), int(bb.findtext("ymin"))
                    x1, y1 = int(bb.findtext("xmax")), int(bb.findtext("ymax"))
                    ax.add_patch(
                        mpatches.Rectangle(
                            (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=1.5
                        )
                    )
            ax.set_title(Path(r["image_path"]).name, fontsize=7)
            ax.axis("off")
        for ax in axes[len(sub):]:
            ax.axis("off")
        fig.suptitle(f"{cls} — sample annotated images")
        fig.tight_layout()
        fig.savefig(fig_dir / f"samples_{cls}.png", dpi=110)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/neu_det"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/reports"))
    args = ap.parse_args()
    summary = audit(args.raw, args.out)
    print(json.dumps({k: v for k, v in summary.items()
                      if not isinstance(v, (list, dict))}, indent=2))
    print("\nProblem counts:")
    for key in ("images_missing_xml", "xml_without_image", "corrupt_images",
                "zero_area_boxes", "out_of_bounds_boxes", "multi_class_images",
                "folder_xml_class_disagree"):
        v = summary.get(key, [])
        print(f"  {key}: {len(v) if isinstance(v, list) else v}")
    print(f"  duplicate_content_groups: {len(summary['duplicate_content_groups'])}")
    print(f"  near_duplicate_groups: {summary['near_duplicate_groups_count']}")
    print(f"\nWrote report to {args.out}")


if __name__ == "__main__":
    main()
