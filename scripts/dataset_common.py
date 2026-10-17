"""Shared helpers for NEU-DET auditing and splitting.

NEU-DET layout (as released):
    <raw>/train/images/<class>/<class>_<n>.jpg
    <raw>/train/annotations/<class>_<n>.xml
    <raw>/validation/images/<class>/<class>_<n>.jpg
    <raw>/validation/annotations/<class>_<n>.xml

The *class folder* is the authoritative image-level label. Pascal VOC XML provides
boxes (possibly several per image, normally all the same class for NEU-DET).
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Box:
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def area(self) -> int:
        return max(0, self.xmax - self.xmin) * max(0, self.ymax - self.ymin)


@dataclass
class Sample:
    image_path: Path
    annotation_path: Path | None
    folder_class: str          # label from the class subfolder (authoritative)
    orig_split: str            # "train" or "validation" as shipped
    width: int | None = None
    height: int | None = None
    depth: int | None = None
    boxes: list[Box] = field(default_factory=list)
    corrupt: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def xml_classes(self) -> list[str]:
        return [b.name for b in self.boxes]


def iter_samples(raw_root: Path) -> list[Sample]:
    """Discover every image and pair it with its annotation by basename stem."""
    raw_root = Path(raw_root)
    samples: list[Sample] = []
    for orig_split in ("train", "validation"):
        img_root = raw_root / orig_split / "images"
        ann_root = raw_root / orig_split / "annotations"
        if not img_root.exists():
            continue
        for img_path in sorted(img_root.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS or not img_path.is_file():
                continue
            folder_class = img_path.parent.name
            ann_path = ann_root / f"{img_path.stem}.xml"
            samples.append(
                Sample(
                    image_path=img_path,
                    annotation_path=ann_path if ann_path.exists() else None,
                    folder_class=folder_class,
                    orig_split=orig_split,
                )
            )
    return samples


def parse_annotation(sample: Sample) -> None:
    """Fill width/height/depth and boxes from the XML; record notes on problems."""
    if sample.annotation_path is None:
        sample.notes.append("missing_xml")
        return
    try:
        root = ET.parse(sample.annotation_path).getroot()
    except ET.ParseError as exc:
        sample.notes.append(f"xml_parse_error:{exc}")
        return
    size = root.find("size")
    if size is not None:
        sample.width = _int(size.findtext("width"))
        sample.height = _int(size.findtext("height"))
        sample.depth = _int(size.findtext("depth"))
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        sample.boxes.append(
            Box(
                name=(obj.findtext("name") or "").strip(),
                xmin=_int(bb.findtext("xmin")) or 0,
                ymin=_int(bb.findtext("ymin")) or 0,
                xmax=_int(bb.findtext("xmax")) or 0,
                ymax=_int(bb.findtext("ymax")) or 0,
            )
        )


def load_image_size(sample: Sample) -> tuple[int, int] | None:
    """Open the image to get real pixel dimensions; mark corrupt on failure."""
    try:
        with Image.open(sample.image_path) as im:
            im.verify()
        with Image.open(sample.image_path) as im:
            return im.size  # (w, h)
    except Exception as exc:  # noqa: BLE001
        sample.corrupt = True
        sample.notes.append(f"corrupt_image:{exc}")
        return None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, hash_size: int = 8) -> str:
    """Difference hash (perceptual) as a hex string; robust to minor changes."""
    with Image.open(path) as im:
        im = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        px = list(im.getdata())
    bits = 0
    idx = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = px[row * (hash_size + 1) + col]
            right = px[row * (hash_size + 1) + col + 1]
            bits = (bits << 1) | int(left > right)
            idx += 1
    return f"{bits:0{(hash_size * hash_size) // 4}x}"


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _int(v: str | None) -> int | None:
    try:
        return int(float(v)) if v is not None else None
    except (TypeError, ValueError):
        return None
