"""Industrial surface defect inspection package."""

__version__ = "0.1.0"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
