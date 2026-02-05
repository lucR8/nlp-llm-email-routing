from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    outputs: Path
    metrics: Path
    figures: Path
    checkpoints: Path


def default_paths(root: Path | None = None) -> Paths:
    root = (root or Path(".")).resolve()
    out = root / "outputs"
    return Paths(
        root=root,
        outputs=out,
        metrics=out / "metrics",
        figures=out / "figures",
        checkpoints=out / "checkpoints",
    )


DEPARTMENTS = [
    "Technical Support",
    "Customer Service",
    "Billing and Payments",
    "Sales and Pre-Sales",
    "General Inquiry",
]
