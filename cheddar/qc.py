from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_DIR_NAMES = {
    "RawData",
    "Data",
    "Output",
    "Work",
    ".private",
    ".idea",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {
    ".DCM",
    ".dcm",
    ".dicom",
    ".nii",
    ".gz",
    ".bval",
    ".bvec",
    ".mif",
    ".log",
}
FORBIDDEN_FILENAMES = {
    ".DS_Store",
    "subject_map.tsv",
    "participants_key.tsv",
}


def is_forbidden_repo_path(path: str | Path) -> bool:
    path = Path(path)
    if any(part in FORBIDDEN_DIR_NAMES for part in path.parts):
        return True
    if path.name in FORBIDDEN_FILENAMES:
        return True
    if path.name.endswith(".nii.gz"):
        return True
    return path.suffix in FORBIDDEN_SUFFIXES


def audit_tracked_paths(paths: Iterable[str | Path]) -> dict[str, Any]:
    forbidden = [str(path) for path in paths if is_forbidden_repo_path(path)]
    return {"ok": not forbidden, "forbidden": forbidden, "count": len(forbidden)}


def data_tree_summary(data_root: Path) -> dict[str, Any]:
    subjects = sorted(path.name for path in data_root.glob("sub-*") if path.is_dir())
    derivatives = data_root / "derivatives"
    pipelines = sorted(path.name for path in derivatives.iterdir() if path.is_dir()) if derivatives.exists() else []
    return {
        "data_root": str(data_root),
        "exists": data_root.exists(),
        "subjects": subjects,
        "derivatives": pipelines,
    }


def write_qc_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

