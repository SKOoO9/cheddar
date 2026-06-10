from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    yaml = None  # type: ignore


@dataclass(frozen=True)
class SeriesRule:
    name: str
    match: re.Pattern[str]
    datatype: str
    suffix: str
    priority: int = 0
    acq: str | None = None
    direction: str | None = None


@dataclass(frozen=True)
class SeriesInfo:
    subject_folder: str
    dicom_dir: Path
    dicom_file: Path
    rule: SeriesRule
    label: str


def load_bidsmap(path: str | Path) -> list[SeriesRule]:
    text = Path(path).read_text(encoding="utf-8")
    payload = (yaml.safe_load(text) if yaml is not None else json.loads(text)) or {}
    rules: list[SeriesRule] = []
    for item in payload.get("series", []):
        direction = item.get("dir")
        rules.append(
            SeriesRule(
                name=str(item["name"]),
                match=re.compile(str(item["match"]), re.IGNORECASE),
                datatype=str(item["datatype"]),
                suffix=str(item["suffix"]),
                priority=int(item.get("priority", 0)),
                acq=item.get("acq"),
                direction=str(direction) if direction is not None else None,
            )
        )
    return sorted(rules, key=lambda rule: rule.priority, reverse=True)


def classify_label(label: str, rules: list[SeriesRule]) -> SeriesRule | None:
    for rule in rules:
        if rule.match.search(label):
            return rule
    return None


def dicom_label(path: Path) -> str:
    stem = path.stem
    parts = stem.split(".")
    if len(parts) >= 5:
        return parts[4]
    return stem


def iter_dicom_dirs(raw_root: Path) -> list[Path]:
    if not raw_root.exists():
        return []
    dirs = []
    for path in sorted(raw_root.rglob("*")):
        if path.is_dir() and path.name.lower() == "dicom" and any(child.suffix.lower() == ".dcm" for child in path.iterdir() if child.is_file()):
            dirs.append(path)
    return dirs


def scan_raw(raw_root: Path, bidsmap_path: str | Path) -> list[SeriesInfo]:
    rules = load_bidsmap(bidsmap_path)
    series: list[SeriesInfo] = []
    for dicom_dir in iter_dicom_dirs(raw_root):
        subject_folder = dicom_dir.parent.name
        for dicom_file in sorted(path for path in dicom_dir.iterdir() if path.is_file() and path.suffix.lower() == ".dcm"):
            label = dicom_label(dicom_file)
            rule = classify_label(label, rules)
            if rule is None:
                continue
            series.append(SeriesInfo(subject_folder, dicom_dir, dicom_file, rule, label))
    return series


def summarize_scan(series: list[SeriesInfo]) -> dict[str, Any]:
    by_subject: dict[str, list[str]] = {}
    for item in series:
        by_subject.setdefault(item.subject_folder, []).append(item.rule.name)
    return {
        "subjects": {
            subject: sorted(labels)
            for subject, labels in sorted(by_subject.items())
        },
        "n_subjects": len(by_subject),
        "n_series": len(series),
    }
