from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dataset_description(name: str, *, generated_by: str | None = None, dataset_type: str = "derivative") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "Name": name,
        "BIDSVersion": "1.10.0",
        "DatasetType": dataset_type,
    }
    if generated_by:
        payload["GeneratedBy"] = [{"Name": generated_by, "Version": "0.1.0"}]
    return payload


def write_dataset_description(root: Path, name: str, *, generated_by: str | None = None, dataset_type: str = "derivative") -> Path:
    path = root / "dataset_description.json"
    write_json(path, dataset_description(name, generated_by=generated_by, dataset_type=dataset_type))
    return path


def provenance(
    *,
    description: str,
    sources: list[str],
    command_log: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "Description": description,
        "Sources": sources,
        "GeneratedBy": [{"Name": "cheddar", "Version": "0.1.0"}],
        "GeneratedAt": datetime.now(timezone.utc).isoformat(),
    }
    if command_log:
        payload["CommandLog"] = command_log
    if extra:
        payload.update(extra)
    return payload

