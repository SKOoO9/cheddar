from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

from .config import ResolvedConfig, SUPPORTED_SYSTEMS
from .runner import command_version


REQUIRED_TOOLS = (
    "dcm2niix",
    "mrconvert",
    "mrcat",
    "mrinfo",
    "dwidenoise",
    "mrdegibbs",
    "dwifslpreproc",
    "dwibiascorrect",
    "dwi2mask",
    "dwiextract",
    "mrmath",
    "fslmerge",
    "fslroi",
    "fslmaths",
    "bet",
    "topup",
    "eddy",
    "N4BiasFieldCorrection",
)
OPTIONAL_TOOLS = ("eddy_cuda", "rsync")


def _tool_name(config: ResolvedConfig, key: str) -> str:
    return str(config.tools.get(key, key))


def _path_status(path: Path, *, create_parent: bool = False) -> dict[str, Any]:
    exists = path.exists()
    writable = False
    target = path if exists else path.parent
    if create_parent:
        target.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
    if target.exists():
        writable = os.access(target, os.W_OK)
    return {"path": str(path), "exists": exists, "parent_writable": writable}


def doctor(config: ResolvedConfig, *, strict: bool = False) -> dict[str, Any]:
    system = platform.system()
    report: dict[str, Any] = {
        "system": system,
        "supported_system": system in SUPPORTED_SYSTEMS,
        "python": sys.version,
        "profile": config.profile_name,
        "paths": {
            "root": _path_status(config.root),
            "raw_root": _path_status(config.raw_root),
            "data_root": _path_status(config.data_root),
            "output_root": _path_status(config.output_root),
            "work_root": _path_status(config.work_root),
        },
        "tools": {},
        "disk": {},
        "strict": strict,
    }

    for key in REQUIRED_TOOLS:
        report["tools"][key] = command_version(_tool_name(config, key))
    for key in OPTIONAL_TOOLS:
        report["tools"][key] = command_version(_tool_name(config, key))

    try:
        usage = shutil.disk_usage(config.root if config.root.exists() else config.root.parent)
        report["disk"] = {
            "total_gb": round(usage.total / 1e9, 3),
            "used_gb": round(usage.used / 1e9, 3),
            "free_gb": round(usage.free / 1e9, 3),
        }
    except Exception as exc:
        report["disk"] = {"error": str(exc)}

    missing_required = [key for key in REQUIRED_TOOLS if not report["tools"][key]["found"]]
    report["missing_required_tools"] = missing_required
    report["ok"] = bool(report["supported_system"] and (not missing_required or not strict))
    return report


def write_doctor_report(config: ResolvedConfig, report: dict[str, Any]) -> Path:
    out = config.output_root / "logs" / f"env_doctor_{config.profile_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
