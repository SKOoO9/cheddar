from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable

from .config import ConfigError, ResolvedConfig
from .runner import CommandRunner, which


@dataclass(frozen=True)
class SyncItem:
    source: Path
    destination: Path
    size: int
    sha256: str | None = None


def _excluded(path: Path, patterns: Iterable[str]) -> bool:
    text = path.as_posix()
    return any(fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(text, pattern) for pattern in patterns)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_copy(source: Path, destination: Path, *, excludes: Iterable[str] = (), checksum: bool = False) -> list[SyncItem]:
    if not source.exists():
        raise FileNotFoundError(f"Sync source does not exist: {source}")
    source = source.resolve()
    destination = destination.resolve()
    items: list[SyncItem] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or _excluded(path.relative_to(source), excludes):
            continue
        rel = path.relative_to(source)
        dst = destination / rel
        if dst.exists() and dst.stat().st_size == path.stat().st_size:
            if not checksum or sha256_file(dst) == sha256_file(path):
                continue
        items.append(SyncItem(path, dst, path.stat().st_size, sha256_file(path) if checksum else None))
    return items


def copy_items(items: Iterable[SyncItem], *, dry_run: bool = False) -> None:
    for item in items:
        if dry_run:
            print(f"DRY RUN: copy {item.source} -> {item.destination}")
            continue
        item.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, item.destination)


def write_manifest(path: Path, items: list[SyncItem], *, mode: str, dry_run: bool) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "dry_run": dry_run,
        "count": len(items),
        "items": [
            {
                "source": str(item.source),
                "destination": str(item.destination),
                "size": item.size,
                "sha256": item.sha256,
            }
            for item in items
        ],
    }
    if dry_run:
        print(json.dumps(payload, indent=2))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sync_tree(
    source: str | Path,
    destination: str | Path,
    *,
    config: ResolvedConfig,
    mode: str,
    dry_run: bool = False,
    checksum: bool = False,
    use_rsync: bool = True,
) -> list[SyncItem]:
    excludes = list(config.sync.get("excludes", []))
    source_path = Path(source).expanduser()
    destination_path = Path(destination).expanduser()
    manifest = config.output_root / "logs" / f"sync_manifest_{mode}.json"

    if not source_path.exists():
        raise ConfigError(
            f"Sync source does not exist: {source_path}. "
            "Set the appropriate source path in config/profiles.yaml or via an environment variable."
        )
    if source_path.resolve() == destination_path.resolve():
        raise ConfigError(f"Sync source and destination are the same path: {source_path}. Skip sync for this profile.")

    rsync_cmd = str(config.tools.get("rsync", "rsync"))
    if use_rsync and which(rsync_cmd):
        runner = CommandRunner(config.output_root / "logs" / f"sync_{mode}.jsonl", dry_run=dry_run)
        args: list[str | Path] = [rsync_cmd, "-a", "--human-readable"]
        if checksum:
            args.append("--checksum")
        for pattern in excludes:
            args.extend(["--exclude", pattern])
        args.extend([str(source_path) + "/", str(destination_path) + "/"])
        runner.run(args)

    items = plan_copy(source_path, destination_path, excludes=excludes, checksum=checksum)
    if not use_rsync or not which(rsync_cmd):
        copy_items(items, dry_run=dry_run)
    write_manifest(manifest, items, mode=mode, dry_run=dry_run)
    return items


def sync_raw(config: ResolvedConfig, *, dry_run: bool = False, checksum: bool = False) -> list[SyncItem]:
    source = config.sync.get("raw_source")
    if not source:
        raise ConfigError(
            "Profile sync.raw_source is not configured. "
            "Set CHEDDAR_RAW_SOURCE=/path/to/incoming/RawData or create a private config/profiles.yaml. "
            "If your data is already in RawData/, skip DA00 and run DA01/scan/convert."
        )
    return sync_tree(source, config.raw_root, config=config, mode="raw", dry_run=dry_run, checksum=checksum)


def sync_data_to_gpu(config: ResolvedConfig, *, dry_run: bool = False, checksum: bool = False) -> list[SyncItem]:
    gpu_root = config.sync.get("gpu_root")
    if not gpu_root:
        raise ConfigError("Profile sync.gpu_root is not configured.")
    return sync_tree(config.data_root, Path(gpu_root) / "Data", config=config, mode="data_to_gpu", dry_run=dry_run, checksum=checksum)


def sync_output_from_gpu(config: ResolvedConfig, *, dry_run: bool = False, checksum: bool = False) -> list[SyncItem]:
    gpu_root = config.sync.get("gpu_root")
    if not gpu_root:
        raise ConfigError("Profile sync.gpu_root is not configured.")
    return sync_tree(Path(gpu_root) / "Output", config.output_root, config=config, mode="output_from_gpu", dry_run=dry_run, checksum=checksum)
