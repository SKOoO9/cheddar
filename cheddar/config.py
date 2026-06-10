from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import platform
import re
import json
from typing import Any, Mapping

try:  # PyYAML is preferred, but example configs are JSON-compatible YAML.
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    yaml = None  # type: ignore


SUPPORTED_SYSTEMS = {"Linux", "Darwin"}
ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]+)\}")


class ConfigError(RuntimeError):
    """Raised when a CHEDDAR configuration cannot be resolved safely."""


def _expand_env_defaults(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default)

    return ENV_DEFAULT_RE.sub(repl, text)


def expand_value(value: Any) -> Any:
    if isinstance(value, str):
        expanded = _expand_env_defaults(value)
        return os.path.expandvars(os.path.expanduser(expanded))
    if isinstance(value, list):
        return [expand_value(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_value(item) for key, item in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        if yaml is not None:
            payload = yaml.safe_load(handle) or {}
        else:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ConfigError(f"Configuration file must contain a mapping: {path}")
    return dict(expand_value(payload))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def _resolve_config_neighbor(config_path: Path, value: str | None, fallback_names: tuple[str, ...]) -> Path:
    if value:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else config_path.parent / candidate
    for name in fallback_names:
        candidate = config_path.parent / name
        if candidate.exists():
            return candidate
    return config_path.parent / fallback_names[0]


def _as_path(root: Path, value: str | Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


@dataclass(frozen=True)
class ResolvedConfig:
    config_path: Path
    profile_name: str
    project_name: str
    dataset_name: str
    default_session: str
    root: Path
    raw_root: Path
    data_root: Path
    output_root: Path
    work_root: Path
    private_root: Path
    private_subject_map: Path
    bidsmap_path: Path
    tools: dict[str, str] = field(default_factory=dict)
    sync: dict[str, Any] = field(default_factory=dict)
    diffusion: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    subject_ids: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_gpu_profile(self) -> bool:
        return "gpu" in self.profile_name.lower()

    def ensure_supported_os(self) -> None:
        system = platform.system()
        if system not in SUPPORTED_SYSTEMS:
            raise ConfigError(f"Unsupported operating system {system!r}; CHEDDAR supports Linux and macOS only.")

    def to_public_dict(self) -> dict[str, Any]:
        """Return a resolved config snapshot without private mappings or credentials."""
        return {
            "config_path": str(self.config_path),
            "profile_name": self.profile_name,
            "project_name": self.project_name,
            "dataset_name": self.dataset_name,
            "default_session": self.default_session,
            "root": str(self.root),
            "raw_root": str(self.raw_root),
            "data_root": str(self.data_root),
            "output_root": str(self.output_root),
            "work_root": str(self.work_root),
            "private_root": str(self.private_root),
            "bidsmap_path": str(self.bidsmap_path),
            "tools": self.tools,
            "sync": self.sync,
            "diffusion": self.diffusion,
            "preprocessing": self.preprocessing,
            "subject_ids": self.subject_ids,
        }


def load_resolved_config(config: str | Path, profile: str = "local") -> ResolvedConfig:
    config_path = Path(config).resolve()
    study = load_yaml(config_path)

    profiles_path = _resolve_config_neighbor(
        config_path,
        study.get("profiles_file"),
        ("profiles.yaml", "profiles.example.yaml"),
    )
    profiles_payload = load_yaml(profiles_path)
    profiles = profiles_payload.get("profiles", {})
    if profile not in profiles:
        available = ", ".join(sorted(profiles)) or "<none>"
        raise ConfigError(f"Profile {profile!r} not found in {profiles_path}; available profiles: {available}")

    profile_payload = profiles[profile] or {}
    if not isinstance(profile_payload, Mapping):
        raise ConfigError(f"Profile {profile!r} must be a mapping in {profiles_path}")

    merged = deep_merge(study, profile_payload)
    paths = dict(merged.get("paths", {}))
    root = Path(str(paths.get("root", "."))).expanduser()
    if not root.is_absolute():
        root = (config_path.parent.parent / root).resolve()
    else:
        root = root.resolve()

    raw_root = _as_path(root, paths.get("raw", "RawData"))
    data_root = _as_path(root, paths.get("data", "Data"))
    output_root = _as_path(root, paths.get("output", "Output"))
    work_root = _as_path(root, paths.get("work", "Work"))
    private_root = _as_path(root, paths.get("private", "Data/.private"))

    privacy = dict(merged.get("privacy", {}))
    private_subject_map = _as_path(root, privacy.get("private_subject_map", "Data/.private/subject_map.tsv"))
    bidsmap_path = _resolve_config_neighbor(
        config_path,
        merged.get("bidsmap_file"),
        ("bidsmap.yaml", "bidsmap.example.yaml"),
    ).resolve()

    resolved = ResolvedConfig(
        config_path=config_path,
        profile_name=profile,
        project_name=str(merged.get("project_name", "CHEDDAR")),
        dataset_name=str(merged.get("dataset_name", "CHEDDAR")),
        default_session=str(merged.get("default_session", "ses-01")),
        root=root,
        raw_root=raw_root,
        data_root=data_root,
        output_root=output_root,
        work_root=work_root,
        private_root=private_root,
        private_subject_map=private_subject_map,
        bidsmap_path=bidsmap_path,
        tools=dict(merged.get("tools", {})),
        sync=dict(merged.get("sync", {})),
        diffusion=dict(merged.get("diffusion", {})),
        preprocessing=dict(merged.get("preprocessing", {})),
        subject_ids=dict(merged.get("subject_ids", {})),
        raw=dict(merged.get("raw", {})),
    )
    resolved.ensure_supported_os()
    return resolved
