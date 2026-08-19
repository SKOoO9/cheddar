from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

from .config import ConfigError


def which(command: str) -> str | None:
    return shutil.which(command)


def shell_join(args: Iterable[str | Path]) -> str:
    rendered: list[str] = []
    for arg in args:
        text = str(arg)
        if not text or any(ch.isspace() for ch in text) or "'" in text:
            rendered.append("'" + text.replace("'", "'\"'\"'") + "'")
        else:
            rendered.append(text)
    return " ".join(rendered)


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    dry_run: bool = False


@dataclass
class CommandRunner:
    log_file: Path | None = None
    dry_run: bool = False
    env: dict[str, str] | None = None
    records: list[dict[str, object]] = field(default_factory=list)

    def run(self, args: Iterable[str | Path], *, cwd: str | Path | None = None, check: bool = True) -> CommandResult:
        cmd = [str(arg) for arg in args]
        record: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": cmd,
            "rendered": shell_join(cmd),
            "cwd": str(cwd) if cwd is not None else None,
            "dry_run": self.dry_run,
        }
        if self.dry_run:
            result = CommandResult(cmd, 0, dry_run=True)
            record["returncode"] = 0
            self._record(record)
            print("DRY RUN:", record["rendered"])
            return result

        executable = cmd[0]
        if "/" in executable:
            executable_path = Path(executable)
            if not executable_path.exists():
                raise ConfigError(
                    f"Command not found: {executable}. Check the tool path in your profile config."
                )
        else:
            path_env = None if self.env is None else self.env.get("PATH")
            if shutil.which(executable, path=path_env) is None:
                raise ConfigError(
                    f"Command not found on PATH: {executable}. "
                    "Run `python -m cheddar env doctor --profile <profile>` to check tools, "
                    "then set the command path in config/profiles.yaml or with the relevant environment variable."
                )

        proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=self.env)
        record.update({"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
        self._record(record)
        if check and proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            tail = "\n".join(detail[-8:])
            raise ConfigError(f"Command failed with exit code {proc.returncode}: {shell_join(cmd)}\n{tail}")
        return CommandResult(cmd, proc.returncode, proc.stdout, proc.stderr)

    def _record(self, record: dict[str, object]) -> None:
        self.records.append(record)
        if self.log_file is None:
            return
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def command_version(command: str, *, args: tuple[str, ...] = ("--version",), timeout: float = 5.0) -> dict[str, object]:
    found = which(command)
    if found is None:
        return {"command": command, "found": False, "path": None, "version": None}
    try:
        proc = subprocess.run([found, *args], text=True, capture_output=True, timeout=timeout)
        text = (proc.stdout or proc.stderr).strip().splitlines()
        version = text[0] if text else ""
        return {"command": command, "found": True, "path": found, "version": version, "returncode": proc.returncode}
    except Exception as exc:
        return {"command": command, "found": True, "path": found, "version": None, "error": str(exc)}
