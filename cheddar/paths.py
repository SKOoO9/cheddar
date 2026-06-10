from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import ResolvedConfig


def participant_id(index: int, *, prefix: str = "sub", width: int = 3) -> str:
    return f"{prefix}-{index:0{width}d}"


@dataclass(frozen=True)
class SubjectRecord:
    participant_id: str
    session_id: str
    source_folder: str


class DataPaths:
    def __init__(self, config: ResolvedConfig) -> None:
        self.config = config

    def subject_dir(self, participant: str, session: str | None = None) -> Path:
        base = self.config.data_root / participant
        return base if session is None else base / session

    def datatype_dir(self, participant: str, session: str, datatype: str) -> Path:
        return self.subject_dir(participant, session) / datatype

    def derivative_dir(self, pipeline: str, participant: str, session: str, datatype: str) -> Path:
        return self.config.data_root / "derivatives" / pipeline / participant / session / datatype

    def output_log_dir(self) -> Path:
        return self.config.output_root / "logs"

    def qc_dir(self, participant: str | None = None, session: str | None = None) -> Path:
        base = self.config.output_root / "qc"
        if participant:
            base /= participant
        if session:
            base /= session
        return base


def read_subject_map(path: Path) -> list[SubjectRecord]:
    if not path.exists():
        return []
    records: list[SubjectRecord] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    required = {"participant_id", "session_id", "source_folder"}
    if not required.issubset(header):
        raise ValueError(f"Subject map {path} must contain columns: {', '.join(sorted(required))}")
    index = {name: header.index(name) for name in required}
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = line.split("\t")
        records.append(
            SubjectRecord(
                participant_id=cells[index["participant_id"]],
                session_id=cells[index["session_id"]],
                source_folder=cells[index["source_folder"]],
            )
        )
    return records


def write_subject_map(path: Path, records: Iterable[SubjectRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["participant_id\tsession_id\tsource_folder"]
    for record in records:
        lines.append(f"{record.participant_id}\t{record.session_id}\t{record.source_folder}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assign_subjects(config: ResolvedConfig, source_folders: Iterable[str], *, dry_run: bool = False) -> list[SubjectRecord]:
    existing = read_subject_map(config.private_subject_map)
    by_source = {(record.source_folder, record.session_id): record for record in existing}
    used_ids = {record.participant_id for record in existing}

    prefix = str(config.subject_ids.get("prefix", "sub"))
    start = int(config.subject_ids.get("start", 1))
    width = int(config.subject_ids.get("width", 3))

    next_index = start

    def next_participant() -> str:
        nonlocal next_index
        while True:
            candidate = participant_id(next_index, prefix=prefix, width=width)
            next_index += 1
            if candidate not in used_ids:
                used_ids.add(candidate)
                return candidate

    records = list(existing)
    for source in sorted(set(source_folders)):
        key = (source, config.default_session)
        if key in by_source:
            continue
        record = SubjectRecord(next_participant(), config.default_session, source)
        by_source[key] = record
        records.append(record)

    if not dry_run:
        write_subject_map(config.private_subject_map, records)
    return records

