from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Any

from .config import ConfigError, ResolvedConfig
from .derivatives import write_dataset_description, write_json
from .dicom import SeriesInfo, classify_label, load_bidsmap, scan_raw
from .paths import SubjectRecord, assign_subjects
from .runner import CommandRunner


FORBIDDEN_JSON_PREFIXES = (
    "Patient",
    "Institution",
    "Referring",
    "Performing",
    "Operators",
)
FORBIDDEN_JSON_KEYS = {
    "AccessionNumber",
    "DeviceSerialNumber",
    "StationName",
    "StudyID",
    "StudyDescription",
    "ProcedureStepDescription",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "DICOMSourceFile",
    "SourceMetadata",
}


def is_forbidden_json_key(key: str) -> bool:
    return key in FORBIDDEN_JSON_KEYS or any(key.startswith(prefix) for prefix in FORBIDDEN_JSON_PREFIXES)


def scrub_sidecar(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not is_forbidden_json_key(key)}


def _entities(record: SubjectRecord, series: SeriesInfo) -> str:
    parts = [record.participant_id, record.session_id]
    if series.rule.acq:
        parts.append(f"acq-{series.rule.acq}")
    if series.rule.direction:
        parts.append(f"dir-{series.rule.direction}")
    return "_".join(parts)


def target_stem(record: SubjectRecord, series: SeriesInfo) -> str:
    return f"{_entities(record, series)}_{series.rule.suffix}"


def target_dir(config: ResolvedConfig, record: SubjectRecord, series: SeriesInfo) -> Path:
    return config.data_root / record.participant_id / record.session_id / series.rule.datatype


def _copy_if_exists(src: Path, dst: Path, *, dry_run: bool = False, force: bool = False) -> bool:
    if not src.exists():
        return False
    if dry_run:
        print(f"DRY RUN: copy {src} -> {dst}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return False
    shutil.copy2(src, dst)
    return True


def _converted_match_outputs(tmp: Path, series: SeriesInfo) -> list[Path]:
    rules = load_bidsmap_from_tmp(series)
    matches: list[Path] = []
    for json_path in sorted(tmp.glob("*.json")):
        label = json_path.stem
        if classify_label(label, rules) and classify_label(label, rules).name == series.rule.name:  # type: ignore[union-attr]
            matches.append(json_path)
    return matches


def load_bidsmap_from_tmp(series: SeriesInfo) -> list[Any]:
    return [series.rule]


def _write_participants(config: ResolvedConfig, records: list[SubjectRecord]) -> None:
    participants = config.data_root / "participants.tsv"
    participants.parent.mkdir(parents=True, exist_ok=True)
    lines = ["participant_id"]
    for participant in sorted({record.participant_id for record in records}):
        lines.append(participant)
    participants.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for record in records:
        sessions = config.data_root / record.participant_id / f"{record.participant_id}_sessions.tsv"
        sessions.parent.mkdir(parents=True, exist_ok=True)
        sessions.write_text("session_id\tsession_order\tdays_from_baseline\n" f"{record.session_id}\t1\t0\n", encoding="utf-8")


def _relative_to_data(path: Path, config: ResolvedConfig) -> str:
    return path.relative_to(config.data_root).as_posix()


def _update_json_sidecar(path: Path, updates: dict[str, Any], *, dry_run: bool = False) -> None:
    if dry_run:
        print(f"DRY RUN: update sidecar {path}: {updates}")
        return
    payload: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload = scrub_sidecar(payload)
    payload.update(updates)
    write_json(path, payload)


def _clean_subject_workdir(config: ResolvedConfig, tmp: Path, *, dry_run: bool = False) -> bool:
    convert_root = config.work_root / "convert"
    tmp_resolved = tmp.resolve(strict=False)
    convert_root_resolved = convert_root.resolve(strict=False)
    if tmp_resolved == convert_root_resolved or not tmp_resolved.is_relative_to(convert_root_resolved):
        raise ConfigError(f"Refusing to clean unsafe conversion work directory: {tmp}")
    if not tmp.exists():
        return False
    if dry_run:
        print(f"DRY RUN: remove temporary conversion directory {tmp}")
        return True
    shutil.rmtree(tmp)
    return True


def _organize_subject_conversion(
    config: ResolvedConfig,
    record: SubjectRecord,
    series_list: list[SeriesInfo],
    tmp: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, list[str]]:
    copied: dict[str, list[str]] = defaultdict(list)
    dwi_targets: list[Path] = []
    fmap_jsons: list[Path] = []
    for series in series_list:
        matches = _converted_match_outputs(tmp, series)
        for json_path in matches:
            base = json_path.with_suffix("")
            nii = base.with_suffix(".nii.gz")
            if not nii.exists():
                nii = base.with_suffix(".nii")
            dst_base = target_dir(config, record, series) / target_stem(record, series)
            dst_nii = dst_base.with_suffix(".nii.gz") if nii.suffix == ".gz" else dst_base.with_suffix(".nii")
            dst_json = dst_base.with_suffix(".json")
            _copy_if_exists(nii, dst_nii, dry_run=dry_run, force=force)
            if json_path.exists() and not dry_run:
                payload = scrub_sidecar(json.loads(json_path.read_text(encoding="utf-8")))
                write_json(dst_json, payload)
            elif dry_run:
                print(f"DRY RUN: scrub/copy {json_path} -> {dst_json}")
            if series.rule.datatype == "dwi":
                _copy_if_exists(base.with_suffix(".bval"), dst_base.with_suffix(".bval"), dry_run=dry_run, force=force)
                _copy_if_exists(base.with_suffix(".bvec"), dst_base.with_suffix(".bvec"), dry_run=dry_run, force=force)
                dwi_targets.append(dst_nii)
                _update_json_sidecar(
                    dst_json,
                    {
                        "PhaseEncodingDirection": config.diffusion.get("phase_encoding_direction", "j"),
                        "TotalReadoutTime": config.diffusion.get("total_readout_time"),
                        "B0FieldSource": [f"{record.participant_id}_{record.session_id}_pepolar"],
                    },
                    dry_run=dry_run,
                )
            if series.rule.datatype == "fmap":
                fmap_jsons.append(dst_json)
                _update_json_sidecar(
                    dst_json,
                    {
                        "PhaseEncodingDirection": config.diffusion.get("reverse_phase_encoding_direction", "j-"),
                        "TotalReadoutTime": config.diffusion.get("total_readout_time"),
                        "B0FieldIdentifier": f"{record.participant_id}_{record.session_id}_pepolar",
                    },
                    dry_run=dry_run,
                )
            copied[series.rule.name].append(str(dst_nii))

    intended = [_relative_to_data(path, config) for path in dwi_targets]
    for fmap_json in fmap_jsons:
        _update_json_sidecar(fmap_json, {"IntendedFor": intended}, dry_run=dry_run)
    return copied


def convert_all(
    config: ResolvedConfig,
    *,
    dry_run: bool = False,
    force: bool = False,
    clean_work: bool = True,
) -> dict[str, Any]:
    series = scan_raw(config.raw_root, config.bidsmap_path)
    source_folders = sorted({item.subject_folder for item in series})
    records = assign_subjects(config, source_folders, dry_run=dry_run)
    record_by_source = {(record.source_folder, record.session_id): record for record in records}
    if not dry_run:
        write_dataset_description(config.data_root, config.dataset_name, dataset_type="raw")
        _write_participants(config, records)

    by_subject: dict[str, list[SeriesInfo]] = defaultdict(list)
    for item in series:
        by_subject[item.subject_folder].append(item)

    summary: dict[str, Any] = {"subjects": {}, "dry_run": dry_run}
    for source, subject_series in sorted(by_subject.items()):
        record = record_by_source[(source, config.default_session)]
        tmp = config.work_root / "convert" / record.participant_id
        runner = CommandRunner(config.output_root / "logs" / f"convert_{record.participant_id}.jsonl", dry_run=dry_run)
        cleaned = _clean_subject_workdir(config, tmp, dry_run=dry_run) if clean_work else False
        if not dry_run:
            tmp.mkdir(parents=True, exist_ok=True)
        dicom_dir = subject_series[0].dicom_dir
        runner.run([
            config.tools.get("dcm2niix", "dcm2niix"),
            "-b",
            "y",
            "-z",
            "y",
            "-o",
            tmp,
            "-f",
            "%p_%s",
            dicom_dir,
        ])
        if dry_run and clean_work:
            copied = {}
        else:
            copied = _organize_subject_conversion(config, record, subject_series, tmp, dry_run=dry_run, force=force)
        summary["subjects"][record.participant_id] = {
            "source_folder": source,
            "work_dir": str(tmp),
            "work_cleaned": cleaned,
            "series": copied,
        }
    return summary
