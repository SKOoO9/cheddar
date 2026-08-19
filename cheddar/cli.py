from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import ConfigError, load_resolved_config
from .convert import convert_all
from .dicom import scan_raw, summarize_scan
from .env import doctor, write_doctor_report
from .qc import audit_tracked_paths, data_tree_summary, write_qc_summary
from .sync import sync_data_to_gpu, sync_output_from_gpu, sync_raw


DEFAULT_CONFIG = "config/study.example.yaml"


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Study configuration YAML. Defaults to config/study.example.yaml.")
    parser.add_argument("--profile", default="local", help="Profile name from profiles YAML. Defaults to local.")


def _resolved(args: argparse.Namespace):
    return load_resolved_config(args.config, args.profile)


def _cmd_env_doctor(args: argparse.Namespace) -> int:
    config = _resolved(args)
    report = doctor(config, strict=args.strict)
    if args.write_report:
        report["report_path"] = str(write_doctor_report(config, report))
    _json_print(report)
    return 0 if report["ok"] else 1


def _cmd_scan(args: argparse.Namespace) -> int:
    config = _resolved(args)
    series = scan_raw(config.raw_root, config.bidsmap_path)
    _json_print(summarize_scan(series))
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    config = _resolved(args)
    if args.sync_command == "raw":
        items = sync_raw(config, dry_run=args.dry_run, checksum=args.checksum)
    elif args.sync_command == "data-to-gpu":
        items = sync_data_to_gpu(config, dry_run=args.dry_run, checksum=args.checksum)
    elif args.sync_command == "output-from-gpu":
        items = sync_output_from_gpu(config, dry_run=args.dry_run, checksum=args.checksum)
    else:
        raise ValueError(f"Unknown sync command: {args.sync_command}")
    _json_print({"mode": args.sync_command, "count": len(items), "dry_run": args.dry_run})
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    config = _resolved(args)
    summary = convert_all(config, dry_run=args.dry_run, force=args.force, clean_work=not args.keep_work)
    _json_print(summary)
    return 0


def _cmd_preprocess(args: argparse.Namespace) -> int:
    from .preprocess_anat import preprocess_anat_all, preprocess_anat_subject
    from .preprocess_dwi import preprocess_dwi_all, preprocess_dwi_subject

    config = _resolved(args)
    results: dict[str, Any] = {}
    if args.subject:
        session = args.session or config.default_session
        if not args.dwi_only:
            results["anat"] = preprocess_anat_subject(config, args.subject, session, dry_run=args.dry_run, force=args.force)
        if not args.anat_only:
            results["dwi"] = preprocess_dwi_subject(config, args.subject, session, dry_run=args.dry_run, force=args.force)
    else:
        if not args.dwi_only:
            results["anat"] = preprocess_anat_all(config, dry_run=args.dry_run, force=args.force)
        if not args.anat_only:
            results["dwi"] = preprocess_dwi_all(config, dry_run=args.dry_run, force=args.force)
    _json_print(results)
    return 0


def _cmd_qc(args: argparse.Namespace) -> int:
    config = _resolved(args)
    payload = {"data": data_tree_summary(config.data_root)}
    out = config.output_root / "qc" / "data_summary.json"
    if not args.dry_run:
        write_qc_summary(out, payload)
        payload["written"] = str(out)
    _json_print(payload)
    return 0


def _cmd_audit_paths(args: argparse.Namespace) -> int:
    payload = audit_tracked_paths(args.paths)
    _json_print(payload)
    return 0 if payload["ok"] else 1


def _cmd_run_all(args: argparse.Namespace) -> int:
    from .preprocess_anat import preprocess_anat_all
    from .preprocess_dwi import preprocess_dwi_all

    config = _resolved(args)
    payload = {
        "convert": convert_all(config, dry_run=args.dry_run, force=args.force, clean_work=not args.keep_work),
        "preprocess_anat": preprocess_anat_all(config, dry_run=args.dry_run, force=args.force),
        "preprocess_dwi": preprocess_dwi_all(config, dry_run=args.dry_run, force=args.force),
    }
    _json_print(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cheddar", description="CHEDDAR data synchronization, conversion, preprocessing, and QC wrapper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("env", help="Environment helpers.")
    env_sub = env_parser.add_subparsers(dest="env_command", required=True)
    doctor_parser = env_sub.add_parser("doctor", help="Check OS, paths, tools, and optional GPU tooling.")
    _add_config_args(doctor_parser)
    doctor_parser.add_argument("--strict", action="store_true", help="Return non-zero if required external tools are missing.")
    doctor_parser.add_argument("--write-report", action="store_true", help="Write JSON report into Output/logs.")
    doctor_parser.set_defaults(func=_cmd_env_doctor)

    scan_parser = subparsers.add_parser("scan", help="Scan RawData and summarize recognized DICOM series.")
    _add_config_args(scan_parser)
    scan_parser.set_defaults(func=_cmd_scan)

    sync_parser = subparsers.add_parser("sync", help="Synchronize raw/data/output trees across mounted paths.")
    sync_sub = sync_parser.add_subparsers(dest="sync_command", required=True)
    for name, help_text in (
        ("raw", "Copy new raw folders into RawData."),
        ("data-to-gpu", "Copy Data to the configured GPU root."),
        ("output-from-gpu", "Copy Output from the configured GPU root."),
    ):
        cmd = sync_sub.add_parser(name, help=help_text)
        _add_config_args(cmd)
        cmd.add_argument("--dry-run", action="store_true", help="Print planned work without copying.")
        cmd.add_argument("--checksum", action="store_true", help="Use SHA256 checksums to detect changed files.")
        cmd.set_defaults(func=_cmd_sync)

    convert_parser = subparsers.add_parser("convert", help="Convert RawData DICOMs into Data.")
    _add_config_args(convert_parser)
    convert_parser.add_argument("--dry-run", action="store_true")
    convert_parser.add_argument("--force", action="store_true")
    convert_parser.add_argument("--keep-work", action="store_true", help="Keep existing Work/convert scratch files before conversion.")
    convert_parser.set_defaults(func=_cmd_convert)

    preproc_parser = subparsers.add_parser("preprocess", help="Run anatomy and DWI preprocessing.")
    _add_config_args(preproc_parser)
    preproc_parser.add_argument("--subject", help="Optional participant id, e.g. sub-001.")
    preproc_parser.add_argument("--session", help="Optional session id. Defaults to config default_session.")
    preproc_parser.add_argument("--anat-only", action="store_true")
    preproc_parser.add_argument("--dwi-only", action="store_true")
    preproc_parser.add_argument("--dry-run", action="store_true")
    preproc_parser.add_argument("--force", action="store_true")
    preproc_parser.set_defaults(func=_cmd_preprocess)

    qc_parser = subparsers.add_parser("qc", help="Write or print lightweight data/QC summaries.")
    _add_config_args(qc_parser)
    qc_parser.add_argument("--dry-run", action="store_true")
    qc_parser.set_defaults(func=_cmd_qc)

    audit_parser = subparsers.add_parser("audit-paths", help="Audit paths for files that must not be committed.")
    audit_parser.add_argument("paths", nargs="+")
    audit_parser.set_defaults(func=_cmd_audit_paths)

    run_parser = subparsers.add_parser("run-all", help="Run convert then preprocessing.")
    _add_config_args(run_parser)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("--keep-work", action="store_true", help="Keep existing Work/convert scratch files before conversion.")
    run_parser.set_defaults(func=_cmd_run_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        parser.error(str(exc))
    return 2
