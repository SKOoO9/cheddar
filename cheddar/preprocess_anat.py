from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ResolvedConfig
from .derivatives import provenance, write_dataset_description, write_json
from .paths import DataPaths
from .runner import CommandRunner


def _first(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[0] if matches else None


def preprocess_anat_subject(config: ResolvedConfig, participant: str, session: str, *, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    paths = DataPaths(config)
    anat_dir = paths.datatype_dir(participant, session, "anat")
    out_dir = paths.derivative_dir("anat-preproc", participant, session, "anat")
    work_dir = config.work_root / "anat-preproc" / participant / session
    runner = CommandRunner(config.output_root / "logs" / f"anat_preproc_{participant}_{session}.jsonl", dry_run=dry_run)

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        write_dataset_description(config.data_root / "derivatives" / "anat-preproc", "CHEDDAR anatomical preprocessing", generated_by="cheddar")

    outputs: dict[str, Any] = {}
    for suffix in ("T1w", "FLAIR"):
        source = _first(anat_dir, f"*_{suffix}.nii*")
        if source is None:
            outputs[suffix] = {"status": "missing"}
            continue

        corrected = out_dir / f"{participant}_{session}_desc-N4_{suffix}.nii.gz"
        bias = out_dir / f"{participant}_{session}_desc-N4BiasField_{suffix}.nii.gz"
        brain = out_dir / f"{participant}_{session}_desc-skullstripped_{suffix}.nii.gz"
        mask = out_dir / f"{participant}_{session}_desc-{suffix}Brain_mask.nii.gz"

        if corrected.exists() and mask.exists() and not force:
            outputs[suffix] = {"status": "skipped", "corrected": str(corrected), "mask": str(mask)}
            continue

        runner.run([
            config.tools.get("N4BiasFieldCorrection", "N4BiasFieldCorrection"),
            "-d",
            "3",
            "-i",
            source,
            "-o",
            f"[{corrected},{bias}]",
        ])
        runner.run([
            config.tools.get("bet", "bet"),
            corrected,
            brain.with_suffix("").with_suffix(""),
            "-m",
        ])
        generated_mask = brain.with_name(brain.name.replace(".nii.gz", "_mask.nii.gz"))
        if not dry_run and generated_mask.exists() and generated_mask != mask:
            generated_mask.replace(mask)
        if dry_run:
            print(f"DRY RUN: normalize BET mask name {generated_mask} -> {mask}")

        sidecar = corrected.with_suffix("").with_suffix(".json")
        if not dry_run:
            write_json(
                sidecar,
                provenance(
                    description=f"N4-corrected {suffix} anatomical image.",
                    sources=[source.relative_to(config.data_root).as_posix()],
                    command_log=str(runner.log_file) if runner.log_file else None,
                    extra={"SkullStripped": False},
                ),
            )
        outputs[suffix] = {"status": "done", "corrected": str(corrected), "brain": str(brain), "mask": str(mask)}

    return outputs


def preprocess_anat_all(config: ResolvedConfig, *, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for subject_dir in sorted(config.data_root.glob("sub-*")):
        if not subject_dir.is_dir():
            continue
        for session_dir in sorted(subject_dir.glob("ses-*")):
            if session_dir.is_dir():
                results[f"{subject_dir.name}/{session_dir.name}"] = preprocess_anat_subject(
                    config,
                    subject_dir.name,
                    session_dir.name,
                    dry_run=dry_run,
                    force=force,
                )
    return results

