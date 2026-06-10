from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ResolvedConfig
from .derivatives import provenance, write_dataset_description, write_json
from .paths import DataPaths
from .runner import CommandRunner


@dataclass(frozen=True)
class DwiInput:
    acq: str
    nifti: Path
    bvec: Path
    bval: Path


def _strip_nii(path: Path) -> Path:
    if path.name.endswith(".nii.gz"):
        return path.with_name(path.name[:-7])
    if path.suffix == ".nii":
        return path.with_suffix("")
    return path


def _load_bvec(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 3:
        return arr
    if arr.shape[1] == 3:
        return arr.T
    raise ValueError(f"bvec file must be 3xN or Nx3: {path} got {arr.shape}")


def _find_dwi_inputs(config: ResolvedConfig, participant: str, session: str) -> list[DwiInput]:
    dwi_dir = DataPaths(config).datatype_dir(participant, session, "dwi")
    acquisitions = list(config.diffusion.get("acquisitions", []))
    inputs: list[DwiInput] = []
    for acq in acquisitions:
        matches = sorted(dwi_dir.glob(f"{participant}_{session}_acq-{acq}_*_dwi.nii*"))
        if not matches:
            continue
        nifti = matches[0]
        base = _strip_nii(nifti)
        inputs.append(DwiInput(acq, nifti, base.with_suffix(".bvec"), base.with_suffix(".bval")))
    return inputs


def _write_merged_gradients(inputs: list[DwiInput], bvec_out: Path, bval_out: Path, volume_table: Path) -> None:
    bvals: list[np.ndarray] = []
    bvecs: list[np.ndarray] = []
    rows = ["volume\tacquisition\tsource_volume\tbval"]
    offset = 0
    for item in inputs:
        vals = np.loadtxt(item.bval, dtype=float).reshape(-1)
        vec = _load_bvec(item.bvec)
        if vec.shape[1] != vals.size:
            raise ValueError(f"bvec/bval volume mismatch for {item.acq}: {vec.shape[1]} vs {vals.size}")
        bvals.append(vals)
        bvecs.append(vec)
        for idx, bval in enumerate(vals):
            rows.append(f"{offset + idx}\t{item.acq}\t{idx}\t{float(bval):.8g}")
        offset += vals.size
    bval_out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(bval_out, np.concatenate(bvals).reshape(1, -1), fmt="%.10g")
    np.savetxt(bvec_out, np.concatenate(bvecs, axis=1), fmt="%.10g")
    volume_table.write_text("\n".join(rows) + "\n", encoding="utf-8")


def preprocess_dwi_subject(config: ResolvedConfig, participant: str, session: str, *, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    inputs = _find_dwi_inputs(config, participant, session)
    if not inputs:
        return {"status": "missing_dwi"}

    paths = DataPaths(config)
    out_dir = paths.derivative_dir("dwi-preproc", participant, session, "dwi")
    work_dir = config.work_root / "dwi-preproc" / participant / session
    runner = CommandRunner(config.output_root / "logs" / f"dwi_preproc_{participant}_{session}.jsonl", dry_run=dry_run)

    stem = f"{participant}_{session}_acq-{config.diffusion.get('merged_acq_label', 'all')}"
    final_dwi = out_dir / f"{stem}_desc-preproc_dwi.nii.gz"
    final_bvec = out_dir / f"{stem}_desc-preproc_dwi.bvec"
    final_bval = out_dir / f"{stem}_desc-preproc_dwi.bval"
    volume_table = out_dir / f"{stem}_desc-volumes.tsv"
    mask = out_dir / f"{stem}_desc-brain_mask.nii.gz"
    brain_dwi = out_dir / f"{stem}_desc-skullstripped_dwi.nii.gz"
    mean_b0 = out_dir / f"{stem}_desc-meanb0_dwi.nii.gz"
    bias_field = out_dir / f"{stem}_desc-N4BiasField_dwi.nii.gz"

    if final_dwi.exists() and final_bvec.exists() and final_bval.exists() and mask.exists() and not force:
        return {"status": "skipped", "dwi": str(final_dwi), "mask": str(mask)}

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        write_dataset_description(config.data_root / "derivatives" / "dwi-preproc", "CHEDDAR DWI preprocessing", generated_by="cheddar")

    denoised: list[Path] = []
    for item in inputs:
        current = item.nifti
        denoise_out = work_dir / f"{item.acq}_desc-denoised_dwi.nii.gz"
        noise_out = out_dir / f"{participant}_{session}_acq-{item.acq}_desc-noise_dwi.nii.gz"
        runner.run([config.tools.get("dwidenoise", "dwidenoise"), current, denoise_out, "-noise", noise_out, "-force"])
        current = denoise_out
        degibbs_out = work_dir / f"{item.acq}_desc-degibbs_dwi.nii.gz"
        runner.run([config.tools.get("mrdegibbs", "mrdegibbs"), current, degibbs_out, "-force"])
        denoised.append(degibbs_out)

    merged = work_dir / f"{stem}_input_merged.nii.gz"
    merged_bvec = work_dir / f"{stem}_input_merged.bvec"
    merged_bval = work_dir / f"{stem}_input_merged.bval"
    runner.run([config.tools.get("fslmerge", "fslmerge"), "-t", merged, *denoised])
    if dry_run:
        print(f"DRY RUN: write merged gradients {merged_bvec}, {merged_bval}, {volume_table}")
    else:
        _write_merged_gradients(inputs, merged_bvec, merged_bval, volume_table)

    b0_first = work_dir / "forward_b0_first.nii.gz"
    runner.run([config.tools.get("fslroi", "fslroi"), merged, b0_first, "0", "1"])
    fmap_dir = paths.datatype_dir(participant, session, "fmap")
    reverse_candidates = sorted(fmap_dir.glob(f"{participant}_{session}_*_epi.nii*"))
    preproc_mif = work_dir / f"{stem}_desc-eddy_dwi.mif"
    eddy_args: list[str | Path] = [
        config.tools.get("dwifslpreproc", "dwifslpreproc"),
        merged,
        preproc_mif,
        "-fslgrad",
        merged_bvec,
        merged_bval,
        "-pe_dir",
        str(config.diffusion.get("phase_encoding_direction", "j")),
        "-readout_time",
        str(config.diffusion.get("total_readout_time", 0.0)),
        "-force",
    ]
    if reverse_candidates:
        reverse_first = work_dir / "reverse_b0_first.nii.gz"
        b0_pair = work_dir / "topup_b0_pair.nii.gz"
        runner.run([config.tools.get("fslroi", "fslroi"), reverse_candidates[0], reverse_first, "0", "1"])
        runner.run([config.tools.get("fslmerge", "fslmerge"), "-t", b0_pair, b0_first, reverse_first])
        eddy_args.extend(["-rpe_pair", "-se_epi", b0_pair])
    else:
        eddy_args.extend(["-rpe_none", "-align_seepi"])
    eddy_options = list(config.preprocessing.get("dwi", {}).get("eddy_options", []))
    if eddy_options:
        eddy_args.extend(["-eddy_options", " ".join(str(option) for option in eddy_options)])
    runner.run(eddy_args)

    biascorr_mif = work_dir / f"{stem}_desc-N4_dwi.mif"
    bias_mif = work_dir / f"{stem}_desc-N4BiasField_dwi.mif"
    runner.run([config.tools.get("dwibiascorrect", "dwibiascorrect"), "ants", preproc_mif, biascorr_mif, "-bias", bias_mif, "-force"])
    runner.run([config.tools.get("mrconvert", "mrconvert"), biascorr_mif, final_dwi, "-export_grad_fsl", final_bvec, final_bval, "-force"])
    runner.run([config.tools.get("mrconvert", "mrconvert"), bias_mif, bias_field, "-force"])
    runner.run([config.tools.get("dwi2mask", "dwi2mask"), biascorr_mif, mask, "-force"])
    runner.run([config.tools.get("dwiextract", "dwiextract"), "-bzero", biascorr_mif, work_dir / "b0_all.mif", "-force"])
    runner.run([config.tools.get("mrmath", "mrmath"), work_dir / "b0_all.mif", "mean", mean_b0, "-axis", "3", "-force"])
    runner.run([config.tools.get("fslmaths", "fslmaths"), final_dwi, "-mas", mask, brain_dwi])

    if not dry_run:
        write_json(
            final_dwi.with_suffix("").with_suffix(".json"),
            provenance(
                description="Denoised, degibbsed, susceptibility/motion/eddy corrected, and N4 bias-corrected merged CHEDDAR DWI.",
                sources=[item.nifti.relative_to(config.data_root).as_posix() for item in inputs],
                command_log=str(runner.log_file) if runner.log_file else None,
                extra={
                    "SkullStripped": False,
                    "PhaseEncodingDirection": config.diffusion.get("phase_encoding_direction", "j"),
                    "TotalReadoutTime": config.diffusion.get("total_readout_time"),
                    "VolumeTable": volume_table.name,
                },
            ),
        )

    return {"status": "done", "dwi": str(final_dwi), "mask": str(mask), "bvec": str(final_bvec), "bval": str(final_bval)}


def preprocess_dwi_all(config: ResolvedConfig, *, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for subject_dir in sorted(config.data_root.glob("sub-*")):
        if not subject_dir.is_dir():
            continue
        for session_dir in sorted(subject_dir.glob("ses-*")):
            if session_dir.is_dir():
                results[f"{subject_dir.name}/{session_dir.name}"] = preprocess_dwi_subject(
                    config,
                    subject_dir.name,
                    session_dir.name,
                    dry_run=dry_run,
                    force=force,
                )
    return results

