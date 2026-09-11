from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import shutil
from typing import Any, Sequence

import numpy as np

from .config import ConfigError, ResolvedConfig
from .derivatives import provenance, write_dataset_description, write_json
from .mati import (
    DEFAULT_ENCODING_GROUP_FIELDS,
    MatiPulseError,
    corrected_mati_pulse,
    encoding_groups,
    load_mati_pulse,
    validate_mati_bvalues,
    write_mati_pulse,
)
from .paths import DataPaths
from .runner import CommandRunner


DWI_STRATEGIES = ("original-groups", "shared-topup")
STRATEGY_ALIASES = {
    "original": "original-groups",
    "original-groups": "original-groups",
    "original_groups": "original-groups",
    "shared": "shared-topup",
    "shared-topup": "shared-topup",
    "shared_topup": "shared-topup",
}


@dataclass(frozen=True)
class DwiInput:
    acq: str
    nifti: Path
    bvec: Path
    bval: Path


@dataclass(frozen=True)
class MergedGradients:
    bvals: np.ndarray
    bvecs: np.ndarray
    volume_rows: list[str]
    first_bzero_index: int
    bzero_count: int


@dataclass(frozen=True)
class DwiGroup:
    label: str
    indices: tuple[int, ...]
    image: Path
    encoding: dict[str, Any] | None = None


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


def _prepare_merged_gradients(inputs: list[DwiInput], bzero_threshold: float) -> MergedGradients:
    if not np.isfinite(bzero_threshold) or bzero_threshold < 0:
        raise ValueError(f"b0 threshold must be a finite non-negative number, got {bzero_threshold}")

    bvals: list[np.ndarray] = []
    bvecs: list[np.ndarray] = []
    rows = ["volume\tacquisition\tsource_volume\tsource_bval\tbval\tis_bzero"]
    offset = 0
    for item in inputs:
        source_vals = np.loadtxt(item.bval, dtype=float).reshape(-1)
        source_vec = _load_bvec(item.bvec)
        if source_vec.shape[1] != source_vals.size:
            raise ValueError(f"bvec/bval volume mismatch for {item.acq}: {source_vec.shape[1]} vs {source_vals.size}")
        if not np.all(np.isfinite(source_vals)):
            raise ValueError(f"bval file contains non-finite values: {item.bval}")
        if np.any(source_vals < 0):
            raise ValueError(f"bval file contains negative values: {item.bval}")

        is_bzero = source_vals <= bzero_threshold
        vals = source_vals.copy()
        vec = source_vec.copy()
        vals[is_bzero] = 0.0
        vec[:, is_bzero] = 0.0
        bvals.append(vals)
        bvecs.append(vec)
        for idx, (source_bval, bval, is_zero) in enumerate(zip(source_vals, vals, is_bzero, strict=True)):
            rows.append(
                f"{offset + idx}\t{item.acq}\t{idx}\t{float(source_bval):.8g}\t"
                f"{float(bval):.8g}\t{str(bool(is_zero)).lower()}"
            )
        offset += source_vals.size

    merged_bvals = np.concatenate(bvals)
    merged_bvecs = np.concatenate(bvecs, axis=1)
    bzero_indices = np.flatnonzero(merged_bvals == 0.0)
    if bzero_indices.size == 0:
        raise ValueError(f"No b=0 volumes found at or below threshold {bzero_threshold:g}")
    return MergedGradients(
        bvals=merged_bvals,
        bvecs=merged_bvecs,
        volume_rows=rows,
        first_bzero_index=int(bzero_indices[0]),
        bzero_count=int(bzero_indices.size),
    )


def _subset_gradients(gradients: MergedGradients, indices: Sequence[int]) -> MergedGradients:
    selected = np.asarray(indices, dtype=int)
    values = gradients.bvals[selected]
    vectors = gradients.bvecs[:, selected]
    bzero_indices = np.flatnonzero(values == 0.0)
    rows = [gradients.volume_rows[0]]
    for new_index, source_index in enumerate(selected):
        fields = gradients.volume_rows[int(source_index) + 1].split("\t")
        fields[0] = str(new_index)
        rows.append("\t".join(fields))
    return MergedGradients(
        bvals=values,
        bvecs=vectors,
        volume_rows=rows,
        first_bzero_index=int(bzero_indices[0]) if bzero_indices.size else -1,
        bzero_count=int(bzero_indices.size),
    )


def _prepend_reference_gradient(gradients: MergedGradients) -> MergedGradients:
    return MergedGradients(
        bvals=np.concatenate([np.array([0.0]), gradients.bvals]),
        bvecs=np.concatenate([np.zeros((3, 1), dtype=float), gradients.bvecs], axis=1),
        volume_rows=gradients.volume_rows,
        first_bzero_index=0,
        bzero_count=gradients.bzero_count + 1,
    )


def _write_merged_gradients(
    gradients: MergedGradients,
    bvec_out: Path,
    bval_out: Path,
    volume_table: Path | None = None,
) -> None:
    bval_out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(bval_out, gradients.bvals.reshape(1, -1), fmt="%.10g")
    np.savetxt(bvec_out, gradients.bvecs, fmt="%.10g")
    if volume_table is not None:
        volume_table.write_text("\n".join(gradients.volume_rows) + "\n", encoding="utf-8")


def _resolve_strategy(config: ResolvedConfig, override: str | None) -> str:
    dwi_settings = dict(config.preprocessing.get("dwi", {}))
    requested = str(override or dwi_settings.get("strategy", "original-groups")).strip().lower()
    strategy = STRATEGY_ALIASES.get(requested)
    if strategy is None:
        raise ConfigError(f"Unknown DWI preprocessing strategy {requested!r}; choose one of: {', '.join(DWI_STRATEGIES)}")
    return strategy


def _mati_pulse_path(config: ResolvedConfig, participant: str, session: str) -> Path | None:
    raw = str(config.diffusion.get("mati_pulse_file", "")).strip()
    if not raw:
        return None
    rendered = raw.format(subject=participant, participant=participant, session=session)
    path = Path(rendered).expanduser()
    if not path.is_absolute():
        path = config.config_path.parent / path
    return path.resolve()


def _phase_encoding_row(direction: str, readout_time: float) -> str:
    normalized = str(direction).strip().lower()
    vectors = {
        "i": (1, 0, 0),
        "i-": (-1, 0, 0),
        "j": (0, 1, 0),
        "j-": (0, -1, 0),
        "k": (0, 0, 1),
        "k-": (0, 0, -1),
    }
    if normalized not in vectors:
        raise ConfigError(f"Unsupported BIDS phase-encoding direction {direction!r}.")
    x, y, z = vectors[normalized]
    return f"{x} {y} {z} {readout_time:.10g}"


def _select_eddy_command(config: ResolvedConfig, dwi_settings: dict[str, Any]) -> str:
    if bool(dwi_settings.get("prefer_eddy_cuda", True)):
        candidate = str(config.tools.get("eddy_cuda", "eddy_cuda"))
        if ("/" in candidate and Path(candidate).exists()) or shutil.which(candidate):
            return candidate
    return str(config.tools.get("eddy", "eddy"))


def _direct_eddy_options(dwi_settings: dict[str, Any]) -> list[str]:
    options: list[str] = []
    for item in dwi_settings.get("eddy_options", []):
        options.extend(shlex.split(str(item)))
    return options


def _original_groups(inputs: Sequence[DwiInput], images: Sequence[Path]) -> list[DwiGroup]:
    groups: list[DwiGroup] = []
    offset = 0
    for item, image in zip(inputs, images, strict=True):
        count = int(np.loadtxt(item.bval, dtype=float).reshape(-1).size)
        indices = tuple(range(offset, offset + count))
        groups.append(DwiGroup(item.acq, indices, image))
        offset += count
    return groups


def _shared_topup_groups(
    payload: dict[str, Any],
    fields: Sequence[str],
    merged: Path,
    work_dir: Path,
    runner: CommandRunner,
    mrconvert: str,
) -> list[DwiGroup]:
    groups: list[DwiGroup] = []
    for item in encoding_groups(payload, fields=fields):
        image = work_dir / f"{item.label}_input.nii.gz"
        selection = ",".join(str(index) for index in item.indices)
        runner.run([mrconvert, merged, image, "-coord", "3", selection, "-force"])
        groups.append(DwiGroup(item.label, item.indices, image, item.encoding))
    return groups


def _prepare_common_reference(
    config: ResolvedConfig,
    participant: str,
    session: str,
    gradients: MergedGradients,
    merged: Path,
    work_dir: Path,
    runner: CommandRunner,
) -> tuple[Path, Path | None]:
    fslroi = config.tools.get("fslroi", "fslroi")
    forward_b0 = work_dir / "forward_b0_reference.nii.gz"
    runner.run([fslroi, merged, forward_b0, str(gradients.first_bzero_index), "1"])

    fmap_dir = DataPaths(config).datatype_dir(participant, session, "fmap")
    reverse_candidates = sorted(fmap_dir.glob(f"{participant}_{session}_*_epi.nii*"))
    if not reverse_candidates:
        return forward_b0, None

    reverse_b0 = work_dir / "reverse_b0_reference.nii.gz"
    pair = work_dir / "topup_b0_pair.nii.gz"
    runner.run([fslroi, reverse_candidates[0], reverse_b0, "0", "1"])
    runner.run([config.tools.get("fslmerge", "fslmerge"), "-t", pair, forward_b0, reverse_b0])
    return forward_b0, pair


def _prepare_group_input(
    config: ResolvedConfig,
    group: DwiGroup,
    gradients: MergedGradients,
    forward_b0: Path,
    work_dir: Path,
    runner: CommandRunner,
) -> tuple[Path, Path, Path, MergedGradients]:
    subset = _subset_gradients(gradients, group.indices)
    with_reference = _prepend_reference_gradient(subset)
    image = work_dir / f"{group.label}_with-reference.nii.gz"
    bvec = work_dir / f"{group.label}_with-reference.bvec"
    bval = work_dir / f"{group.label}_with-reference.bval"
    runner.run([config.tools.get("fslmerge", "fslmerge"), "-t", image, forward_b0, group.image])
    if not runner.dry_run:
        _write_merged_gradients(with_reference, bvec, bval)
    else:
        print(f"DRY RUN: write reference-prepended gradients {bvec}, {bval}")
    return image, bvec, bval, subset


def _run_original_group(
    config: ResolvedConfig,
    dwi_settings: dict[str, Any],
    group: DwiGroup,
    gradients: MergedGradients,
    forward_b0: Path,
    topup_pair: Path | None,
    work_dir: Path,
    runner: CommandRunner,
) -> Path:
    image, bvec, bval, _ = _prepare_group_input(config, group, gradients, forward_b0, work_dir, runner)
    with_reference = work_dir / f"{group.label}_desc-eddy_with-reference.mif"
    corrected = work_dir / f"{group.label}_desc-eddy_dwi.mif"
    command: list[str | Path] = [
        config.tools.get("dwifslpreproc", "dwifslpreproc"),
        image,
        with_reference,
        "-fslgrad",
        bvec,
        bval,
        "-pe_dir",
        str(config.diffusion.get("phase_encoding_direction", "j")),
        "-readout_time",
        str(config.diffusion.get("total_readout_time", 0.0)),
        "-force",
    ]
    if topup_pair is None:
        command.append("-rpe_none")
    else:
        command.extend(["-rpe_pair", "-se_epi", topup_pair, "-align_seepi"])
    options = " ".join(str(option) for option in dwi_settings.get("eddy_options", [])).strip()
    if options:
        command.extend(["-eddy_options", f"{options} "])
    runner.run(command)
    runner.run([config.tools.get("mrconvert", "mrconvert"), with_reference, corrected, "-coord", "3", "1:end", "-force"])
    return corrected


def _run_shared_topup(
    config: ResolvedConfig,
    dwi_settings: dict[str, Any],
    topup_pair: Path,
    work_dir: Path,
    runner: CommandRunner,
) -> tuple[Path, Path, Path]:
    readout_time = float(config.diffusion.get("total_readout_time", 0.0))
    if readout_time <= 0:
        raise ConfigError("shared-topup requires a positive diffusion.total_readout_time.")
    acqparams = work_dir / "topup_acqparams.txt"
    forward = str(config.diffusion.get("phase_encoding_direction", "j"))
    reverse = str(config.diffusion.get("reverse_phase_encoding_direction", "j-"))
    forward_row = _phase_encoding_row(forward, readout_time)
    reverse_row = _phase_encoding_row(reverse, readout_time)
    if not runner.dry_run:
        acqparams.write_text(forward_row + "\n" + reverse_row + "\n", encoding="utf-8")
    else:
        print(f"DRY RUN: write TOPUP acquisition parameters {acqparams}")

    topup_base = work_dir / "shared_topup"
    topup_corrected = work_dir / "shared_topup_corrected.nii.gz"
    topup_field = work_dir / "shared_topup_field.nii.gz"
    runner.run(
        [
            config.tools.get("topup", "topup"),
            f"--imain={topup_pair}",
            f"--datain={acqparams}",
            f"--config={dwi_settings.get('topup_config', 'b02b0.cnf')}",
            f"--out={topup_base}",
            f"--iout={topup_corrected}",
            f"--fout={topup_field}",
        ]
    )
    topup_mean = work_dir / "shared_topup_mean.nii.gz"
    brain = work_dir / "shared_topup_mean_brain.nii.gz"
    mask = work_dir / "shared_topup_mean_brain_mask.nii.gz"
    runner.run([config.tools.get("fslmaths", "fslmaths"), topup_corrected, "-Tmean", topup_mean])
    runner.run(
        [
            config.tools.get("bet", "bet"),
            topup_mean,
            brain,
            "-m",
            "-f",
            str(dwi_settings.get("topup_mask_fraction", 0.2)),
        ]
    )
    return topup_base, acqparams, mask


def _run_shared_eddy_group(
    config: ResolvedConfig,
    dwi_settings: dict[str, Any],
    group: DwiGroup,
    gradients: MergedGradients,
    forward_b0: Path,
    topup_base: Path,
    acqparams: Path,
    mask: Path,
    work_dir: Path,
    runner: CommandRunner,
) -> Path:
    image, bvec, bval, subset = _prepare_group_input(config, group, gradients, forward_b0, work_dir, runner)
    index_file = work_dir / f"{group.label}_eddy_index.txt"
    if not runner.dry_run:
        index_file.write_text(" ".join(["1"] * (subset.bvals.size + 1)) + "\n", encoding="utf-8")
    else:
        print(f"DRY RUN: write eddy index {index_file}")

    eddy_base = work_dir / f"{group.label}_eddy"
    command = [
        _select_eddy_command(config, dwi_settings),
        f"--imain={image}",
        f"--mask={mask}",
        f"--acqp={acqparams}",
        f"--index={index_file}",
        f"--bvecs={bvec}",
        f"--bvals={bval}",
        f"--topup={topup_base}",
        f"--out={eddy_base}",
        *_direct_eddy_options(dwi_settings),
    ]
    runner.run(command)

    eddy_image = Path(f"{eddy_base}.nii.gz")
    if not runner.dry_run and not eddy_image.exists():
        uncompressed = Path(f"{eddy_base}.nii")
        if uncompressed.exists():
            eddy_image = uncompressed
        else:
            raise ConfigError(f"eddy completed without writing {eddy_base}.nii.gz or {eddy_base}.nii")
    corrected = work_dir / f"{group.label}_desc-eddy_dwi.mif"
    runner.run(
        [
            config.tools.get("mrconvert", "mrconvert"),
            eddy_image,
            corrected,
            "-fslgrad",
            Path(f"{eddy_base}.eddy_rotated_bvecs"),
            bval,
            "-coord",
            "3",
            "1:end",
            "-force",
        ]
    )
    return corrected


def _combine_corrected_gradients(
    config: ResolvedConfig,
    groups: Sequence[DwiGroup],
    corrected_images: Sequence[Path],
    work_dir: Path,
    runner: CommandRunner,
) -> tuple[Path, Path]:
    if len(groups) != len(corrected_images):
        raise ValueError("Corrected DWI groups and images do not have the same length.")

    group_gradient_files: list[tuple[DwiGroup, Path, Path]] = []
    for group, image in zip(groups, corrected_images, strict=True):
        bvec = work_dir / f"{group.label}_desc-eddy_dwi.bvec"
        bval = work_dir / f"{group.label}_desc-eddy_dwi.bval"
        runner.run(
            [
                config.tools.get("mrinfo", "mrinfo"),
                image,
                "-export_grad_fsl",
                bvec,
                bval,
            ]
        )
        group_gradient_files.append((group, bvec, bval))

    combined_bvec = work_dir / "corrected_groups.bvec"
    combined_bval = work_dir / "corrected_groups.bval"
    if runner.dry_run:
        print(f"DRY RUN: concatenate corrected group gradients into {combined_bvec}, {combined_bval}")
        return combined_bvec, combined_bval

    bvecs: list[np.ndarray] = []
    bvals: list[np.ndarray] = []
    for group, bvec, bval in group_gradient_files:
        vectors = _load_bvec(bvec)
        values = np.loadtxt(bval, dtype=float).reshape(-1)
        expected = len(group.indices)
        if vectors.shape[1] != expected or values.size != expected:
            raise ConfigError(
                f"Corrected gradients for {group.label} contain {vectors.shape[1]} b-vectors and "
                f"{values.size} b-values; expected {expected} of each."
            )
        bvecs.append(vectors)
        bvals.append(values)

    np.savetxt(combined_bvec, np.concatenate(bvecs, axis=1), fmt="%.10g")
    np.savetxt(combined_bval, np.concatenate(bvals).reshape(1, -1), fmt="%.10g")
    return combined_bvec, combined_bval


def preprocess_dwi_subject(
    config: ResolvedConfig,
    participant: str,
    session: str,
    *,
    dry_run: bool = False,
    force: bool = False,
    strategy: str | None = None,
) -> dict[str, Any]:
    inputs = _find_dwi_inputs(config, participant, session)
    if not inputs:
        return {"status": "missing_dwi"}

    dwi_settings = dict(config.preprocessing.get("dwi", {}))
    selected_strategy = _resolve_strategy(config, strategy)
    proc_label = "origgroups" if selected_strategy == "original-groups" else "sharedtopup"
    bzero_threshold = float(dwi_settings.get("b0_threshold", 50.0))
    gradients = _prepare_merged_gradients(inputs, bzero_threshold)

    pulse_path = _mati_pulse_path(config, participant, session)
    pulse_payload: dict[str, Any] | None = None
    if pulse_path is not None:
        try:
            pulse_payload = load_mati_pulse(pulse_path, expected_nacq=int(gradients.bvals.size))
            validate_mati_bvalues(
                pulse_payload,
                gradients.bvals,
                absolute_tolerance=float(dwi_settings.get("mati_bval_absolute_tolerance", 50.0)),
                relative_tolerance=float(dwi_settings.get("mati_bval_relative_tolerance", 0.05)),
            )
        except MatiPulseError as exc:
            raise ConfigError(str(exc)) from exc
    if selected_strategy == "shared-topup" and pulse_payload is None:
        raise ConfigError(
            "shared-topup requires diffusion.mati_pulse_file so eddy groups can be derived from MATI encoding fields."
        )

    paths = DataPaths(config)
    out_dir = paths.derivative_dir("dwi-preproc", participant, session, "dwi")
    work_dir = config.work_root / "dwi-preproc" / participant / session / proc_label
    runner = CommandRunner(
        config.output_root / "logs" / f"dwi_preproc_{participant}_{session}_{proc_label}.jsonl",
        dry_run=dry_run,
    )

    base_stem = f"{participant}_{session}_acq-{config.diffusion.get('merged_acq_label', 'all')}_proc-{proc_label}"
    final_dwi = out_dir / f"{base_stem}_desc-preproc_dwi.nii.gz"
    final_bvec = out_dir / f"{base_stem}_desc-preproc_dwi.bvec"
    final_bval = out_dir / f"{base_stem}_desc-preproc_dwi.bval"
    final_pulse = out_dir / f"{base_stem}_desc-preproc_pulse.json"
    volume_table = out_dir / f"{base_stem}_desc-volumes.tsv"
    mask = out_dir / f"{base_stem}_desc-brain_mask.nii.gz"
    brain_dwi = out_dir / f"{base_stem}_desc-skullstripped_dwi.nii.gz"
    mean_b0 = out_dir / f"{base_stem}_desc-meanb0_dwi.nii.gz"
    bias_field = out_dir / f"{base_stem}_desc-N4BiasField_dwi.nii.gz"

    expected = [final_dwi, final_bvec, final_bval, mask]
    if pulse_payload is not None:
        expected.append(final_pulse)
    if all(path.exists() for path in expected) and not force:
        result = {
            "status": "skipped",
            "strategy": selected_strategy,
            "dwi": str(final_dwi),
            "mask": str(mask),
        }
        if pulse_payload is not None:
            result["pulse"] = str(final_pulse)
        return result

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        write_dataset_description(
            config.data_root / "derivatives" / "dwi-preproc",
            "CHEDDAR DWI preprocessing",
            generated_by="cheddar",
        )

    denoised: list[Path] = []
    for item in inputs:
        denoise_out = work_dir / f"{item.acq}_desc-denoised_dwi.nii.gz"
        noise_out = out_dir / f"{participant}_{session}_acq-{item.acq}_desc-noise_dwi.nii.gz"
        runner.run([config.tools.get("dwidenoise", "dwidenoise"), item.nifti, denoise_out, "-noise", noise_out, "-force"])
        degibbs_out = work_dir / f"{item.acq}_desc-degibbs_dwi.nii.gz"
        runner.run([config.tools.get("mrdegibbs", "mrdegibbs"), denoise_out, degibbs_out, "-force"])
        denoised.append(degibbs_out)

    merged = work_dir / f"{base_stem}_input_merged.nii.gz"
    merged_bvec = work_dir / f"{base_stem}_input_merged.bvec"
    merged_bval = work_dir / f"{base_stem}_input_merged.bval"
    runner.run([config.tools.get("fslmerge", "fslmerge"), "-t", merged, *denoised])
    if not dry_run:
        _write_merged_gradients(gradients, merged_bvec, merged_bval)
    else:
        print(
            f"DRY RUN: normalize {gradients.bzero_count} b=0 volumes at or below "
            f"{bzero_threshold:g} s/mm^2 and write {merged_bvec}, {merged_bval}"
        )

    forward_b0, topup_pair = _prepare_common_reference(
        config, participant, session, gradients, merged, work_dir, runner
    )

    if selected_strategy == "original-groups":
        groups = _original_groups(inputs, denoised)
    else:
        group_fields = tuple(dwi_settings.get("encoding_group_fields", DEFAULT_ENCODING_GROUP_FIELDS))
        try:
            groups = _shared_topup_groups(
                pulse_payload or {},
                group_fields,
                merged,
                work_dir,
                runner,
                str(config.tools.get("mrconvert", "mrconvert")),
            )
        except MatiPulseError as exc:
            raise ConfigError(str(exc)) from exc

    corrected_groups: list[Path] = []
    if selected_strategy == "original-groups":
        for group in groups:
            corrected_groups.append(
                _run_original_group(
                    config, dwi_settings, group, gradients, forward_b0, topup_pair, work_dir, runner
                )
            )
    else:
        if topup_pair is None:
            raise ConfigError("shared-topup requires a reverse phase-encoded EPI image in the session fmap folder.")
        topup_base, acqparams, topup_mask = _run_shared_topup(
            config, dwi_settings, topup_pair, work_dir, runner
        )
        for group in groups:
            corrected_groups.append(
                _run_shared_eddy_group(
                    config,
                    dwi_settings,
                    group,
                    gradients,
                    forward_b0,
                    topup_base,
                    acqparams,
                    topup_mask,
                    work_dir,
                    runner,
                )
            )

    output_order = tuple(index for group in groups for index in group.indices)
    ordered_gradients = _subset_gradients(gradients, output_order)
    if not dry_run:
        volume_table.write_text("\n".join(ordered_gradients.volume_rows) + "\n", encoding="utf-8")
    else:
        print(f"DRY RUN: write final source-volume provenance {volume_table}")

    corrected_bvec, corrected_bval = _combine_corrected_gradients(
        config, groups, corrected_groups, work_dir, runner
    )
    concatenated_mif = work_dir / f"{base_stem}_desc-eddy_image.mif"
    preproc_mif = work_dir / f"{base_stem}_desc-eddy_dwi.mif"
    runner.run(
        [config.tools.get("mrcat", "mrcat"), *corrected_groups, concatenated_mif, "-axis", "3", "-force"]
    )
    runner.run(
        [
            config.tools.get("mrconvert", "mrconvert"),
            concatenated_mif,
            preproc_mif,
            "-fslgrad",
            corrected_bvec,
            corrected_bval,
            "-force",
        ]
    )
    biascorr_mif = work_dir / f"{base_stem}_desc-N4_dwi.mif"
    bias_mif = work_dir / f"{base_stem}_desc-N4BiasField_dwi.mif"
    runner.run(
        [
            config.tools.get("dwibiascorrect", "dwibiascorrect"),
            "ants",
            preproc_mif,
            biascorr_mif,
            "-bias",
            bias_mif,
            "-force",
        ]
    )
    runner.run(
        [
            config.tools.get("mrconvert", "mrconvert"),
            biascorr_mif,
            final_dwi,
            "-export_grad_fsl",
            final_bvec,
            final_bval,
            "-force",
        ]
    )
    runner.run([config.tools.get("mrconvert", "mrconvert"), bias_mif, bias_field, "-force"])
    runner.run([config.tools.get("dwi2mask", "dwi2mask"), biascorr_mif, mask, "-force"])
    runner.run([config.tools.get("dwiextract", "dwiextract"), "-bzero", biascorr_mif, work_dir / "b0_all.mif", "-force"])
    runner.run([config.tools.get("mrmath", "mrmath"), work_dir / "b0_all.mif", "mean", mean_b0, "-axis", "3", "-force"])
    runner.run([config.tools.get("fslmaths", "fslmaths"), final_dwi, "-mas", mask, brain_dwi])

    if not dry_run and pulse_payload is not None:
        try:
            corrected_pulse = corrected_mati_pulse(
                pulse_payload,
                output_order,
                bvecs=_load_bvec(final_bvec),
                bvals_smm2=np.loadtxt(final_bval, dtype=float),
            )
            write_mati_pulse(final_pulse, corrected_pulse)
        except MatiPulseError as exc:
            raise ConfigError(str(exc)) from exc

    if not dry_run:
        group_payload = [
            {
                "Label": group.label,
                "Volumes": len(group.indices),
                **({"MATIEncoding": group.encoding} if group.encoding is not None else {}),
            }
            for group in groups
        ]
        write_json(
            final_dwi.with_suffix("").with_suffix(".json"),
            provenance(
                description="Denoised, degibbsed, susceptibility/motion/eddy corrected, and N4 bias-corrected DWI groups recombined for model fitting.",
                sources=[item.nifti.relative_to(config.data_root).as_posix() for item in inputs],
                command_log=str(runner.log_file) if runner.log_file else None,
                extra={
                    "SkullStripped": False,
                    "PreprocessingStrategy": selected_strategy,
                    "ProcessingGroups": group_payload,
                    "PhaseEncodingDirection": config.diffusion.get("phase_encoding_direction", "j"),
                    "TotalReadoutTime": config.diffusion.get("total_readout_time"),
                    "VolumeTable": volume_table.name,
                    "MATIPulse": final_pulse.name if pulse_payload is not None else None,
                    "B0Threshold": bzero_threshold,
                    "B0VolumesNormalized": gradients.bzero_count,
                    "SourceGradientFilesUnchanged": True,
                    "GradientDirectionsRotatedByEddy": True,
                },
            ),
        )

    result = {
        "status": "done",
        "strategy": selected_strategy,
        "groups": [group.label for group in groups],
        "dwi": str(final_dwi),
        "mask": str(mask),
        "bvec": str(final_bvec),
        "bval": str(final_bval),
    }
    if pulse_payload is not None:
        result["pulse"] = str(final_pulse)
    return result


def preprocess_dwi_all(
    config: ResolvedConfig,
    *,
    dry_run: bool = False,
    force: bool = False,
    strategy: str | None = None,
) -> dict[str, Any]:
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
                    strategy=strategy,
                )
    return results
