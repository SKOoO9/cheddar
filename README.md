# CHEDDAR

CHEDDAR is a project wrapper for diffusion MRI data synchronization, DICOM-to-`Data/` organization, preprocessing, brain masking, skull stripping, QC, and model-ready outputs.

This repository is code-only. Do **not** commit or upload PHI, raw DICOMs, NIfTI files, generated derivatives, logs, sync manifests, private subject maps, real server paths, credentials, or local configuration files.

## What Belongs Here

- Python wrapper code in `cheddar/`
- ms3t-style launcher scripts in `scripts/`
- sanitized example configuration in `config/*.example.yaml`
- tests and public documentation

## What Never Belongs Here

- `RawData/`
- `Data/`
- `Output/`
- `Work/`
- `.private/`
- `*.DCM`, `*.dcm`, `*.nii`, `*.nii.gz`, `*.bval`, `*.bvec`
- real participant mappings, logs, credentials, or real machine paths

## Daily Use

The wrapper is designed to feel like the existing DA-style project pipelines:

```bash
python scripts/DA00_sync_cheddar_data.py --profile smb-server --dry-run
python scripts/DA01_convert_cheddar_data.py --profile gpu-server
python scripts/DA02_preprocess_cheddar_data.py --profile gpu-server
python scripts/DA03_qc_cheddar_data.py --profile gpu-server
python scripts/DA_cheddar.py --profile gpu-server
```

The same logic is available through the module CLI:

```bash
python -m cheddar env doctor --config config/study.example.yaml --profile local
python -m cheddar sync raw --config config/study.example.yaml --profile local --dry-run
python -m cheddar convert --config config/study.example.yaml --profile local --dry-run
python -m cheddar preprocess --config config/study.example.yaml --profile local --dry-run
```

DA02 provides two DWI strategies:

```bash
# One dwifslpreproc TOPUP+eddy run for each original scanner acquisition.
python scripts/DA02_preprocess_cheddar_data.py --profile gpu-server \
  --dwi-only --dwi-strategy original-groups

# One shared TOPUP estimate, followed by eddy runs grouped by MATI pulse fields.
python scripts/DA02_preprocess_cheddar_data.py --profile gpu-server \
  --dwi-only --dwi-strategy shared-topup
```

`original-groups` is the default. Both strategies prepend the same selected forward b=0 reference to each temporary eddy input, remove it after correction, export and concatenate the rotated gradient tables in the same order as the corrected image groups, and then estimate one N4 field on the combined data. Strategy-specific `proc-origgroups` and `proc-sharedtopup` filenames allow both results to coexist.

`shared-topup` requires `diffusion.mati_pulse_file`. Set it in a private `config/study.yaml` or through `CHEDDAR_MATI_PULSE`; relative paths are resolved beside the study config. The pulse must contain one acquisition entry per input DWI volume in the order given by `diffusion.acquisitions`. When it includes `b`, DA02 checks those MATI values against the converted b-values before grouping, using configurable absolute and relative tolerances. `config/mati_pulse.example.json` is an illustrative schema, not the CHEDDAR scanner protocol.

The grouping fields default to MATI's `shape`, `n`, `delta`, `Delta`, `trise`, `TE`, `TR`, and `FA`. They can be changed with `preprocessing.dwi.encoding_group_fields`. The shared strategy runs FSL TOPUP once and passes its result to separate eddy calls for those groups.

`convert` cleans each subject's temporary `Work/convert/sub-*` folder before running `dcm2niix`, so reruns do not accumulate stale intermediate files. Add `--keep-work` only when you want to inspect previous scratch outputs while debugging conversion.

During DWI preprocessing, values at or below `preprocessing.dwi.b0_threshold` are treated as nominal b=0 volumes. CHEDDAR preserves the converted source gradients, writes exact zero b-values and b-vectors only to working and derivative files, and records both source and normalized b-values in the derivative volume table. The example threshold is `50 s/mm^2`; choose a study-specific value below the lowest intentionally acquired diffusion-weighted shell.

## Design Boundary

`cheddar` is a wrapper/orchestrator. It does not implement DTI, DKI, NODDI, IMPULSED, or future CHEDDAR model fitting. Later, MATI should consume:

- preprocessed DWI
- the derivative MATI `DiffusionPulseSequence` JSON through `mati fit --pulse`
- corrected `.bval/.bvec` for FSL/MRtrix interoperability and QC
- brain mask
- source-volume provenance table

and write fitted maps into `Data/derivatives/fit-*`.

When `diffusion.mati_pulse_file` is configured, DA02 writes a derivative `_pulse.json` with the final volume order, b-values converted from `s/mm^2` to MATI's `ms/um^2`, and `gdir` replaced by eddy's rotated directions. Timing and waveform fields remain those supplied by the MATI pulse. Use the pulse JSON alone for advanced MATI diffusion fitting:

```bash
mati fit --model MODEL --data SUB_SES_DWI.nii.gz \
  --pulse SUB_SES_PULSE.json --mask SUB_SES_MASK.nii.gz --out OUTPUT/fit
```

## Supported Systems

Linux and macOS are supported. Windows is intentionally unsupported.
