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

`convert` cleans each subject's temporary `Work/convert/sub-*` folder before running `dcm2niix`, so reruns do not accumulate stale intermediate files. Add `--keep-work` only when you want to inspect previous scratch outputs while debugging conversion.

## Design Boundary

`cheddar` is a wrapper/orchestrator. It does not implement DTI, DKI, NODDI, IMPULSED, or future CHEDDAR model fitting. Later, MATI should consume:

- preprocessed DWI
- corrected `.bval/.bvec`
- brain mask
- volume/acquisition table
- protocol metadata

and write fitted maps into `Data/derivatives/fit-*`.

## Supported Systems

Linux and macOS are supported. Windows is intentionally unsupported.
