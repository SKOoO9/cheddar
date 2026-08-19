from __future__ import annotations

import contextlib
import io
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from cheddar.config import ConfigError, load_resolved_config
from cheddar.convert import _clean_subject_workdir, convert_all
from cheddar.dicom import SeriesInfo, SeriesRule
from cheddar.paths import SubjectRecord


REPO = Path(__file__).resolve().parents[1]


class ConvertWorkdirTests(unittest.TestCase):
    def test_clean_subject_workdir_removes_only_subject_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict("os.environ", {"CHEDDAR_ROOT": tmp}, clear=False):
                config = load_resolved_config(REPO / "config" / "study.example.yaml", "local")
            subject_tmp = config.work_root / "convert" / "sub-001"
            subject_tmp.mkdir(parents=True)
            stale = subject_tmp / "stale.json"
            stale.write_text("{}", encoding="utf-8")

            cleaned = _clean_subject_workdir(config, subject_tmp)

            self.assertTrue(cleaned)
            self.assertFalse(subject_tmp.exists())
            self.assertTrue((config.work_root / "convert").exists())

    def test_clean_subject_workdir_rejects_convert_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict("os.environ", {"CHEDDAR_ROOT": tmp}, clear=False):
                config = load_resolved_config(REPO / "config" / "study.example.yaml", "local")

            with self.assertRaises(ConfigError):
                _clean_subject_workdir(config, config.work_root / "convert")

    def test_clean_dry_run_does_not_read_stale_work_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict("os.environ", {"CHEDDAR_ROOT": tmp}, clear=False):
                config = load_resolved_config(REPO / "config" / "study.example.yaml", "local")
            subject_tmp = config.work_root / "convert" / "sub-001"
            subject_tmp.mkdir(parents=True)
            (subject_tmp / "stale_T1w.json").write_text("{}", encoding="utf-8")

            rule = SeriesRule(
                name="T1w",
                match=re.compile("stale", re.IGNORECASE),
                datatype="anat",
                suffix="T1w",
            )
            series = [
                SeriesInfo(
                    subject_folder="Source_001",
                    dicom_dir=Path(tmp) / "RawData" / "Source_001" / "DICOM",
                    dicom_file=Path(tmp) / "RawData" / "Source_001" / "DICOM" / "image.dcm",
                    rule=rule,
                    label="stale_T1w",
                )
            ]
            records = [SubjectRecord("sub-001", "ses-01", "Source_001")]

            with mock.patch("cheddar.convert.scan_raw", return_value=series):
                with mock.patch("cheddar.convert.assign_subjects", return_value=records):
                    with contextlib.redirect_stdout(io.StringIO()):
                        summary = convert_all(config, dry_run=True, clean_work=True)

            self.assertEqual(summary["subjects"]["sub-001"]["series"], {})
            self.assertTrue(summary["subjects"]["sub-001"]["work_cleaned"])


if __name__ == "__main__":
    unittest.main()
