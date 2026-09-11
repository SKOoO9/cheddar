from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from cheddar.dicom import scan_raw, summarize_scan


REPO = Path(__file__).resolve().parents[1]


class DicomScanTests(unittest.TestCase):
    def test_scan_mock_philips_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dicom = Path(tmp) / "RawData" / "PRELIMINARY" / "SYNTHETIC_001" / "DICOM"
            dicom.mkdir(parents=True)
            for name in (
                "SYNTHETIC_001.02.01.10-00-00.WIP_MPRAGE.01.DCM",
                "SYNTHETIC_001.03.01.10-00-00.WIP_T2FLAIR.01.DCM",
                "SYNTHETIC_001.04.01.10-00-00.WIP_b0neg.01.DCM",
                "SYNTHETIC_001.05.01.10-00-00.WIP_tcos1.01.DCM",
                "SYNTHETIC_001.07.01.10-00-00.WIP_tpgse_1.01.DCM",
            ):
                (dicom / name).write_bytes(b"mock")
            series = scan_raw(Path(tmp) / "RawData", REPO / "config" / "bidsmap.example.yaml")
            summary = summarize_scan(series)
            self.assertEqual(summary["n_subjects"], 1)
            self.assertIn("tcos1", summary["subjects"]["SYNTHETIC_001"])
            self.assertIn("tpgse1", summary["subjects"]["SYNTHETIC_001"])


if __name__ == "__main__":
    unittest.main()
