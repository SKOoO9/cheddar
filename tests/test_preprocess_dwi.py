from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from cheddar.config import load_resolved_config
from cheddar.preprocess_dwi import (
    DwiInput,
    _prepare_merged_gradients,
    _write_merged_gradients,
    preprocess_dwi_subject,
)
from cheddar.runner import CommandRunner


REPO = Path(__file__).resolve().parents[1]


class DwiGradientTests(unittest.TestCase):
    def test_nominal_bzero_values_and_vectors_are_normalized_in_derivatives_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_bval = root / "source.bval"
            source_bvec = root / "source.bvec"
            source_bval.write_text("100 0.1 250\n", encoding="utf-8")
            source_bvec.write_text("1 0.4 0\n0 0.5 1\n0 0.6 0\n", encoding="utf-8")
            source_bval_before = source_bval.read_text(encoding="utf-8")
            source_bvec_before = source_bvec.read_text(encoding="utf-8")
            inputs = [DwiInput("test", root / "source.nii.gz", source_bvec, source_bval)]

            gradients = _prepare_merged_gradients(inputs, 50.0)

            self.assertEqual(gradients.first_bzero_index, 1)
            self.assertEqual(gradients.bzero_count, 1)
            np.testing.assert_allclose(gradients.bvals, [100.0, 0.0, 250.0])
            np.testing.assert_allclose(gradients.bvecs[:, 1], [0.0, 0.0, 0.0])

            output_bval = root / "derived" / "merged.bval"
            output_bvec = root / "derived" / "merged.bvec"
            volume_table = root / "derived" / "volumes.tsv"
            _write_merged_gradients(gradients, output_bvec, output_bval, volume_table)

            self.assertEqual(source_bval.read_text(encoding="utf-8"), source_bval_before)
            self.assertEqual(source_bvec.read_text(encoding="utf-8"), source_bvec_before)
            self.assertIn("source_bval\tbval\tis_bzero", volume_table.read_text(encoding="utf-8"))
            self.assertIn("0.1\t0\ttrue", volume_table.read_text(encoding="utf-8"))

    def test_no_bzero_at_threshold_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bval = root / "source.bval"
            bvec = root / "source.bvec"
            bval.write_text("100 250\n", encoding="utf-8")
            bvec.write_text("1 0\n0 1\n0 0\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "No b=0 volumes"):
                _prepare_merged_gradients([DwiInput("test", root / "source.nii.gz", bvec, bval)], 50.0)


class DwiCommandTests(unittest.TestCase):
    def _config_with_dwi(self, root: Path, *, reverse: bool, bvals: str = "0.1 100 250\n"):
        with mock.patch.dict("os.environ", {"CHEDDAR_ROOT": str(root)}, clear=False):
            config = load_resolved_config(REPO / "config" / "study.example.yaml", "local")

        dwi_dir = config.data_root / "sub-001" / "ses-01" / "dwi"
        dwi_dir.mkdir(parents=True)
        stem = dwi_dir / "sub-001_ses-01_acq-tcos1_dir-j_dwi"
        Path(f"{stem}.nii.gz").touch()
        Path(f"{stem}.bval").write_text(bvals, encoding="utf-8")
        Path(f"{stem}.bvec").write_text("1 0.4 0\n0 0.5 1\n0 0.6 0\n", encoding="utf-8")
        if reverse:
            fmap_dir = config.data_root / "sub-001" / "ses-01" / "fmap"
            fmap_dir.mkdir(parents=True)
            (fmap_dir / "sub-001_ses-01_dir-jminus_epi.nii.gz").touch()
        return config

    def _run_and_get_commands(self, config) -> list[list[str]]:
        with mock.patch.object(CommandRunner, "run", autospec=True) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                preprocess_dwi_subject(config, "sub-001", "ses-01", dry_run=True)
        return [[str(arg) for arg in call.args[1]] for call in run.call_args_list]

    def test_rpe_pair_uses_selected_bzero_alignment_and_valid_eddy_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            commands = self._run_and_get_commands(self._config_with_dwi(Path(tmp), reverse=True))

        forward_bzero = next(command for command in commands if command[0] == "fslroi" and "forward_b0" in command[2])
        self.assertEqual(forward_bzero[-2:], ["0", "1"])
        eddy = next(command for command in commands if command[0] == "dwifslpreproc")
        self.assertIn("-rpe_pair", eddy)
        self.assertIn("-align_seepi", eddy)
        self.assertEqual(eddy[eddy.index("-eddy_options") + 1], "--repol ")

    def test_later_bzero_is_selected_without_invalid_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config_with_dwi(Path(tmp), reverse=True, bvals="100 0.1 250\n")
            commands = self._run_and_get_commands(config)

        forward_bzero = next(command for command in commands if command[0] == "fslroi" and "forward_b0" in command[2])
        self.assertEqual(forward_bzero[-2:], ["1", "1"])
        eddy = next(command for command in commands if command[0] == "dwifslpreproc")
        self.assertIn("-rpe_pair", eddy)
        self.assertNotIn("-align_seepi", eddy)

    def test_rpe_none_does_not_request_se_epi_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            commands = self._run_and_get_commands(self._config_with_dwi(Path(tmp), reverse=False))

        eddy = next(command for command in commands if command[0] == "dwifslpreproc")
        self.assertIn("-rpe_none", eddy)
        self.assertNotIn("-align_seepi", eddy)
        self.assertNotIn("-se_epi", eddy)


if __name__ == "__main__":
    unittest.main()
