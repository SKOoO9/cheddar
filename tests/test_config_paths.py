from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from cheddar.config import load_resolved_config
from cheddar.paths import assign_subjects, participant_id, read_subject_map


REPO = Path(__file__).resolve().parents[1]


class ConfigPathTests(unittest.TestCase):
    def test_example_config_uses_profile_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["CHEDDAR_ROOT"] = tmp
            config = load_resolved_config(REPO / "config" / "study.example.yaml", "local")
            self.assertEqual(config.root, Path(tmp).resolve())
            self.assertEqual(config.raw_root, Path(tmp).resolve() / "RawData")
            self.assertEqual(config.data_root, Path(tmp).resolve() / "Data")

    def test_participant_ids_and_subject_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["CHEDDAR_ROOT"] = tmp
            config = load_resolved_config(REPO / "config" / "study.example.yaml", "local")
            self.assertEqual(participant_id(1), "sub-001")
            records = assign_subjects(config, ["SYNTHETIC_A", "SYNTHETIC_B"], dry_run=False)
            self.assertEqual([record.participant_id for record in records], ["sub-001", "sub-002"])
            reloaded = read_subject_map(config.private_subject_map)
            self.assertEqual(len(reloaded), 2)
            self.assertEqual(reloaded[0].session_id, "ses-01")


if __name__ == "__main__":
    unittest.main()
