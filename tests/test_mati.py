from __future__ import annotations

import unittest

import numpy as np

from cheddar.mati import MatiPulseError, corrected_mati_pulse, encoding_groups, validate_mati_bvalues


class MatiPulseTests(unittest.TestCase):
    def _payload(self) -> dict[str, object]:
        return {
            "Nacq": 4,
            "TE": [128, 128, 128, 128],
            "TR": 13200,
            "FA": 90,
            "delta": [40, 12, 40, 12],
            "Delta": [50, 74, 50, 74],
            "shape": ["tcos", "tpgse", "tcos", "tpgse"],
            "b": [0, 0, 0.25, 1.0],
            "n": [1, 0, 1, 0],
            "trise": 0.9,
            "gdir": [[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "gamma": 26.75,
        }

    def test_encoding_groups_follow_mati_fields_and_keep_source_order(self) -> None:
        groups = encoding_groups(self._payload())

        self.assertEqual([group.indices for group in groups], [(0, 2), (1, 3)])
        self.assertEqual(groups[0].encoding["shape"], "tcos")
        self.assertEqual(groups[1].encoding["shape"], "tpgse")

    def test_corrected_pulse_uses_output_order_and_rotated_gradients(self) -> None:
        corrected = corrected_mati_pulse(
            self._payload(),
            [0, 2, 1, 3],
            bvecs=np.array([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]),
            bvals_smm2=np.array([0, 250, 0, 1000]),
        )

        self.assertEqual(corrected["shape"], ["tcos", "tcos", "tpgse", "tpgse"])
        self.assertEqual(corrected["b"], [0.0, 0.25, 0.0, 1.0])
        self.assertEqual(corrected["gdir"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertNotIn("G", corrected)

    def test_mati_bvalues_guard_against_wrong_volume_order(self) -> None:
        validate_mati_bvalues(self._payload(), np.array([0.0, 0.0, 250.0, 1000.0]))

        with self.assertRaisesRegex(MatiPulseError, "volume 1"):
            validate_mati_bvalues(self._payload(), np.array([0.0, 1000.0, 250.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
