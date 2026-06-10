from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from cheddar.sync import plan_copy


REPO = Path(__file__).resolve().parents[1]


class SyncCliTests(unittest.TestCase):
    def test_plan_copy_excludes_data_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            dest = Path(tmp) / "dest"
            source.mkdir()
            (source / "keep.txt").write_text("ok", encoding="utf-8")
            (source / ".DS_Store").write_text("noise", encoding="utf-8")
            items = plan_copy(source, dest, excludes=[".DS_Store"])
            self.assertEqual([item.source.name for item in items], ["keep.txt"])

    def test_cli_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "cheddar", "sync", "--help"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("raw", proc.stdout)


if __name__ == "__main__":
    unittest.main()

