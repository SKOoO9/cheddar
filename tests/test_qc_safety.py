from __future__ import annotations

from pathlib import Path
import unittest

from cheddar.qc import audit_tracked_paths, is_forbidden_repo_path


REPO = Path(__file__).resolve().parents[1]


class SafetyTests(unittest.TestCase):
    def test_forbidden_data_paths(self) -> None:
        self.assertTrue(is_forbidden_repo_path("RawData/PRELIMINARY/Xu/file.DCM"))
        self.assertTrue(is_forbidden_repo_path("Data/sub-001/ses-01/dwi/file.nii.gz"))
        self.assertTrue(is_forbidden_repo_path("Data/.private/subject_map.tsv"))
        self.assertFalse(is_forbidden_repo_path("cheddar/config.py"))

    def test_repo_tree_contains_no_data_files(self) -> None:
        paths = [
            path.relative_to(REPO).as_posix()
            for path in REPO.rglob("*")
            if path.is_file()
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and path.suffix != ".pyc"
        ]
        audit = audit_tracked_paths(paths)
        self.assertTrue(audit["ok"], audit["forbidden"])


if __name__ == "__main__":
    unittest.main()
