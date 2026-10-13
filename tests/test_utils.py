from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.utils import (
    PCA_DIMS,
    components_for_threshold,
    compute_error_rate,
    git_commit_hash,
    json_default,
    package_versions,
    relative_dimensionality_reduction,
    sha256_file,
    write_json,
)


class UtilsTest(unittest.TestCase):
    def test_experiment_dimensions_are_fixed(self) -> None:
        self.assertEqual(PCA_DIMS, (1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200))

    def test_error_rate_is_complement_of_accuracy(self) -> None:
        self.assertAlmostEqual(compute_error_rate(0.875), 0.125)

    def test_relative_dimensionality_reduction(self) -> None:
        self.assertAlmostEqual(
            relative_dimensionality_reduction(50, 4096), 1.0 - 50.0 / 4096.0
        )

    def test_components_for_threshold_returns_first_crossing(self) -> None:
        cumulative = [0.4, 0.81, 0.93, 0.97]
        self.assertEqual(components_for_threshold(cumulative, 0.80), 2)
        self.assertEqual(components_for_threshold(cumulative, 0.95), 4)
        self.assertIsNone(components_for_threshold(cumulative, 0.99))

    def test_sha256_file_matches_standard_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.txt"
            path.write_bytes(b"EE6222\n")
            expected = hashlib.sha256(b"EE6222\n").hexdigest()
            self.assertEqual(sha256_file(path), expected)

    def test_numeric_helpers_reject_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            compute_error_rate(1.1)
        with self.assertRaises(ValueError):
            relative_dimensionality_reduction(0, 4096)
        with self.assertRaises(ValueError):
            relative_dimensionality_reduction(4097, 4096)
        with self.assertRaises(ValueError):
            components_for_threshold([0.5], 0.0)

    def test_json_helpers_handle_scientific_values_and_unknown_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.json"
            write_json(
                path,
                {
                    "path": path,
                    "array": np.array([1, 2]),
                    "scalar": np.float32(0.5),
                },
            )
            payload = path.read_text(encoding="utf-8")
            self.assertIn('"array": [', payload)
            self.assertEqual(json_default(Path("x")), "x")
            with self.assertRaises(TypeError):
                json_default(object())

    def test_package_and_git_helpers_record_safe_fallbacks(self) -> None:
        versions = package_versions(["definitely_not_an_installed_package"])
        self.assertIsNone(versions["definitely_not_an_installed_package"])
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(git_commit_hash(Path(tmpdir)))


if __name__ == "__main__":
    unittest.main()
