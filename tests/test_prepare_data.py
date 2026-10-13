from __future__ import annotations

import unittest

import numpy as np

from src.prepare_data import audit_dataset, build_split_audit


class PrepareDataTest(unittest.TestCase):
    def test_audit_dataset_reports_shapes_and_finite_values(self) -> None:
        X = np.arange(12, dtype=np.float32).reshape(4, 3)
        y = np.array([0, 0, 1, 1], dtype=np.int64)
        images = X.reshape(4, 1, 3)

        audit = audit_dataset(X, y, images)

        self.assertEqual(audit["X_shape"], [4, 3])
        self.assertEqual(audit["images_shape"], [4, 1, 3])
        self.assertEqual(audit["number_of_classes"], 2)
        self.assertEqual(audit["nan_count"], 0)
        self.assertEqual(audit["inf_count"], 0)
        self.assertTrue(audit["integrity_checks"]["sample_count_matches_labels"])

    def test_split_audit_detects_overlap(self) -> None:
        y = np.array([0, 0, 1, 1], dtype=np.int64)
        audit = build_split_audit(y, np.array([0, 1, 2]), np.array([2, 3]))

        self.assertFalse(audit["checks"]["no_index_overlap"])
        self.assertEqual(audit["train_size"], 3)
        self.assertEqual(audit["test_size"], 2)


if __name__ == "__main__":
    unittest.main()

