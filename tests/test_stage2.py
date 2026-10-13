from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

import numpy as np

from src.run_stage2 import (
    LDA_DIMS,
    PCA_PRE_LDA_DIM,
    build_stage2_summary,
    choose_best_observed,
    fit_pca_preprocessing,
    run_lda_sweep,
    validate_lda_dimension,
)


def _logger() -> logging.Logger:
    logger = logging.getLogger("ee6222.stage2.test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


class Stage2HelpersTest(unittest.TestCase):
    def test_fixed_dimensions_and_legal_dimension_validation(self) -> None:
        self.assertEqual(LDA_DIMS, (1, 2, 5, 10, 15, 20, 25, 30, 35, 39))
        self.assertEqual(PCA_PRE_LDA_DIM, 280)
        validate_lda_dimension(2, n_features=6, n_classes=3)
        with self.assertRaises(ValueError):
            validate_lda_dimension(3, n_features=6, n_classes=3)
        with self.assertRaises(ValueError):
            validate_lda_dimension(7, n_features=6, n_classes=10)

    def test_train_only_pca_and_lda_output_shapes(self) -> None:
        rng = np.random.default_rng(7)
        X_train = rng.normal(size=(12, 6))
        X_test = rng.normal(size=(4, 6))
        y_train = np.repeat([0, 1, 2], 4)
        y_test = np.array([0, 1, 2, 0])

        X_train_pca, X_test_pca, pca, _ = fit_pca_preprocessing(
            X_train, X_test, n_components=4, logger=_logger()
        )
        self.assertEqual(X_train_pca.shape, (12, 4))
        self.assertEqual(X_test_pca.shape, (4, 4))
        self.assertEqual(pca.n_components_, 4)

        with patch("src.run_stage2.LDA_DIMS", (1, 2)):
            rows, failures = run_lda_sweep(
                X_train_pca,
                X_test_pca,
                y_train,
                y_test,
                logger=_logger(),
                method="pca280_lda_1nn",
                input_dimension=6,
                intermediate_dimension=4,
                pca_fit_time_sec=0.1,
                pca_pre_explained_variance=0.8,
            )
        self.assertEqual(failures, [])
        self.assertEqual([row["output_dimension"] for row in rows], [1, 2])
        self.assertTrue(all(row["train_output_shape"] == [12, row["output_dimension"]] for row in rows))
        self.assertTrue(all(row["test_output_shape"] == [4, row["output_dimension"]] for row in rows))
        self.assertTrue(all(row["preprocessing_fit_scope"] == "X_train only" for row in rows))

    def test_best_observed_uses_smallest_dimension_on_tie(self) -> None:
        rows = [
            {"output_dimension": 39, "accuracy": 0.95},
            {"output_dimension": 10, "accuracy": 0.95},
            {"output_dimension": 5, "accuracy": 0.90},
        ]
        self.assertEqual(choose_best_observed(rows)["output_dimension"], 10)

    def test_stage2_summary_contains_fixed_test_size_caution(self) -> None:
        raw = {"accuracy": 0.90, "error_rate": 0.10, "dimension": 4096}
        best_pca = {"output_dimension": 75, "accuracy": 0.90, "error_rate": 0.10}
        direct = [
            {"output_dimension": 10, "accuracy": 0.95, "error_rate": 0.05},
            {"output_dimension": 39, "accuracy": 0.95, "error_rate": 0.05},
        ]
        pca_lda = [
            {"output_dimension": 5, "accuracy": 0.925, "error_rate": 0.075},
            {"output_dimension": 10, "accuracy": 0.925, "error_rate": 0.075},
        ]
        summary = build_stage2_summary(
            raw,
            best_pca,
            direct,
            pca_lda,
            pca_pre_explained_variance=0.99,
            failures=[],
        )
        self.assertEqual(summary["direct_lda"]["smallest_tested_dimension_at_max"], 10)
        self.assertEqual(summary["pca_lda"]["smallest_tested_dimension_at_max"], 5)
        self.assertIn("only 80 samples", summary["statistical_caution"])
        self.assertEqual(len(summary["stage3_candidate_shortlist"]), 4)


if __name__ == "__main__":
    unittest.main()
