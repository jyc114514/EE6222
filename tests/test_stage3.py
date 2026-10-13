from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.run_stage3 import (
    LDA_DIMS,
    PCA_DIMS,
    REPRESENTATIVE_CONFIGS,
    ROBUSTNESS_SEEDS,
    build_seed_split,
)
from src.stage3_analysis import (
    aggregate_results,
    confidence_interval_95,
    make_paired_comparison,
    smallest_dimension_within_tolerance,
)


class Stage3AnalysisTest(unittest.TestCase):
    def test_protocol_constants_are_fixed(self) -> None:
        self.assertEqual(ROBUSTNESS_SEEDS, tuple(range(100, 120)))
        self.assertEqual(len(PCA_DIMS), 12)
        self.assertEqual(len(LDA_DIMS), 10)
        self.assertEqual(REPRESENTATIVE_CONFIGS["raw"]["output_dimension"], 4096)
        self.assertEqual(REPRESENTATIVE_CONFIGS["pca75"]["output_dimension"], 75)
        self.assertEqual(REPRESENTATIVE_CONFIGS["lda25"]["output_dimension"], 25)
        self.assertEqual(REPRESENTATIVE_CONFIGS["pca280_lda20"]["intermediate_dimension"], 280)

    def test_each_seed_split_is_stratified_and_reusable(self) -> None:
        y = np.repeat(np.arange(40), 10)
        first = build_seed_split(y, seed=3)
        second = build_seed_split(y, seed=3)
        self.assertTrue(np.array_equal(first["train_indices"], second["train_indices"]))
        self.assertTrue(np.array_equal(first["test_indices"], second["test_indices"]))
        self.assertEqual(first["train_indices"].size, 320)
        self.assertEqual(first["test_indices"].size, 80)
        self.assertTrue(first["checks"]["no_index_overlap"])
        self.assertTrue(first["checks"]["all_classes_have_8_train_2_test"])

    def test_confidence_interval_is_distinct_from_standard_deviation(self) -> None:
        result = confidence_interval_95([0.8, 0.9, 1.0, 0.9, 0.8])
        self.assertAlmostEqual(result["mean"], 0.88)
        self.assertGreater(result["std"], 0.0)
        self.assertLess(result["ci95_lower"], result["mean"])
        self.assertGreater(result["ci95_upper"], result["mean"])

    def test_aggregation_and_near_peak_dimension(self) -> None:
        rows = []
        for seed, accuracy in enumerate([0.9, 1.0, 0.9]):
            rows.append(
                {
                    "seed": seed,
                    "method": "pca_1nn",
                    "output_dimension": 20,
                    "intermediate_dimension": None,
                    "accuracy": accuracy,
                    "error_rate": 1.0 - accuracy,
                    "n_correct": int(accuracy * 10),
                    "explained_variance": 0.75,
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "method": "pca_1nn",
                    "output_dimension": 75,
                    "intermediate_dimension": None,
                    "accuracy": accuracy,
                    "error_rate": 1.0 - accuracy,
                    "n_correct": int(accuracy * 10),
                    "explained_variance": 0.92,
                }
            )
        summary = aggregate_results(pd.DataFrame(rows))
        self.assertEqual(len(summary), 2)
        self.assertTrue((summary["n_seeds"] == 3).all())
        self.assertEqual(smallest_dimension_within_tolerance(summary, "pca_1nn"), 20)

    def test_paired_comparison_counts_wins_ties_losses(self) -> None:
        frame = pd.DataFrame(
            [
                {"seed": 0, "method": "raw_1nn", "output_dimension": 4096, "intermediate_dimension": None, "accuracy": 0.8},
                {"seed": 0, "method": "lda_1nn", "output_dimension": 25, "intermediate_dimension": None, "accuracy": 0.9},
                {"seed": 1, "method": "raw_1nn", "output_dimension": 4096, "intermediate_dimension": None, "accuracy": 0.9},
                {"seed": 1, "method": "lda_1nn", "output_dimension": 25, "intermediate_dimension": None, "accuracy": 0.9},
                {"seed": 2, "method": "raw_1nn", "output_dimension": 4096, "intermediate_dimension": None, "accuracy": 1.0},
                {"seed": 2, "method": "lda_1nn", "output_dimension": 25, "intermediate_dimension": None, "accuracy": 0.9},
            ]
        )
        comparison = make_paired_comparison(
            frame,
            name="LDA25 vs Raw",
            left=REPRESENTATIVE_CONFIGS["lda25"],
            right=REPRESENTATIVE_CONFIGS["raw"],
        )
        self.assertEqual(comparison["wins"], 1)
        self.assertEqual(comparison["ties"], 1)
        self.assertEqual(comparison["losses"], 1)
        self.assertAlmostEqual(comparison["mean_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()

