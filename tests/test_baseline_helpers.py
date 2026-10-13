from __future__ import annotations

import unittest

from src.run_baseline import choose_best_pca_result


class BaselineHelpersTest(unittest.TestCase):
    def test_choose_best_result_prefers_smallest_dimension_on_tie(self) -> None:
        rows = [
            {"dimension": 20, "accuracy": 0.8},
            {"dimension": 5, "accuracy": 0.8},
            {"dimension": 10, "accuracy": 0.7},
        ]
        best = choose_best_pca_result(rows)
        self.assertEqual(best["dimension"], 5)
        self.assertEqual(best["accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()

