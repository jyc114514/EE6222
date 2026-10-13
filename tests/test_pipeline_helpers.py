from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import run_baseline, validate_results
from src.audit_environment import collect_environment, environment_text
from src.prepare_data import (
    DatasetBundle,
    audit_dataset,
    build_split_audit,
    dataset_audit_text,
)


def _logger() -> logging.Logger:
    logger = logging.getLogger("ee6222.test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


class PipelineHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = _logger()
        self.X_train = np.array(
            [
                [0.0, 0.0, 0.1, 0.0, 0.1, 0.0],
                [0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0, 0.0, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [1.0, 1.0, 0.9, 1.0, 0.9, 1.0],
                [0.9, 1.0, 1.0, 0.9, 1.0, 1.0],
                [1.0, 0.9, 1.0, 1.0, 1.0, 0.9],
                [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            ],
            dtype=np.float64,
        )
        self.X_test = np.array(
            [[0.05, 0.05, 0.05, 0.05, 0.05, 0.05], [0.95] * 6],
            dtype=np.float64,
        )
        self.y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        self.y_test = np.array([0, 1], dtype=np.int64)

    def test_core_pipeline_helpers_generate_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = run_baseline.run_raw_baseline(
                self.X_train, self.X_test, self.y_train, self.y_test, self.logger
            )
            with patch.object(run_baseline, "PCA_DIMS", (1, 2, 5)), patch.object(
                validate_results, "PCA_DIMS", (1, 2, 5)
            ):
                pca_rows, skipped = run_baseline.run_pca_sweep(
                    self.X_train,
                    self.X_test,
                    self.y_train,
                    self.y_test,
                    self.logger,
                )
            variance, fit_time = run_baseline.run_full_explained_variance(
                self.X_train, self.logger
            )
            self.assertEqual(skipped, [])
            self.assertEqual(len(pca_rows), 3)
            self.assertGreaterEqual(fit_time, 0.0)

            figures_dir = root / "results" / "figures"
            images = np.stack(
                [
                    np.zeros((2, 3)),
                    np.zeros((2, 3)),
                    np.ones((2, 3)),
                    np.ones((2, 3)),
                ]
            )
            labels = np.array([0, 0, 1, 1])
            run_baseline.generate_figures(
                pca_rows,
                raw["accuracy"],
                variance,
                images,
                labels,
                figures_dir,
                self.logger,
            )
            for name in run_baseline.FIGURE_NAMES:
                self.assertTrue((figures_dir / name).is_file())

            dataset_audit = audit_dataset(
                np.zeros((10, 3), dtype=np.float32),
                np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64),
                np.zeros((10, 1, 3)),
            )
            train_indices = np.array([0, 1, 2, 3, 5, 6, 7, 8])
            test_indices = np.array([4, 9])
            split_audit = build_split_audit(
                np.concatenate([self.y_train, self.y_test]),
                train_indices,
                test_indices,
            )
            summary = run_baseline.build_summary(
                raw,
                pca_rows,
                variance,
                dataset_audit,
                split_audit,
                [],
                fit_time,
            )
            self.assertIn("observations", summary)
            self.assertIn("best_tested_dimension", summary["pca_baseline"])
            text = run_baseline.summary_text(summary, pca_rows)
            self.assertIn("Raw baseline", text)
            run_baseline.write_report_notes(root, summary)
            run_baseline.write_results_csv(root, [raw, *pca_rows])
            run_baseline.write_explained_variance_csv(root, variance)
            self.assertTrue((root / "report" / "notes.md").is_file())
            self.assertTrue((root / "results" / "tables" / "baseline_results.csv").is_file())

    def test_validation_passes_for_complete_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for relative in (
                "audit/environment_audit.txt",
                "audit/environment_audit.json",
                "audit/dataset_audit.txt",
                "audit/dataset_audit.json",
                "audit/split_audit.json",
                "audit/run_manifest.json",
                "results/tables/baseline_results.csv",
                "results/tables/pca_explained_variance.csv",
                "results/baseline_summary.txt",
                "results/baseline_summary.json",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            for name in run_baseline.FIGURE_NAMES:
                path = root / "results" / "figures" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")

            y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
            dataset_audit = audit_dataset(
                np.zeros((10, 3), dtype=np.float32), y, np.zeros((10, 1, 3))
            )
            dataset_audit["overall_integrity_pass"] = True
            split_audit = build_split_audit(
                y,
                np.array([0, 1, 2, 3, 5, 6, 7, 8]),
                np.array([4, 9]),
            )
            rows = [
                {
                    "method": "raw_1nn",
                    "dimension": 3,
                    "accuracy": 0.5,
                    "error_rate": 0.5,
                    "n_correct": 2,
                    "n_incorrect": 2,
                    "explained_variance": None,
                    "fit_time_sec": 0.1,
                    "predict_time_sec": 0.1,
                    "total_time_sec": 0.2,
                    "random_state": 42,
                    "train_size": 8,
                    "test_size": 2,
                    "classifier": run_baseline.CLASSIFIER_NAME,
                    "metric": "euclidean",
                }
            ]
            for dimension in (1, 2, 5):
                rows.append(
                    {
                        "method": "pca_1nn",
                        "dimension": dimension,
                        "accuracy": 0.5,
                        "error_rate": 0.5,
                        "n_correct": 2,
                        "n_incorrect": 2,
                        "explained_variance": 0.5,
                        "fit_time_sec": 0.1,
                        "predict_time_sec": 0.1,
                        "total_time_sec": 0.2,
                        "random_state": 42,
                        "train_size": 8,
                        "test_size": 2,
                        "classifier": run_baseline.CLASSIFIER_NAME,
                        "metric": "euclidean",
                    }
                )
            variance = pd.DataFrame(
                {
                    "component": [1, 2, 3],
                    "explained_variance_ratio": [0.5, 0.3, 0.2],
                    "cumulative_explained_variance_ratio": [0.5, 0.8, 1.0],
                }
            )
            with patch.object(validate_results, "PCA_DIMS", (1, 2, 5)):
                report = validate_results.validate_outputs(
                    root,
                    dataset_audit,
                    split_audit,
                    rows,
                    variance,
                    run_baseline.FIGURE_NAMES,
                )
            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "PASS")
            self.assertTrue("PASS" in validate_results.validation_text(report))

    def test_environment_audit_is_renderable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = collect_environment(Path(tmpdir))
            rendered = environment_text(audit)
            self.assertIn("Python version", rendered)
            self.assertIn("CPU only", rendered)

    def test_run_experiment_orchestrates_a_complete_small_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            X = np.vstack(
                [
                    np.zeros((5, 6), dtype=np.float64),
                    np.ones((5, 6), dtype=np.float64),
                ]
            )
            y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
            images = X.reshape(10, 2, 3)
            train_indices = np.array([0, 1, 2, 3, 5, 6, 7, 8])
            test_indices = np.array([4, 9])
            fake_audit = audit_dataset(X, y, images)
            fake_audit["overall_integrity_pass"] = True
            fake_audit["source"] = "sklearn.datasets.fetch_olivetti_faces"
            fake_audit["data_home"] = str(root / "data" / "sklearn_cache")
            fake_split = build_split_audit(y, train_indices, test_indices)

            def fake_prepare(project_root: Path, logger: logging.Logger):
                audit_dir = project_root / "audit"
                audit_dir.mkdir(parents=True, exist_ok=True)
                run_baseline.write_json(audit_dir / "dataset_audit.json", fake_audit)
                (audit_dir / "dataset_audit.txt").write_text(
                    dataset_audit_text(fake_audit), encoding="utf-8"
                )
                run_baseline.write_json(audit_dir / "split_audit.json", fake_split)
                return (
                    DatasetBundle(
                        X=X,
                        y=y,
                        images=images,
                        source=fake_audit["source"],
                        data_home=root / "data" / "sklearn_cache",
                    ),
                    fake_audit,
                    fake_split,
                )

            def fake_environment(project_root: Path):
                audit_dir = project_root / "audit"
                audit_dir.mkdir(parents=True, exist_ok=True)
                environment = {
                    "hostname": "test-host",
                    "user": "test-user",
                    "python": {
                        "version": "3.10.12",
                        "executable": "/tmp/test-python",
                    },
                    "packages": {"scikit-learn": "1.7.2"},
                    "venv": {"python_exists": True},
                }
                run_baseline.write_json(audit_dir / "environment_audit.json", environment)
                (audit_dir / "environment_audit.txt").write_text(
                    "EE6222 test environment\n", encoding="utf-8"
                )
                return environment

            with patch.object(run_baseline, "PCA_DIMS", (1, 2, 5)), patch.object(
                validate_results, "PCA_DIMS", (1, 2, 5)
            ), patch.object(
                run_baseline, "load_and_prepare_dataset", side_effect=fake_prepare
            ), patch.object(
                run_baseline, "write_environment_audit", side_effect=fake_environment
            ):
                exit_code = run_baseline.run_experiment(root)

            self.assertEqual(exit_code, 0)
            self.assertIn(
                "PASS",
                (root / "audit" / "validation_report.txt").read_text(encoding="utf-8"),
            )
            self.assertTrue((root / "results" / "baseline_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
