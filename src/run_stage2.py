"""Run the additive EE6222 Stage 2 LDA and PCA-to-LDA experiments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

if __package__ in {None, ""}:  # support the requested ``python src/run_stage2.py``
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stage2_validation import (
        EXPECTED_STAGE1_ACCURACIES,
        RESULT_COLUMNS as VALIDATION_RESULT_COLUMNS,
        validate_stage2_outputs,
        validation_text,
    )
    from utils import (
        RANDOM_STATE,
        compute_error_rate,
        git_commit_hash,
        package_versions,
        relative_dimensionality_reduction,
        sha256_file,
        source_sha256,
        write_json,
    )
else:
    from .stage2_validation import (
        EXPECTED_STAGE1_ACCURACIES,
        RESULT_COLUMNS as VALIDATION_RESULT_COLUMNS,
        validate_stage2_outputs,
        validation_text,
    )
    from .utils import (
        RANDOM_STATE,
        compute_error_rate,
        git_commit_hash,
        package_versions,
        relative_dimensionality_reduction,
        sha256_file,
        source_sha256,
        write_json,
    )


LDA_DIMS: tuple[int, ...] = (1, 2, 5, 10, 15, 20, 25, 30, 35, 39)
PCA_PRE_LDA_DIM = 280
LDA_SOLVER = "svd"
CLASSIFIER_NAME = "KNeighborsClassifier(n_neighbors=1)"
METRIC_NAME = "euclidean"
STAGE2_FIGURE_NAMES = (
    "lda_accuracy_vs_dimension.png",
    "method_comparison_accuracy.png",
    "method_comparison_error_rate.png",
)
STAGE2_COLUMNS = [
    "method",
    "input_dimension",
    "intermediate_dimension",
    "output_dimension",
    "accuracy",
    "error_rate",
    "n_correct",
    "n_incorrect",
    "explained_variance",
    "pca_pre_explained_variance",
    "solver",
    "classifier",
    "metric",
    "pca_fit_time_sec",
    "lda_fit_time_sec",
    "transform_time_sec",
    "classifier_fit_time_sec",
    "prediction_time_sec",
    "fit_time_sec",
    "predict_time_sec",
    "total_time_sec",
    "random_state",
    "train_size",
    "test_size",
    "preprocessing_fit_scope",
    "pca_fit_scope",
    "train_output_shape",
    "test_output_shape",
]


@dataclass(frozen=True)
class Stage1State:
    """Immutable in-memory view of the frozen Stage 1 artifacts."""

    project_root: Path
    X: np.ndarray
    y: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    dataset_audit: dict[str, Any]
    split_audit: dict[str, Any]
    baseline_frame: pd.DataFrame
    baseline_sha256: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_logging(log_path: Path) -> logging.Logger:
    """Create a fresh Stage 2 file-and-console logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ee6222.assignment1.stage2")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def load_stage1_state(project_root: Path) -> Stage1State:
    """Read existing Stage 1 arrays/metadata without refetching or rewriting them."""
    project_root = project_root.resolve()
    required = (
        project_root / "data" / "processed" / "X.npy",
        project_root / "data" / "processed" / "y.npy",
        project_root / "audit" / "dataset_audit.json",
        project_root / "audit" / "split_audit.json",
        project_root / "audit" / "run_manifest.json",
        project_root / "audit" / "validation_report.txt",
        project_root / "results" / "tables" / "baseline_results.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Stage 1 is incomplete; missing frozen artifacts: " + ", ".join(missing)
        )
    validation_lines = (
        project_root / "audit" / "validation_report.txt"
    ).read_text(encoding="utf-8").splitlines()
    if not validation_lines or validation_lines[0] != "PASS":
        raise RuntimeError("Stage 1 validation is not PASS; Stage 2 will not proceed")

    X = np.load(project_root / "data" / "processed" / "X.npy", allow_pickle=False)
    y = np.load(project_root / "data" / "processed" / "y.npy", allow_pickle=False)
    dataset_audit = _read_json(project_root / "audit" / "dataset_audit.json")
    split_audit = _read_json(project_root / "audit" / "split_audit.json")
    baseline_path = project_root / "results" / "tables" / "baseline_results.csv"
    baseline_frame = pd.read_csv(baseline_path)
    baseline_sha256 = sha256_file(baseline_path)

    if X.shape != (400, 4096) or y.shape != (400,):
        raise RuntimeError(f"Unexpected Stage 1 array shapes: X={X.shape}, y={y.shape}")
    train_indices = np.asarray(split_audit["train_indices"], dtype=np.int64)
    test_indices = np.asarray(split_audit["test_indices"], dtype=np.int64)
    if train_indices.size != 320 or test_indices.size != 80:
        raise RuntimeError(
            f"Unexpected frozen split sizes: train={train_indices.size}, "
            f"test={test_indices.size}"
        )
    if np.intersect1d(train_indices, test_indices).size:
        raise RuntimeError("Frozen Stage 1 split has train/test index overlap")

    for (method, dimension), expected_accuracy in EXPECTED_STAGE1_ACCURACIES.items():
        matches = baseline_frame[
            (baseline_frame["method"] == method)
            & (baseline_frame["dimension"] == dimension)
        ]
        if len(matches) != 1 or not np.isclose(
            float(matches.iloc[0]["accuracy"]), expected_accuracy, atol=1e-12
        ):
            raise RuntimeError(
                f"Frozen Stage 1 accuracy mismatch for {method} d={dimension}: "
                f"expected {expected_accuracy}"
            )
    return Stage1State(
        project_root=project_root,
        X=X,
        y=y,
        train_indices=train_indices,
        test_indices=test_indices,
        dataset_audit=dataset_audit,
        split_audit=split_audit,
        baseline_frame=baseline_frame,
        baseline_sha256=baseline_sha256,
    )


def validate_lda_dimension(dimension: int, n_features: int, n_classes: int) -> None:
    """Raise when an LDA output dimension exceeds the legal supervised rank."""
    maximum = min(int(n_features), int(n_classes) - 1)
    if dimension < 1 or dimension > maximum:
        raise ValueError(
            f"LDA dimension {dimension} is illegal; legal range is 1..{maximum}"
        )


def choose_best_observed(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Choose max observed accuracy, breaking ties by the smallest output dimension."""
    candidates = list(rows)
    if not candidates:
        raise ValueError("at least one result row is required")
    return min(
        candidates,
        key=lambda row: (
            -float(row["accuracy"]),
            int(row.get("output_dimension", row.get("dimension"))),
        ),
    )


def _stage2_result_row(
    *,
    method: str,
    input_dimension: int,
    intermediate_dimension: int | None,
    output_dimension: int,
    accuracy: float,
    n_correct: int,
    n_incorrect: int,
    solver: str,
    pca_fit_time_sec: float | None,
    lda_fit_time_sec: float,
    transform_time_sec: float,
    classifier_fit_time_sec: float,
    prediction_time_sec: float,
    total_time_sec: float,
    train_size: int,
    test_size: int,
    pca_pre_explained_variance: float | None,
    pca_fit_scope: str | None,
    train_output_shape: list[int],
    test_output_shape: list[int],
) -> dict[str, Any]:
    pca_fit_value = float(pca_fit_time_sec or 0.0)
    return {
        "method": method,
        "input_dimension": int(input_dimension),
        "intermediate_dimension": (
            int(intermediate_dimension) if intermediate_dimension is not None else None
        ),
        "output_dimension": int(output_dimension),
        "accuracy": float(accuracy),
        "error_rate": compute_error_rate(accuracy),
        "n_correct": int(n_correct),
        "n_incorrect": int(n_incorrect),
        "explained_variance": None,
        "pca_pre_explained_variance": (
            float(pca_pre_explained_variance)
            if pca_pre_explained_variance is not None
            else None
        ),
        "solver": solver,
        "classifier": CLASSIFIER_NAME,
        "metric": METRIC_NAME,
        "pca_fit_time_sec": (
            float(pca_fit_time_sec) if pca_fit_time_sec is not None else None
        ),
        "lda_fit_time_sec": float(lda_fit_time_sec),
        "transform_time_sec": float(transform_time_sec),
        "classifier_fit_time_sec": float(classifier_fit_time_sec),
        "prediction_time_sec": float(prediction_time_sec),
        "fit_time_sec": float(pca_fit_value + lda_fit_time_sec + classifier_fit_time_sec),
        "predict_time_sec": float(transform_time_sec + prediction_time_sec),
        "total_time_sec": float(total_time_sec),
        "random_state": RANDOM_STATE,
        "train_size": int(train_size),
        "test_size": int(test_size),
        "preprocessing_fit_scope": "X_train only",
        "pca_fit_scope": pca_fit_scope,
        "train_output_shape": [int(value) for value in train_output_shape],
        "test_output_shape": [int(value) for value in test_output_shape],
    }


def fit_pca_preprocessing(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, PCA, dict[str, float]]:
    """Fit PCA only on training data and transform train/test separately."""
    maximum = min(X_train.shape[0], X_train.shape[1])
    if n_components < 1 or n_components > maximum:
        raise ValueError(
            f"PCA preprocessing dimension {n_components} is illegal; legal maximum is {maximum}"
        )
    pca = PCA(n_components=n_components, svd_solver="full")
    fit_start = time.perf_counter()
    pca.fit(X_train)
    fit_time = time.perf_counter() - fit_start
    transform_start = time.perf_counter()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    transform_time = time.perf_counter() - transform_start
    if X_train_pca.shape != (X_train.shape[0], n_components) or X_test_pca.shape != (
        X_test.shape[0],
        n_components,
    ):
        raise RuntimeError(
            f"PCA preprocessing shape failure: train={X_train_pca.shape}, "
            f"test={X_test_pca.shape}"
        )
    if not np.isfinite(X_train_pca).all() or not np.isfinite(X_test_pca).all():
        raise RuntimeError("PCA preprocessing generated non-finite values")
    logger.info(
        "PCA preprocessing: dimension=%d fit=%.4fs transform=%.4fs cumulative_variance=%.9f",
        n_components,
        fit_time,
        transform_time,
        float(np.sum(pca.explained_variance_ratio_)),
    )
    return X_train_pca, X_test_pca, pca, {
        "fit_time_sec": float(fit_time),
        "transform_time_sec": float(transform_time),
    }


def run_lda_sweep(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    logger: logging.Logger,
    *,
    method: str,
    input_dimension: int,
    intermediate_dimension: int | None = None,
    pca_fit_time_sec: float | None = None,
    pca_pre_explained_variance: float | None = None,
    pca_fit_scope: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run fixed direct-LDA or PCA-preprocessed-LDA dimensions."""
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    max_features = int(X_train.shape[1])
    n_classes = int(np.unique(y_train).size)
    for dimension in LDA_DIMS:
        try:
            validate_lda_dimension(dimension, max_features, n_classes)
            total_start = time.perf_counter()
            lda = LinearDiscriminantAnalysis(
                solver=LDA_SOLVER,
                n_components=dimension,
            )
            lda_fit_start = time.perf_counter()
            lda.fit(X_train, y_train)
            lda_fit_time = time.perf_counter() - lda_fit_start
            transform_start = time.perf_counter()
            X_train_lda = lda.transform(X_train)
            X_test_lda = lda.transform(X_test)
            transform_time = time.perf_counter() - transform_start
            expected_train_shape = (X_train.shape[0], dimension)
            expected_test_shape = (X_test.shape[0], dimension)
            if X_train_lda.shape != expected_train_shape or X_test_lda.shape != expected_test_shape:
                raise RuntimeError(
                    f"LDA output shape mismatch: train={X_train_lda.shape}, "
                    f"test={X_test_lda.shape}, expected={expected_train_shape}/{expected_test_shape}"
                )
            if not np.isfinite(X_train_lda).all() or not np.isfinite(X_test_lda).all():
                raise RuntimeError("LDA generated non-finite transformed values")

            classifier = KNeighborsClassifier(n_neighbors=1, metric=METRIC_NAME)
            classifier_fit_start = time.perf_counter()
            classifier.fit(X_train_lda, y_train)
            classifier_fit_time = time.perf_counter() - classifier_fit_start
            prediction_start = time.perf_counter()
            predictions = classifier.predict(X_test_lda)
            prediction_time = time.perf_counter() - prediction_start
            total_time = time.perf_counter() - total_start
            accuracy = float(accuracy_score(y_test, predictions))
            n_correct = int(np.sum(predictions == y_test))
            row = _stage2_result_row(
                method=method,
                input_dimension=input_dimension,
                intermediate_dimension=intermediate_dimension,
                output_dimension=dimension,
                accuracy=accuracy,
                n_correct=n_correct,
                n_incorrect=int(y_test.size - n_correct),
                solver=LDA_SOLVER,
                pca_fit_time_sec=pca_fit_time_sec,
                lda_fit_time_sec=lda_fit_time,
                transform_time_sec=transform_time,
                classifier_fit_time_sec=classifier_fit_time,
                prediction_time_sec=prediction_time,
                total_time_sec=total_time + float(pca_fit_time_sec or 0.0),
                train_size=int(y_train.size),
                test_size=int(y_test.size),
                pca_pre_explained_variance=pca_pre_explained_variance,
                pca_fit_scope=pca_fit_scope,
                train_output_shape=list(X_train_lda.shape),
                test_output_shape=list(X_test_lda.shape),
            )
            rows.append(row)
            logger.info(
                "%s d=%d: accuracy=%.6f error_rate=%.6f lda_fit=%.4fs "
                "transform=%.4fs predict=%.4fs",
                method,
                dimension,
                row["accuracy"],
                row["error_rate"],
                lda_fit_time,
                transform_time,
                prediction_time,
            )
        except Exception as exc:  # record each numerical/configuration failure explicitly
            failure = {
                "method": method,
                "dimension": int(dimension),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            logger.exception(
                "%s d=%d failed; solver remains fixed at %s",
                method,
                dimension,
                LDA_SOLVER,
            )
    return rows, failures


def _none_if_nan(value: Any) -> Any:
    return None if pd.isna(value) else value


def baseline_to_all_method_rows(baseline_frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Map frozen Stage 1 rows into the Stage 2 comparison schema."""
    rows: list[dict[str, Any]] = []
    for record in baseline_frame.to_dict(orient="records"):
        method = str(record["method"])
        dimension = int(record["dimension"])
        is_pca = method == "pca_1nn"
        rows.append(
            {
                "method": method,
                "input_dimension": 4096,
                "intermediate_dimension": None,
                "output_dimension": dimension,
                "accuracy": float(record["accuracy"]),
                "error_rate": float(record["error_rate"]),
                "n_correct": int(record["n_correct"]),
                "n_incorrect": int(record["n_incorrect"]),
                "explained_variance": (
                    _none_if_nan(record.get("explained_variance")) if is_pca else None
                ),
                "pca_pre_explained_variance": None,
                "solver": None,
                "classifier": record["classifier"],
                "metric": record["metric"],
                "pca_fit_time_sec": _none_if_nan(record.get("pca_fit_time_sec")),
                "lda_fit_time_sec": None,
                "transform_time_sec": _none_if_nan(record.get("transform_time_sec")),
                "classifier_fit_time_sec": _none_if_nan(record.get("classifier_fit_time_sec")),
                "prediction_time_sec": _none_if_nan(record.get("prediction_time_sec")),
                "fit_time_sec": float(record["fit_time_sec"]),
                "predict_time_sec": float(record["predict_time_sec"]),
                "total_time_sec": float(record["total_time_sec"]),
                "random_state": int(record["random_state"]),
                "train_size": int(record["train_size"]),
                "test_size": int(record["test_size"]),
                "preprocessing_fit_scope": "X_train only" if is_pca else "none",
                "pca_fit_scope": "X_train only" if is_pca else None,
                "train_output_shape": None,
                "test_output_shape": None,
            }
        )
    return rows


def _best_dimension(rows: list[dict[str, Any]]) -> int:
    return int(choose_best_observed(rows)["output_dimension"])


def build_stage2_summary(
    raw_row: dict[str, Any],
    best_pca_row: dict[str, Any],
    direct_rows: list[dict[str, Any]],
    pca_lda_rows: list[dict[str, Any]],
    pca_pre_explained_variance: float,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize observed Stage 2 performance without calling dimensions optimal."""
    raw_accuracy = float(raw_row["accuracy"])
    raw_dimension = int(raw_row.get("input_dimension", raw_row.get("dimension", 4096)))
    best_direct = choose_best_observed(direct_rows)
    best_pca_lda = choose_best_observed(pca_lda_rows)
    best_pca_dimension = int(
        best_pca_row.get("output_dimension", best_pca_row.get("dimension"))
    )
    best_pca_accuracy = float(best_pca_row["accuracy"])
    direct_accuracy = float(best_direct["accuracy"])
    pca_lda_accuracy = float(best_pca_lda["accuracy"])

    direct_max = max(float(row["accuracy"]) for row in direct_rows)
    pca_lda_max = max(float(row["accuracy"]) for row in pca_lda_rows)
    direct_peak_dims = [
        int(row["output_dimension"])
        for row in direct_rows
        if float(row["accuracy"]) == direct_max
    ]
    pca_lda_peak_dims = [
        int(row["output_dimension"])
        for row in pca_lda_rows
        if float(row["accuracy"]) == pca_lda_max
    ]

    direct_comparison = (
        f"Best observed direct LDA accuracy is {direct_accuracy:.6f}, "
        f"difference vs raw is {direct_accuracy - raw_accuracy:+.6f}, "
        f"and difference vs best Stage 1 PCA is {direct_accuracy - best_pca_accuracy:+.6f}."
    )
    pca_lda_comparison = (
        f"Best observed PCA(280)->LDA accuracy is {pca_lda_accuracy:.6f}, "
        f"difference vs raw is {pca_lda_accuracy - raw_accuracy:+.6f}, "
        f"and difference vs direct LDA is {pca_lda_accuracy - direct_accuracy:+.6f}."
    )
    observations = [
        direct_comparison,
        pca_lda_comparison,
        f"Direct LDA peak accuracy occurs at tested dimensions {direct_peak_dims}; "
        f"the smallest tested dimension at that peak is {min(direct_peak_dims)}.",
        f"PCA(280)->LDA peak accuracy occurs at tested dimensions {pca_lda_peak_dims}; "
        f"the smallest tested dimension at that peak is {min(pca_lda_peak_dims)}.",
        f"The fixed PCA preprocessing retains {pca_pre_explained_variance:.6f} "
        "of training-set variance before LDA.",
        "No solver or dimension was changed in response to test accuracy; all requested "
        "Stage 2 dimensions were fixed before evaluation.",
    ]
    if failures:
        observations.append(f"Numerical/configuration failures were recorded: {failures}.")
    else:
        observations.append("All requested direct-LDA and PCA(280)->LDA dimensions completed.")

    shortlist = [
        {
            "label": "Raw 4096D + 1-NN",
            "method": "raw_1nn",
            "output_dimension": raw_dimension,
            "accuracy": raw_accuracy,
            "reason": "frozen reference baseline",
        },
        {
            "label": f"PCA {best_pca_dimension}D + 1-NN",
            "method": "pca_1nn",
            "output_dimension": best_pca_dimension,
            "accuracy": best_pca_accuracy,
            "reason": "smallest Stage 1 PCA point achieving the maximum observed PCA accuracy",
        },
        {
            "label": f"Direct LDA {min(direct_peak_dims)}D + 1-NN",
            "method": "lda_1nn",
            "output_dimension": min(direct_peak_dims),
            "accuracy": direct_accuracy,
            "reason": "smallest tested direct-LDA dimension at the maximum observed accuracy",
        },
        {
            "label": f"PCA(280)->LDA {min(pca_lda_peak_dims)}D + 1-NN",
            "method": "pca280_lda_1nn",
            "intermediate_dimension": PCA_PRE_LDA_DIM,
            "output_dimension": min(pca_lda_peak_dims),
            "accuracy": pca_lda_accuracy,
            "reason": "smallest tested PCA(280)->LDA dimension at the maximum observed accuracy",
        },
    ]
    return {
        "experiment": "EE6222 Assignment 1 - Stage 2 LDA extension",
        "direct_lda": {
            "solver": LDA_SOLVER,
            "tested_dimensions": [int(row["output_dimension"]) for row in direct_rows],
            "maximum_observed_accuracy": direct_accuracy,
            "smallest_tested_dimension_at_max": min(direct_peak_dims),
            "peak_dimensions": direct_peak_dims,
            "error_rate": float(best_direct["error_rate"]),
            "difference_vs_raw": direct_accuracy - raw_accuracy,
            "difference_vs_best_pca": direct_accuracy - best_pca_accuracy,
            "dimensionality_reduction": relative_dimensionality_reduction(
                min(direct_peak_dims), raw_dimension
            ),
        },
        "pca_lda": {
            "pca_preprocessing_dimension": PCA_PRE_LDA_DIM,
            "pca_pre_explained_variance": float(pca_pre_explained_variance),
            "tested_final_lda_dimensions": [
                int(row["output_dimension"]) for row in pca_lda_rows
            ],
            "maximum_observed_accuracy": pca_lda_accuracy,
            "smallest_tested_dimension_at_max": min(pca_lda_peak_dims),
            "peak_dimensions": pca_lda_peak_dims,
            "error_rate": float(best_pca_lda["error_rate"]),
            "difference_vs_raw": pca_lda_accuracy - raw_accuracy,
            "difference_vs_direct_lda": pca_lda_accuracy - direct_accuracy,
            "dimensionality_reduction_by_final_representation": relative_dimensionality_reduction(
                min(pca_lda_peak_dims), raw_dimension
            ),
        },
        "best_stage1_reference": {
            "dimension": best_pca_dimension,
            "accuracy": best_pca_accuracy,
        },
        "statistical_caution": (
            "Because the test set contains only 80 samples, a 1.25 percentage-point "
            "accuracy difference corresponds to only one test image. Small observed "
            "differences should therefore not be over-interpreted before robustness "
            "experiments across multiple splits."
        ),
        "stage3_candidate_shortlist": shortlist,
        "stage3_shortlist_status": (
            "Proposal only: four configurations selected from fixed Stage 2 observations; "
            "no multi-seed robustness was run in Stage 2."
        ),
        "observations": observations,
        "failures": failures,
    }


def summary_text(
    summary: dict[str, Any],
    direct_rows: list[dict[str, Any]],
    pca_lda_rows: list[dict[str, Any]],
) -> str:
    """Render the Stage 2 summary, including both complete sweep tables."""
    direct = summary["direct_lda"]
    pca_lda = summary["pca_lda"]
    lines = [
        "EE6222 Assignment 1 - Stage 2 Summary",
        "",
        "Direct LDA + 1-NN (solver=svd):",
        "  dimension | accuracy | error_rate",
    ]
    lines.extend(
        f"  {row['output_dimension']:9d} | {row['accuracy']:.12f} | {row['error_rate']:.12f}"
        for row in direct_rows
    )
    lines.extend(
        [
            f"  maximum observed accuracy: {direct['maximum_observed_accuracy']:.12f}",
            f"  smallest tested dimension at maximum: {direct['smallest_tested_dimension_at_max']}",
            f"  difference vs raw: {direct['difference_vs_raw']:+.12f}",
            f"  difference vs best PCA: {direct['difference_vs_best_pca']:+.12f}",
            "",
            "PCA(280) -> LDA + 1-NN:",
            "  PCA pre-dimension | final LDA dimension | accuracy | error_rate",
        ]
    )
    lines.extend(
        f"  {PCA_PRE_LDA_DIM:17d} | {row['output_dimension']:19d} | "
        f"{row['accuracy']:.12f} | {row['error_rate']:.12f}"
        for row in pca_lda_rows
    )
    lines.extend(
        [
            f"  PCA preprocessing cumulative explained variance: {pca_lda['pca_pre_explained_variance']:.12f}",
            f"  maximum observed accuracy: {pca_lda['maximum_observed_accuracy']:.12f}",
            f"  smallest tested final dimension at maximum: {pca_lda['smallest_tested_dimension_at_max']}",
            f"  difference vs raw: {pca_lda['difference_vs_raw']:+.12f}",
            f"  difference vs direct LDA: {pca_lda['difference_vs_direct_lda']:+.12f}",
            "",
            "Stage 3 candidate shortlist (not run here):",
        ]
    )
    lines.extend(
        f"  - {candidate['label']}: accuracy={candidate['accuracy']:.12f}; "
        f"{candidate['reason']}"
        for candidate in summary["stage3_candidate_shortlist"]
    )
    lines.extend(
        [
            "",
            "Statistical caution:",
            f"  {summary['statistical_caution']}",
            "",
            "Observations:",
        ]
    )
    lines.extend(f"  - {observation}" for observation in summary["observations"])
    if summary["failures"]:
        lines.extend(["", f"Failures: {summary['failures']}"])
    return "\n".join(lines) + "\n"


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_stage2_figures(
    baseline_frame: pd.DataFrame,
    direct_rows: list[dict[str, Any]],
    pca_lda_rows: list[dict[str, Any]],
    figures_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate the requested Stage 2 accuracy/error comparison figures."""
    raw_accuracy = float(
        baseline_frame.loc[baseline_frame["method"] == "raw_1nn", "accuracy"].iloc[0]
    )
    pca_frame = baseline_frame[baseline_frame["method"] == "pca_1nn"]
    direct_dimensions = [int(row["output_dimension"]) for row in direct_rows]
    direct_accuracy = [float(row["accuracy"]) for row in direct_rows]
    pca_lda_dimensions = [int(row["output_dimension"]) for row in pca_lda_rows]
    pca_lda_accuracy = [float(row["accuracy"]) for row in pca_lda_rows]

    fig, ax = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    ax.plot(
        direct_dimensions,
        direct_accuracy,
        marker="o",
        linewidth=2,
        label="Direct LDA + 1-NN",
    )
    ax.plot(
        pca_lda_dimensions,
        pca_lda_accuracy,
        marker="s",
        linewidth=2,
        label="PCA(280) -> LDA + 1-NN",
    )
    ax.axhline(
        raw_accuracy,
        color="tab:red",
        linestyle="--",
        linewidth=1.8,
        label="Raw 4096D + 1-NN baseline",
    )
    ax.set_xlabel("LDA Reduced Dimensionality")
    ax.set_ylabel("Test Classification Accuracy")
    ax.set_title("LDA Accuracy vs Dimension")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(sorted(set(direct_dimensions + pca_lda_dimensions)))
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, figures_dir / "lda_accuracy_vs_dimension.png")

    fig, ax = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    ax.plot(
        pca_frame["dimension"],
        pca_frame["accuracy"],
        marker="o",
        linewidth=1.8,
        label="PCA + 1-NN",
    )
    ax.plot(
        direct_dimensions,
        direct_accuracy,
        marker="^",
        linewidth=1.8,
        label="Direct LDA + 1-NN",
    )
    ax.plot(
        pca_lda_dimensions,
        pca_lda_accuracy,
        marker="s",
        linewidth=1.8,
        label="PCA(280) -> LDA + 1-NN",
    )
    ax.axhline(
        raw_accuracy,
        color="tab:red",
        linestyle="--",
        linewidth=1.8,
        label="Raw 4096D + 1-NN baseline",
    )
    ax.set_xlabel("Output Representation Dimensionality")
    ax.set_ylabel("Test Classification Accuracy")
    ax.set_title("Four-Method Accuracy Comparison")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, figures_dir / "method_comparison_accuracy.png")

    fig, ax = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    ax.plot(
        pca_frame["dimension"],
        1.0 - pca_frame["accuracy"],
        marker="o",
        linewidth=1.8,
        label="PCA + 1-NN",
    )
    ax.plot(
        direct_dimensions,
        [1.0 - value for value in direct_accuracy],
        marker="^",
        linewidth=1.8,
        label="Direct LDA + 1-NN",
    )
    ax.plot(
        pca_lda_dimensions,
        [1.0 - value for value in pca_lda_accuracy],
        marker="s",
        linewidth=1.8,
        label="PCA(280) -> LDA + 1-NN",
    )
    ax.axhline(
        1.0 - raw_accuracy,
        color="tab:red",
        linestyle="--",
        linewidth=1.8,
        label="Raw 4096D + 1-NN baseline",
    )
    ax.set_xlabel("Output Representation Dimensionality")
    ax.set_ylabel("Classification Error Rate")
    ax.set_title("Four-Method Error Rate Comparison")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, figures_dir / "method_comparison_error_rate.png")
    logger.info("Stage 2 figures generated under %s", figures_dir)


def write_stage2_notes(project_root: Path, summary: dict[str, Any]) -> None:
    """Append factual Stage 2 notes while preserving the Stage 1 section."""
    notes_path = project_root / "report" / "notes.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    stage2_marker = "\n## Stage 2 - LDA Experiments\n"
    prefix = existing.split(stage2_marker, maxsplit=1)[0].rstrip()
    direct = summary["direct_lda"]
    pca_lda = summary["pca_lda"]
    stage2_text = [
        "## Stage 2 - LDA Experiments",
        "",
        f"Direct LDA settings: solver={direct['solver']}, fixed dimensions "
        f"{direct['tested_dimensions']}.",
        f"PCA->LDA settings: PCA preprocessing dimension={pca_lda['pca_preprocessing_dimension']}, "
        f"fixed final LDA dimensions={pca_lda['tested_final_lda_dimensions']}.",
        f"Best direct LDA result: smallest tested d={direct['smallest_tested_dimension_at_max']} "
        f"at maximum observed accuracy {direct['maximum_observed_accuracy']:.12f}.",
        f"Best PCA->LDA result: smallest tested final d={pca_lda['smallest_tested_dimension_at_max']} "
        f"at maximum observed accuracy {pca_lda['maximum_observed_accuracy']:.12f}.",
        f"Comparison against raw: direct difference {direct['difference_vs_raw']:+.12f}; "
        f"PCA->LDA difference {pca_lda['difference_vs_raw']:+.12f}.",
        f"Comparison against PCA: direct difference {direct['difference_vs_best_pca']:+.12f}; "
        f"PCA->LDA comparison is recorded against direct LDA as "
        f"{pca_lda['difference_vs_direct_lda']:+.12f}.",
        "Interesting trend:",
        *[f"- {observation}" for observation in summary["observations"]],
        "Potential explanation: LDA uses class labels, whereas PCA retains unsupervised "
        "variance; the observed curves are descriptive for this fixed split only.",
        f"Statistical caution: {summary['statistical_caution']}",
    ]
    notes_path.write_text(
        (prefix + "\n\n" if prefix else "") + "\n".join(stage2_text) + "\n",
        encoding="utf-8",
    )


def build_stage2_manifest(
    state: Stage1State,
    environment_audit: dict[str, Any],
    baseline_before_hash: str,
    baseline_after_hash: str,
    pca_pre_dimension: int,
    pca_pre_shapes: tuple[list[int], list[int]],
    pca_pre_fit_scope: str,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Capture Stage 2 provenance and the Stage 1 preservation boundary."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": environment_audit.get("hostname"),
        "user": environment_audit.get("user"),
        "project_path": str(state.project_root),
        "python_version": environment_audit["python"]["version"],
        "python_executable": environment_audit["python"]["executable"],
        "package_versions": package_versions(
            ("numpy", "scipy", "sklearn", "pandas", "matplotlib", "joblib")
        ),
        "dataset": {
            "source": state.dataset_audit.get("source"),
            "data_home": state.dataset_audit.get("data_home"),
            "X_shape": state.dataset_audit.get("X_shape"),
            "y_shape": state.dataset_audit.get("y_shape"),
        },
        "train_test_split": {
            "random_state": RANDOM_STATE,
            "test_size": 0.20,
            "train_size": int(state.train_indices.size),
            "test_size": int(state.test_indices.size),
            "index_hash_source": "audit/split_audit.json",
            "no_index_overlap": bool(np.intersect1d(state.train_indices, state.test_indices).size == 0),
        },
        "lda_dimensions": list(LDA_DIMS),
        "lda_solver": LDA_SOLVER,
        "pca_pre_lda_dimension": pca_pre_dimension,
        "pca_pre_lda_shapes": {
            "train": pca_pre_shapes[0],
            "test": pca_pre_shapes[1],
        },
        "pca_pre_lda_fit_scope": pca_pre_fit_scope,
        "classifier": {
            "class": "sklearn.neighbors.KNeighborsClassifier",
            "n_neighbors": 1,
            "metric": METRIC_NAME,
        },
        "device": "CPU only",
        "gpu_used": False,
        "command_used": "python src/run_stage2.py",
        "actual_argv": [str(value) for value in sys.argv],
        "stage1_baseline_results_sha256_before": baseline_before_hash,
        "stage1_baseline_results_sha256_after": baseline_after_hash,
        "stage1_baseline_preserved": baseline_before_hash == baseline_after_hash,
        "failures": failures,
        "git_commit_hash": git_commit_hash(state.project_root),
        "source_code_sha256": source_sha256(state.project_root),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=STAGE2_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")
    return frame


def _write_all_method_csv(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=STAGE2_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")
    return frame


def run_experiment(project_root: Path) -> int:
    """Run Stage 2 against frozen Stage 1 artifacts and return an exit code."""
    project_root = project_root.resolve()
    logger = configure_logging(project_root / "results" / "logs" / "stage2_run.log")
    logger.info("Starting additive EE6222 Stage 2 CPU-only experiment")
    logger.info("Project root: %s", project_root)
    try:
        state = load_stage1_state(project_root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.exception("Stage 1 preservation gate failed; Stage 2 not started")
        return 2
    baseline_before_hash = state.baseline_sha256
    environment_path = project_root / "audit" / "environment_audit.json"
    environment_audit = _read_json(environment_path)
    train_indices = state.train_indices
    test_indices = state.test_indices
    X_train = state.X[train_indices]
    X_test = state.X[test_indices]
    y_train = state.y[train_indices]
    y_test = state.y[test_indices]
    logger.info(
        "Frozen Stage 1 state loaded: X=%s, train=%d, test=%d, baseline_sha256=%s",
        state.X.shape,
        y_train.size,
        y_test.size,
        baseline_before_hash,
    )

    direct_rows, direct_failures = run_lda_sweep(
        X_train,
        X_test,
        y_train,
        y_test,
        logger,
        method="lda_1nn",
        input_dimension=int(X_train.shape[1]),
    )
    X_train_pca, X_test_pca, pca, pca_timing = fit_pca_preprocessing(
        X_train,
        X_test,
        n_components=PCA_PRE_LDA_DIM,
        logger=logger,
    )
    pca_pre_shapes = (list(X_train_pca.shape), list(X_test_pca.shape))
    pca_pre_explained_variance = float(np.sum(pca.explained_variance_ratio_))
    pca_lda_rows, pca_lda_failures = run_lda_sweep(
        X_train_pca,
        X_test_pca,
        y_train,
        y_test,
        logger,
        method="pca280_lda_1nn",
        input_dimension=int(X_train.shape[1]),
        intermediate_dimension=PCA_PRE_LDA_DIM,
        pca_fit_time_sec=pca_timing["fit_time_sec"],
        pca_pre_explained_variance=pca_pre_explained_variance,
        pca_fit_scope="X_train only",
    )
    failures = [*direct_failures, *pca_lda_failures]
    if not direct_rows or not pca_lda_rows:
        logger.error("At least one Stage 2 sweep produced no successful rows")

    stage2_rows = [*direct_rows, *pca_lda_rows]
    all_method_rows = [
        *baseline_to_all_method_rows(state.baseline_frame),
        *stage2_rows,
    ]
    stage2_frame = _write_csv(
        project_root / "results" / "tables" / "stage2_results.csv", stage2_rows
    )
    all_method_frame = _write_all_method_csv(
        project_root / "results" / "tables" / "all_method_results.csv",
        all_method_rows,
    )
    logger.info("Stage 2 result tables generated")
    generate_stage2_figures(
        state.baseline_frame,
        direct_rows,
        pca_lda_rows,
        project_root / "results" / "figures",
        logger,
    )

    raw_matches = state.baseline_frame[state.baseline_frame["method"] == "raw_1nn"]
    pca_matches = state.baseline_frame[state.baseline_frame["method"] == "pca_1nn"]
    raw_row = raw_matches.iloc[0].to_dict()
    best_pca_row = min(
        pca_matches.to_dict(orient="records"),
        key=lambda row: (-float(row["accuracy"]), int(row["dimension"])),
    )
    summary = build_stage2_summary(
        raw_row,
        best_pca_row,
        direct_rows,
        pca_lda_rows,
        pca_pre_explained_variance,
        failures,
    )
    write_json(project_root / "results" / "stage2_summary.json", summary)
    (project_root / "results" / "stage2_summary.txt").write_text(
        summary_text(summary, direct_rows, pca_lda_rows), encoding="utf-8"
    )
    write_stage2_notes(project_root, summary)
    logger.info("Stage 2 summary and report notes generated")

    baseline_after_hash = sha256_file(
        project_root / "results" / "tables" / "baseline_results.csv"
    )
    manifest = build_stage2_manifest(
        state,
        environment_audit,
        baseline_before_hash,
        baseline_after_hash,
        PCA_PRE_LDA_DIM,
        pca_pre_shapes,
        "X_train only",
        failures,
    )
    write_json(project_root / "audit" / "stage2_run_manifest.json", manifest)
    logger.info("Stage 2 run manifest generated")

    validation = validate_stage2_outputs(
        project_root,
        baseline_before_hash=baseline_before_hash,
        baseline_after_hash=baseline_after_hash,
        baseline_frame=state.baseline_frame,
        stage2_frame=stage2_frame,
        all_method_frame=all_method_frame,
        direct_rows=direct_rows,
        pca_lda_rows=pca_lda_rows,
        lda_dims=LDA_DIMS,
        dataset_audit=state.dataset_audit,
        split_audit=state.split_audit,
        pca_pre_dimension=PCA_PRE_LDA_DIM,
        pca_pre_shapes=pca_pre_shapes,
        pca_pre_fit_scope="X_train only",
        failures=failures,
        figure_names=STAGE2_FIGURE_NAMES,
    )
    validation_path = project_root / "audit" / "stage2_validation_report.txt"
    validation_json_path = project_root / "audit" / "stage2_validation_report.json"
    write_json(validation_json_path, validation)
    validation_path.write_text(validation_text(validation), encoding="utf-8")
    # The validator intentionally does not require its own file to exist before
    # computing the status; this final check proves both report files were written.
    report_files_written = validation_path.is_file() and validation_json_path.is_file()
    validation["checks"].append(
        {
            "name": "file:stage2_validation_reports",
            "passed": report_files_written,
            "detail": "text and JSON validation reports exist and are non-empty"
            if report_files_written
            else "validation report file write failed",
        }
    )
    validation["failed_checks"] = [
        item for item in validation["checks"] if not item["passed"]
    ]
    validation["passed"] = not validation["failed_checks"]
    validation["status"] = "PASS" if validation["passed"] else "FAIL"
    write_json(validation_json_path, validation)
    validation_path.write_text(validation_text(validation), encoding="utf-8")
    if validation["passed"]:
        logger.info("Stage 2 validation passed")
        return 0
    logger.error("Stage 2 validation failed: %s", validation["failed_checks"])
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Assignment project root (default: this file's parent project)",
    )
    return parser.parse_args()


def main() -> int:
    return run_experiment(parse_args().project_root)


if __name__ == "__main__":
    raise SystemExit(main())
