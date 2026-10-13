"""Run the fixed-seed Stage 3 robustness and PCA->LDA diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import sys
import time
import traceback
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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if __package__ in {None, ""}:  # support ``python src/run_stage3.py``
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_stage2 import load_stage1_state
    from stage2_validation import EXPECTED_STAGE1_ACCURACIES
    from stage3_analysis import (
        DIAGNOSTIC_LARGE_GAP,
        DIAGNOSTIC_NEAR_TOLERANCE,
        DIAGNOSTIC_RESIDUAL_GAP,
        aggregate_results,
        diagnostic_interpretation,
        make_paired_comparison,
        smallest_dimension_within_tolerance,
    )
    from utils import (
        PCA_DIMS as STAGE1_PCA_DIMS,
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
    from .run_stage2 import load_stage1_state
    from .stage2_validation import EXPECTED_STAGE1_ACCURACIES
    from .stage3_analysis import (
        DIAGNOSTIC_LARGE_GAP,
        DIAGNOSTIC_NEAR_TOLERANCE,
        DIAGNOSTIC_RESIDUAL_GAP,
        aggregate_results,
        diagnostic_interpretation,
        make_paired_comparison,
        smallest_dimension_within_tolerance,
    )
    from .utils import (
        PCA_DIMS as STAGE1_PCA_DIMS,
        RANDOM_STATE,
        compute_error_rate,
        git_commit_hash,
        package_versions,
        relative_dimensionality_reduction,
        sha256_file,
        source_sha256,
        write_json,
    )


ROBUSTNESS_SEEDS: tuple[int, ...] = tuple(range(100, 120))
PCA_DIMS: tuple[int, ...] = STAGE1_PCA_DIMS
LDA_DIMS: tuple[int, ...] = (1, 2, 5, 10, 15, 20, 25, 30, 35, 39)
PCA_PRE_DIMS: tuple[int, ...] = (280, 319)
PCA_PRE_LDA_DIM = 280
PCA319_PRE_LDA_DIM = 319
LDA_SOLVER = "svd"
CLASSIFIER_NAME = "KNeighborsClassifier(n_neighbors=1)"
METRIC_NAME = "euclidean"
REPRESENTATIVE_CONFIGS: dict[str, dict[str, Any]] = {
    "raw": {
        "method": "raw_1nn",
        "output_dimension": 4096,
        "intermediate_dimension": None,
    },
    "pca75": {
        "method": "pca_1nn",
        "output_dimension": 75,
        "intermediate_dimension": None,
    },
    "lda25": {
        "method": "lda_1nn",
        "output_dimension": 25,
        "intermediate_dimension": None,
    },
    "pca280_lda20": {
        "method": "pca280_lda_1nn",
        "output_dimension": 20,
        "intermediate_dimension": 280,
    },
}
PAIRED_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("PCA75 - Raw", "pca75", "raw"),
    ("LDA25 - Raw", "lda25", "raw"),
    ("PCA280->LDA20 - Raw", "pca280_lda20", "raw"),
    (
        "PCA319->LDA25 - Direct LDA25",
        "pca319_lda25",
        "lda25",
    ),
    ("PCA280-only - Raw", "pca280_only", "raw"),
)
FIGURE_NAMES = (
    "stage3_pca_mean_accuracy.png",
    "stage3_lda_mean_accuracy.png",
    "stage3_main_method_comparison.png",
    "stage3_lda25_vs_raw_paired.png",
    "stage3_pca_lda_diagnostic.png",
)
RESULT_COLUMNS = [
    "seed",
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
    "train_size",
    "test_size",
    "preprocessing_fit_scope",
    "pca_fit_scope",
    "fit_strategy",
    "train_output_shape",
    "test_output_shape",
    "train_index_hash",
    "test_index_hash",
    "split_id",
]


@dataclass(frozen=True)
class SeedRun:
    """Rows and split evidence produced by one paired seed."""

    seed: int
    rows: list[dict[str, Any]]
    split_audit: dict[str, Any]
    failures: list[dict[str, Any]]


def configure_logging(log_path: Path) -> logging.Logger:
    """Create a concise Stage 3 file-and-console logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ee6222.assignment1.stage3")
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


def _index_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def build_seed_split(y: np.ndarray, seed: int) -> dict[str, Any]:
    """Build one deterministic stratified split and audit its class counts."""
    labels = np.asarray(y)
    indices = np.arange(labels.size, dtype=np.int64)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.20,
        stratify=labels,
        random_state=int(seed),
    )
    class_counts = {
        str(label): {
            "train": int(np.sum(labels[train_indices] == label)),
            "test": int(np.sum(labels[test_indices] == label)),
        }
        for label in np.unique(labels)
    }
    checks = {
        "no_index_overlap": bool(np.intersect1d(train_indices, test_indices).size == 0),
        "sizes_sum_to_total": bool(train_indices.size + test_indices.size == labels.size),
        "train_size_320": bool(train_indices.size == 320),
        "test_size_80": bool(test_indices.size == 80),
        "all_classes_have_8_train_2_test": bool(
            all(item["train"] == 8 and item["test"] == 2 for item in class_counts.values())
        ),
    }
    return {
        "seed": int(seed),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_size": int(train_indices.size),
        "test_size": int(test_indices.size),
        "class_counts": class_counts,
        "train_index_hash": _index_hash(train_indices),
        "test_index_hash": _index_hash(test_indices),
        "checks": checks,
        "overall_pass": bool(all(checks.values())),
    }


def _result_row(
    *,
    seed: int,
    method: str,
    input_dimension: int,
    intermediate_dimension: int | None,
    output_dimension: int,
    accuracy: float,
    n_correct: int,
    n_incorrect: int,
    solver: str | None,
    pca_fit_time_sec: float | None,
    lda_fit_time_sec: float | None,
    transform_time_sec: float,
    classifier_fit_time_sec: float,
    prediction_time_sec: float,
    total_time_sec: float,
    train_size: int,
    test_size: int,
    explained_variance: float | None,
    pca_pre_explained_variance: float | None,
    pca_fit_scope: str | None,
    fit_strategy: str,
    train_output_shape: list[int],
    test_output_shape: list[int],
    split_audit: dict[str, Any],
) -> dict[str, Any]:
    pca_fit_value = float(pca_fit_time_sec or 0.0)
    lda_fit_value = float(lda_fit_time_sec or 0.0)
    return {
        "seed": int(seed),
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
        "explained_variance": (
            float(explained_variance) if explained_variance is not None else None
        ),
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
        "lda_fit_time_sec": (
            float(lda_fit_time_sec) if lda_fit_time_sec is not None else None
        ),
        "transform_time_sec": float(transform_time_sec),
        "classifier_fit_time_sec": float(classifier_fit_time_sec),
        "prediction_time_sec": float(prediction_time_sec),
        "fit_time_sec": float(pca_fit_value + lda_fit_value + classifier_fit_time_sec),
        "predict_time_sec": float(transform_time_sec + prediction_time_sec),
        "total_time_sec": float(total_time_sec),
        "train_size": int(train_size),
        "test_size": int(test_size),
        "preprocessing_fit_scope": "X_train only" if solver != "none" else "none",
        "pca_fit_scope": pca_fit_scope,
        "fit_strategy": fit_strategy,
        "train_output_shape": [int(value) for value in train_output_shape],
        "test_output_shape": [int(value) for value in test_output_shape],
        "train_index_hash": split_audit["train_index_hash"],
        "test_index_hash": split_audit["test_index_hash"],
        "split_id": f"seed_{int(seed)}",
    }


def _evaluate_knn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, int, float, float]:
    classifier = KNeighborsClassifier(n_neighbors=1, metric=METRIC_NAME)
    fit_start = time.perf_counter()
    classifier.fit(X_train, y_train)
    fit_time = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    predictions = classifier.predict(X_test)
    prediction_time = time.perf_counter() - predict_start
    accuracy = float(accuracy_score(y_test, predictions))
    return accuracy, int(np.sum(predictions == y_test)), float(fit_time), float(prediction_time)


def _run_raw(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    split_audit: dict[str, Any],
) -> dict[str, Any]:
    total_start = time.perf_counter()
    accuracy, n_correct, classifier_fit, prediction_time = _evaluate_knn(
        X_train, X_test, y_train, y_test
    )
    total = time.perf_counter() - total_start
    return _result_row(
        seed=seed,
        method="raw_1nn",
        input_dimension=X_train.shape[1],
        intermediate_dimension=None,
        output_dimension=X_train.shape[1],
        accuracy=accuracy,
        n_correct=n_correct,
        n_incorrect=y_test.size - n_correct,
        solver="none",
        pca_fit_time_sec=None,
        lda_fit_time_sec=None,
        transform_time_sec=0.0,
        classifier_fit_time_sec=classifier_fit,
        prediction_time_sec=prediction_time,
        total_time_sec=total,
        train_size=y_train.size,
        test_size=y_test.size,
        explained_variance=None,
        pca_pre_explained_variance=None,
        pca_fit_scope=None,
        fit_strategy="no preprocessing",
        train_output_shape=list(X_train.shape),
        test_output_shape=list(X_test.shape),
        split_audit=split_audit,
    )


def _run_pca_dimension(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    dimension: int,
    split_audit: dict[str, Any],
) -> dict[str, Any]:
    total_start = time.perf_counter()
    pca = PCA(n_components=dimension, svd_solver="full")
    fit_start = time.perf_counter()
    pca.fit(X_train)
    pca_fit_time = time.perf_counter() - fit_start
    transform_start = time.perf_counter()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    transform_time = time.perf_counter() - transform_start
    if X_train_pca.shape != (X_train.shape[0], dimension) or X_test_pca.shape != (
        X_test.shape[0],
        dimension,
    ):
        raise RuntimeError(f"PCA shape mismatch at d={dimension}: {X_train_pca.shape}/{X_test_pca.shape}")
    accuracy, n_correct, classifier_fit, prediction_time = _evaluate_knn(
        X_train_pca, X_test_pca, y_train, y_test
    )
    return _result_row(
        seed=seed,
        method="pca_1nn",
        input_dimension=X_train.shape[1],
        intermediate_dimension=None,
        output_dimension=dimension,
        accuracy=accuracy,
        n_correct=n_correct,
        n_incorrect=y_test.size - n_correct,
        solver="pca_full",
        pca_fit_time_sec=pca_fit_time,
        lda_fit_time_sec=None,
        transform_time_sec=transform_time,
        classifier_fit_time_sec=classifier_fit,
        prediction_time_sec=prediction_time,
        total_time_sec=time.perf_counter() - total_start,
        train_size=y_train.size,
        test_size=y_test.size,
        explained_variance=float(np.sum(pca.explained_variance_ratio_)),
        pca_pre_explained_variance=None,
        pca_fit_scope="X_train only",
        fit_strategy="independent PCA fit per dimension per seed",
        train_output_shape=list(X_train_pca.shape),
        test_output_shape=list(X_test_pca.shape),
        split_audit=split_audit,
    )


def _run_lda_curve(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    split_audit: dict[str, Any],
    *,
    method: str,
    input_dimension: int,
    intermediate_dimension: int | None = None,
    pca_fit_time_sec: float | None = None,
    pca_pre_explained_variance: float | None = None,
    pca_fit_scope: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dimension in LDA_DIMS:
        total_start = time.perf_counter()
        try:
            maximum = min(X_train.shape[1], np.unique(y_train).size - 1)
            if dimension < 1 or dimension > maximum:
                raise ValueError(f"illegal LDA dimension {dimension}; legal maximum={maximum}")
            lda = LinearDiscriminantAnalysis(solver=LDA_SOLVER, n_components=dimension)
            lda_fit_start = time.perf_counter()
            lda.fit(X_train, y_train)
            lda_fit_time = time.perf_counter() - lda_fit_start
            transform_start = time.perf_counter()
            X_train_lda = lda.transform(X_train)
            X_test_lda = lda.transform(X_test)
            transform_time = time.perf_counter() - transform_start
            if X_train_lda.shape != (X_train.shape[0], dimension) or X_test_lda.shape != (
                X_test.shape[0],
                dimension,
            ):
                raise RuntimeError(
                    f"LDA shape mismatch at d={dimension}: "
                    f"{X_train_lda.shape}/{X_test_lda.shape}"
                )
            if not np.isfinite(X_train_lda).all() or not np.isfinite(X_test_lda).all():
                raise RuntimeError(f"LDA non-finite output at d={dimension}")
            accuracy, n_correct, classifier_fit, prediction_time = _evaluate_knn(
                X_train_lda, X_test_lda, y_train, y_test
            )
            rows.append(
                _result_row(
                    seed=seed,
                    method=method,
                    input_dimension=input_dimension,
                    intermediate_dimension=intermediate_dimension,
                    output_dimension=dimension,
                    accuracy=accuracy,
                    n_correct=n_correct,
                    n_incorrect=y_test.size - n_correct,
                    solver=LDA_SOLVER,
                    pca_fit_time_sec=pca_fit_time_sec,
                    lda_fit_time_sec=lda_fit_time,
                    transform_time_sec=transform_time,
                    classifier_fit_time_sec=classifier_fit,
                    prediction_time_sec=prediction_time,
                    total_time_sec=time.perf_counter() - total_start + float(pca_fit_time_sec or 0.0),
                    train_size=y_train.size,
                    test_size=y_test.size,
                    explained_variance=None,
                    pca_pre_explained_variance=pca_pre_explained_variance,
                    pca_fit_scope=pca_fit_scope,
                    fit_strategy=(
                        "independent LDA fit per dimension per seed"
                        if pca_fit_time_sec is None
                        else "one PCA fit per seed; independent LDA fit per dimension"
                    ),
                    train_output_shape=list(X_train_lda.shape),
                    test_output_shape=list(X_test_lda.shape),
                    split_audit=split_audit,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "seed": int(seed),
                    "method": method,
                    "dimension": int(dimension),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    return rows, failures


def _fit_pca_control(
    X_train: np.ndarray,
    X_test: np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, PCA, dict[str, float]]:
    maximum = min(X_train.shape[0], X_train.shape[1])
    if dimension < 1 or dimension > maximum:
        raise ValueError(f"PCA control dimension {dimension} is illegal; maximum={maximum}")
    pca = PCA(n_components=dimension, svd_solver="full")
    fit_start = time.perf_counter()
    pca.fit(X_train)
    fit_time = time.perf_counter() - fit_start
    transform_start = time.perf_counter()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    transform_time = time.perf_counter() - transform_start
    if X_train_pca.shape != (X_train.shape[0], dimension) or X_test_pca.shape != (
        X_test.shape[0],
        dimension,
    ):
        raise RuntimeError(f"PCA control shape mismatch at d={dimension}")
    if not np.isfinite(X_train_pca).all() or not np.isfinite(X_test_pca).all():
        raise RuntimeError(f"PCA control non-finite output at d={dimension}")
    return X_train_pca, X_test_pca, pca, {
        "fit_time_sec": float(fit_time),
        "transform_time_sec": float(transform_time),
    }


def _run_pca_only_control(
    X_train_pca: np.ndarray,
    X_test_pca: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    split_audit: dict[str, Any],
    *,
    dimension: int,
    pca: PCA,
    pca_timing: dict[str, float],
) -> dict[str, Any]:
    total_start = time.perf_counter()
    accuracy, n_correct, classifier_fit, prediction_time = _evaluate_knn(
        X_train_pca, X_test_pca, y_train, y_test
    )
    return _result_row(
        seed=seed,
        method=f"pca{dimension}_1nn",
        input_dimension=4096,
        intermediate_dimension=None,
        output_dimension=dimension,
        accuracy=accuracy,
        n_correct=n_correct,
        n_incorrect=y_test.size - n_correct,
        solver="pca_full",
        pca_fit_time_sec=pca_timing["fit_time_sec"],
        lda_fit_time_sec=None,
        transform_time_sec=pca_timing["transform_time_sec"],
        classifier_fit_time_sec=classifier_fit,
        prediction_time_sec=prediction_time,
        total_time_sec=time.perf_counter() - total_start + pca_timing["fit_time_sec"],
        train_size=y_train.size,
        test_size=y_test.size,
        explained_variance=float(np.sum(pca.explained_variance_ratio_)),
        pca_pre_explained_variance=None,
        pca_fit_scope="X_train only",
        fit_strategy=f"one PCA({dimension}) fit per seed; no LDA",
        train_output_shape=list(X_train_pca.shape),
        test_output_shape=list(X_test_pca.shape),
        split_audit=split_audit,
    )


def run_seed(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    logger: logging.Logger,
) -> SeedRun:
    """Run all 44 paired rows for one seed with fresh current-split models."""
    split_audit = build_seed_split(y, seed)
    X_train = X[split_audit["train_indices"]]
    X_test = X[split_audit["test_indices"]]
    y_train = y[split_audit["train_indices"]]
    y_test = y[split_audit["test_indices"]]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    rows.append(_run_raw(X_train, X_test, y_train, y_test, seed, split_audit))
    for dimension in PCA_DIMS:
        try:
            rows.append(
                _run_pca_dimension(
                    X_train, X_test, y_train, y_test, seed, dimension, split_audit
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "seed": int(seed),
                    "method": "pca_1nn",
                    "dimension": int(dimension),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    direct_rows, direct_failures = _run_lda_curve(
        X_train,
        X_test,
        y_train,
        y_test,
        seed,
        split_audit,
        method="lda_1nn",
        input_dimension=X_train.shape[1],
    )
    rows.extend(direct_rows)
    failures.extend(direct_failures)

    try:
        X_train_280, X_test_280, pca280, pca280_timing = _fit_pca_control(
            X_train, X_test, PCA_PRE_LDA_DIM
        )
        rows.append(
            _run_pca_only_control(
                X_train_280,
                X_test_280,
                y_train,
                y_test,
                seed,
                split_audit,
                dimension=PCA_PRE_LDA_DIM,
                pca=pca280,
                pca_timing=pca280_timing,
            )
        )
        pca280_rows, pca280_failures = _run_lda_curve(
            X_train_280,
            X_test_280,
            y_train,
            y_test,
            seed,
            split_audit,
            method="pca280_lda_1nn",
            input_dimension=X_train.shape[1],
            intermediate_dimension=PCA_PRE_LDA_DIM,
            pca_fit_time_sec=pca280_timing["fit_time_sec"],
            pca_pre_explained_variance=float(np.sum(pca280.explained_variance_ratio_)),
            pca_fit_scope="X_train only",
        )
        rows.extend(pca280_rows)
        failures.extend(pca280_failures)
    except Exception as exc:
        failures.append(
            {
                "seed": int(seed),
                "method": "pca280_control",
                "dimension": PCA_PRE_LDA_DIM,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    try:
        X_train_319, X_test_319, pca319, pca319_timing = _fit_pca_control(
            X_train, X_test, PCA319_PRE_LDA_DIM
        )
        pca319_rows, pca319_failures = _run_lda_curve(
            X_train_319,
            X_test_319,
            y_train,
            y_test,
            seed,
            split_audit,
            method="pca319_lda_1nn",
            input_dimension=X_train.shape[1],
            intermediate_dimension=PCA319_PRE_LDA_DIM,
            pca_fit_time_sec=pca319_timing["fit_time_sec"],
            pca_pre_explained_variance=float(np.sum(pca319.explained_variance_ratio_)),
            pca_fit_scope="X_train only",
        )
        rows.extend(pca319_rows)
        failures.extend(pca319_failures)
    except Exception as exc:
        failures.append(
            {
                "seed": int(seed),
                "method": "pca319_control",
                "dimension": PCA319_PRE_LDA_DIM,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    logger.info(
        "seed=%d complete: rows=%d failures=%d train_hash=%s test_hash=%s",
        seed,
        len(rows),
        len(failures),
        split_audit["train_index_hash"][:12],
        split_audit["test_index_hash"][:12],
    )
    return SeedRun(seed=seed, rows=rows, split_audit=split_audit, failures=failures)


def _config_stats(aggregate: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    mask = (aggregate["method"] == config["method"]) & (
        aggregate["output_dimension"] == config["output_dimension"]
    )
    intermediate = config.get("intermediate_dimension")
    if intermediate is None:
        mask &= aggregate["intermediate_dimension"].isna()
    else:
        mask &= aggregate["intermediate_dimension"] == intermediate
    matches = aggregate.loc[mask]
    if len(matches) != 1:
        raise ValueError(f"representative configuration has {len(matches)} rows: {config}")
    row = matches.iloc[0].to_dict()
    # Pandas promotes a column containing None to float NaN. Keep JSON output
    # standards-compliant by converting only scalar NaNs to None at the
    # serialization boundary; CSV aggregation retains its conventional blanks.
    for key, value in list(row.items()):
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            row[key] = None
    return row


def _config_from_summary_key(key: str) -> dict[str, Any]:
    if key == "pca319_lda25":
        return {"method": "pca319_lda_1nn", "output_dimension": 25, "intermediate_dimension": 319}
    if key == "pca280_only":
        return {"method": "pca280_1nn", "output_dimension": 280, "intermediate_dimension": None}
    return REPRESENTATIVE_CONFIGS[key]


def build_stage3_summary(
    aggregate: pd.DataFrame,
    paired: list[dict[str, Any]],
    runtime_sec: float,
    failures: list[dict[str, Any]],
    actual_rows: int,
) -> dict[str, Any]:
    """Build the final descriptive summary and conservative diagnostic interpretation."""
    representative = {
        key: _config_stats(aggregate, config)
        for key, config in REPRESENTATIVE_CONFIGS.items()
    }
    diagnostic_keys = {
        "pca280_only": _config_from_summary_key("pca280_only"),
        "pca319_lda25": _config_from_summary_key("pca319_lda25"),
    }
    diagnostic_stats = {
        key: _config_stats(aggregate, config) for key, config in diagnostic_keys.items()
    }
    metrics = {
        "raw_mean_accuracy": float(representative["raw"]["mean_accuracy"]),
        "pca280_only_mean_accuracy": float(diagnostic_stats["pca280_only"]["mean_accuracy"]),
        "direct_lda25_mean_accuracy": float(representative["lda25"]["mean_accuracy"]),
        "pca280_lda20_mean_accuracy": float(representative["pca280_lda20"]["mean_accuracy"]),
        "pca319_lda25_mean_accuracy": float(diagnostic_stats["pca319_lda25"]["mean_accuracy"]),
    }
    curve_methods = {
        "pca": "pca_1nn",
        "direct_lda": "lda_1nn",
        "pca280_lda": "pca280_lda_1nn",
        "pca319_lda": "pca319_lda_1nn",
    }
    curve_findings: dict[str, Any] = {}
    for label, method in curve_methods.items():
        subset = aggregate[aggregate["method"] == method]
        max_mean = float(subset["mean_accuracy"].max())
        peak_dims = [
            int(value)
            for value in subset.loc[
                np.isclose(subset["mean_accuracy"], max_mean), "output_dimension"
            ]
        ]
        curve_findings[label] = {
            "maximum_mean_accuracy": max_mean,
            "peak_dimensions": peak_dims,
            "smallest_dimension_within_1_percentage_point": smallest_dimension_within_tolerance(
                aggregate, method, tolerance=0.01
            ),
        }

    return {
        "experiment": "EE6222 Assignment 1 - Stage 3 Robustness and Diagnostics",
        "protocol": {
            "split_type": "20 repeated stratified train/test splits",
            "seeds": list(ROBUSTNESS_SEEDS),
            "train_size": 320,
            "test_size": 80,
            "paired_design": "all methods share the exact split for each seed",
            "pca_dimensions": list(PCA_DIMS),
            "lda_dimensions": list(LDA_DIMS),
            "pca_preprocessing_controls": list(PCA_PRE_DIMS),
            "randomness_source": "train_test_split random_state only",
        },
        "representative_configurations": representative,
        "diagnostic_controls": diagnostic_stats,
        "curve_findings": curve_findings,
        "paired_comparisons": paired,
        "diagnostic_metrics": metrics,
        "diagnostic_interpretation": diagnostic_interpretation(metrics),
        "diagnostic_thresholds": {
            "near_tolerance": DIAGNOSTIC_NEAR_TOLERANCE,
            "large_gap": DIAGNOSTIC_LARGE_GAP,
            "residual_gap": DIAGNOSTIC_RESIDUAL_GAP,
        },
        "runtime": {
            "total_sec": float(runtime_sec),
            "mean_per_seed_sec": float(runtime_sec / len(ROBUSTNESS_SEEDS)),
        },
        "row_counts": {
            "expected": 880,
            "actual": int(actual_rows),
            "failures": int(len(failures)),
        },
        "statistical_caution": (
            "These are 20 repeated stratified train/test splits, not 20 independent datasets. "
            "The test set contains 80 samples, so one prediction changes accuracy by 1.25 "
            "percentage points; paired differences and confidence intervals are descriptive."
        ),
        "freeze_recommendation": "PENDING_FINAL_VALIDATION",
        "validation_status": "PENDING",
    }


def summary_text(summary: dict[str, Any]) -> str:
    """Render a compact Stage 3 handoff."""
    lines = [
        "EE6222 Assignment 1 - Stage 3 Summary",
        "",
        "Protocol:",
        f"  seeds: {summary['protocol']['seeds']}",
        "  split: 20 repeated stratified train/test splits, 320 train / 80 test",
        "  paired design: all methods share the exact split for each seed",
        f"  rows: {summary['row_counts']['actual']} / {summary['row_counts']['expected']}",
        "",
        "Representative configurations (mean accuracy, sample SD, 95% t CI, min, max):",
    ]
    for label, stats_row in summary["representative_configurations"].items():
        lines.append(
            f"  {label}: mean={stats_row['mean_accuracy']:.6f}, "
            f"std={stats_row['std_accuracy']:.6f}, "
            f"CI95=[{stats_row['ci95_lower']:.6f}, {stats_row['ci95_upper']:.6f}], "
            f"min={stats_row['min_accuracy']:.6f}, max={stats_row['max_accuracy']:.6f}"
        )
    lines.extend(["", "Paired comparisons (left minus right):"])
    for comparison in summary["paired_comparisons"]:
        lines.append(
            f"  {comparison['comparison']}: mean_delta={comparison['mean_delta']:+.6f}, "
            f"wins/ties/losses={comparison['wins']}/{comparison['ties']}/{comparison['losses']}"
        )
    lines.extend(["", "Curve findings:"])
    for label, finding in summary["curve_findings"].items():
        lines.append(
            f"  {label}: max_mean={finding['maximum_mean_accuracy']:.6f}, "
            f"peak_dimensions={finding['peak_dimensions']}, "
            f"smallest within 1pp={finding['smallest_dimension_within_1_percentage_point']}"
        )
    lines.extend(["", "PCA->LDA diagnostic controls:"])
    for key, stats_row in summary["diagnostic_controls"].items():
        lines.append(
            f"  {key}: mean={stats_row['mean_accuracy']:.6f}, "
            f"std={stats_row['std_accuracy']:.6f}, "
            f"CI95=[{stats_row['ci95_lower']:.6f}, {stats_row['ci95_upper']:.6f}]"
        )
    lines.extend(["", "Diagnostic interpretation:"])
    lines.extend(f"  - {statement}" for statement in summary["diagnostic_interpretation"])
    lines.extend(
        [
            "",
            "Statistical caution:",
            f"  {summary['statistical_caution']}",
            "",
            f"Validation: {summary['validation_status']}",
            f"Freeze recommendation: {summary['freeze_recommendation']}",
        ]
    )
    if summary["row_counts"]["failures"]:
        lines.append(f"Failures: {summary['row_counts']['failures']}")
    return "\n".join(lines) + "\n"


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_figures(
    aggregate: pd.DataFrame,
    raw_mean: float,
    paired: list[dict[str, Any]],
    raw_frame: pd.DataFrame,
    figures_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate the five required Stage 3 figures."""
    pca = aggregate[aggregate["method"] == "pca_1nn"]
    direct = aggregate[aggregate["method"] == "lda_1nn"]
    pca280 = aggregate[aggregate["method"] == "pca280_lda_1nn"]
    pca319 = aggregate[aggregate["method"] == "pca319_lda_1nn"]
    fig, ax = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    ax.plot(pca["output_dimension"], pca["mean_accuracy"], marker="o", linewidth=2, label="PCA + 1-NN")
    ax.fill_between(
        pca["output_dimension"].to_numpy(dtype=float),
        (pca["mean_accuracy"] - pca["std_accuracy"]).to_numpy(dtype=float),
        (pca["mean_accuracy"] + pca["std_accuracy"]).to_numpy(dtype=float),
        alpha=0.2,
        label="PCA mean ± 1 SD",
    )
    ax.axhline(raw_mean, color="tab:red", linestyle="--", linewidth=1.8, label="Raw mean accuracy")
    ax.set_xlabel("Reduced Dimensionality")
    ax.set_ylabel("Mean Test Classification Accuracy")
    ax.set_title("PCA Mean Accuracy Across 20 Stratified Splits")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, figures_dir / "stage3_pca_mean_accuracy.png")

    fig, ax = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    for frame, label, marker in (
        (direct, "Direct LDA", "o"),
        (pca280, "PCA(280) -> LDA", "s"),
        (pca319, "PCA(319) -> LDA", "^"),
    ):
        ax.plot(frame["output_dimension"], frame["mean_accuracy"], marker=marker, linewidth=2, label=label)
        ax.fill_between(
            frame["output_dimension"].to_numpy(dtype=float),
            (frame["mean_accuracy"] - frame["std_accuracy"]).to_numpy(dtype=float),
            (frame["mean_accuracy"] + frame["std_accuracy"]).to_numpy(dtype=float),
            alpha=0.12,
        )
    ax.axhline(raw_mean, color="tab:red", linestyle="--", linewidth=1.8, label="Raw mean accuracy")
    ax.set_xlabel("Final LDA Dimensionality")
    ax.set_ylabel("Mean Test Classification Accuracy")
    ax.set_title("LDA Mean Accuracy Across 20 Stratified Splits")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(list(LDA_DIMS))
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, figures_dir / "stage3_lda_mean_accuracy.png")

    representative_labels = ["Raw 4096D", "PCA75", "LDA25", "PCA280->LDA20"]
    representative_keys = ["raw", "pca75", "lda25", "pca280_lda20"]
    representative_stats = [
        _config_stats(aggregate, REPRESENTATIVE_CONFIGS[key]) for key in representative_keys
    ]
    fig, ax = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    means = [float(row["mean_accuracy"]) for row in representative_stats]
    stds = [float(row["std_accuracy"]) for row in representative_stats]
    ax.errorbar(
        np.arange(len(means)),
        means,
        yerr=stds,
        fmt="o",
        capsize=5,
        linewidth=2,
        markersize=8,
        color="tab:blue",
    )
    ax.set_xticks(np.arange(len(means)), representative_labels)
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Representative Configuration Comparison (Mean ± 1 SD)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    _save_figure(fig, figures_dir / "stage3_main_method_comparison.png")

    raw_rep = raw_frame[raw_frame["method"] == "raw_1nn"]
    lda_rep = raw_frame[(raw_frame["method"] == "lda_1nn") & (raw_frame["output_dimension"] == 25)]
    paired_frame = raw_rep[["seed", "accuracy"]].rename(columns={"accuracy": "raw_accuracy"}).merge(
        lda_rep[["seed", "accuracy"]].rename(columns={"accuracy": "lda_accuracy"}),
        on="seed",
        validate="one_to_one",
    )
    fig, ax = plt.subplots(figsize=(7.0, 6.5), constrained_layout=True)
    ax.scatter(paired_frame["raw_accuracy"], paired_frame["lda_accuracy"], s=55, alpha=0.85)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="tab:red", label="y = x")
    ax.set_xlabel("Raw 4096D Accuracy")
    ax.set_ylabel("Direct LDA25 Accuracy")
    ax.set_title("Paired LDA25 vs Raw Across 20 Splits")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, figures_dir / "stage3_lda25_vs_raw_paired.png")

    fig, ax = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    for frame, label, marker in (
        (direct, "Direct LDA", "o"),
        (pca280, "PCA(280) -> LDA", "s"),
        (pca319, "PCA(319) -> LDA", "^"),
    ):
        ax.plot(frame["output_dimension"], frame["mean_accuracy"], marker=marker, linewidth=2, label=label)
    pca280_only = aggregate[aggregate["method"] == "pca280_1nn"]
    ax.errorbar(
        [280],
        [float(pca280_only["mean_accuracy"].iloc[0])],
        yerr=[float(pca280_only["std_accuracy"].iloc[0])],
        fmt="D",
        color="tab:brown",
        capsize=5,
        label="PCA280-only + 1-NN",
    )
    ax.axhline(raw_mean, color="tab:red", linestyle="--", linewidth=1.8, label="Raw mean accuracy")
    ax.set_xlabel("Final Representation / LDA Dimensionality")
    ax.set_ylabel("Mean Test Classification Accuracy")
    ax.set_title("PCA->LDA Diagnostic Controls")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.0, 300.0)
    ax.set_xticks([0, 25, 50, 100, 200, 280])
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, figures_dir / "stage3_pca_lda_diagnostic.png")
    logger.info("Stage 3 figures generated under %s", figures_dir)


def write_stage3_notes(project_root: Path, summary: dict[str, Any]) -> None:
    """Append Stage 3 facts while preserving Stage 1 and Stage 2 notes."""
    notes_path = project_root / "report" / "notes.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    marker = "\n## Stage 3 – Robustness\n"
    prefix = existing.split(marker, maxsplit=1)[0].rstrip()
    lines = [
        "## Stage 3 – Robustness",
        "",
        "Protocol: 20 repeated stratified train/test splits.",
        f"Seeds: {summary['protocol']['seeds']}.",
        "Repeated stratified holdout: 320 train / 80 test per seed.",
        "Paired design: every method uses the same split indices for a given seed.",
        "",
        "Raw robustness:",
        f"- mean={summary['representative_configurations']['raw']['mean_accuracy']:.6f}, "
        f"std={summary['representative_configurations']['raw']['std_accuracy']:.6f}.",
        "PCA robustness:",
        f"- predefined PCA75 mean={summary['representative_configurations']['pca75']['mean_accuracy']:.6f}, "
        f"std={summary['representative_configurations']['pca75']['std_accuracy']:.6f}.",
        "Direct LDA robustness:",
        f"- predefined LDA25 mean={summary['representative_configurations']['lda25']['mean_accuracy']:.6f}, "
        f"std={summary['representative_configurations']['lda25']['std_accuracy']:.6f}.",
        "PCA280→LDA robustness:",
        f"- predefined PCA280→LDA20 mean={summary['representative_configurations']['pca280_lda20']['mean_accuracy']:.6f}, "
        f"std={summary['representative_configurations']['pca280_lda20']['std_accuracy']:.6f}.",
        "",
        "Paired comparisons:",
        *[
            f"- {item['comparison']}: mean delta={item['mean_delta']:+.6f}; "
            f"wins/ties/losses={item['wins']}/{item['ties']}/{item['losses']}."
            for item in summary["paired_comparisons"]
        ],
        "",
        "Diagnostic controls:",
        f"- PCA280-only mean={summary['diagnostic_controls']['pca280_only']['mean_accuracy']:.6f}.",
        f"- PCA319→LDA25 mean={summary['diagnostic_controls']['pca319_lda25']['mean_accuracy']:.6f}.",
        *[f"- {statement}" for statement in summary["diagnostic_interpretation"]],
        "",
        "Interpretation: Stage 3 statements are descriptive evidence from repeated paired "
        "splits, not proof of causality or performance on independent datasets.",
        f"Limitations: {summary['statistical_caution']}",
        "Final recommended report findings: use the paired representative comparison, "
        "the mean curves with SD bands, and the PCA280/PCA319 diagnostic contrast.",
        "",
        f"Freeze decision: {summary['freeze_recommendation']}.",
    ]
    notes_path.write_text(
        (prefix + "\n\n" if prefix else "") + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _hash_files(project_root: Path, relative_paths: Iterable[str]) -> dict[str, str]:
    return {
        relative: sha256_file(project_root / relative)
        for relative in relative_paths
        if (project_root / relative).is_file()
    }


PRESERVED_PATHS = (
    "results/tables/baseline_results.csv",
    "results/tables/pca_explained_variance.csv",
    "results/tables/stage2_results.csv",
    "results/tables/all_method_results.csv",
    "results/baseline_summary.txt",
    "results/baseline_summary.json",
    "results/stage2_summary.txt",
    "results/stage2_summary.json",
    "results/figures/pca_accuracy_vs_dimension.png",
    "results/figures/pca_error_rate_vs_dimension.png",
    "results/figures/pca_cumulative_explained_variance.png",
    "results/figures/lda_accuracy_vs_dimension.png",
    "results/figures/method_comparison_accuracy.png",
    "results/figures/method_comparison_error_rate.png",
    "audit/run_manifest.json",
    "audit/stage2_run_manifest.json",
    "audit/validation_report.txt",
    "audit/stage2_validation_report.txt",
)


def run_experiment(project_root: Path) -> int:
    """Execute all fixed Stage 3 seeds and finalize artifacts."""
    project_root = project_root.resolve()
    logger = configure_logging(project_root / "results" / "logs" / "stage3_run.log")
    logger.info("Starting EE6222 Stage 3 CPU-only robustness experiment")
    before_hashes = _hash_files(project_root, PRESERVED_PATHS)
    try:
        state = load_stage1_state(project_root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.exception("Frozen Stage 1 gate failed; Stage 3 not started")
        return 2
    if set(before_hashes) != set(PRESERVED_PATHS):
        logger.error("Required Stage 1/Stage 2 preservation files are missing")
        return 2
    logger.info("Frozen Stage 1/Stage 2 hashes captured: %d files", len(before_hashes))
    start = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    split_audits: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    progress_path = project_root / "audit" / "stage3_progress.json"
    progress_csv_path = project_root / "results" / "tables" / "stage3_robustness_progress.csv"
    for index, seed in enumerate(ROBUSTNESS_SEEDS, start=1):
        try:
            result = run_seed(state.X, state.y, seed, logger)
            all_rows.extend(result.rows)
            split_audits.append(result.split_audit)
            failures.extend(result.failures)
        except Exception as exc:
            failure = {
                "seed": int(seed),
                "method": "seed_run",
                "dimension": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            logger.exception("seed=%d failed at seed-run level; continuing", seed)
        progress = {
            "status": "RUNNING",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed_seed_count": index,
            "completed_seeds": list(ROBUSTNESS_SEEDS[:index]),
            "actual_rows": len(all_rows),
            "failure_count": len(failures),
        }
        write_json(progress_path, progress)
        pd.DataFrame(all_rows, columns=RESULT_COLUMNS).to_csv(
            progress_csv_path, index=False, float_format="%.12g"
        )
        logger.info(
            "progress %d/%d seeds: rows=%d failures=%d",
            index,
            len(ROBUSTNESS_SEEDS),
            len(all_rows),
            len(failures),
        )

    runtime_sec = time.perf_counter() - start
    raw_frame = pd.DataFrame(all_rows, columns=RESULT_COLUMNS)
    raw_path = project_root / "results" / "tables" / "stage3_robustness_raw.csv"
    raw_frame.to_csv(raw_path, index=False, float_format="%.12g")
    aggregate = aggregate_results(raw_frame) if not raw_frame.empty else pd.DataFrame()
    aggregate_path = project_root / "results" / "tables" / "stage3_robustness_summary.csv"
    aggregate.to_csv(aggregate_path, index=False, float_format="%.12g")

    paired: list[dict[str, Any]] = []
    paired_configs = dict(REPRESENTATIVE_CONFIGS)
    paired_configs.update(
        {
            "pca319_lda25": _config_from_summary_key("pca319_lda25"),
            "pca280_only": _config_from_summary_key("pca280_only"),
        }
    )
    if not raw_frame.empty:
        for name, left_key, right_key in PAIRED_COMPARISONS:
            paired.append(
                make_paired_comparison(
                    raw_frame,
                    name=name,
                    left=paired_configs[left_key],
                    right=paired_configs[right_key],
                )
            )
    paired_frame = pd.DataFrame(
        [{key: value for key, value in item.items() if key != "seed_deltas"} for item in paired]
    )
    paired_frame.to_csv(
        project_root / "results" / "tables" / "stage3_paired_comparisons.csv",
        index=False,
        float_format="%.12g",
    )
    summary = build_stage3_summary(
        aggregate,
        paired,
        runtime_sec,
        failures,
        len(raw_frame),
    )
    raw_mean = float(
        _config_stats(aggregate, REPRESENTATIVE_CONFIGS["raw"])["mean_accuracy"]
    )
    generate_figures(
        aggregate,
        raw_mean,
        paired,
        raw_frame,
        project_root / "results" / "figures",
        logger,
    )
    write_json(project_root / "audit" / "stage3_split_audits.json", split_audits)
    write_json(project_root / "results" / "stage3_summary.json", summary)
    (project_root / "results" / "stage3_summary.txt").write_text(
        summary_text(summary), encoding="utf-8"
    )
    write_stage3_notes(project_root, summary)

    after_hashes = _hash_files(project_root, PRESERVED_PATHS)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "project_path": str(project_root),
        "python_executable": sys.executable,
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "package_versions": package_versions(
            ("numpy", "scipy", "sklearn", "pandas", "matplotlib", "joblib")
        ),
        "dataset": {
            "source": state.dataset_audit.get("source"),
            "X_shape": state.X.shape,
            "y_shape": state.y.shape,
        },
        "seeds": list(ROBUSTNESS_SEEDS),
        "split_strategy": "train_test_split(test_size=0.20, stratify=y, random_state=seed)",
        "train_size": 320,
        "test_size": 80,
        "pca_dimensions": list(PCA_DIMS),
        "lda_dimensions": list(LDA_DIMS),
        "pca_pre_lda_controls": list(PCA_PRE_DIMS),
        "lda_solver": LDA_SOLVER,
        "classifier": CLASSIFIER_NAME,
        "metric": METRIC_NAME,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "command_used": "python src/run_stage3.py",
        "expected_experiment_rows": 880,
        "actual_experiment_rows": len(raw_frame),
        "failures": failures,
        "runtime_sec": runtime_sec,
        "mean_runtime_per_seed_sec": runtime_sec / len(ROBUSTNESS_SEEDS),
        "paired_design": True,
        "fit_scope": "all PCA/LDA estimators fit on the current seed's X_train (and y_train for LDA)",
        "diagnostic_thresholds": {
            "near_tolerance": DIAGNOSTIC_NEAR_TOLERANCE,
            "large_gap": DIAGNOSTIC_LARGE_GAP,
            "residual_gap": DIAGNOSTIC_RESIDUAL_GAP,
        },
        "stage1_stage2_hashes_before": before_hashes,
        "stage1_stage2_hashes_after": after_hashes,
        "stage1_stage2_unchanged": before_hashes == after_hashes,
        "source_code_sha256": source_sha256(project_root),
        "git_commit_hash": git_commit_hash(project_root),
    }
    write_json(project_root / "audit" / "stage3_run_manifest.json", manifest)

    split_pass = bool(split_audits) and all(item["overall_pass"] for item in split_audits)
    method_counts = raw_frame["method"].value_counts().to_dict() if not raw_frame.empty else {}
    stage3_validation = {
        "status": "PENDING",
        "passed": False,
        "checks": [],
        "failed_checks": [],
        "expected_rows": 880,
        "actual_rows": len(raw_frame),
        "method_counts": method_counts,
        "split_pass": split_pass,
    }
    # Final validation is delegated to the structured validator below; the
    # provisional report is written first so all required artifacts exist.
    write_json(project_root / "audit" / "stage3_validation_report.json", stage3_validation)
    (project_root / "audit" / "stage3_validation_report.txt").write_text(
        "PENDING\nStage 3 validation is being finalized.\n", encoding="utf-8"
    )
    if __package__ in {None, ""}:
        from stage3_validation import validate_stage3_outputs, validation_text
    else:
        from .stage3_validation import validate_stage3_outputs, validation_text

    validation = validate_stage3_outputs(
        project_root,
        before_hashes=before_hashes,
        after_hashes=after_hashes,
        raw_frame=raw_frame,
        aggregate_frame=aggregate,
        paired_frame=paired_frame,
        split_audits=split_audits,
        failures=failures,
        seeds=ROBUSTNESS_SEEDS,
        pca_dims=PCA_DIMS,
        lda_dims=LDA_DIMS,
        figure_names=FIGURE_NAMES,
    )
    validation["checks"].append(
        {
            "name": "file:stage3_validation_reports",
            "passed": True,
            "detail": "text and JSON validation reports written",
        }
    )
    validation["failed_checks"] = [
        item for item in validation["checks"] if not item["passed"]
    ]
    validation["passed"] = not validation["failed_checks"]
    validation["status"] = "PASS" if validation["passed"] else "FAIL"
    write_json(project_root / "audit" / "stage3_validation_report.json", validation)
    (project_root / "audit" / "stage3_validation_report.txt").write_text(
        validation_text(validation), encoding="utf-8"
    )

    summary["validation_status"] = validation["status"]
    summary["freeze_recommendation"] = (
        "EXPERIMENTS READY TO FREEZE"
        if validation["passed"]
        else "NOT READY TO FREEZE"
    )
    write_json(project_root / "results" / "stage3_summary.json", summary)
    (project_root / "results" / "stage3_summary.txt").write_text(
        summary_text(summary), encoding="utf-8"
    )
    write_stage3_notes(project_root, summary)
    progress = {
        "status": "COMPLETE" if validation["passed"] else "COMPLETE_WITH_VALIDATION_FAILURE",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_seed_count": len(ROBUSTNESS_SEEDS),
        "completed_seeds": list(ROBUSTNESS_SEEDS),
        "actual_rows": len(raw_frame),
        "failure_count": len(failures),
        "validation_status": validation["status"],
    }
    write_json(project_root / "audit" / "stage3_progress.json", progress)
    if validation["passed"]:
        logger.info("Stage 3 validation passed; experiments ready to freeze")
        return 0
    logger.error("Stage 3 validation failed: %s", validation["failed_checks"])
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
