"""Run the leakage-safe Olivetti raw/PCA + 1-NN Stage 1 experiment."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

if __package__ in {None, ""}:  # support the requested ``python src/run_baseline.py``
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_environment import write_environment_audit
    from prepare_data import DatasetLoadError, load_and_prepare_dataset
    from utils import (
        ORIGINAL_DIMENSION,
        PCA_DIMS,
        RANDOM_STATE,
        compute_error_rate,
        components_for_threshold,
        git_commit_hash,
        package_versions,
        relative_dimensionality_reduction,
        source_sha256,
        write_json,
    )
    from validate_results import validate_outputs, validation_text
else:
    from .audit_environment import write_environment_audit
    from .prepare_data import DatasetLoadError, load_and_prepare_dataset
    from .utils import (
        ORIGINAL_DIMENSION,
        PCA_DIMS,
        RANDOM_STATE,
        compute_error_rate,
        components_for_threshold,
        git_commit_hash,
        package_versions,
        relative_dimensionality_reduction,
        source_sha256,
        write_json,
    )
    from .validate_results import validate_outputs, validation_text


LOGGER_NAME = "ee6222.assignment1"
CLASSIFIER_NAME = "KNeighborsClassifier(n_neighbors=1)"
METRIC_NAME = "euclidean"
PACKAGE_IMPORTS = ("numpy", "scipy", "sklearn", "pandas", "matplotlib", "joblib")
FIGURE_NAMES = (
    "pca_accuracy_vs_dimension.png",
    "pca_error_rate_vs_dimension.png",
    "pca_cumulative_explained_variance.png",
    "olivetti_sample_faces.png",
)
RESULT_COLUMNS = [
    "method",
    "dimension",
    "accuracy",
    "error_rate",
    "n_correct",
    "n_incorrect",
    "explained_variance",
    "fit_time_sec",
    "predict_time_sec",
    "total_time_sec",
    "random_state",
    "train_size",
    "test_size",
    "classifier",
    "metric",
    "pca_fit_time_sec",
    "transform_time_sec",
    "classifier_fit_time_sec",
    "prediction_time_sec",
]


def configure_logging(log_path: Path) -> logging.Logger:
    """Create a fresh file-and-console logger for one experiment run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
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


def close_logging(logger: logging.Logger) -> None:
    """Close handlers after a completed run so temporary/output files can be moved."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def choose_best_pca_result(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Choose the highest accuracy, breaking ties by the smallest dimension."""
    candidates = list(rows)
    if not candidates:
        raise ValueError("at least one PCA result is required")
    return min(
        candidates,
        key=lambda row: (-float(row["accuracy"]), int(row["dimension"])),
    )


def _base_result_row(
    *,
    method: str,
    dimension: int,
    accuracy: float,
    n_correct: int,
    n_incorrect: int,
    fit_time_sec: float,
    predict_time_sec: float,
    total_time_sec: float,
    train_size: int,
    test_size: int,
    explained_variance: float | None,
    pca_fit_time_sec: float | None = None,
    transform_time_sec: float | None = None,
    classifier_fit_time_sec: float | None = None,
    prediction_time_sec: float | None = None,
) -> dict[str, Any]:
    """Build one normalized row for the result table."""
    return {
        "method": method,
        "dimension": int(dimension),
        "accuracy": float(accuracy),
        "error_rate": compute_error_rate(accuracy),
        "n_correct": int(n_correct),
        "n_incorrect": int(n_incorrect),
        "explained_variance": (
            float(explained_variance) if explained_variance is not None else None
        ),
        "fit_time_sec": float(fit_time_sec),
        "predict_time_sec": float(predict_time_sec),
        "total_time_sec": float(total_time_sec),
        "random_state": RANDOM_STATE,
        "train_size": int(train_size),
        "test_size": int(test_size),
        "classifier": CLASSIFIER_NAME,
        "metric": METRIC_NAME,
        "pca_fit_time_sec": (
            float(pca_fit_time_sec) if pca_fit_time_sec is not None else None
        ),
        "transform_time_sec": (
            float(transform_time_sec) if transform_time_sec is not None else None
        ),
        "classifier_fit_time_sec": (
            float(classifier_fit_time_sec)
            if classifier_fit_time_sec is not None
            else None
        ),
        "prediction_time_sec": (
            float(prediction_time_sec) if prediction_time_sec is not None else None
        ),
    }


def run_raw_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Fit and evaluate Euclidean 1-NN on raw pixels."""
    total_start = time.perf_counter()
    classifier = KNeighborsClassifier(n_neighbors=1, metric=METRIC_NAME)
    fit_start = time.perf_counter()
    classifier.fit(X_train, y_train)
    fit_time = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    predictions = classifier.predict(X_test)
    predict_time = time.perf_counter() - predict_start
    total_time = time.perf_counter() - total_start
    accuracy = float(accuracy_score(y_test, predictions))
    n_correct = int(np.sum(predictions == y_test))
    row = _base_result_row(
        method="raw_1nn",
        dimension=int(X_train.shape[1]),
        accuracy=accuracy,
        n_correct=n_correct,
        n_incorrect=int(y_test.size - n_correct),
        fit_time_sec=fit_time,
        predict_time_sec=predict_time,
        total_time_sec=total_time,
        train_size=int(y_train.size),
        test_size=int(y_test.size),
        explained_variance=None,
        classifier_fit_time_sec=fit_time,
        prediction_time_sec=predict_time,
    )
    logger.info(
        "Raw baseline: dimension=%d accuracy=%.6f error_rate=%.6f correct=%d/%d",
        row["dimension"],
        row["accuracy"],
        row["error_rate"],
        row["n_correct"],
        row["test_size"],
    )
    return row


def run_pca_sweep(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Fit independent training-only PCA models at the fixed dimensions."""
    max_components = min(X_train.shape[0], X_train.shape[1])
    rows: list[dict[str, Any]] = []
    skipped: list[int] = []
    for dimension in PCA_DIMS:
        if dimension > max_components:
            skipped.append(dimension)
            logger.warning(
                "Skipping PCA dimension=%d; legal maximum is %d",
                dimension,
                max_components,
            )
            continue
        total_start = time.perf_counter()
        pca = PCA(n_components=dimension, svd_solver="full")
        pca_fit_start = time.perf_counter()
        pca.fit(X_train)
        pca_fit_time = time.perf_counter() - pca_fit_start
        transform_start = time.perf_counter()
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        transform_time = time.perf_counter() - transform_start
        if X_train_pca.shape[1] != dimension or X_test_pca.shape[1] != dimension:
            raise RuntimeError(
                f"PCA shape check failed for d={dimension}: "
                f"train={X_train_pca.shape}, test={X_test_pca.shape}"
            )

        classifier = KNeighborsClassifier(n_neighbors=1, metric=METRIC_NAME)
        classifier_fit_start = time.perf_counter()
        classifier.fit(X_train_pca, y_train)
        classifier_fit_time = time.perf_counter() - classifier_fit_start
        prediction_start = time.perf_counter()
        predictions = classifier.predict(X_test_pca)
        prediction_time = time.perf_counter() - prediction_start
        total_time = time.perf_counter() - total_start
        accuracy = float(accuracy_score(y_test, predictions))
        n_correct = int(np.sum(predictions == y_test))
        cumulative_variance = float(np.sum(pca.explained_variance_ratio_))
        row = _base_result_row(
            method="pca_1nn",
            dimension=dimension,
            accuracy=accuracy,
            n_correct=n_correct,
            n_incorrect=int(y_test.size - n_correct),
            fit_time_sec=pca_fit_time + classifier_fit_time,
            predict_time_sec=transform_time + prediction_time,
            total_time_sec=total_time,
            train_size=int(y_train.size),
            test_size=int(y_test.size),
            explained_variance=cumulative_variance,
            pca_fit_time_sec=pca_fit_time,
            transform_time_sec=transform_time,
            classifier_fit_time_sec=classifier_fit_time,
            prediction_time_sec=prediction_time,
        )
        rows.append(row)
        logger.info(
            "PCA d=%d: accuracy=%.6f error_rate=%.6f explained_variance=%.6f "
            "pca_fit=%.4fs transform=%.4fs classifier_fit=%.4fs predict=%.4fs",
            dimension,
            row["accuracy"],
            row["error_rate"],
            row["explained_variance"],
            pca_fit_time,
            transform_time,
            classifier_fit_time,
            prediction_time,
        )
    return rows, skipped


def run_full_explained_variance(
    X_train: np.ndarray, logger: logging.Logger
) -> tuple[pd.DataFrame, float]:
    """Fit one rank-compatible PCA on training data for the EV curve."""
    max_components = min(X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=max_components, svd_solver="full")
    start = time.perf_counter()
    pca.fit(X_train)
    fit_time = time.perf_counter() - start
    ratios = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    cumulative = np.cumsum(ratios)
    frame = pd.DataFrame(
        {
            "component": np.arange(1, ratios.size + 1, dtype=np.int64),
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance_ratio": cumulative,
        }
    )
    logger.info(
        "Full training-only PCA fitted: components=%d fit_time=%.4fs final_cumulative=%.9f",
        max_components,
        fit_time,
        float(cumulative[-1]),
    )
    return frame, fit_time


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_figures(
    pca_rows: list[dict[str, Any]],
    raw_accuracy: float,
    explained_variance: pd.DataFrame,
    images: np.ndarray,
    labels: np.ndarray,
    figures_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate the three required curves and a small dataset sanity figure."""
    dimensions = [int(row["dimension"]) for row in pca_rows]
    accuracies = [float(row["accuracy"]) for row in pca_rows]
    errors = [float(row["error_rate"]) for row in pca_rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.plot(dimensions, accuracies, marker="o", linewidth=2, label="PCA + 1-NN")
    ax.axhline(
        raw_accuracy,
        color="tab:red",
        linestyle="--",
        linewidth=1.8,
        label="Raw 4096D + 1-NN baseline",
    )
    ax.set_xlabel("Reduced Dimensionality")
    ax.set_ylabel("Test Classification Accuracy")
    ax.set_title("PCA Accuracy vs Reduced Dimensionality")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(dimensions)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, figures_dir / "pca_accuracy_vs_dimension.png")

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.plot(dimensions, errors, marker="o", linewidth=2, label="PCA + 1-NN")
    ax.axhline(
        compute_error_rate(raw_accuracy),
        color="tab:red",
        linestyle="--",
        linewidth=1.8,
        label="Raw 4096D + 1-NN baseline",
    )
    ax.set_xlabel("Reduced Dimensionality")
    ax.set_ylabel("Classification Error Rate")
    ax.set_title("Classification Error Rate vs Dimension")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(dimensions)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, figures_dir / "pca_error_rate_vs_dimension.png")

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.plot(
        explained_variance["component"],
        explained_variance["cumulative_explained_variance_ratio"],
        linewidth=2,
        color="tab:green",
        label="Cumulative explained variance",
    )
    for threshold, color in ((0.90, "tab:orange"), (0.95, "tab:purple"), (0.99, "tab:brown")):
        ax.axhline(
            threshold,
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=f"{int(threshold * 100)}% threshold",
        )
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title("PCA Cumulative Explained Variance (Training Data Only)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, figures_dir / "pca_cumulative_explained_variance.png")

    selected_indices: list[int] = []
    for label in np.unique(labels)[:6]:
        selected_indices.append(int(np.flatnonzero(labels == label)[0]))
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.5), constrained_layout=True)
    for axis, index in zip(axes.flat, selected_indices):
        axis.imshow(images[index], cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"Identity {int(labels[index])}")
        axis.axis("off")
    for axis in list(axes.flat)[len(selected_indices) :]:
        axis.axis("off")
    fig.suptitle("Olivetti Faces Dataset Samples")
    _save_figure(fig, figures_dir / "olivetti_sample_faces.png")
    logger.info("Figures generated under %s", figures_dir)


def _threshold_components(frame: pd.DataFrame) -> dict[str, int | None]:
    cumulative = frame["cumulative_explained_variance_ratio"].tolist()
    return {
        f"{int(threshold * 100)}%": components_for_threshold(cumulative, threshold)
        for threshold in (0.80, 0.90, 0.95, 0.99)
    }


def build_summary(
    raw_row: dict[str, Any],
    pca_rows: list[dict[str, Any]],
    explained_variance: pd.DataFrame,
    dataset_audit: dict[str, Any],
    split_audit: dict[str, Any],
    skipped_dimensions: list[int],
    full_pca_fit_time_sec: float,
) -> dict[str, Any]:
    """Create objective, machine-readable findings from the fixed sweep."""
    best = choose_best_pca_result(pca_rows)
    raw_accuracy = float(raw_row["accuracy"])
    best_accuracy = float(best["accuracy"])
    threshold_components = _threshold_components(explained_variance)
    dimensions = [int(row["dimension"]) for row in pca_rows]
    accuracies = [float(row["accuracy"]) for row in pca_rows]
    max_accuracy = max(accuracies)
    peak_count = sum(accuracy == max_accuracy for accuracy in accuracies)
    largest = pca_rows[-1]
    variance_95 = threshold_components["95%"]
    if variance_95 is None:
        relationship = "95% explained variance was not reached by the full training rank."
    elif int(best["dimension"]) == variance_95:
        relationship = (
            f"The best tested classification dimension ({best['dimension']}) equals "
            f"the 95% variance dimension ({variance_95})."
        )
    else:
        relationship = (
            f"The best tested classification dimension ({best['dimension']}) differs "
            f"from the 95% variance dimension ({variance_95})."
        )

    if best_accuracy > raw_accuracy:
        raw_comparison = (
            f"Best tested PCA accuracy exceeds raw accuracy by "
            f"{best_accuracy - raw_accuracy:.6f}."
        )
    elif best_accuracy < raw_accuracy:
        raw_comparison = (
            f"Best tested PCA accuracy is below raw accuracy by "
            f"{raw_accuracy - best_accuracy:.6f}."
        )
    else:
        raw_comparison = "Best tested PCA accuracy equals raw accuracy."

    observations = [
        f"Across {len(pca_rows)} tested PCA dimensions ({dimensions[0]} to {dimensions[-1]}), "
        f"accuracy ranged from {min(accuracies):.6f} to {max_accuracy:.6f}; "
        f"the peak occurred at d={best['dimension']} ({peak_count} tested point(s) at the peak).",
        raw_comparison,
        f"At the largest tested dimension d={largest['dimension']}, accuracy was "
        f"{float(largest['accuracy']):.6f} and cumulative explained variance was "
        f"{float(largest['explained_variance']):.6f}.",
        relationship,
    ]
    if skipped_dimensions:
        observations.append(f"Dimensions skipped because they exceeded training rank: {skipped_dimensions}.")
    else:
        observations.append("All requested PCA dimensions were legal for the 320-sample training split.")

    return {
        "experiment": "EE6222 Assignment 1 - Stage 1 Raw/PCA + 1-NN",
        "raw_baseline": {
            "method": raw_row["method"],
            "dimension": int(raw_row["dimension"]),
            "accuracy": raw_accuracy,
            "error_rate": float(raw_row["error_rate"]),
            "n_correct": int(raw_row["n_correct"]),
            "n_incorrect": int(raw_row["n_incorrect"]),
        },
        "pca_baseline": {
            "tested_dimensions": dimensions,
            "best_tested_dimension": int(best["dimension"]),
            "best_accuracy": best_accuracy,
            "best_error_rate": float(best["error_rate"]),
            "accuracy_difference_vs_raw": best_accuracy - raw_accuracy,
            "relative_dimensionality_reduction": relative_dimensionality_reduction(
                int(best["dimension"]), int(raw_row["dimension"])
            ),
            "skipped_dimensions": skipped_dimensions,
        },
        "explained_variance": {
            "full_pca_components": int(len(explained_variance)),
            "components_required": threshold_components,
            "full_pca_fit_time_sec": float(full_pca_fit_time_sec),
        },
        "protocol": {
            "random_state": RANDOM_STATE,
            "test_size": 0.20,
            "train_size": int(split_audit["train_size"]),
            "test_size_count": int(split_audit["test_size"]),
            "classifier": CLASSIFIER_NAME,
            "metric": METRIC_NAME,
            "raw_normalization": "none",
            "pca_fit_scope": "X_train only",
            "test_set_role": "final evaluation only; fixed dimensions were specified before evaluation",
        },
        "dataset": {
            "source": dataset_audit.get("source"),
            "X_shape": dataset_audit["X_shape"],
            "y_shape": dataset_audit["y_shape"],
            "images_shape": dataset_audit["images_shape"],
            "class_count": dataset_audit["number_of_classes"],
        },
        "observations": observations,
    }


def summary_text(summary: dict[str, Any], pca_rows: list[dict[str, Any]]) -> str:
    """Render the summary JSON into a compact text handoff."""
    raw = summary["raw_baseline"]
    pca = summary["pca_baseline"]
    variance = summary["explained_variance"]
    lines = [
        "EE6222 Assignment 1 - Stage 1 Baseline Summary",
        "",
        "Raw baseline:",
        f"  dimension: {raw['dimension']}",
        f"  accuracy: {raw['accuracy']:.12f}",
        f"  error rate: {raw['error_rate']:.12f}",
        f"  correct / incorrect: {raw['n_correct']} / {raw['n_incorrect']}",
        "",
        "PCA + 1-NN sweep:",
        "  dimension | accuracy | error_rate | cumulative_explained_variance",
    ]
    lines.extend(
        f"  {row['dimension']:9d} | {row['accuracy']:.12f} | "
        f"{row['error_rate']:.12f} | {row['explained_variance']:.12f}"
        for row in pca_rows
    )
    lines.extend(
        [
            "",
            "Best tested PCA:",
            f"  dimension: {pca['best_tested_dimension']}",
            f"  accuracy: {pca['best_accuracy']:.12f}",
            f"  error rate: {pca['best_error_rate']:.12f}",
            f"  accuracy difference vs raw: {pca['accuracy_difference_vs_raw']:.12f}",
            f"  relative dimensionality reduction: {pca['relative_dimensionality_reduction']:.12f}",
            "",
            "Components required for cumulative explained variance:",
        ]
    )
    lines.extend(
        f"  {threshold}: {components if components is not None else 'not reached'}"
        for threshold, components in variance["components_required"].items()
    )
    lines.extend(["", "Observational findings:"])
    lines.extend(f"  - {observation}" for observation in summary["observations"])
    lines.extend(
        [
            "",
            "Leakage statement:",
            "  PCA was fitted separately on X_train for every tested dimension.",
            "  The test set was transformed and evaluated only after fitting.",
            "  No feature-wise scaler or test-driven hyperparameter selection was used.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report_notes(project_root: Path, summary: dict[str, Any]) -> None:
    """Write factual notes for a future report without drafting the report itself."""
    raw = summary["raw_baseline"]
    pca = summary["pca_baseline"]
    lines = [
        "# EE6222 Assignment 1 - Stage 1 Report Notes",
        "",
        "These are machine-generated experiment facts, not a formal assignment report.",
        "",
        "- Dataset: sklearn Olivetti Faces; X shape "
        f"{summary['dataset']['X_shape']}, y shape {summary['dataset']['y_shape']}; "
        f"{summary['dataset']['class_count']} identities.",
        "- Train/test protocol: stratified `test_size=0.20`, `random_state=42`; "
        f"{summary['protocol']['train_size']} train and {summary['protocol']['test_size_count']} test samples.",
        f"- Raw baseline: 4096D Euclidean 1-NN, accuracy {raw['accuracy']:.12f}, "
        f"error rate {raw['error_rate']:.12f}.",
        f"- Best PCA baseline: d={pca['best_tested_dimension']}, accuracy "
        f"{pca['best_accuracy']:.12f}, difference vs raw "
        f"{pca['accuracy_difference_vs_raw']:.12f}.",
        "- Observed trend:",
        *[f"  - {observation}" for observation in summary["observations"]],
        "- Potential discussion points: compare the classification-optimal dimension "
        "with the variance-retention thresholds, while keeping the fixed test evaluation "
        "boundary explicit.",
    ]
    notes_path = project_root / "report" / "notes.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results_csv(project_root: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    path = project_root / "results" / "tables" / "baseline_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")


def write_explained_variance_csv(project_root: Path, frame: pd.DataFrame) -> None:
    path = project_root / "results" / "tables" / "pca_explained_variance.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")


def build_run_manifest(
    project_root: Path,
    environment_audit: dict[str, Any],
    dataset_audit: dict[str, Any],
    split_audit: dict[str, Any],
) -> dict[str, Any]:
    """Capture provenance needed to reproduce this run."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": environment_audit["hostname"],
        "user": environment_audit["user"],
        "project_path": str(project_root.resolve()),
        "python_version": environment_audit["python"]["version"],
        "python_executable": environment_audit["python"]["executable"],
        "package_versions": package_versions(PACKAGE_IMPORTS),
        "dataset_source": dataset_audit.get("source"),
        "dataset_data_home": dataset_audit.get("data_home"),
        "dataset_shape": {
            "X": dataset_audit["X_shape"],
            "y": dataset_audit["y_shape"],
            "images": dataset_audit["images_shape"],
        },
        "train_test_split": {
            "random_state": RANDOM_STATE,
            "test_size": 0.20,
            "train_size": split_audit["train_size"],
            "test_size": split_audit["test_size"],
            "class_counts": split_audit["class_counts"],
            "no_index_overlap": split_audit["checks"]["no_index_overlap"],
        },
        "pca_dimension_list": list(PCA_DIMS),
        "classifier_configuration": {
            "class": "sklearn.neighbors.KNeighborsClassifier",
            "n_neighbors": 1,
            "metric": METRIC_NAME,
        },
        "normalization": "none",
        "device": "CPU only",
        "gpu_used": False,
        "command_used": "python src/run_baseline.py",
        "actual_argv": [str(value) for value in sys.argv],
        "git_commit_hash": git_commit_hash(project_root),
        "source_code_sha256": source_sha256(project_root),
    }


def _write_dataset_failure(project_root: Path, error: Exception) -> None:
    failure = {
        "status": "FAIL",
        "source": "sklearn.datasets.fetch_olivetti_faces",
        "error_type": type(error).__name__,
        "error": str(error),
    }
    write_json(project_root / "audit" / "dataset_audit.json", failure)
    (project_root / "audit" / "dataset_audit.txt").write_text(
        "FAIL\nEE6222 Assignment 1 - Dataset Load Failure\n"
        f"{type(error).__name__}: {error}\n",
        encoding="utf-8",
    )
    (project_root / "audit" / "validation_report.txt").write_text(
        "FAIL\nEE6222 Assignment 1 - Stage 1 Validation Report\n"
        f"[FAIL] dataset_load: {type(error).__name__}: {error}\n",
        encoding="utf-8",
    )


def run_experiment(project_root: Path) -> int:
    """Execute the complete Stage 1 pipeline and return a process exit code."""
    project_root = project_root.resolve()
    for relative in (
        "audit",
        "data/sklearn_cache",
        "data/processed",
        "results/figures",
        "results/tables",
        "results/logs",
        "results/models",
        "report",
    ):
        (project_root / relative).mkdir(parents=True, exist_ok=True)
    logger = configure_logging(project_root / "results" / "logs" / "baseline_run.log")
    logger.info("Starting EE6222 Stage 1 CPU-only baseline experiment")
    logger.info("Project root: %s", project_root)

    environment_audit = write_environment_audit(project_root)
    logger.info(
        "Environment audited: Python %s; scikit-learn=%s; venv=%s",
        environment_audit["python"]["version"],
        environment_audit["packages"].get("scikit-learn"),
        environment_audit["venv"]["python_exists"],
    )
    try:
        bundle, dataset_audit, split_audit = load_and_prepare_dataset(project_root, logger)
    except DatasetLoadError as exc:
        logger.exception("Dataset load failed; no synthetic fallback will be used")
        _write_dataset_failure(project_root, exc)
        return 2

    train_indices = np.asarray(split_audit["train_indices"], dtype=np.int64)
    test_indices = np.asarray(split_audit["test_indices"], dtype=np.int64)
    X_train = bundle.X[train_indices]
    X_test = bundle.X[test_indices]
    y_train = bundle.y[train_indices]
    y_test = bundle.y[test_indices]

    raw_row = run_raw_baseline(X_train, X_test, y_train, y_test, logger)
    pca_rows, skipped_dimensions = run_pca_sweep(
        X_train, X_test, y_train, y_test, logger
    )
    if not pca_rows:
        raise RuntimeError("No valid PCA dimensions remained after legality checks")
    explained_variance, full_pca_fit_time = run_full_explained_variance(X_train, logger)

    all_rows = [raw_row, *pca_rows]
    write_results_csv(project_root, all_rows)
    write_explained_variance_csv(project_root, explained_variance)
    logger.info("CSV tables generated")
    generate_figures(
        pca_rows,
        float(raw_row["accuracy"]),
        explained_variance,
        bundle.images,
        bundle.y,
        project_root / "results" / "figures",
        logger,
    )

    summary = build_summary(
        raw_row,
        pca_rows,
        explained_variance,
        dataset_audit,
        split_audit,
        skipped_dimensions,
        full_pca_fit_time,
    )
    write_json(project_root / "results" / "baseline_summary.json", summary)
    (project_root / "results" / "baseline_summary.txt").write_text(
        summary_text(summary, pca_rows), encoding="utf-8"
    )
    write_report_notes(project_root, summary)
    logger.info("Baseline summary and report notes generated")

    manifest = build_run_manifest(
        project_root, environment_audit, dataset_audit, split_audit
    )
    write_json(project_root / "audit" / "run_manifest.json", manifest)
    logger.info("Run manifest generated")

    validation = validate_outputs(
        project_root,
        dataset_audit,
        split_audit,
        all_rows,
        explained_variance,
        FIGURE_NAMES,
    )
    write_json(project_root / "audit" / "validation_report.json", validation)
    (project_root / "audit" / "validation_report.txt").write_text(
        validation_text(validation), encoding="utf-8"
    )
    if validation["passed"]:
        logger.info("Validation passed")
        close_logging(logger)
        return 0
    logger.error("Validation failed: %s", validation["failed_checks"])
    close_logging(logger)
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
