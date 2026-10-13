"""Load, audit, and split the official Olivetti Faces dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

try:
    from .utils import (
        EXPECTED_CLASSES,
        EXPECTED_SAMPLES,
        EXPECTED_SAMPLES_PER_CLASS,
        RANDOM_STATE,
        TEST_SIZE,
        write_json,
    )
except ImportError:  # direct execution: ``python src/prepare_data.py``
    from utils import (
        EXPECTED_CLASSES,
        EXPECTED_SAMPLES,
        EXPECTED_SAMPLES_PER_CLASS,
        RANDOM_STATE,
        TEST_SIZE,
        write_json,
    )


class DatasetLoadError(RuntimeError):
    """Raised when the official dataset cannot be loaded or audited."""


@dataclass(frozen=True)
class DatasetBundle:
    """The raw arrays and provenance needed by the baseline experiment."""

    X: np.ndarray
    y: np.ndarray
    images: np.ndarray
    source: str
    data_home: Path


def _class_distribution(y: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts)}


def audit_dataset(X: np.ndarray, y: np.ndarray, images: np.ndarray) -> dict[str, Any]:
    """Return structural, numerical, and integrity checks for a dataset."""
    X_array = np.asarray(X)
    y_array = np.asarray(y)
    images_array = np.asarray(images)
    sample_count = int(X_array.shape[0]) if X_array.ndim >= 1 else 0
    labels_sample_count = int(y_array.shape[0]) if y_array.ndim >= 1 else 0
    labels, counts = np.unique(y_array, return_counts=True)
    samples_per_class = [int(count) for count in counts]

    numeric_array = X_array.astype(np.float64, copy=False)
    nan_count = int(np.isnan(numeric_array).sum())
    inf_count = int(np.isinf(numeric_array).sum())
    if X_array.ndim == 2 and sample_count > 0:
        zero_variance_features = int(np.sum(np.var(numeric_array, axis=0) == 0.0))
        unique_rows = int(np.unique(X_array, axis=0).shape[0])
    else:
        zero_variance_features = None
        unique_rows = None

    integrity_checks = {
        "X_is_2d": bool(X_array.ndim == 2),
        "labels_are_1d": bool(y_array.ndim == 1),
        "images_are_3d": bool(images_array.ndim == 3),
        "sample_count_matches_labels": bool(sample_count == labels_sample_count),
        "no_nan": bool(nan_count == 0),
        "no_inf": bool(inf_count == 0),
        "labels_are_finite": bool(
            np.issubdtype(y_array.dtype, np.number)
            and np.isfinite(y_array.astype(np.float64, copy=False)).all()
        ),
        "images_sample_count_matches_X": bool(
            images_array.ndim >= 1 and images_array.shape[0] == sample_count
        ),
        "X_matches_flattened_images": bool(
            X_array.ndim == 2
            and images_array.ndim == 3
            and X_array.shape[1] == int(np.prod(images_array.shape[1:]))
        ),
        "expected_sample_count": bool(sample_count == EXPECTED_SAMPLES),
        "expected_feature_count": bool(
            X_array.ndim == 2 and X_array.shape[1] == 64 * 64
        ),
        "expected_class_count": bool(len(labels) == EXPECTED_CLASSES),
        "expected_samples_per_class": bool(
            len(samples_per_class) == EXPECTED_CLASSES
            and all(count == EXPECTED_SAMPLES_PER_CLASS for count in samples_per_class)
        ),
    }
    warnings: list[str] = []
    if not integrity_checks["expected_sample_count"]:
        warnings.append(f"Expected {EXPECTED_SAMPLES} samples, found {sample_count}.")
    if not integrity_checks["expected_feature_count"]:
        observed_features = X_array.shape[1] if X_array.ndim == 2 else "N/A"
        warnings.append(f"Expected 4096 features, found {observed_features}.")
    if not integrity_checks["expected_class_count"]:
        warnings.append(f"Expected {EXPECTED_CLASSES} classes, found {len(labels)}.")
    if not integrity_checks["expected_samples_per_class"]:
        warnings.append(
            "Expected 10 samples per class for all 40 classes; "
            f"observed counts {samples_per_class}."
        )
    if nan_count or inf_count:
        warnings.append(f"Non-finite feature values found: NaN={nan_count}, Inf={inf_count}.")

    return {
        "X_shape": [int(value) for value in X_array.shape],
        "y_shape": [int(value) for value in y_array.shape],
        "images_shape": [int(value) for value in images_array.shape],
        "dtype": str(X_array.dtype),
        "y_dtype": str(y_array.dtype),
        "number_of_samples": sample_count,
        "number_of_features": (
            int(X_array.shape[1]) if X_array.ndim == 2 else None
        ),
        "number_of_classes": int(len(labels)),
        "class_distribution": _class_distribution(y_array),
        "samples_per_class": samples_per_class,
        "global_min": float(np.min(numeric_array)) if numeric_array.size else None,
        "global_max": float(np.max(numeric_array)) if numeric_array.size else None,
        "global_mean": float(np.mean(numeric_array)) if numeric_array.size else None,
        "global_std": float(np.std(numeric_array)) if numeric_array.size else None,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_variance_feature_count": zero_variance_features,
        "duplicate_row_count": (
            sample_count - unique_rows
            if unique_rows is not None
            else None
        ),
        "integrity_checks": integrity_checks,
        "overall_integrity_pass": bool(all(integrity_checks.values())),
        "warnings": warnings,
    }


def dataset_audit_text(audit: dict[str, Any]) -> str:
    """Render the dataset audit as a human-readable text file."""
    lines = [
        "EE6222 Assignment 1 - Olivetti Faces Dataset Audit",
        f"Overall integrity pass: {audit['overall_integrity_pass']}",
        f"X shape: {audit['X_shape']}",
        f"y shape: {audit['y_shape']}",
        f"images shape: {audit['images_shape']}",
        f"dtype: {audit['dtype']}",
        f"number of samples: {audit['number_of_samples']}",
        f"number of features: {audit['number_of_features']}",
        f"number of classes: {audit['number_of_classes']}",
        f"samples per class: {audit['samples_per_class']}",
        f"global min: {audit['global_min']}",
        f"global max: {audit['global_max']}",
        f"global mean: {audit['global_mean']}",
        f"global std: {audit['global_std']}",
        f"NaN count: {audit['nan_count']}",
        f"Inf count: {audit['inf_count']}",
        f"zero-variance feature count: {audit['zero_variance_feature_count']}",
        f"duplicate row count: {audit['duplicate_row_count']}",
        "",
        "Integrity checks:",
    ]
    lines.extend(
        f"  {name}: {value}" for name, value in audit["integrity_checks"].items()
    )
    lines.append("")
    lines.append("Warnings:")
    lines.extend(f"  - {warning}" for warning in audit["warnings"] or ["none"])
    return "\n".join(lines) + "\n"


def build_split_audit(
    y: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray
) -> dict[str, Any]:
    """Audit split sizes, class coverage, and index disjointness."""
    labels = np.asarray(y)
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    unique_labels = np.unique(labels)
    class_counts = {
        str(label): {
            "train": int(np.sum(labels[train] == label)),
            "test": int(np.sum(labels[test] == label)),
        }
        for label in unique_labels
    }
    no_overlap = bool(np.intersect1d(train, test).size == 0)
    checks = {
        "no_index_overlap": no_overlap,
        "sizes_sum_to_total": bool(train.size + test.size == labels.size),
        "every_class_in_train": bool(all(item["train"] > 0 for item in class_counts.values())),
        "every_class_in_test": bool(all(item["test"] > 0 for item in class_counts.values())),
        "expected_train_size": bool(train.size == int(round(labels.size * (1 - TEST_SIZE)))),
        "expected_test_size": bool(test.size == int(round(labels.size * TEST_SIZE))),
    }
    return {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_size": int(train.size),
        "test_size": int(test.size),
        "total_size": int(labels.size),
        "train_indices": [int(value) for value in train],
        "test_indices": [int(value) for value in test],
        "class_counts": class_counts,
        "checks": checks,
        "overall_pass": bool(all(checks.values())),
    }


def make_split_indices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create the fixed, stratified train/test split from labels only."""
    indices = np.arange(np.asarray(y).shape[0], dtype=np.int64)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=np.asarray(y),
        random_state=RANDOM_STATE,
    )
    return train_indices, test_indices


def load_and_prepare_dataset(
    project_root: Path, logger: logging.Logger | None = None
) -> tuple[DatasetBundle, dict[str, Any], dict[str, Any]]:
    """Fetch, audit, persist, and split Olivetti Faces using a controlled cache."""
    data_home = project_root / "data" / "sklearn_cache"
    processed_dir = project_root / "data" / "processed"
    audit_dir = project_root / "audit"
    data_home.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info("Loading sklearn Olivetti Faces with data_home=%s", data_home)
    try:
        dataset = fetch_olivetti_faces(
            data_home=str(data_home),
            shuffle=False,
            download_if_missing=True,
        )
    except Exception as exc:
        raise DatasetLoadError(
            "Unable to load sklearn Olivetti Faces from the controlled cache. "
            f"The dataset may be absent and network access may be blocked: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    X = np.asarray(dataset.data)
    y = np.asarray(dataset.target)
    images = np.asarray(dataset.images)
    audit = audit_dataset(X, y, images)
    audit["source"] = "sklearn.datasets.fetch_olivetti_faces"
    audit["data_home"] = str(data_home)
    audit["processed_arrays"] = {
        "X_path": str(processed_dir / "X.npy"),
        "y_path": str(processed_dir / "y.npy"),
        "X_shape": [int(value) for value in X.shape],
        "y_shape": [int(value) for value in y.shape],
        "X_dtype": str(X.dtype),
        "y_dtype": str(y.dtype),
    }
    write_json(audit_dir / "dataset_audit.json", audit)
    (audit_dir / "dataset_audit.txt").write_text(
        dataset_audit_text(audit), encoding="utf-8"
    )
    np.save(processed_dir / "X.npy", X)
    np.save(processed_dir / "y.npy", y)
    if logger:
        logger.info(
            "Dataset loaded: X=%s, y=%s, images=%s, classes=%d",
            X.shape,
            y.shape,
            images.shape,
            audit["number_of_classes"],
        )
        for warning in audit["warnings"]:
            logger.warning("Dataset audit warning: %s", warning)

    train_indices, test_indices = make_split_indices(y)
    split_audit = build_split_audit(y, train_indices, test_indices)
    write_json(audit_dir / "split_audit.json", split_audit)
    if logger:
        logger.info(
            "Split created: train=%d, test=%d, overlap=%s",
            split_audit["train_size"],
            split_audit["test_size"],
            not split_audit["checks"]["no_index_overlap"],
        )

    bundle = DatasetBundle(
        X=X,
        y=y,
        images=images,
        source="sklearn.datasets.fetch_olivetti_faces",
        data_home=data_home,
    )
    return bundle, audit, split_audit


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    bundle, audit, split_audit = load_and_prepare_dataset(project_root)
    print(f"Loaded {bundle.X.shape[0]} samples with {bundle.X.shape[1]} features.")
    print(f"Dataset audit pass: {audit['overall_integrity_pass']}")
    print(f"Split audit pass: {split_audit['overall_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

