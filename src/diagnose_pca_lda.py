from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PCA_PRE_DIMS = (50, 75, 100, 150, 200, 240, 260, 280, 319)
FINAL_LDA_DIM = 20
OUTER_SPLIT_SEED = 42
FOLD_SELECTION_SEED = 6222
LDA_SOLVER = "svd"
LDA_TOL = 1e-4
KNN_NEIGHBORS = 1
METRIC = "euclidean"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ee6222.pca_lda_diagnostic")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(stream)
    return logger


def within_svd_diagnostics(
    X: np.ndarray, y: np.ndarray, tol: float = LDA_TOL
) -> tuple[dict[str, Any], np.ndarray]:
    """Reproduce sklearn 1.7.2 SVD solver's within-class preprocessing and spectrum."""
    labels = np.unique(y)
    n_samples, n_features = X.shape
    n_classes = labels.size
    class_means = np.vstack([X[y == label].mean(axis=0) for label in labels])
    centered = np.concatenate(
        [X[y == label] - class_means[i] for i, label in enumerate(labels)],
        axis=0,
    )
    within_std = centered.std(axis=0)
    zero_std = within_std == 0
    within_std[zero_std] = 1.0
    factor = np.asarray(1.0 / (n_samples - n_classes), dtype=X.dtype)
    scaled = np.sqrt(factor) * (centered / within_std)
    _, singular_values, vt = svd(scaled, full_matrices=False, check_finite=False)

    rank_tol = int(np.sum(singular_values > tol))
    if singular_values.size and singular_values[0] > 0:
        machine_cutoff = (
            np.finfo(singular_values.dtype).eps
            * max(scaled.shape)
            * float(singular_values[0])
        )
        rank_machine = int(np.sum(singular_values > machine_cutoff))
    else:
        machine_cutoff = 0.0
        rank_machine = 0
    rank_numpy = int(np.linalg.matrix_rank(scaled))
    if rank_tol:
        min_kept = float(singular_values[rank_tol - 1])
        cond_kept = float(singular_values[0] / singular_values[rank_tol - 1])
    else:
        min_kept = None
        cond_kept = None

    diagnostics = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_classes": int(n_classes),
        "within_rank_upper_bound": int(min(n_features, n_samples - n_classes)),
        "within_std_zero_count": int(np.sum(zero_std)),
        "within_singular_count": int(singular_values.size),
        "within_rank_sklearn_tol": rank_tol,
        "within_rank_machine_epsilon": rank_machine,
        "within_rank_numpy_default": rank_numpy,
        "sklearn_within_tol": float(tol),
        "within_singular_max": float(singular_values[0]) if singular_values.size else 0.0,
        "within_singular_min_kept": min_kept,
        "within_condition_kept": cond_kept,
        "within_covariance_condition_kept": (
            float(cond_kept**2) if cond_kept is not None else None
        ),
        "within_count_near_tol_10x": int(
            np.sum((singular_values >= tol / 10.0) & (singular_values <= tol * 10.0))
        ),
        "machine_rank_cutoff": float(machine_cutoff),
    }
    return diagnostics, singular_values


def between_svd_rank(lda: LinearDiscriminantAnalysis, within_diag: dict[str, Any]) -> dict[str, Any]:
    """Check sklearn's relative tolerance for the between-class SVD."""
    # The fitted object exposes the exact internal rank through the scaling matrix.
    lda_rank = int(lda.scalings_.shape[1])
    return {
        "between_rank_sklearn_relative_tol": lda_rank,
        "between_rank_upper_bound": int(min(len(lda.classes_) - 1, lda.n_features_in_)),
        "lda_scalings_shape": [int(v) for v in lda.scalings_.shape],
        "lda_explained_variance_ratio_length": int(len(lda.explained_variance_ratio_)),
        "lda_tol": float(lda.tol),
    }


def deterministic_internal_validation_indices(y_train: np.ndarray) -> np.ndarray:
    """Choose exactly one fixed internal validation image per identity."""
    rng = np.random.default_rng(FOLD_SELECTION_SEED)
    selected: list[int] = []
    for label in np.unique(y_train):
        members = np.flatnonzero(y_train == label)
        if members.size != 8:
            raise RuntimeError(f"seed-42 training class {label} has {members.size} rows, expected 8")
        selected.append(int(rng.choice(members)))
    return np.asarray(selected, dtype=np.int64)


def nearest_same_different(
    Z_train: np.ndarray, y_train: np.ndarray, z_valid: np.ndarray, y_valid: int
) -> tuple[float, float]:
    distances = np.linalg.norm(Z_train - z_valid.reshape(1, -1), axis=1)
    same = distances[y_train == y_valid]
    different = distances[y_train != y_valid]
    return float(np.min(same)), float(np.min(different))


def fit_eval_one_fold(
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    valid_local_index: int,
    pca_dimension: int,
    fold_id: int,
) -> tuple[dict[str, Any], np.ndarray]:
    train_mask = np.ones(y_outer_train.size, dtype=bool)
    train_mask[valid_local_index] = False
    X_inner = X_outer_train[train_mask]
    y_inner = y_outer_train[train_mask]
    X_valid = X_outer_train[valid_local_index : valid_local_index + 1]
    y_valid = y_outer_train[valid_local_index : valid_local_index + 1]

    if X_inner.shape != (319, 4096) or y_inner.size != 319:
        raise RuntimeError(f"Unexpected inner split shape: {X_inner.shape}, {y_inner.shape}")
    if np.unique(y_inner).size != 40:
        raise RuntimeError("Every internal training fold must retain all 40 classes")

    pca = PCA(n_components=pca_dimension, svd_solver="full")
    pca.fit(X_inner)
    X_inner_pca = pca.transform(X_inner)
    X_valid_pca = pca.transform(X_valid)
    if X_inner_pca.shape != (319, pca_dimension) or X_valid_pca.shape != (1, pca_dimension):
        raise RuntimeError("PCA output dimensions do not match the requested pre-dimension")

    lda = LinearDiscriminantAnalysis(
        solver=LDA_SOLVER,
        n_components=FINAL_LDA_DIM,
        tol=LDA_TOL,
    )
    lda.fit(X_inner_pca, y_inner)
    Z_inner = lda.transform(X_inner_pca)
    Z_valid = lda.transform(X_valid_pca)
    if Z_inner.shape != (319, FINAL_LDA_DIM) or Z_valid.shape != (1, FINAL_LDA_DIM):
        raise RuntimeError(f"LDA transform shape mismatch: {Z_inner.shape}, {Z_valid.shape}")
    if not np.isfinite(Z_inner).all() or not np.isfinite(Z_valid).all():
        raise RuntimeError("LDA transform produced non-finite coordinates")

    knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric=METRIC)
    knn.fit(Z_inner, y_inner)
    pred = int(knn.predict(Z_valid)[0])
    same_distance, different_distance = nearest_same_different(
        Z_inner, y_inner, Z_valid[0], int(y_valid[0])
    )

    train_coordinate_std = Z_inner.std(axis=0, ddof=0)
    safe_scale = np.where(train_coordinate_std == 0, 1.0, train_coordinate_std)
    Z_inner_scaled = (Z_inner - Z_inner.mean(axis=0)) / safe_scale
    Z_valid_scaled = (Z_valid - Z_inner.mean(axis=0)) / safe_scale
    scaled_knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric=METRIC)
    scaled_knn.fit(Z_inner_scaled, y_inner)
    scaled_pred = int(scaled_knn.predict(Z_valid_scaled)[0])
    scaled_same_distance, scaled_different_distance = nearest_same_different(
        Z_inner_scaled, y_inner, Z_valid_scaled[0], int(y_valid[0])
    )

    within_diag, singular_values = within_svd_diagnostics(X_inner_pca, y_inner)
    between_diag = between_svd_rank(lda, within_diag)
    expected_within_bound = min(X_inner_pca.shape[1], X_inner.shape[0] - np.unique(y_inner).size)
    if within_diag["within_rank_upper_bound"] != expected_within_bound:
        raise RuntimeError("Within-class rank bound mismatch")

    result = {
        "fold_id": int(fold_id),
        "heldout_class": int(y_valid[0]),
        "inner_train_n": int(X_inner.shape[0]),
        "inner_valid_n": 1,
        "pca_pre_dimension": int(pca_dimension),
        "lda_output_dimension": FINAL_LDA_DIM,
        "pca_fit_scope": "inner training rows only",
        "lda_fit_scope": "inner training rows and labels only",
        "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "pca_last_component_variance": float(pca.explained_variance_ratio_[-1]),
        "pca_train_feature_std_min": float(X_inner_pca.std(axis=0, ddof=0).min()),
        "pca_train_feature_std_max": float(X_inner_pca.std(axis=0, ddof=0).max()),
        "prediction_euclidean_1nn": pred,
        "prediction_train_std_euclidean_1nn": scaled_pred,
        "correct_euclidean_1nn": int(pred == int(y_valid[0])),
        "correct_train_std_euclidean_1nn": int(scaled_pred == int(y_valid[0])),
        "nearest_same_distance": same_distance,
        "nearest_different_distance": different_distance,
        "nearest_same_to_different_ratio": same_distance / different_distance if different_distance else None,
        "nearest_same_distance_train_standardized": scaled_same_distance,
        "nearest_different_distance_train_standardized": scaled_different_distance,
        "nearest_same_to_different_ratio_train_standardized": (
            scaled_same_distance / scaled_different_distance if scaled_different_distance else None
        ),
        "lda_coord_std_min": float(train_coordinate_std.min()),
        "lda_coord_std_median": float(np.median(train_coordinate_std)),
        "lda_coord_std_max": float(train_coordinate_std.max()),
        "lda_coord_std_ratio": (
            float(train_coordinate_std.max() / train_coordinate_std[train_coordinate_std > 0].min())
            if np.any(train_coordinate_std > 0)
            else None
        ),
        "lda_output_rank": int(Z_inner.shape[1]),
        **within_diag,
        **between_diag,
    }
    return result, singular_values


def full_outer_train_spectra(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    configurations: list[tuple[str, int | None, np.ndarray]] = [("raw", None, X_train)]
    for dimension in PCA_PRE_DIMS:
        if dimension > X_train.shape[0]:
            raise RuntimeError(f"PCA dimension {dimension} exceeds outer train size {X_train.shape[0]}")
        pca = PCA(n_components=dimension, svd_solver="full")
        pca.fit(X_train)
        transformed = pca.transform(X_train)
        configurations.append((f"pca{dimension}", dimension, transformed))

    for method, dimension, features in configurations:
        diag, spectrum = within_svd_diagnostics(features, y_train)
        lda = LinearDiscriminantAnalysis(
            solver=LDA_SOLVER,
            n_components=FINAL_LDA_DIM,
            tol=LDA_TOL,
        )
        lda.fit(features, y_train)
        between = between_svd_rank(lda, diag)
        summary.append(
            {
                "method": method,
                "pca_pre_dimension": dimension,
                "outer_train_n": int(features.shape[0]),
                "outer_train_features": int(features.shape[1]),
                "outer_train_class_count": int(np.unique(y_train).size),
                "outer_within_rank_upper_bound": int(features.shape[0] - np.unique(y_train).size),
                "pca_cumulative_explained_variance": (
                    float(np.sum(pca.explained_variance_ratio_)) if dimension is not None else None
                ),
                **diag,
                **between,
            }
        )
        for index, value in enumerate(spectrum, start=1):
            rows.append(
                {
                    "method": method,
                    "pca_pre_dimension": dimension,
                    "singular_index": index,
                    "singular_value": float(value),
                    "relative_to_max": float(value / spectrum[0]) if spectrum[0] else 0.0,
                    "above_sklearn_within_tol": bool(value > LDA_TOL),
                }
            )
    return rows, summary


def run(project_root: Path) -> int:
    root = project_root.resolve()
    output_dir = root / "results" / "diagnostics"
    audit_dir = root / "audit"
    log_path = root / "results" / "logs" / "pca_lda_internal_diagnostic.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(log_path)
    start = time.perf_counter()

    X_all = np.load(root / "data" / "processed" / "X.npy", mmap_mode="r", allow_pickle=False)
    y_all = np.load(root / "data" / "processed" / "y.npy", mmap_mode="r", allow_pickle=False)
    split = json.loads((root / "audit" / "split_audit.json").read_text(encoding="utf-8"))
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    # The script intentionally never reads or indexes the seed-42 test_indices field.
    X_train = np.asarray(X_all[train_indices])
    y_train = np.asarray(y_all[train_indices])
    if X_train.shape != (320, 4096) or y_train.shape != (320,):
        raise RuntimeError(f"Unexpected seed-42 outer training shape: {X_train.shape}/{y_train.shape}")
    if not np.array_equal(np.unique(y_train, return_counts=True)[1], np.repeat(8, 40)):
        raise RuntimeError("Expected eight training examples per identity")

    valid_indices = deterministic_internal_validation_indices(y_train)
    if valid_indices.size != 40 or np.unique(valid_indices).size != 40:
        raise RuntimeError("Expected 40 unique within-training validation cases")
    if not np.array_equal(np.sort(y_train[valid_indices]), np.unique(y_train)):
        raise RuntimeError("Internal validation cases must cover each class exactly once")

    logger.info(
        "Starting training-only diagnostic: outer train=%s, heldouts=%d, dimensions=%s",
        X_train.shape,
        valid_indices.size,
        PCA_PRE_DIMS,
    )
    cv_rows: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    for fold_id, valid_local_index in enumerate(valid_indices, start=1):
        for dimension in PCA_PRE_DIMS:
            row, fold_spectrum = fit_eval_one_fold(
                X_train, y_train, int(valid_local_index), dimension, fold_id
            )
            cv_rows.append(row)
            for index, value in enumerate(fold_spectrum, start=1):
                spectra.append(
                    {
                        "fold_id": fold_id,
                        "heldout_class": row["heldout_class"],
                        "pca_pre_dimension": dimension,
                        "singular_index": index,
                        "singular_value": float(value),
                        "relative_to_max": (
                            float(value / fold_spectrum[0]) if fold_spectrum.size and fold_spectrum[0] else 0.0
                        ),
                        "above_sklearn_within_tol": bool(value > LDA_TOL),
                    }
                )
        logger.info("completed internal fold %d/%d", fold_id, len(valid_indices))

    cv_frame = pd.DataFrame(cv_rows)
    spectrum_frame = pd.DataFrame(spectra)
    group = cv_frame.groupby("pca_pre_dimension", sort=True)
    summary = group.agg(
        n_internal_validation=("fold_id", "count"),
        accuracy_euclidean_1nn=("correct_euclidean_1nn", "mean"),
        sd_correctness_euclidean_1nn=("correct_euclidean_1nn", "std"),
        accuracy_train_std_euclidean_1nn=("correct_train_std_euclidean_1nn", "mean"),
        within_rank_tol_mean=("within_rank_sklearn_tol", "mean"),
        within_rank_tol_min=("within_rank_sklearn_tol", "min"),
        within_rank_tol_max=("within_rank_sklearn_tol", "max"),
        within_rank_machine_mean=("within_rank_machine_epsilon", "mean"),
        within_singular_max_mean=("within_singular_max", "mean"),
        within_singular_min_kept_median=("within_singular_min_kept", "median"),
        within_condition_kept_median=("within_condition_kept", "median"),
        between_rank_mean=("between_rank_sklearn_relative_tol", "mean"),
        lda_coord_std_ratio_median=("lda_coord_std_ratio", "median"),
        nearest_same_to_different_ratio_median=("nearest_same_to_different_ratio", "median"),
        nearest_same_to_different_ratio_train_standardized_median=(
            "nearest_same_to_different_ratio_train_standardized",
            "median",
        ),
        pca_explained_variance_mean=("pca_explained_variance", "mean"),
        pca_last_component_variance_mean=("pca_last_component_variance", "mean"),
    ).reset_index()
    cv_frame.to_csv(output_dir / "pca_lda_internal_cv_rows.csv", index=False, float_format="%.12g")
    summary.to_csv(output_dir / "pca_lda_internal_cv_summary.csv", index=False, float_format="%.12g")
    spectrum_frame.to_csv(output_dir / "pca_lda_internal_cv_singular_spectra.csv", index=False, float_format="%.12g")

    outer_spectrum_rows, outer_spectrum_summary = full_outer_train_spectra(X_train, y_train)
    pd.DataFrame(outer_spectrum_rows).to_csv(
        output_dir / "pca_lda_seed42_outer_train_singular_spectra.csv",
        index=False,
        float_format="%.12g",
    )
    pd.DataFrame(outer_spectrum_summary).to_csv(
        output_dir / "pca_lda_seed42_outer_train_rank_summary.csv",
        index=False,
        float_format="%.12g",
    )

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.errorbar(
        summary["pca_pre_dimension"],
        summary["accuracy_euclidean_1nn"],
        yerr=summary["sd_correctness_euclidean_1nn"],
        marker="o",
        capsize=3,
        label="LDA20 + Euclidean 1-NN",
    )
    ax.plot(
        summary["pca_pre_dimension"],
        summary["accuracy_train_std_euclidean_1nn"],
        marker="s",
        linestyle="--",
        label="LDA20 output standardized on inner train + Euclidean 1-NN",
    )
    ax.set_xlabel("PCA intermediate dimension")
    ax.set_ylabel("Accuracy on 40 held-out training examples")
    ax.set_title("Training-only PCA→LDA diagnostic (one internal holdout per class)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "pca_lda_internal_accuracy.png", dpi=220)
    plt.close(fig)

    outer_spectra = pd.DataFrame(outer_spectrum_rows)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for method in ("raw", "pca50", "pca150", "pca280", "pca319"):
        selected = outer_spectra[outer_spectra["method"] == method]
        if not selected.empty:
            ax.semilogy(
                selected["singular_index"],
                np.maximum(selected["singular_value"], np.finfo(float).tiny),
                label=method,
            )
    ax.axhline(LDA_TOL, color="black", linestyle=":", label="svd within tol = 1e-4")
    ax.set_xlabel("Within-class scaled singular index")
    ax.set_ylabel("Singular value (log scale)")
    ax.set_title("LDA SVD within-class spectrum on the full seed-42 training split")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "pca_lda_within_singular_spectrum.png", dpi=220)
    plt.close(fig)

    runtime = time.perf_counter() - start
    finite = bool(np.isfinite(cv_frame.select_dtypes(include=[np.number]).to_numpy()).all())
    rank_bounds = bool(
        (
            cv_frame["within_rank_sklearn_tol"]
            <= cv_frame["within_rank_upper_bound"]
        ).all()
    )
    output_shapes = bool((cv_frame["lda_output_rank"] == FINAL_LDA_DIM).all())
    scope = bool(
        (cv_frame["pca_fit_scope"] == "inner training rows only").all()
        and (cv_frame["lda_fit_scope"] == "inner training rows and labels only").all()
    )
    validation = {
        "status": "PASS" if finite and rank_bounds and output_shapes and scope and len(cv_frame) == 360 else "FAIL",
        "checks": {
            "internal_folds_cover_each_class_once": True,
            "all_folds_use_319_training_rows": bool((cv_frame["inner_train_n"] == 319).all()),
            "all_rows_use_one_internal_validation_case": bool((cv_frame["inner_valid_n"] == 1).all()),
            "finite_metrics": finite,
            "within_rank_not_above_theoretical_bound": rank_bounds,
            "all_lda_output_shapes_are_20": output_shapes,
            "pca_and_lda_fit_only_on_inner_training_rows": scope,
            "matrix_rows_40_folds_x_9_dimensions": len(cv_frame) == 360,
            "uses_outer_test_indices_or_labels": False,
        },
        "failed_checks": [],
        "rows": int(len(cv_frame)),
        "outer_train_n": int(X_train.shape[0]),
        "classes": int(np.unique(y_train).size),
        "outer_within_rank_upper_bound": int(X_train.shape[0] - np.unique(y_train).size),
        "inner_within_rank_upper_bound": int(319 - np.unique(y_train).size),
    }
    if validation["status"] != "PASS":
        validation["failed_checks"] = [
            name for name, passed in validation["checks"].items()
            if passed is False
        ]
    (audit_dir / "pca_lda_internal_diagnostic_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "input_X_shape": list(X_train.shape),
        "input_X_dtype": str(X_train.dtype),
        "outer_split_seed": OUTER_SPLIT_SEED,
        "outer_train_indices_sha256": hashlib.sha256(train_indices.tobytes()).hexdigest(),
        "heldout_selection_seed": FOLD_SELECTION_SEED,
        "heldout_internal_validation_n": int(valid_indices.size),
        "heldout_internal_validation_class_counts": {
            str(label): int(np.sum(y_train[valid_indices] == label))
            for label in np.unique(y_train)
        },
        "pca_pre_dimensions": list(PCA_PRE_DIMS),
        "lda_output_dimension": FINAL_LDA_DIM,
        "lda_solver": LDA_SOLVER,
        "lda_tol": LDA_TOL,
        "classifier": "KNeighborsClassifier(n_neighbors=1, metric='euclidean')",
        "spectrum_formula": "sqrt(1/(n_samples-n_classes)) * (within_class_centered_X / within_class_feature_std)",
        "within_rank_threshold": "singular_value > tol (absolute)",
        "between_rank_threshold": "singular_value > tol * largest_between_singular_value",
        "final_test_data_used": False,
        "runtime_sec": runtime,
        "source_sha256": sha256_file(Path(__file__)),
        "validation_status": validation["status"],
    }
    (audit_dir / "pca_lda_internal_diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("completed diagnostic in %.2f sec; status=%s", runtime, validation["status"])
    print(summary.to_string(index=False))
    print(json.dumps(validation, indent=2))
    return 0 if validation["status"] == "PASS" else 1


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    return run(args.project_root)


if __name__ == "__main__":
    raise SystemExit(main())

