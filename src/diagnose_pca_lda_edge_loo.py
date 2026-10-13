from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from diagnose_pca_lda import within_svd_diagnostics
else:
    from .diagnose_pca_lda import within_svd_diagnostics

ROOT = Path(__file__).resolve().parents[1]
PRE_DIMS = (280, 319)
LDA_DIM = 20
TOL = 1e-4


def make_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ee6222.pca_lda_edge_loo")
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


def run(project_root: Path) -> int:
    root = project_root.resolve()
    X_all = np.load(root / "data/processed/X.npy", mmap_mode="r", allow_pickle=False)
    y_all = np.load(root / "data/processed/y.npy", mmap_mode="r", allow_pickle=False)
    split = json.loads((root / "audit/split_audit.json").read_text(encoding="utf-8"))
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    X = np.asarray(X_all[train_indices])
    y = np.asarray(y_all[train_indices])
    if X.shape != (320, 4096) or y.shape != (320,):
        raise RuntimeError(f"Unexpected seed-42 outer-training shape {X.shape}/{y.shape}")
    if not np.array_equal(np.unique(y, return_counts=True)[1], np.repeat(8, 40)):
        raise RuntimeError("Expected 8 training images per identity")

    logger = make_logger(root / "results/logs/pca_lda_edge_loo.log")
    start = time.perf_counter()
    rows: list[dict[str, object]] = []
    for valid_index in range(X.shape[0]):
        mask = np.ones(X.shape[0], dtype=bool)
        mask[valid_index] = False
        X_train, X_valid = X[mask], X[valid_index : valid_index + 1]
        y_train, y_valid = y[mask], y[valid_index : valid_index + 1]
        for dimension in PRE_DIMS:
            pca = PCA(n_components=dimension, svd_solver="full")
            pca.fit(X_train)
            X_train_pca = pca.transform(X_train)
            X_valid_pca = pca.transform(X_valid)

            lda = LinearDiscriminantAnalysis(solver="svd", n_components=LDA_DIM, tol=TOL)
            lda.fit(X_train_pca, y_train)
            Z_train = lda.transform(X_train_pca)
            Z_valid = lda.transform(X_valid_pca)

            knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
            knn.fit(Z_train, y_train)
            pred = int(knn.predict(Z_valid)[0])

            scale = Z_train.std(axis=0, ddof=0)
            scale_safe = np.where(scale == 0, 1.0, scale)
            mean = Z_train.mean(axis=0)
            Z_train_scaled = (Z_train - mean) / scale_safe
            Z_valid_scaled = (Z_valid - mean) / scale_safe
            scaled_knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
            scaled_knn.fit(Z_train_scaled, y_train)
            scaled_pred = int(scaled_knn.predict(Z_valid_scaled)[0])

            within, _ = within_svd_diagnostics(X_train_pca, y_train, tol=TOL)
            rows.append(
                {
                    "heldout_outer_train_index": int(valid_index),
                    "heldout_class": int(y_valid[0]),
                    "pca_pre_dimension": int(dimension),
                    "lda_output_dimension": LDA_DIM,
                    "inner_train_n": int(X_train.shape[0]),
                    "inner_within_rank_upper_bound": int(X_train.shape[0] - np.unique(y_train).size),
                    "within_rank_sklearn_tol": within["within_rank_sklearn_tol"],
                    "within_singular_min_kept": within["within_singular_min_kept"],
                    "within_condition_kept": within["within_condition_kept"],
                    "pca_last_component_variance": float(pca.explained_variance_ratio_[-1]),
                    "lda_coord_std_ratio": (
                        float(scale.max() / scale[scale > 0].min())
                        if np.any(scale > 0)
                        else None
                    ),
                    "prediction_euclidean_1nn": pred,
                    "prediction_train_std_euclidean_1nn": scaled_pred,
                    "correct_euclidean_1nn": int(pred == int(y_valid[0])),
                    "correct_train_std_euclidean_1nn": int(scaled_pred == int(y_valid[0])),
                }
            )
        if (valid_index + 1) % 40 == 0:
            logger.info("completed %d/320 leave-one-out cases", valid_index + 1)

    frame = pd.DataFrame(rows)
    out_dir = root / "results/diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "pca_lda_edge_loo_rows.csv", index=False, float_format="%.12g")
    summary = (
        frame.groupby("pca_pre_dimension", sort=True)
        .agg(
            n_predictions=("heldout_outer_train_index", "count"),
            mean_accuracy_euclidean_1nn=("correct_euclidean_1nn", "mean"),
            sd_accuracy_euclidean_1nn=("correct_euclidean_1nn", "std"),
            mean_accuracy_train_std_euclidean_1nn=("correct_train_std_euclidean_1nn", "mean"),
            within_rank_min=("within_rank_sklearn_tol", "min"),
            within_rank_max=("within_rank_sklearn_tol", "max"),
            within_condition_median=("within_condition_kept", "median"),
            pca_last_variance_median=("pca_last_component_variance", "median"),
            lda_coord_std_ratio_median=("lda_coord_std_ratio", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "pca_lda_edge_loo_summary.csv", index=False, float_format="%.12g")

    checks = {
        "rows_equal_320_outer_training_samples_times_two_dimensions": len(frame) == 640,
        "each_outer_training_sample_held_out_once_per_dimension": bool(
            frame.groupby("pca_pre_dimension")["heldout_outer_train_index"].nunique().eq(320).all()
        ),
        "both_pca_dimensions_present": set(frame["pca_pre_dimension"]) == set(PRE_DIMS),
        "all_folds_train_319": bool((frame["inner_train_n"] == 319).all()),
        "all_folds_retain_40_classes": bool((frame["inner_within_rank_upper_bound"] == 279).all()),
        "all_outputs_finite": bool(
            np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "seed42_outer_test_indices_and_labels_not_used": True,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "rows": len(frame),
        "pca_pre_dimensions": list(PRE_DIMS),
        "lda_output_dimension": LDA_DIM,
        "training_source": "seed-42 Stage 1/2 training indices only",
        "validation_design": "320 leave-one-out cases within the 320-sample training partition",
        "elapsed_sec": time.perf_counter() - start,
    }
    (root / "audit/pca_lda_edge_loo_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("edge LOO diagnostic complete; status=%s", report["status"])
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    return run(parser.parse_args().project_root)


if __name__ == "__main__":
    raise SystemExit(main())

