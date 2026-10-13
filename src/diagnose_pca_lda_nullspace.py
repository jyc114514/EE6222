from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PCA_DIMS = (280, 319)
SPLIT_SEED = 20261012
N_OUTER = 5
LDA_TOL = 1e-4


def measure(X: np.ndarray, y: np.ndarray, split_name: str, pca_dim: int) -> dict[str, object]:
    pca = PCA(n_components=pca_dim, svd_solver="full").fit(X)
    Z = pca.transform(X)
    labels = np.unique(y)
    n, p = Z.shape
    c = labels.size
    means = np.vstack([Z[y == label].mean(axis=0) for label in labels])
    centered = np.concatenate([Z[y == label] - means[i] for i, label in enumerate(labels)])
    within_std = centered.std(axis=0)
    within_std[within_std == 0] = 1.0
    within_scaled = np.sqrt(1.0 / (n - c)) * (centered / within_std)
    _, singular, vt = svd(within_scaled, full_matrices=False, check_finite=False)
    retained = int(np.sum(singular > LDA_TOL))

    priors = np.full(c, 1.0 / c)
    xbar = priors @ means
    between_scaled = (
        np.sqrt(n * priors / (c - 1))[:, None] * (means - xbar) / within_std
    )
    retained_energy = float(np.square(between_scaled @ vt[:retained].T).sum())
    dropped_energy = float(np.square(between_scaled @ vt[retained:].T).sum())
    total_energy = retained_energy + dropped_energy
    return {
        "split": split_name,
        "pca_pre_dimension": pca_dim,
        "n_samples": n,
        "n_classes": c,
        "n_features": p,
        "within_rank_upper_bound": min(p, n - c),
        "within_singular_count": int(singular.size),
        "within_rank_sklearn_tol_1e-4": retained,
        "within_discarded_dimensions": int(p - retained),
        "within_min_retained_singular": float(singular[retained - 1]),
        "within_max_singular": float(singular[0]),
        "within_condition_retained": float(singular[0] / singular[retained - 1]),
        "between_center_energy_dropped_by_within_rank_truncation": dropped_energy,
        "between_center_energy_retained": retained_energy,
        "between_center_energy_dropped_fraction": (
            dropped_energy / total_energy if total_energy else 0.0
        ),
        "pca_cumulative_variance": float(pca.explained_variance_ratio_.sum()),
        "pca_last_component_variance_ratio": float(pca.explained_variance_ratio_[-1]),
    }


def main() -> int:
    X = np.load(ROOT / "data/processed/X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(ROOT / "data/processed/y.npy", mmap_mode="r", allow_pickle=False)
    rows: list[dict[str, object]] = []

    split_audit = json.loads((ROOT / "audit/split_audit.json").read_text(encoding="utf-8"))
    train = np.asarray(split_audit["train_indices"], dtype=np.int64)
    rows.extend(measure(np.asarray(X[train]), np.asarray(y[train]), "seed42_outer_train", d) for d in PCA_DIMS)

    outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SPLIT_SEED)
    for fold, (train_idx, _) in enumerate(outer.split(X, y), start=1):
        for dimension in PCA_DIMS:
            rows.append(measure(np.asarray(X[train_idx]), np.asarray(y[train_idx]), f"nested_outer_train_{fold}", dimension))

    frame = pd.DataFrame(rows)
    out_path = ROOT / "results/diagnostics/pca_lda_within_nullspace.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False, float_format="%.12g")
    expected_splits = {"seed42_outer_train", *(f"nested_outer_train_{i}" for i in range(1, N_OUTER + 1))}
    checks = {
        "six_training_partitions": set(frame["split"]) == expected_splits,
        "both_pca_dimensions_present_per_partition": bool(
            frame.groupby("split")["pca_pre_dimension"].nunique().eq(len(PCA_DIMS)).all()
        ),
        "within_rank_respects_theoretical_bound": bool(
            (frame["within_rank_sklearn_tol_1e-4"] <= frame["within_rank_upper_bound"]).all()
        ),
        "pca319_has_39_dropped_within_directions": bool(
            (frame[frame.pca_pre_dimension == 319]["within_discarded_dimensions"] == 39).all()
        ),
        "pca280_is_full_within_rank": bool(
            (frame[frame.pca_pre_dimension == 280]["within_rank_sklearn_tol_1e-4"] == 280).all()
        ),
        "all_numeric_outputs_finite": bool(np.isfinite(frame.select_dtypes(include=[np.number])).all().all()),
    }
    validation = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "within_svd_tol": LDA_TOL,
        "source": str(out_path.relative_to(ROOT)),
    }
    (ROOT / "audit/pca_lda_within_nullspace_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False))
    print(json.dumps(validation, indent=2))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
