"""Statistical aggregation and paired-comparison helpers for Stage 3."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


DIAGNOSTIC_NEAR_TOLERANCE = 0.02
DIAGNOSTIC_LARGE_GAP = 0.10
DIAGNOSTIC_RESIDUAL_GAP = 0.05


def confidence_interval_95(values: Iterable[float]) -> dict[str, float | int]:
    """Return sample SD and a two-sided Student-t 95% CI for a mean."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("at least two observations are required for a CI")
    if not np.isfinite(array).all():
        raise ValueError("CI values must be finite")
    n = int(array.size)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1))
    standard_error = std / np.sqrt(n)
    t_critical = float(stats.t.ppf(0.975, df=n - 1))
    margin = t_critical * standard_error
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": float(standard_error),
        "t_critical": t_critical,
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
    }


def _optional_numeric_summary(values: pd.Series) -> tuple[float | None, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return None, None
    return float(np.mean(numeric)), float(np.std(numeric, ddof=1)) if numeric.size > 1 else 0.0


def aggregate_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-level rows by method and output representation dimension."""
    if frame.empty:
        raise ValueError("cannot aggregate an empty result frame")
    required = {"seed", "method", "output_dimension", "accuracy", "error_rate", "n_correct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing aggregation columns: {sorted(missing)}")
    working = frame.copy()
    if "intermediate_dimension" not in working:
        working["intermediate_dimension"] = np.nan
    if "input_dimension" not in working:
        working["input_dimension"] = np.nan
    grouping = ["method", "intermediate_dimension", "output_dimension"]
    rows: list[dict[str, Any]] = []
    for keys, group in working.groupby(grouping, dropna=False, sort=True):
        method, intermediate, output = keys
        accuracy_stats = confidence_interval_95(group["accuracy"].to_numpy(dtype=float))
        error_stats = confidence_interval_95(group["error_rate"].to_numpy(dtype=float))
        mean_ev, std_ev = _optional_numeric_summary(group.get("explained_variance", pd.Series(dtype=float)))
        rows.append(
            {
                "method": method,
                "input_dimension": (
                    int(group["input_dimension"].dropna().iloc[0])
                    if group["input_dimension"].notna().any()
                    else None
                ),
                "intermediate_dimension": (
                    int(intermediate) if pd.notna(intermediate) else None
                ),
                "output_dimension": int(output),
                "dimension": int(output),
                "n_seeds": int(len(group)),
                "mean_accuracy": accuracy_stats["mean"],
                "std_accuracy": accuracy_stats["std"],
                "min_accuracy": float(group["accuracy"].min()),
                "max_accuracy": float(group["accuracy"].max()),
                "median_accuracy": float(group["accuracy"].median()),
                "ci95_lower": accuracy_stats["ci95_lower"],
                "ci95_upper": accuracy_stats["ci95_upper"],
                "mean_error_rate": error_stats["mean"],
                "std_error_rate": error_stats["std"],
                "mean_correct": float(group["n_correct"].mean()),
                "mean_explained_variance": mean_ev,
                "std_explained_variance": std_ev,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["method", "intermediate_dimension", "output_dimension"],
        na_position="first",
    ).reset_index(drop=True)


def _config_mask(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    mask = (frame["method"] == config["method"]) & (
        frame["output_dimension"] == config["output_dimension"]
    )
    intermediate = config.get("intermediate_dimension")
    if intermediate is None:
        mask &= frame["intermediate_dimension"].isna()
    else:
        mask &= frame["intermediate_dimension"] == intermediate
    return mask


def make_paired_comparison(
    frame: pd.DataFrame,
    *,
    name: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """Compute seed-paired left-minus-right accuracy statistics."""
    working = frame.copy()
    if "intermediate_dimension" not in working:
        working["intermediate_dimension"] = np.nan
    left_frame = working.loc[_config_mask(working, left), ["seed", "accuracy"]].rename(
        columns={"accuracy": "left_accuracy"}
    )
    right_frame = working.loc[_config_mask(working, right), ["seed", "accuracy"]].rename(
        columns={"accuracy": "right_accuracy"}
    )
    if left_frame["seed"].duplicated().any() or right_frame["seed"].duplicated().any():
        raise ValueError(f"comparison {name} requires one row per seed and configuration")
    paired = left_frame.merge(right_frame, on="seed", how="inner", validate="one_to_one")
    if len(paired) != len(left_frame) or len(paired) != len(right_frame):
        raise ValueError(f"comparison {name} has unmatched seeds")
    if paired.empty:
        raise ValueError(f"comparison {name} has no paired observations")
    deltas = paired["left_accuracy"].to_numpy(dtype=float) - paired["right_accuracy"].to_numpy(dtype=float)
    ties = np.isclose(deltas, 0.0, atol=1e-12)
    return {
        "comparison": name,
        "left_method": left["method"],
        "left_output_dimension": int(left["output_dimension"]),
        "right_method": right["method"],
        "right_output_dimension": int(right["output_dimension"]),
        "n_pairs": int(deltas.size),
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
        "median_delta": float(np.median(deltas)),
        "min_delta": float(np.min(deltas)),
        "max_delta": float(np.max(deltas)),
        "wins": int(np.sum(deltas > 1e-12)),
        "ties": int(np.sum(ties)),
        "losses": int(np.sum(deltas < -1e-12)),
        "delta_ci95_lower": confidence_interval_95(deltas)["ci95_lower"]
        if deltas.size > 1
        else float(deltas[0]),
        "delta_ci95_upper": confidence_interval_95(deltas)["ci95_upper"]
        if deltas.size > 1
        else float(deltas[0]),
        "seed_deltas": [
            {"seed": int(seed), "delta": float(delta)}
            for seed, delta in zip(paired["seed"], deltas)
        ],
    }


def smallest_dimension_within_tolerance(
    summary: pd.DataFrame, method: str, tolerance: float = 0.01
) -> int:
    """Return the smallest tested dimension within ``tolerance`` of max mean."""
    subset = summary[summary["method"] == method]
    if subset.empty:
        raise ValueError(f"no summary rows for method {method}")
    maximum = float(subset["mean_accuracy"].max())
    # Include values exactly on the descriptive boundary despite binary
    # floating-point representation (for example, 0.665625 - 0.655625).
    boundary = maximum - float(tolerance) - 1e-12
    candidates = subset[subset["mean_accuracy"] >= boundary]
    return int(candidates["output_dimension"].min())


def diagnostic_interpretation(metrics: dict[str, float]) -> list[str]:
    """Return conservative, thresholded interpretations of PCA->LDA controls."""
    raw = metrics["raw_mean_accuracy"]
    pca280_only = metrics["pca280_only_mean_accuracy"]
    direct = metrics["direct_lda25_mean_accuracy"]
    pca280_lda = metrics["pca280_lda20_mean_accuracy"]
    pca319_lda = metrics["pca319_lda25_mean_accuracy"]
    statements: list[str] = []
    if (
        abs(pca280_only - raw) <= DIAGNOSTIC_NEAR_TOLERANCE
        and raw - pca280_lda >= DIAGNOSTIC_LARGE_GAP
    ):
        statements.append(
            "PCA280-only remains close to raw while PCA280->LDA is substantially worse; "
            "the degradation is unlikely to be explained solely by PCA280 harming "
            "nearest-neighbor classification and is consistent with a PCA/LDA subspace interaction."
        )
    if (
        abs(pca319_lda - direct) <= DIAGNOSTIC_NEAR_TOLERANCE
        and direct - pca280_lda >= DIAGNOSTIC_LARGE_GAP
    ):
        statements.append(
            "PCA319->LDA is close to direct LDA while PCA280->LDA is substantially worse; "
            "this is consistent with the hypothesis that truncation near 280 components "
            "removes low-variance directions useful for discrimination."
        )
    elif direct - pca319_lda >= DIAGNOSTIC_RESIDUAL_GAP:
        statements.append(
            "PCA319->LDA remains meaningfully below direct LDA; the 280-dimensional "
            "truncation alone cannot explain the full degradation, and explicit PCA "
            "subspace or numerical effects remain possible."
        )
    if not statements:
        statements.append(
            "The predefined controls do not isolate a single explanation; the observed "
            "PCA->LDA difference remains unresolved within the stated diagnostic comparisons."
        )
    return statements
