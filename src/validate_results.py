"""Fail-closed consistency checks for the Stage 1 generated artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from .utils import PCA_DIMS
except ImportError:  # direct execution compatibility
    from utils import PCA_DIMS


RESULT_COLUMNS = {
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
}


def _add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, detail: str
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def validate_outputs(
    project_root: Path,
    dataset_audit: dict[str, Any],
    split_audit: dict[str, Any],
    result_rows: Iterable[dict[str, Any]],
    explained_variance: pd.DataFrame,
    figure_names: Iterable[str],
) -> dict[str, Any]:
    """Validate generated values, schemas, and required files."""
    checks: list[dict[str, Any]] = []
    result_frame = pd.DataFrame(list(result_rows))
    required_files = [
        project_root / "audit" / "environment_audit.txt",
        project_root / "audit" / "environment_audit.json",
        project_root / "audit" / "dataset_audit.txt",
        project_root / "audit" / "dataset_audit.json",
        project_root / "audit" / "split_audit.json",
        project_root / "audit" / "run_manifest.json",
        project_root / "results" / "tables" / "baseline_results.csv",
        project_root / "results" / "tables" / "pca_explained_variance.csv",
        project_root / "results" / "baseline_summary.txt",
        project_root / "results" / "baseline_summary.json",
    ]
    required_files.extend(project_root / "results" / "figures" / name for name in figure_names)
    for path in required_files:
        _add_check(
            checks,
            f"file:{path.relative_to(project_root)}",
            path.is_file() and path.stat().st_size > 0,
            "exists and is non-empty" if path.is_file() else "missing or empty",
        )

    _add_check(
        checks,
        "dataset:2d_X",
        dataset_audit["X_shape"] and len(dataset_audit["X_shape"]) == 2,
        f"X_shape={dataset_audit['X_shape']}",
    )
    _add_check(
        checks,
        "dataset:sample_label_count",
        dataset_audit["integrity_checks"]["sample_count_matches_labels"],
        f"X samples={dataset_audit['number_of_samples']}, labels={dataset_audit['y_shape']}",
    )
    _add_check(
        checks,
        "dataset:no_nan_inf",
        dataset_audit["nan_count"] == 0 and dataset_audit["inf_count"] == 0,
        f"NaN={dataset_audit['nan_count']}, Inf={dataset_audit['inf_count']}",
    )
    _add_check(
        checks,
        "dataset:official_shape_and_classes",
        bool(dataset_audit["overall_integrity_pass"]),
        f"classes={dataset_audit['number_of_classes']}, counts={dataset_audit['samples_per_class']}",
    )

    for name, passed in split_audit["checks"].items():
        _add_check(checks, f"split:{name}", bool(passed), str(passed))

    _add_check(
        checks,
        "results:required_columns",
        RESULT_COLUMNS.issubset(result_frame.columns),
        f"missing={sorted(RESULT_COLUMNS.difference(result_frame.columns))}",
    )
    _add_check(
        checks,
        "results:row_count",
        len(result_frame) == 1 + len(tuple(PCA_DIMS)),
        f"rows={len(result_frame)}, expected={1 + len(PCA_DIMS)}",
    )
    _add_check(
        checks,
        "results:methods",
        set(result_frame.get("method", [])) == {"raw_1nn", "pca_1nn"},
        f"methods={sorted(set(result_frame.get('method', [])))}",
    )

    if not result_frame.empty and "accuracy" in result_frame:
        accuracy = pd.to_numeric(result_frame["accuracy"], errors="coerce").to_numpy()
        error_rate = pd.to_numeric(result_frame["error_rate"], errors="coerce").to_numpy()
        finite = bool(np.isfinite(accuracy).all() and np.isfinite(error_rate).all())
        bounds = bool(((accuracy >= 0.0) & (accuracy <= 1.0)).all())
        complement = bool(np.allclose(error_rate, 1.0 - accuracy, atol=1e-10))
    else:
        finite = bounds = complement = False
    _add_check(checks, "results:finite_metrics", finite, f"finite={finite}")
    _add_check(checks, "results:accuracy_bounds", bounds, f"bounds={bounds}")
    _add_check(checks, "results:error_complement", complement, f"allclose={complement}")

    pca_frame = explained_variance
    pca_columns = {
        "component",
        "explained_variance_ratio",
        "cumulative_explained_variance_ratio",
    }
    _add_check(
        checks,
        "variance:required_columns",
        pca_columns.issubset(pca_frame.columns),
        f"missing={sorted(pca_columns.difference(pca_frame.columns))}",
    )
    if pca_columns.issubset(pca_frame.columns) and not pca_frame.empty:
        ratios = pca_frame["explained_variance_ratio"].to_numpy(dtype=float)
        cumulative = pca_frame["cumulative_explained_variance_ratio"].to_numpy(dtype=float)
        variance_ok = bool(
            np.isfinite(ratios).all()
            and np.isfinite(cumulative).all()
            and (ratios >= -1e-12).all()
            and (ratios <= 1.0 + 1e-12).all()
            and (cumulative >= -1e-12).all()
            and (cumulative <= 1.0 + 1e-10).all()
            and np.all(np.diff(cumulative) >= -1e-12)
        )
        components_ok = bool(
            np.array_equal(
                pca_frame["component"].to_numpy(dtype=int),
                np.arange(1, len(pca_frame) + 1),
            )
        )
    else:
        variance_ok = components_ok = False
    _add_check(checks, "variance:finite_monotonic_bounded", variance_ok, str(variance_ok))
    _add_check(checks, "variance:sequential_components", components_ok, str(components_ok))

    passed = bool(all(item["passed"] for item in checks))
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "checks": checks,
        "failed_checks": [item for item in checks if not item["passed"]],
    }


def validation_text(report: dict[str, Any]) -> str:
    """Render the validation report with an explicit first-line status."""
    lines = [
        f"{report['status']}",
        "EE6222 Assignment 1 - Stage 1 Validation Report",
        "",
    ]
    lines.extend(
        f"[{('PASS' if item['passed'] else 'FAIL')}] {item['name']}: {item['detail']}"
        for item in report["checks"]
    )
    if report["failed_checks"]:
        lines.extend(
            [
                "",
                "Failed checks:",
                *[f"- {item['name']}: {item['detail']}" for item in report["failed_checks"]],
            ]
        )
    return "\n".join(lines) + "\n"

