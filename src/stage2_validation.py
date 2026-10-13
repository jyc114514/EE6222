"""Validation gates for the additive EE6222 Stage 2 artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPECTED_STAGE1_ACCURACIES = {
    ("raw_1nn", 4096): 0.925,
    ("pca_1nn", 1): 0.125,
    ("pca_1nn", 2): 0.300,
    ("pca_1nn", 5): 0.800,
    ("pca_1nn", 10): 0.875,
    ("pca_1nn", 20): 0.9125,
    ("pca_1nn", 30): 0.9125,
    ("pca_1nn", 40): 0.9125,
    ("pca_1nn", 50): 0.9125,
    ("pca_1nn", 75): 0.925,
    ("pca_1nn", 100): 0.925,
    ("pca_1nn", 150): 0.925,
    ("pca_1nn", 200): 0.925,
}

RESULT_COLUMNS = {
    "method",
    "input_dimension",
    "intermediate_dimension",
    "output_dimension",
    "accuracy",
    "error_rate",
    "n_correct",
    "n_incorrect",
    "explained_variance",
    "solver",
    "classifier",
    "metric",
    "fit_time_sec",
    "predict_time_sec",
    "total_time_sec",
    "random_state",
    "train_size",
    "test_size",
}


def _add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, detail: str
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _finite_metric_check(rows: Iterable[dict[str, Any]]) -> bool:
    for row in rows:
        numeric = [
            row.get("accuracy"),
            row.get("error_rate"),
            row.get("n_correct"),
            row.get("n_incorrect"),
            row.get("fit_time_sec"),
            row.get("predict_time_sec"),
            row.get("total_time_sec"),
        ]
        if not all(np.isfinite(float(value)) for value in numeric):
            return False
        if not 0.0 <= float(row["accuracy"]) <= 1.0:
            return False
        if not np.isclose(float(row["error_rate"]), 1.0 - float(row["accuracy"]), atol=1e-10):
            return False
    return True


def _row_shape_checks(
    rows: Iterable[dict[str, Any]], train_size: int, test_size: int
) -> bool:
    for row in rows:
        output_dimension = int(row["output_dimension"])
        if row.get("train_output_shape") != [train_size, output_dimension]:
            return False
        if row.get("test_output_shape") != [test_size, output_dimension]:
            return False
    return True


def validate_stage2_outputs(
    project_root: Path,
    *,
    baseline_before_hash: str,
    baseline_after_hash: str,
    baseline_frame: pd.DataFrame,
    stage2_frame: pd.DataFrame,
    all_method_frame: pd.DataFrame,
    direct_rows: list[dict[str, Any]],
    pca_lda_rows: list[dict[str, Any]],
    lda_dims: tuple[int, ...],
    dataset_audit: dict[str, Any],
    split_audit: dict[str, Any],
    pca_pre_dimension: int,
    pca_pre_shapes: tuple[list[int], list[int]],
    pca_pre_fit_scope: str,
    failures: list[dict[str, Any]],
    figure_names: tuple[str, ...],
) -> dict[str, Any]:
    """Run fail-closed checks without changing any experiment artifacts."""
    checks: list[dict[str, Any]] = []
    required_files = [
        project_root / "results" / "tables" / "stage2_results.csv",
        project_root / "results" / "tables" / "all_method_results.csv",
        project_root / "results" / "stage2_summary.txt",
        project_root / "results" / "stage2_summary.json",
        project_root / "audit" / "stage2_run_manifest.json",
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
        "baseline:hash_unchanged",
        baseline_before_hash == baseline_after_hash,
        f"before={baseline_before_hash}, after={baseline_after_hash}",
    )
    stage1_validation = project_root / "audit" / "validation_report.txt"
    _add_check(
        checks,
        "baseline:stage1_validation_pass",
        stage1_validation.is_file()
        and stage1_validation.read_text(encoding="utf-8").splitlines()[:1] == ["PASS"],
        "existing audit/validation_report.txt starts with PASS",
    )
    baseline_accuracy_ok = True
    missing_baseline: list[str] = []
    for (method, dimension), expected_accuracy in EXPECTED_STAGE1_ACCURACIES.items():
        matches = baseline_frame[
            (baseline_frame["method"] == method)
            & (baseline_frame["dimension"] == dimension)
        ]
        if len(matches) != 1 or not np.isclose(
            float(matches.iloc[0]["accuracy"]), expected_accuracy, atol=1e-12
        ):
            baseline_accuracy_ok = False
            missing_baseline.append(f"{method}:{dimension}")
    _add_check(
        checks,
        "baseline:accuracy_values_preserved",
        baseline_accuracy_ok,
        "all frozen Stage 1 raw/PCA accuracy values match"
        if baseline_accuracy_ok
        else f"mismatch={missing_baseline}",
    )
    _add_check(
        checks,
        "baseline:row_count_preserved",
        len(baseline_frame) == len(EXPECTED_STAGE1_ACCURACIES),
        f"rows={len(baseline_frame)}, expected={len(EXPECTED_STAGE1_ACCURACIES)}",
    )

    _add_check(
        checks,
        "dataset:shape_and_finite",
        dataset_audit.get("X_shape") == [400, 4096]
        and dataset_audit.get("y_shape") == [400]
        and dataset_audit.get("nan_count") == 0
        and dataset_audit.get("inf_count") == 0,
        f"X={dataset_audit.get('X_shape')}, y={dataset_audit.get('y_shape')}, "
        f"NaN={dataset_audit.get('nan_count')}, Inf={dataset_audit.get('inf_count')}",
    )
    _add_check(
        checks,
        "split:reused_and_disjoint",
        split_audit.get("train_size") == 320
        and split_audit.get("test_size") == 80
        and split_audit.get("overall_pass") is True
        and split_audit["checks"]["no_index_overlap"] is True,
        f"train={split_audit.get('train_size')}, test={split_audit.get('test_size')}, "
        f"overall_pass={split_audit.get('overall_pass')}",
    )

    expected_lda_set = set(lda_dims)
    direct_dimensions = {int(row["output_dimension"]) for row in direct_rows}
    pca_lda_dimensions = {int(row["output_dimension"]) for row in pca_lda_rows}
    _add_check(
        checks,
        "lda:all_requested_dimensions_executed",
        not failures
        and direct_dimensions == expected_lda_set
        and pca_lda_dimensions == expected_lda_set,
        f"direct={sorted(direct_dimensions)}, pca_lda={sorted(pca_lda_dimensions)}, "
        f"failures={failures}",
    )
    _add_check(
        checks,
        "lda:legal_dimensions",
        all(1 <= int(row["output_dimension"]) <= 39 for row in [*direct_rows, *pca_lda_rows]),
        "all output dimensions satisfy d <= min(features, classes - 1)=39",
    )
    _add_check(
        checks,
        "lda:finite_metrics_and_shapes",
        _finite_metric_check([*direct_rows, *pca_lda_rows])
        and _row_shape_checks([*direct_rows, *pca_lda_rows], 320, 80),
        "finite bounded metrics and expected train/test output shapes",
    )
    _add_check(
        checks,
        "lda:training_only_scope",
        all(
            row.get("preprocessing_fit_scope") == "X_train only"
            for row in [*direct_rows, *pca_lda_rows]
        ),
        "all LDA rows declare X_train-only fitting",
    )

    _add_check(
        checks,
        "pca_lda:preprocessing_dimension",
        pca_pre_dimension == 280
        and pca_pre_dimension <= min(320, 4096)
        and pca_pre_shapes == ([320, 280], [80, 280]),
        f"pre_dim={pca_pre_dimension}, shapes={pca_pre_shapes}",
    )
    _add_check(
        checks,
        "pca_lda:training_only_pca_scope",
        pca_pre_fit_scope == "X_train only"
        and all(row.get("pca_fit_scope") == "X_train only" for row in pca_lda_rows),
        "PCA(280) fit scope is X_train only",
    )

    _add_check(
        checks,
        "stage2:schema",
        RESULT_COLUMNS.issubset(stage2_frame.columns),
        f"missing={sorted(RESULT_COLUMNS.difference(stage2_frame.columns))}",
    )
    _add_check(
        checks,
        "stage2:row_count_and_methods",
        len(stage2_frame) == 2 * len(lda_dims)
        and set(stage2_frame.get("method", []))
        == {"lda_1nn", "pca280_lda_1nn"},
        f"rows={len(stage2_frame)}, methods={sorted(set(stage2_frame.get('method', [])))}",
    )
    _add_check(
        checks,
        "all_methods:row_count_and_methods",
        len(all_method_frame) == len(baseline_frame) + 2 * len(lda_dims)
        and set(all_method_frame.get("method", []))
        == {"raw_1nn", "pca_1nn", "lda_1nn", "pca280_lda_1nn"},
        f"rows={len(all_method_frame)}, methods={sorted(set(all_method_frame.get('method', [])))}",
    )

    passed = bool(all(item["passed"] for item in checks))
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "checks": checks,
        "failed_checks": [item for item in checks if not item["passed"]],
    }


def validation_text(report: dict[str, Any]) -> str:
    """Render the validation report with a machine-readable first line."""
    lines = [
        str(report["status"]),
        "EE6222 Assignment 1 - Stage 2 Validation Report",
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
