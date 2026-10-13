"""Fail-closed validation for Stage 3 robustness and diagnostic artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RESULT_COLUMNS = {
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
    "solver",
    "classifier",
    "metric",
    "fit_time_sec",
    "predict_time_sec",
    "total_time_sec",
    "train_size",
    "test_size",
}
EXPECTED_METHOD_COUNTS = {
    "raw_1nn": 20,
    "pca_1nn": 240,
    "lda_1nn": 200,
    "pca280_lda_1nn": 200,
    "pca280_1nn": 20,
    "pca319_lda_1nn": 200,
}


def _add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, detail: str
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _finite_metrics(frame: pd.DataFrame) -> bool:
    numeric_columns = [
        "accuracy",
        "error_rate",
        "n_correct",
        "n_incorrect",
        "fit_time_sec",
        "predict_time_sec",
        "total_time_sec",
    ]
    if not set(numeric_columns).issubset(frame.columns):
        return False
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            return False
    accuracy = frame["accuracy"].to_numpy(dtype=float)
    error_rate = frame["error_rate"].to_numpy(dtype=float)
    return bool(
        ((accuracy >= 0.0) & (accuracy <= 1.0)).all()
        and np.allclose(error_rate, 1.0 - accuracy, atol=1e-10)
    )


def _paired_hashes(frame: pd.DataFrame, seeds: tuple[int, ...]) -> bool:
    if not {"train_index_hash", "test_index_hash", "split_id"}.issubset(frame.columns):
        return False
    for seed in seeds:
        group = frame[frame["seed"] == seed]
        if len(group) != 44:
            return False
        if group["train_index_hash"].nunique() != 1:
            return False
        if group["test_index_hash"].nunique() != 1:
            return False
        if group["split_id"].nunique() != 1 or group["split_id"].iloc[0] != f"seed_{seed}":
            return False
    return True


def validate_stage3_outputs(
    project_root: Path,
    *,
    before_hashes: dict[str, str],
    after_hashes: dict[str, str],
    raw_frame: pd.DataFrame,
    aggregate_frame: pd.DataFrame,
    paired_frame: pd.DataFrame,
    split_audits: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    seeds: tuple[int, ...],
    pca_dims: tuple[int, ...],
    lda_dims: tuple[int, ...],
    figure_names: tuple[str, ...],
) -> dict[str, Any]:
    """Return structured PASS/FAIL evidence for all Stage 3 acceptance gates."""
    checks: list[dict[str, Any]] = []
    required_files = [
        project_root / "results" / "tables" / "stage3_robustness_raw.csv",
        project_root / "results" / "tables" / "stage3_robustness_summary.csv",
        project_root / "results" / "tables" / "stage3_paired_comparisons.csv",
        project_root / "results" / "stage3_summary.txt",
        project_root / "results" / "stage3_summary.json",
        project_root / "audit" / "stage3_run_manifest.json",
        project_root / "audit" / "stage3_split_audits.json",
        project_root / "audit" / "stage3_progress.json",
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
        "preservation:stage1_stage2_hashes_unchanged",
        before_hashes == after_hashes and set(before_hashes) == set(after_hashes),
        "all frozen Stage 1/Stage 2 hashes match before and after Stage 3"
        if before_hashes == after_hashes
        else f"before={before_hashes}, after={after_hashes}",
    )
    for relative, label in (
        ("audit/validation_report.txt", "stage1"),
        ("audit/stage2_validation_report.txt", "stage2"),
    ):
        path = project_root / relative
        _add_check(
            checks,
            f"preservation:{label}_validation_pass",
            path.is_file() and path.read_text(encoding="utf-8").splitlines()[:1] == ["PASS"],
            f"{relative} starts with PASS",
        )

    dataset_audit_path = project_root / "audit" / "dataset_audit.json"
    dataset_audit = (
        json.loads(dataset_audit_path.read_text(encoding="utf-8"))
        if dataset_audit_path.is_file()
        else {}
    )
    _add_check(
        checks,
        "dataset:official_shape_and_finite",
        raw_frame.shape[0] > 0
        and dataset_audit.get("X_shape") == [400, 4096]
        and dataset_audit.get("nan_count") == 0
        and dataset_audit.get("inf_count") == 0,
        "X=[400,4096], finite feature values",
    )

    split_seed_set = {int(item.get("seed")) for item in split_audits}
    split_valid = len(split_audits) == len(seeds) and split_seed_set == set(seeds)
    for item in split_audits:
        checks_for_seed = item.get("checks", {})
        split_valid = split_valid and item.get("overall_pass") is True and all(
            checks_for_seed.get(name) is True
            for name in (
                "no_index_overlap",
                "sizes_sum_to_total",
                "train_size_320",
                "test_size_80",
                "all_classes_have_8_train_2_test",
            )
        )
    _add_check(
        checks,
        "split:all_seeds_stratified_and_disjoint",
        split_valid,
        f"audits={len(split_audits)}, seeds={sorted(split_seed_set)}",
    )

    _add_check(
        checks,
        "rows:expected_count",
        len(raw_frame) == 880,
        f"actual={len(raw_frame)}, expected=880",
    )
    method_counts = raw_frame["method"].value_counts().to_dict() if "method" in raw_frame else {}
    _add_check(
        checks,
        "rows:method_counts",
        method_counts == EXPECTED_METHOD_COUNTS,
        f"actual={method_counts}, expected={EXPECTED_METHOD_COUNTS}",
    )
    _add_check(
        checks,
        "rows:required_schema",
        RESULT_COLUMNS.issubset(raw_frame.columns),
        f"missing={sorted(RESULT_COLUMNS.difference(raw_frame.columns))}",
    )
    _add_check(
        checks,
        "rows:finite_accuracy_and_error",
        _finite_metrics(raw_frame),
        "all metrics finite, bounded, and error_rate=1-accuracy",
    )
    _add_check(
        checks,
        "rows:paired_split_hashes",
        _paired_hashes(raw_frame, seeds),
        "each seed has one shared train/test hash across its 44 method rows",
    )

    dimensions_valid = True
    for method, group in raw_frame.groupby("method"):
        output = group["output_dimension"].to_numpy(dtype=int)
        if method == "raw_1nn":
            dimensions_valid &= set(output) == {4096}
        elif method == "pca_1nn":
            dimensions_valid &= set(output) == set(pca_dims)
        elif method == "lda_1nn":
            dimensions_valid &= set(output) == set(lda_dims) and bool((output <= 39).all())
        elif method == "pca280_1nn":
            dimensions_valid &= set(output) == {280}
        elif method == "pca280_lda_1nn":
            dimensions_valid &= set(output) == set(lda_dims) and bool((output <= 39).all())
            dimensions_valid &= set(group["intermediate_dimension"].dropna().astype(int)) == {280}
        elif method == "pca319_lda_1nn":
            dimensions_valid &= set(output) == set(lda_dims) and bool((output <= 39).all())
            dimensions_valid &= set(group["intermediate_dimension"].dropna().astype(int)) == {319}
        else:
            dimensions_valid = False
    _add_check(checks, "dimensions:fixed_and_legal", dimensions_valid, "all predefined dimensions present")

    scope_valid = bool(
        (raw_frame.loc[raw_frame["method"] == "raw_1nn", "preprocessing_fit_scope"] == "none").all()
        and (
            raw_frame.loc[raw_frame["method"] != "raw_1nn", "preprocessing_fit_scope"]
            == "X_train only"
        ).all()
        and (
            raw_frame.loc[
                raw_frame["method"].isin(
                    ["pca_1nn", "pca280_1nn", "pca280_lda_1nn", "pca319_lda_1nn"]
                ),
                "pca_fit_scope",
            ]
            == "X_train only"
        ).all()
    )
    _add_check(checks, "leakage:training_only_scopes", scope_valid, "all fitted transforms declare current X_train only")

    _add_check(
        checks,
        "aggregation:20_seed_groups",
        not aggregate_frame.empty
        and "n_seeds" in aggregate_frame
        and (aggregate_frame["n_seeds"] == 20).all(),
        f"groups={len(aggregate_frame)}",
    )
    required_paired = {
        "PCA75 - Raw",
        "LDA25 - Raw",
        "PCA280->LDA20 - Raw",
        "PCA319->LDA25 - Direct LDA25",
        "PCA280-only - Raw",
    }
    _add_check(
        checks,
        "paired:required_comparisons",
        set(paired_frame.get("comparison", [])) == required_paired
        and (paired_frame.get("n_pairs", pd.Series(dtype=int)) == 20).all(),
        f"comparisons={sorted(set(paired_frame.get('comparison', [])))}",
    )
    _add_check(
        checks,
        "failures:none",
        not failures,
        f"failure_count={len(failures)}",
    )

    passed = bool(all(item["passed"] for item in checks))
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "checks": checks,
        "failed_checks": [item for item in checks if not item["passed"]],
    }


def validation_text(report: dict[str, Any]) -> str:
    lines = [
        str(report["status"]),
        "EE6222 Assignment 1 - Stage 3 Validation Report",
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
