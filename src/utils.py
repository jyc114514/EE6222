"""Shared configuration and small, testable helpers for Stage 1."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

PCA_DIMS: tuple[int, ...] = (1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200)
RANDOM_STATE = 42
TEST_SIZE = 0.20
ORIGINAL_DIMENSION = 64 * 64
EXPECTED_SAMPLES = 400
EXPECTED_CLASSES = 40
EXPECTED_SAMPLES_PER_CLASS = 10


def compute_error_rate(accuracy: float) -> float:
    """Return the classification error rate for an accuracy in [0, 1]."""
    accuracy_value = float(accuracy)
    if not 0.0 <= accuracy_value <= 1.0:
        raise ValueError(f"accuracy must be in [0, 1], got {accuracy_value}")
    return 1.0 - accuracy_value


def relative_dimensionality_reduction(
    reduced_dimension: int, original_dimension: int
) -> float:
    """Return the fraction of dimensions removed by a reduction."""
    if reduced_dimension <= 0 or original_dimension <= 0:
        raise ValueError("dimensions must be positive")
    if reduced_dimension > original_dimension:
        raise ValueError("reduced dimension cannot exceed original dimension")
    return 1.0 - (float(reduced_dimension) / float(original_dimension))


def components_for_threshold(
    cumulative_variance: Sequence[float], threshold: float
) -> int | None:
    """Return the first component count reaching ``threshold`` or ``None``."""
    threshold_value = float(threshold)
    if not 0.0 < threshold_value <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    for component, cumulative in enumerate(cumulative_variance, start=1):
        if float(cumulative) >= threshold_value:
            return component
    return None


def sha256_file(path: Path) -> str:
    """Hash a file without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(project_root: Path) -> dict[str, str]:
    """Return SHA-256 hashes for tracked-in-scope Python source files."""
    source_root = project_root / "src"
    return {
        str(path.relative_to(project_root)): sha256_file(path)
        for path in sorted(source_root.glob("*.py"))
        if path.is_file()
    }


def package_versions(package_names: Iterable[str]) -> dict[str, str | None]:
    """Import packages and return versions, recording missing packages safely."""
    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            module = importlib.import_module(name)
        except Exception:
            versions[name] = None
        else:
            versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def git_commit_hash(project_root: Path) -> str | None:
    """Return the project-local Git commit, never an ancestor repository's hash."""
    if not (project_root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def json_default(value: Any) -> Any:
    """Convert common scientific Python values into JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic, human-readable JSON and create its parent first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

