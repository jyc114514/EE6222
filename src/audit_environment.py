"""Collect a non-destructive environment audit for the assignment workspace."""

from __future__ import annotations

import getpass
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

try:
    from .utils import package_versions, write_json
except ImportError:  # direct execution: ``python src/audit_environment.py``
    from utils import package_versions, write_json


PACKAGE_IMPORTS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
    "pytest": "pytest",
    "coverage": "coverage",
}


def _run_command(argv: Sequence[str]) -> str:
    """Run a fixed diagnostic command and preserve failures as text."""
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        return f"EXIT_CODE={completed.returncode}\n{output}".strip()
    return output


def collect_environment(project_root: Path) -> dict[str, object]:
    """Collect environment metadata without changing host state."""
    package_status = package_versions(PACKAGE_IMPORTS.values())
    package_versions_by_name = {
        display_name: package_status[import_name]
        for display_name, import_name in PACKAGE_IMPORTS.items()
    }
    commands = {
        "hostname": _run_command(["hostname"]),
        "whoami": _run_command(["whoami"]),
        "pwd": _run_command(["pwd"]),
        "date": _run_command(["date"]),
        "uname": _run_command(["uname", "-a"]),
        "python_version": _run_command(["python", "--version"]),
        "python3_version": _run_command(["python3", "--version"]),
        "which_python": shutil.which("python"),
        "which_python3": shutil.which("python3"),
        "lscpu": _run_command(["lscpu"]),
        "free_h": _run_command(["free", "-h"]),
        "df_home": _run_command(["df", "-h", str(Path.home())]),
    }
    return {
        "audit_phase": "final_environment_audit_after_project_dependency_setup",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "cwd": str(Path.cwd()),
        "project_root": str(project_root.resolve()),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "venv": {
            "path": str(project_root / ".venv"),
            "python_exists": (project_root / ".venv" / "bin" / "python").exists(),
        },
        "packages": package_versions_by_name,
        "commands": commands,
        "execution_policy": {
            "device": "CPU only",
            "gpu_used": False,
            "system_python_modified": False,
        },
    }


def environment_text(audit: dict[str, object]) -> str:
    """Render the structured audit into a concise text companion."""
    python_info = audit["python"]
    packages = audit["packages"]
    commands = audit["commands"]
    lines = [
        "EE6222 Assignment 1 - Environment Audit",
        f"Audit phase: {audit['audit_phase']}",
        f"Timestamp (UTC): {audit['timestamp_utc']}",
        f"Hostname: {audit['hostname']}",
        f"User: {audit['user']}",
        f"Project root: {audit['project_root']}",
        f"Python executable: {python_info['executable']}",
        f"Python version: {python_info['version']}",
        f"Python implementation: {python_info['implementation']}",
        f"Project .venv Python exists: {audit['venv']['python_exists']}",
        "",
        "Package versions:",
    ]
    lines.extend(f"  {name}: {version or 'MISSING'}" for name, version in packages.items())
    lines.extend(
        [
            "",
            "Command evidence:",
            f"  hostname: {commands['hostname']}",
            f"  whoami: {commands['whoami']}",
            f"  pwd: {commands['pwd']}",
            f"  date: {commands['date']}",
            f"  uname: {commands['uname']}",
            f"  python --version: {commands['python_version']}",
            f"  python3 --version: {commands['python3_version']}",
            f"  which python: {commands['which_python'] or 'MISSING'}",
            f"  which python3: {commands['which_python3'] or 'MISSING'}",
            "",
            "lscpu:",
            str(commands["lscpu"]),
            "",
            "free -h:",
            str(commands["free_h"]),
            "",
            "df -h project home:",
            str(commands["df_home"]),
            "",
            "Execution policy:",
            "  device: CPU only",
            "  GPU used: False",
            "  system Python modified: False",
        ]
    )
    return "\n".join(lines) + "\n"


def write_environment_audit(project_root: Path) -> dict[str, object]:
    """Collect and save both JSON and text audit artifacts."""
    audit = collect_environment(project_root)
    audit_dir = project_root / "audit"
    write_json(audit_dir / "environment_audit.json", audit)
    (audit_dir / "environment_audit.txt").write_text(
        environment_text(audit), encoding="utf-8"
    )
    return audit


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    write_environment_audit(project_root)
    print(f"Environment audit written under {project_root / 'audit'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

