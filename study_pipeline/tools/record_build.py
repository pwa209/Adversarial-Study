"""Validate this local build and record exact source/output provenance.

Runs unit tests, a synthetic demonstration and static release-hash checks only.
It never runs MATLAB, connects to a server or submits a research job.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def command(arguments: list[str], root: Path, log_path: Path) -> str:
    result = subprocess.run([sys.executable, *arguments], cwd=root, capture_output=True,
                            text=True, shell=False, check=False)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Build check failed; inspect {log_path}")
    return result.stdout


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logs = root / "validation" / stamp
    logs.mkdir(parents=True, exist_ok=False)
    command(["-m", "unittest", "discover", "-s", "tests", "-v"], root, logs / "tests.txt")
    test_log = (logs / "tests.txt").read_text(encoding="utf-8")
    matched = re.search(r"Ran (\d+) tests", test_log)
    if matched is None or not re.search(r"^OK\s*$", test_log, re.MULTILINE):
        raise RuntimeError("Test completion could not be verified from the log")
    demo_stdout = command(["run_study.py", "demo", "--output", str(root / "examples" / f"validated_{stamp}")], root, logs / "demo.txt")
    demo_result = json.loads(demo_stdout)
    static_stdout = command(["run_study.py", "verify-instrument"], root, logs / "static_release.txt")
    static_result = json.loads(static_stdout)
    excluded = {"__pycache__", "runs", "data", "qualification", "contracts", "examples", "validation", ".venv"}
    source_files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_file() and path.name != "BUILD_MANIFEST.json" and not set(relative.parts) & excluded:
            source_files[relative.as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    record = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "status": "python_and_synthetic_validation_passed",
        "python": sys.version, "numpy": np.__version__,
        "tests_passed": int(matched[1]), "test_log": str(logs / "tests.txt"),
        "test_log_sha256": digest(logs / "tests.txt"),
        "synthetic_demo": demo_result, "static_release_verification": static_result,
        "matlab_executed": False, "nftsim_executed": False,
        "human_data_analyzed": False, "cluster_jobs_submitted": False,
        "source_files": source_files,
    }
    output = root / "BUILD_MANIFEST.json"
    if output.exists():
        archive = logs / "previous_BUILD_MANIFEST.json"
        archive.write_bytes(output.read_bytes())
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "tests_passed": record["tests_passed"],
                      "manifest": str(output), "synthetic_demo": demo_result["status"]}, indent=2))


if __name__ == "__main__":
    main()
