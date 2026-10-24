"""Compile and numerically qualify pristine NFTsim v2 on a scheduled CPU job.

Runtime qualification is separate from the functional hsyn diagnostic and from
any empirical observation map. No source is patched by this helper.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .figshare_source import BASELINE, DOI, REQUIRED_SOURCE, sha256_file

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def require_scheduled_job() -> str:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("Native build/run requires a scheduled Slurm job (SLURM_JOB_ID)")
    return job_id


def verify_release(release_dir: Path) -> dict:
    release_dir = Path(release_dir).resolve()
    manifest = json.loads((release_dir / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "pinned_payload_verified" or manifest.get("doi") != DOI:
        raise ValueError("Native release must have a pinned v2 source acquisition manifest")
    records = manifest.get("files", [])
    paths = set()
    for record in records:
        relative = record.get("path", "")
        candidate = release_dir / relative
        if not relative or candidate.resolve().is_relative_to(release_dir) is False or candidate.is_symlink():
            raise ValueError("Invalid manifest source path")
        if relative in paths or not candidate.is_file() or sha256_file(candidate) != record.get("sha256"):
            raise ValueError("Published source changed or is missing: " + relative)
        paths.add(relative)
    if not {"src/" + value for value in REQUIRED_SOURCE} <= paths:
        raise ValueError("Required published C++ files missing")
    if "configuration_files/" + BASELINE not in paths:
        raise ValueError("Published baseline config missing")
    actual_sources = {str(path.relative_to(release_dir)).replace("\\", "/")
                      for path in (release_dir / "src").glob("*") if path.suffix in {".cpp", ".h"}}
    recorded_sources = {value for value in paths if value.startswith("src/")}
    if actual_sources != recorded_sources:
        raise ValueError("Unrecorded or missing source files in src/")
    return manifest


def parse_output(path: Path, *, expected_rows: int | None = None,
                 expected_interval: float | None = None) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if "=============================================" not in text:
        raise ValueError("Native output must preserve its configuration/header separator")
    lines = text.rsplit("=============================================", 1)[-1].strip().splitlines()
    nonempty = [value for value in lines if value.strip()]
    if len(nonempty) < 3 or not nonempty[0].split() or nonempty[0].split()[0].lower() != "time":
        raise ValueError("NFTsim output lacks Time header and node row after config separator")
    columns = nonempty[0].split()
    try:
        node_ids = [int(value) for value in nonempty[1].split()]
    except ValueError as exc:
        raise ValueError("Invalid native output node header") from exc
    if len(node_ids) != len(columns) - 1 or any(value < 1 for value in node_ids):
        raise ValueError("Native output node header differs from column count")
    values = []
    for line in nonempty[2:]:
        fields = line.split()
        if len(fields) != len(columns):
            raise ValueError("Malformed numeric NFTsim row")
        try:
            values.append([float(value) for value in fields])
        except ValueError as exc:
            raise ValueError("Non-numeric NFTsim output") from exc
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2 or not np.isfinite(array).all():
        raise ValueError("Empty/non-finite native output")
    if expected_rows is not None and len(array) != expected_rows:
        raise ValueError(f"Native output rows {len(array)} differ from expected {expected_rows}")
    differences = np.diff(array[:, 0])
    if not np.all(differences > 0):
        raise ValueError("Output times are not strictly increasing")
    if expected_interval is not None and not np.allclose(differences, expected_interval, rtol=1e-7, atol=1e-10):
        raise ValueError("Native output sampling interval differs from configuration")
    return {"columns": columns, "values": array, "rows": len(array),
            "first_time": float(array[0, 0]), "last_time": float(array[-1, 0]),
            "interval": float(np.median(differences)), "sha256": sha256_file(path)}


def _mask_comments(text: str) -> str:
    return re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", lambda hit: "".join("\n" if c == "\n" else " " for c in hit[0]), text)


def perturb_config(text: str, kind: str, factor: float = 2.0) -> tuple[str, dict]:
    if not np.isfinite(factor) or factor <= 0 or factor == 1:
        raise ValueError("Diagnostic factor must be finite positive and different from one")
    masked = _mask_comments(text)
    if kind == "hsyn":
        pattern = re.compile(r"^\s*Dendrite 1:\s*Scaled[^\n]*?\bhsyn:\s*(" + FLOAT + r")", re.MULTILINE)
    elif kind == "coupling":
        pattern = re.compile(r"^\s*Coupling 1:\s*Map\s*-\s*nu:\s*(" + FLOAT + r")", re.MULTILINE)
    else:
        raise ValueError("Only explicit Dendrite 1 hsyn / Coupling 1 nu diagnostics supported")
    matches = list(pattern.finditer(masked))
    if len(matches) != 1:
        raise ValueError("Diagnostic parameter must address exactly one configuration field")
    hit = matches[0]
    before = float(hit[1])
    if before == 0 or not np.isfinite(before):
        raise ValueError("Diagnostic parameter must be finite nonzero")
    after = before * factor
    altered = text[:hit.start(1)] + format(after, ".17g") + text[hit.end(1):]
    return altered, {"parameter": "Dendrite 1.hsyn" if kind == "hsyn" else "Coupling 1.nu",
                     "before": before, "after": after, "factor": factor}


def compare_outputs(reference: dict, alternative: dict) -> dict:
    if reference["columns"] != alternative["columns"] or reference["values"].shape != alternative["values"].shape:
        raise ValueError("Diagnostic native outputs differ in shape/columns")
    if not np.array_equal(reference["values"][:, 0], alternative["values"][:, 0]):
        raise ValueError("Diagnostic output times changed")
    a, b = reference["values"][:, 1:], alternative["values"][:, 1:]
    delta = b - a
    result = {"exactly_equal": bool(np.array_equal(a, b)),
              "max_absolute_difference": float(np.max(np.abs(delta))),
              "relative_l2_difference": float(np.linalg.norm(delta) / max(np.linalg.norm(a), 1e-300))}
    return result


def _execute(argv: list[str], log_path: Path, *, cwd: Path, timeout: int) -> dict:
    start = time.monotonic()
    with log_path.open("x", encoding="utf-8") as handle:
        try:
            process = subprocess.run(argv, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                                     timeout=timeout, check=False, shell=False)
            code, error = process.returncode, ""
        except (subprocess.TimeoutExpired, OSError) as exc:
            code, error = None, type(exc).__name__ + ": " + str(exc)
    return {"argv": argv, "returncode": code, "error": error,
            "seconds": time.monotonic() - start, "log_sha256": sha256_file(log_path)}


def qualify_runtime(release_dir: Path, output_dir: Path, *, compiler: str = "g++",
                    timeout: int = 600) -> dict:
    job_id = require_scheduled_job()
    release_dir = Path(release_dir).resolve()
    manifest = verify_release(release_dir)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {"status": "native_runtime_not_qualified", "slurm_job_id": job_id,
              "release_doi": DOI, "source_manifest_sha256": sha256_file(release_dir / "SOURCE_MANIFEST.json"),
              "published_source_unchanged": True, "source_files": manifest["files"],
              "observation_mapping": "not_qualified", "clinical_validation": "not_established",
              "parameter_identifiability": "not_established",
              "seed_policy": "published default mt19937_64 auto-ID seeds; independent executable processes; repeat tested on this binary/runtime",
              "interpretation": "finite repeated simulator output is execution evidence only; no human mechanism or EEG forward-map claim"}
    try:
        compiler_path = shutil.which(compiler)
        if compiler_path is None:
            raise ValueError("C++ compiler unavailable: " + compiler)
        compiler_probe = subprocess.run([compiler_path, "--version"], text=True, capture_output=True, timeout=30, check=False)
        result["compiler_version"] = compiler_probe.stdout.splitlines()[:3]
        executable = output_dir / "nftsim"
        sources = sorted(str(path) for path in (release_dir / "src").glob("*.cpp"))
        command = [compiler_path, "-std=c++11", "-O2", "-Wall", "-Wextra", "-pedantic", "-I", str(release_dir / "src"), *sources, "-o", str(executable)]
        result["build"] = _execute(command, output_dir / "build.log", cwd=release_dir, timeout=timeout)
        if result["build"]["returncode"] != 0 or not executable.is_file():
            raise ValueError("Pristine published source did not compile; inspect build.log")
        result["executable_sha256"] = sha256_file(executable)
        baseline = release_dir / "configuration_files" / BASELINE
        result["baseline_sha256"] = sha256_file(baseline)
        result["runs"] = []
        outputs = {}
        inputs = {"baseline_1": baseline, "baseline_2": baseline}
        for kind, factor in (("hsyn", 2.0), ("coupling", 1.05)):
            altered, change = perturb_config(baseline.read_text(encoding="utf-8"), kind, factor)
            config_path = output_dir / (kind + "_diagnostic.conf")
            config_path.write_text(altered, encoding="utf-8")
            inputs[kind] = config_path
            result[kind + "_configuration_change"] = change | {"config_sha256": sha256_file(config_path)}
        for label, config_path in inputs.items():
            output_path = output_dir / (label + ".output")
            run = _execute([str(executable), "-i", str(config_path), "-o", str(output_path)],
                           output_dir / (label + ".log"), cwd=release_dir, timeout=timeout)
            run["label"] = label
            result["runs"].append(run)
            if run["returncode"] != 0:
                if label in {"hsyn", "coupling"}:
                    run["diagnostic_status"] = "failed; no functional conclusion"
                    continue
                raise ValueError("Native simulation failed: " + label)
            try:
                parsed = parse_output(output_path, expected_rows=5000, expected_interval=.002)
            except ValueError as exc:
                if label in {"hsyn", "coupling"}:
                    run["diagnostic_status"] = "invalid output; no functional conclusion"
                    run["diagnostic_error"] = str(exc)
                    continue
                raise
            outputs[label] = parsed
            run["output"] = {key: value for key, value in parsed.items() if key != "values"}
        result["repeat"] = compare_outputs(outputs["baseline_1"], outputs["baseline_2"])
        if not result["repeat"]["exactly_equal"]:
            raise ValueError("Repeated pristine baseline outputs differ")
        result["status"] = "pristine_native_baseline_finite_reproducible"
        result["hsyn_sensitivity"] = compare_outputs(outputs["baseline_1"], outputs["hsyn"]) if "hsyn" in outputs else None
        result["coupling_positive_control"] = compare_outputs(outputs["baseline_1"], outputs["coupling"]) if "coupling" in outputs else None
        result["scaled_gain_diagnostic"] = (
            "no_hsyn_effect_with_nonzero_coupling_control_in_tested_baseline"
            if result["hsyn_sensitivity"] and result["coupling_positive_control"] and result["hsyn_sensitivity"]["exactly_equal"] and not result["coupling_positive_control"]["exactly_equal"]
            else "hsyn_test_inconclusive" if result["hsyn_sensitivity"] is None or result["hsyn_sensitivity"]["exactly_equal"]
            else "hsyn_changed_native_output_in_tested_baseline")
        result["time_axis"] = "published Output Start=10; printed time is relative to output start; 5000 samples at 500 Hz"
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    with (output_dir / "qualification.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; receipts are immutable")
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--timeout", default=600, type=int)
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("timeout must be positive")
    result = qualify_runtime(args.release_dir, args.output_dir, compiler=args.compiler, timeout=args.timeout)
    print(json.dumps({key: result.get(key) for key in ("status", "scaled_gain_diagnostic", "error")}))
    return 0 if result["status"] == "pristine_native_baseline_finite_reproducible" else 2


if __name__ == "__main__":
    raise SystemExit(main())
