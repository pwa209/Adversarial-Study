"""Scheduled native MATLAB reference qualification and frozen batch inference."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .figshare_source import sha256_file
from .nftsim_runtime import require_scheduled_job


def validate_reference(signal_csv: Path, reference: dict, sfreq: float) -> dict:
    required = {"reference_id", "participant_id", "source_id", "channel", "source_recording_path",
                "source_recording_sha256", "signal_sha256", "start_sample", "stop_sample", "sfreq_hz"}
    if not required <= set(reference):
        raise ValueError("Empirical reference provenance missing fields: " + ", ".join(sorted(required - set(reference))))
    if reference.get("reference_kind") != "empirical_eeg" or reference.get("synthetic") is not False or reference.get("continuity_verified") is not True:
        raise ValueError("Reference must declare empirical EEG, synthetic=false and verified continuity")
    for field in ("reference_id", "participant_id", "source_id", "channel"):
        if not isinstance(reference[field], str) or not reference[field].strip():
            raise ValueError("Empty provenance identifier: " + field)
    for field in ("source_recording_sha256", "signal_sha256"):
        if not isinstance(reference[field], str) or not re.fullmatch(r"[a-f0-9]{64}", reference[field]):
            raise ValueError("Invalid provenance SHA256: " + field)
    signal_csv = Path(signal_csv).resolve(strict=True)
    raw_path = Path(reference["source_recording_path"]).resolve(strict=True)
    if not signal_csv.is_file() or not raw_path.is_file() or raw_path == signal_csv:
        raise ValueError("Raw recording and exported signal must be distinct actual files")
    if sha256_file(signal_csv) != reference["signal_sha256"] or sha256_file(raw_path) != reference["source_recording_sha256"]:
        raise ValueError("Empirical source/export hash differs from provenance")
    if not math.isfinite(sfreq) or sfreq <= 90 or isinstance(reference["sfreq_hz"], bool) or not isinstance(reference["sfreq_hz"], (int, float)) or float(reference["sfreq_hz"]) != sfreq:
        raise ValueError("Reference sampling frequency mismatch or insufficient bandwidth")
    start, stop = reference["start_sample"], reference["stop_sample"]
    if isinstance(start, bool) or isinstance(stop, bool) or not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start < stop:
        raise ValueError("Reference sample span must be nonnegative integer start/exclusive stop")
    signal = np.loadtxt(signal_csv, delimiter=",", dtype=float)
    if signal.ndim > 1 and min(signal.shape) != 1:
        raise ValueError("Export must be one univariate channel")
    signal = signal.reshape(-1)
    if len(signal) != stop - start or abs(len(signal) / sfreq - 10) > .01:
        raise ValueError("Export/sample span must represent genuine continuous ten seconds")
    if not np.isfinite(signal).all() or len(signal) < 2 or np.std(signal) == 0:
        raise ValueError("Reference signal must be finite and nonconstant")
    return reference | {"source_recording_path": str(raw_path), "signal_path": str(signal_csv),
                        "samples": len(signal), "verification": "source/export hashes and declared continuous sample span verified; empirical identity supplied by provenance"}


def matlab_string(value: str | Path) -> str:
    text = str(value)
    if "\n" in text or "\r" in text or "\x00" in text:
        raise ValueError("MATLAB arguments cannot contain control characters")
    return "'" + text.replace("'", "''") + "'"


def matlab_command(mode: str, *, release_dir: Path, output_json: Path | None = None,
                   signal_csv: Path | None = None, sfreq: float | None = None,
                   reference_json: Path | None = None, layers: list[str] | None = None,
                   jobs_csv: Path | None = None, output_csv: Path | None = None,
                   qualification_json: Path | None = None) -> str:
    native_dir = Path(__file__).resolve().parent
    matlab_dir = native_dir.parent / "matlab"
    preamble = "addpath(" + matlab_string(matlab_dir) + "); addpath(" + matlab_string(native_dir) + "); "
    if mode == "qualify":
        layer_arg = "{" + ",".join(matlab_string(value) for value in (layers or [])) + "}"
        return preamble + "qualify_empirical_native(" + ",".join([
            matlab_string(signal_csv), format(float(sfreq), ".17g"), matlab_string(release_dir),
            matlab_string(output_json), matlab_string(reference_json), layer_arg]) + ");"
    if mode == "batch":
        return preamble + "batch_predict(" + ",".join(matlab_string(value) for value in
            (jobs_csv, output_csv, release_dir, qualification_json)) + ");"
    raise ValueError("Unknown MATLAB worker mode")


def run_matlab(command: str, log_path: Path, *, matlab: str, timeout: int) -> dict:
    job_id = require_scheduled_job()
    executable = shutil.which(matlab)
    if executable is None:
        raise ValueError("MATLAB executable unavailable; load a MATLAB module")
    if timeout < 1:
        raise ValueError("timeout must be positive")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    start = time.monotonic()
    with Path(log_path).open("x", encoding="utf-8") as handle:
        try:
            result = subprocess.run([executable, "-batch", command], stdout=handle,
                                    stderr=subprocess.STDOUT, env=environment,
                                    timeout=timeout, check=False, shell=False)
            code, error = result.returncode, ""
        except (subprocess.TimeoutExpired, OSError) as exc:
            code, error = None, type(exc).__name__ + ": " + str(exc)
    return {"slurm_job_id": job_id, "returncode": code, "error": error,
            "seconds": time.monotonic() - start, "matlab_executable": executable,
            "cuda_visible_devices": "", "loaded_modules": os.environ.get("LOADEDMODULES", "").split(":"),
            "log_sha256": sha256_file(log_path), "command": command,
            "scope": "native numerical runtime correspondence only; clinical validity not established"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["qualify", "batch"], default="qualify")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--signal-csv", type=Path)
    parser.add_argument("--sfreq", type=float)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--layers", default="", help="Actual layer names, comma separated; default selects native convolution/fully-connected layers plus pre-bound predecessor")
    parser.add_argument("--jobs-csv", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--qualification-json", type=Path)
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    require_scheduled_job()
    release_dir = args.release_dir.resolve(strict=True)
    if args.mode == "qualify":
        if any(value is None for value in (args.signal_csv, args.sfreq, args.reference_json, args.output_json)):
            parser.error("qualify requires signal-csv, sfreq, reference-json and output-json")
        reference = validate_reference(args.signal_csv, json.loads(args.reference_json.read_text(encoding="utf-8-sig")), args.sfreq)
        output = args.output_json.resolve()
        if output.exists():
            raise ValueError("Preserve existing qualification; use a new output filename")
        output.parent.mkdir(parents=True, exist_ok=True)
        provenance_path = Path(str(output) + ".reference.json")
        with provenance_path.open("x", encoding="utf-8") as handle:
            json.dump(reference, handle, indent=2, allow_nan=False)
            handle.write("\n")
        command = matlab_command("qualify", release_dir=release_dir, output_json=output,
            signal_csv=args.signal_csv.resolve(), sfreq=args.sfreq, reference_json=provenance_path,
            layers=[value.strip() for value in args.layers.split(",") if value.strip()])
    else:
        if any(value is None for value in (args.jobs_csv, args.output_csv, args.qualification_json)):
            parser.error("batch requires jobs-csv, output-csv and qualification-json")
        output = args.output_csv.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise ValueError("Preserve existing batch output; use a new filename")
        qualification = json.loads(args.qualification_json.read_text(encoding="utf-8"))
        if qualification.get("reference_kind") != "empirical_eeg" or qualification.get("activation_repeats_verified") is not True:
            raise ValueError("Batch requires the empirical native activation qualification receipt")
        command = matlab_command("batch", release_dir=release_dir,
            jobs_csv=args.jobs_csv.resolve(strict=True), output_csv=output,
            qualification_json=args.qualification_json.resolve(strict=True))
    log_path = Path(str(output) + ".matlab.log")
    receipt = run_matlab(command, log_path, matlab=args.matlab, timeout=args.timeout)
    receipt["mode"] = args.mode
    if output.is_file():
        receipt["output_sha256"] = sha256_file(output)
    with Path(str(output) + ".worker.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: receipt.get(key) for key in ("mode", "returncode", "seconds", "error")}))
    return 0 if receipt["returncode"] == 0 and output.is_file() else 2


if __name__ == "__main__":
    raise SystemExit(main())
