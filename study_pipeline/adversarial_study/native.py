"""A file bridge to the unmodified released MATLAB predictor.

No detector is implemented or retrained in Python. The bridge exports real
single-channel ten-second windows, checks native receipts, and preserves clipped
and pre-clipping values separately. Qualification confirms runtime behavior, not
the checkpoint variant, clinical calibration, or validity in a new population.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .core import StudyError, read_csv, read_json, sha256_file, write_csv, write_json

CHECKPOINT_SHA256 = "d8b5f89e9c958bddf0804e445a21ca8160a04f23494c1a94bd3e0e9bd134dd17"
CHECKPOINT_MD5 = "03a9c7f599c5630cd71bddfd3a542600"
RELEASE_FILES = ("trained_consciousness_detector.mat", "predict_consciousness_level.m",
                 "remap_gcs_scale.m", "BoundedOutputLayer.m")
PINNED_HASHES = {
    "trained_consciousness_detector.mat": CHECKPOINT_SHA256,
    "predict_consciousness_level.m": "a9b6073428f86bc55b93666867d5d63e5af4de8310fae41e395b05ad73e6fa23",
    "remap_gcs_scale.m": "9b1a8bfdfd1e0cd36425ab73f5d97232215ea9c482d9fd57fccc16c704f90c68",
    "BoundedOutputLayer.m": "96aeb138b1ddf6dc5c874c38501ad5015db6d3636b4ba1aaede98996c547033f",
}


def verify_release(release_dir: Path, expected_checkpoint_sha256: str | None = None) -> dict:
    paths = {name: Path(release_dir) / name for name in RELEASE_FILES}
    for name, path in paths.items():
        if not path.is_file():
            raise StudyError(f"Missing native release file: {path}")
    checkpoint = paths[RELEASE_FILES[0]]
    sha = sha256_file(checkpoint)
    md5 = hashlib.md5(checkpoint.read_bytes()).hexdigest()
    expected = expected_checkpoint_sha256 or CHECKPOINT_SHA256
    if sha != expected or md5 != CHECKPOINT_MD5:
        raise StudyError("Checkpoint differs from the pinned Figshare v2 artifact")
    for name, path in paths.items():
        if sha256_file(path) != PINNED_HASHES[name]:
            raise StudyError(f"Native release file differs from pinned v2: {name}")
    return {"status": "static_release_hash_verified_not_native_execution",
            "release": "10.6084/m9.figshare.30906557.v2",
            "checkpoint_sha256": sha, "checkpoint_md5": md5,
            "files": {name: sha256_file(path) for name, path in paths.items()},
            "variant_identity": "unresolved", "clinical_calibration": "not_established"}


def export_jobs(windows: list[dict], output_dir: Path) -> Path:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise StudyError("Native job directory must be empty; retain earlier exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    used: set[tuple[str, str]] = set()
    for window in windows:
        data = np.asarray(window["data"], dtype=float)
        names = window["channel_names"]
        fs = float(window["sfreq_hz"])
        if data.ndim != 2 or len(names) != data.shape[0] or not math.isfinite(fs) or fs <= 90:
            raise StudyError("Native jobs require channels x samples and sampling above 90 Hz")
        if abs(data.shape[1] / fs - 10.0) > 0.01 or not np.isfinite(data).all():
            raise StudyError("Native input must be a finite, genuine ten-second window")
        for channel, signal in zip(names, data):
            key = (str(window["episode_id"]), str(channel))
            if key in used:
                raise StudyError(f"Duplicate episode/channel job: {key}")
            used.add(key)
            if np.std(signal, ddof=1) <= np.finfo(float).eps:
                raise StudyError("Constant input cannot be standardized by the native predictor")
            job_id = hashlib.sha256((key[0] + "\0" + key[1]).encode()).hexdigest()[:24]
            input_path = output_dir / f"{job_id}.csv"
            np.savetxt(input_path, signal, delimiter=",", fmt="%.17g")
            rows.append({"job_id": job_id, "episode_id": key[0], "channel_name": key[1],
                         "participant_id": window["participant_id"], "source_id": window["source_id"],
                         "input_path": str(input_path), "sfreq_hz": fs,
                         "signal_sha256": sha256_file(input_path)})
    if not rows:
        raise StudyError("No qualified windows to export")
    jobs = output_dir / "native_jobs.csv"
    write_csv(jobs, rows)
    write_json(output_dir / "export.json", {"jobs": len(rows), "window_count": len(windows),
               "jobs_sha256": sha256_file(jobs), "preprocessing": "unchanged native MATLAB wrapper",
               "score_semantics": "GCS prediction and linear remapping, not awareness probability"})
    return jobs


def import_scores(scores_path: Path, jobs_path: Path, qualification_path: Path | None = None) -> list[dict]:
    if qualification_path is None:
        raise StudyError("Native scores require a runtime qualification receipt")
    qualification = read_json(qualification_path)
    if qualification.get("status") != "native_forward_pass_reproducible":
        raise StudyError("Native runtime has not passed repeated inference qualification")
    if qualification.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise StudyError("Qualification checkpoint differs from pinned v2")
    for key, name in (("wrapper_sha256", "predict_consciousness_level.m"),
                      ("mapping_sha256", "remap_gcs_scale.m"), ("bound_layer_sha256", "BoundedOutputLayer.m")):
        if qualification.get(key) != PINNED_HASHES[name]:
            raise StudyError(f"Qualification {key} differs from pinned release")
    receipt_path = Path(str(scores_path) + ".receipt.json")
    receipt = read_json(receipt_path)
    if receipt.get("status") != "native_batch_completed":
        raise StudyError("Batch receipt is not a completed native batch")
    for key, path in (("scores_sha256", scores_path), ("jobs_sha256", jobs_path),
                      ("qualification_sha256", qualification_path)):
        if receipt.get(key) != sha256_file(path):
            raise StudyError(f"Native batch {key} is missing or changed")
    if receipt.get("checkpoint_sha256") != qualification["checkpoint_sha256"]:
        raise StudyError("Batch checkpoint differs from qualified checkpoint")
    if receipt.get("wrapper_sha256") != qualification.get("wrapper_sha256"):
        raise StudyError("Batch preprocessing wrapper differs from qualification")
    jobs = read_csv(jobs_path, ("job_id", "episode_id", "channel_name", "input_path", "signal_sha256"))
    expected = {row["job_id"]: row for row in jobs}
    if len(expected) != len(jobs):
        raise StudyError("Duplicate native job IDs")
    if receipt.get("jobs") != len(jobs) or receipt.get("successful_jobs") != len(jobs):
        raise StudyError("Batch receipt does not certify a complete successful job set")
    scores = read_csv(scores_path, ("job_id", "gcs", "score", "status", "signal_sha256"))
    seen: set[str] = set()
    result = []
    for row in scores:
        job_id = row["job_id"]
        if job_id not in expected or job_id in seen:
            raise StudyError(f"Unknown or duplicate score job: {job_id}")
        seen.add(job_id)
        job = expected[job_id]
        path = Path(job["input_path"])
        if sha256_file(path) != job["signal_sha256"] or row["signal_sha256"] != job["signal_sha256"]:
            raise StudyError(f"Native signal changed for {job_id}")
        if row["status"] != "ok":
            raise StudyError(f"Native scoring failed for {job_id}: {row.get('error', '')}")
        gcs, score = float(row["gcs"]), float(row["score"])
        if not (math.isfinite(gcs) and math.isfinite(score) and 3 <= gcs <= 15 and 0 <= score <= 1):
            raise StudyError(f"Invalid native score for {job_id}")
        if abs(score - (gcs - 3) / 12) > 1e-7:
            raise StudyError(f"Score/GCS remapping mismatch for {job_id}")
        preclip = row.get("preclip_gcs", "")
        raw = float(preclip) if preclip else None
        if raw is not None and (not math.isfinite(raw) or abs(min(max(raw, 3), 15) - gcs) > 1e-5):
            raise StudyError(f"Pre-clipping output correspondence failed for {job_id}")
        result.append({**job, "gcs": gcs, "score": score, "preclip_gcs": raw,
                       "saturated": gcs <= 3 + 1e-7 or gcs >= 15 - 1e-7,
                       "score_origin": "native_frozen_v2", "checkpoint_sha256": CHECKPOINT_SHA256})
    if seen != set(expected):
        raise StudyError("Native score table does not contain every exported job")
    return result
