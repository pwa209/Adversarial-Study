"""Configuration, immutable run records, source census and evaluation boundaries.

Participant IDs must be globally deidentified across acquisitions. A metadata
census never opens EEG or report/outcome tables. Development commands cannot
consume protected cohorts. Hash checks protect the recorded specification; they
do not prove that analysts have never seen outcomes outside this programme.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class StudyError(ValueError):
    """An actionable data, scientific-contract or configuration error."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise StudyError(f"Expected a JSON object: {path}")
    return value


def write_json(path: str | Path, value: Any) -> None:
    """Strict JSON: NaN/Infinity must be resolved instead of hidden in a report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(target)


def read_csv(path: str | Path, required: tuple[str, ...] = ()) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise StudyError(f"{path}: missing columns {sorted(missing)}")
        rows = []
        for number, row in enumerate(reader, start=2):
            if None in row:
                raise StudyError(f"{path}:{number}: too many fields")
            if any(v is None for v in row.values()):
                raise StudyError(f"{path}:{number}: missing CSV fields")
            rows.append({k: v.strip() for k, v in row.items()})
        return rows


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or sorted({key for row in rows for key in row})
    if not names:
        raise StudyError("An empty CSV needs explicit field names")
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k, "") for k in names})


def number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StudyError(f"{name}: expected a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise StudyError(f"{name}: expected a finite number")
    return result


def flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1", "yes"}:
        return True
    if str(value).lower() in {"false", "0", "no", ""}:
        return False
    raise StudyError(f"Invalid boolean: {value!r}")


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    cfg = read_json(config_path)
    if cfg.get("schema_version") != 1:
        raise StudyError("Configuration schema_version must be 1")
    if cfg.get("mode") not in {"development", "empirical"}:
        raise StudyError("mode must be development or empirical")
    if not cfg.get("study_id"):
        raise StudyError("study_id is required")
    seed = cfg.get("seed", 20261004)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise StudyError("seed must be a nonnegative integer")
    cfg.setdefault("seed", seed)
    cfg["_config_path"] = str(config_path)
    cfg["_base"] = str((config_path.parent / cfg.get("path_base", ".")).resolve())
    cfg["_config_sha256"] = sha256_file(config_path)
    return cfg


def resolve_path(cfg: dict, key: str, must_exist: bool = False) -> Path:
    value = cfg.get("paths", {}).get(key)
    if not isinstance(value, str) or not value:
        raise StudyError(f"Set paths.{key} in the configuration")
    result = (Path(cfg["_base"]) / value).resolve()
    if must_exist and not result.exists():
        raise StudyError(f"paths.{key} does not exist: {result}")
    return result


RECORDING_COLUMNS = (
    "recording_id", "participant_id", "source_id", "cohort_id", "site_id", "session_id",
    "role", "signal_path", "signal_format", "modality", "units", "duration_s",
    "sfreq_hz", "channel_count", "detector_exposure", "analyst_exposure", "synthetic",
)


def validate_recordings(rows: list[dict]) -> None:
    ids: set[str] = set()
    participant_roles: dict[str, set[str]] = defaultdict(set)
    cohort_roles: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        missing = [k for k in RECORDING_COLUMNS if not row.get(k)]
        if missing:
            raise StudyError(f"Recording missing required metadata {missing}")
        rid = row["recording_id"]
        if rid in ids:
            raise StudyError(f"Duplicate recording_id: {rid}")
        ids.add(rid)
        if row["role"] not in {"development", "evaluation", "diagnostic", "clinical_reference"}:
            raise StudyError(f"{rid}: invalid role")
        for key in ("detector_exposure", "analyst_exposure"):
            if row[key] not in {"yes", "no", "unknown"}:
                raise StudyError(f"{rid}: {key} must be yes/no/unknown")
        for key in ("duration_s", "sfreq_hz", "channel_count"):
            if number(row[key], f"{rid}.{key}") <= 0:
                raise StudyError(f"{rid}.{key} must be positive")
        if not number(row["channel_count"], f"{rid}.channel_count").is_integer():
            raise StudyError(f"{rid}.channel_count must be an integer")
        flag(row["synthetic"])
        participant_roles[row["participant_id"]].add(row["role"])
        cohort_roles[row["cohort_id"]].add(row["role"])
        if row["role"] in {"evaluation", "diagnostic"}:
            if row["analyst_exposure"] != "no" or row["detector_exposure"] != "no":
                raise StudyError(f"{rid}: protected data require confirmed no detector/analyst exposure")
    for unit, mapping in (("participant", participant_roles), ("cohort", cohort_roles)):
        for identifier, roles in mapping.items():
            if len(roles) > 1:
                raise StudyError(f"{unit} {identifier} crosses assigned roles: {sorted(roles)}")


def census(rows: list[dict]) -> dict:
    validate_recordings(rows)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["source_id"], row["role"])].append(row)
    return {
        "status": "metadata_census_only",
        "independent_participants": len({r["participant_id"] for r in rows}),
        "recordings": len(rows),
        "groups": [
            {"source_id": source, "role": role, "recordings": len(items),
             "participants": len({r["participant_id"] for r in items}),
             "cohorts": sorted({r["cohort_id"] for r in items}),
             "declared_dense_channel_recordings": sum(int(float(r["channel_count"])) >= 32 for r in items),
             "detector_exposure": dict(Counter(r["detector_exposure"] for r in items))}
            for (source, role), items in sorted(groups.items())
        ],
        "qualification": "Metadata counts do not establish compatible windows or informative people.",
    }


def select_recordings(rows: list[dict], role: str, mode: str) -> list[dict]:
    validate_recordings(rows)
    if mode == "development" and role in {"evaluation", "diagnostic"}:
        raise StudyError("Development mode cannot select protected cohorts")
    selected = [r for r in rows if r["role"] == role]
    if not selected:
        raise StudyError(f"No recordings assigned role {role}")
    if mode == "empirical" and any(flag(r["synthetic"]) for r in selected):
        raise StudyError("Synthetic recordings cannot enter an empirical run")
    if mode == "development" and any(not flag(r["synthetic"]) for r in selected):
        raise StudyError("Development mode is reserved for synthetic software checks; use empirical mode with the development role for human discovery")
    return selected


def code_hashes(package_dir: Path | None = None) -> dict[str, str]:
    directory = package_dir or Path(__file__).parent
    result = {"python/" + p.name: sha256_file(p) for p in sorted(directory.glob("*.py"))}
    for path in sorted((directory.parent / "matlab").glob("*.m")):
        result["matlab/" + path.name] = sha256_file(path)
    return result


def validate_analysis_contract(cfg: dict) -> None:
    contract = cfg.get("analysis", {})
    needed = ("claim_id", "primary_contrast_name", "value_column", "condition_column",
              "target_value", "comparator_value", "construct", "timing_definition")
    unset = [k for k in needed if contract.get(k) in {None, "", "UNSET"}]
    if unset:
        raise StudyError(f"Specify the analysis contract before inference: {unset}")
    if contract["target_value"] == contract["comparator_value"]:
        raise StudyError("Target and comparator must differ")
    if contract.get("independent_unit") != "participant":
        raise StudyError("This implementation's inferential unit is participant")
    if int(contract.get("minimum_episodes_per_condition", 1)) < 1:
        raise StudyError("minimum_episodes_per_condition must be positive")
    if int(contract.get("bootstrap_reps", 2000)) < 100:
        raise StudyError("Use at least 100 bootstrap replicates for an inference report")
    if contract.get("equivalence_margin") is not None:
        if number(contract["equivalence_margin"], "equivalence_margin") <= 0:
            raise StudyError("equivalence_margin must be positive")


def freeze_analysis(cfg: dict, manifest_path: Path, output_path: Path) -> dict:
    """Record an immutable specification before opening protected signal/outcomes."""
    if cfg["mode"] != "empirical":
        raise StudyError("An empirical freeze requires mode=empirical")
    validate_analysis_contract(cfg)
    rows = read_csv(manifest_path, RECORDING_COLUMNS)
    validate_recordings(rows)
    protected = [r for r in rows if r["role"] in {"evaluation", "diagnostic"}]
    if not protected:
        raise StudyError("Assign protected cohorts before freezing the analysis")
    if any(flag(r["synthetic"]) for r in rows):
        raise StudyError("A synthetic manifest cannot form an empirical freeze")
    from . import native
    release = native.verify_release(resolve_path(cfg, "release_dir", True))
    qualification_path = resolve_path(cfg, "qualification", True)
    qualification = read_json(qualification_path)
    if qualification.get("status") != "native_forward_pass_reproducible" or qualification.get("checkpoint_sha256") != release["checkpoint_sha256"]:
        raise StudyError("Qualify native runtime before freezing the empirical programme")
    artifacts = {"qualification": {"path": str(qualification_path), "sha256": sha256_file(qualification_path)}}
    if cfg["analysis"].get("enable_prediction"):
        model_path = resolve_path(cfg, "prediction_model", True)
        artifacts["prediction_model"] = {"path": str(model_path), "sha256": sha256_file(model_path)}
    if cfg.get("mechanisms", {}).get("enabled"):
        model_path = resolve_path(cfg, "mechanism_model", True)
        artifacts["mechanism_model"] = {"path": str(model_path), "sha256": sha256_file(model_path)}
    if any(r["role"] == "diagnostic" for r in protected):
        diagnostic = cfg.get("diagnostic", {})
        needed = ("outcome_column", "treatment_column", "active_value", "control_value", "manipulation_definition")
        if any(not diagnostic.get(k) for k in needed) or diagnostic.get("active_value") == diagnostic.get("control_value"):
            raise StudyError("Specify the diagnostic manipulation, outcome and arms before freezing diagnostic cohorts")
    receipt = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": cfg["study_id"], "config_sha256": cfg["_config_sha256"],
        "manifest_sha256": sha256_file(manifest_path), "code_sha256": code_hashes(),
        "protected_recording_ids": sorted(r["recording_id"] for r in protected),
        "analysis": cfg["analysis"],
        "artifacts": artifacts, "release_files": release["files"],
        "status": "frozen_specification_not_preregistration_or_results",
    }
    if output_path.exists():
        raise StudyError("Freeze receipt already exists; preserve it and create a new version")
    write_json(output_path, receipt)
    return receipt


def verify_freeze(cfg: dict, manifest_path: Path, freeze_path: Path) -> dict:
    receipt = read_json(freeze_path)
    checks = {"study_id": cfg["study_id"], "config_sha256": cfg["_config_sha256"],
              "manifest_sha256": sha256_file(manifest_path), "code_sha256": code_hashes()}
    for key, current in checks.items():
        if receipt.get(key) != current:
            raise StudyError(f"Frozen {key} changed; protected evaluation cannot proceed")
    for item in receipt.get("artifacts", {}).values():
        if sha256_file(item["path"]) != item["sha256"]:
            raise StudyError("A frozen qualification or prediction artifact changed")
    from . import native
    release = native.verify_release(resolve_path(cfg, "release_dir", True))
    if receipt.get("release_files") != release["files"]:
        raise StudyError("Frozen instrument release changed")
    validate_analysis_contract(cfg)
    return receipt


def new_run(cfg: dict, stage: str, role: str = "development", inputs: list[Path] | None = None) -> Path:
    base = resolve_path(cfg, "output_dir")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = base / f"{stamp}_{stage}_{role}"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "run.json", {
        "study_id": cfg["study_id"], "stage": stage, "role": role, "mode": cfg["mode"],
        "synthetic": cfg["mode"] == "development", "created_utc": stamp,
        "config_sha256": cfg["_config_sha256"], "code_sha256": code_hashes(),
        "inputs": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in inputs or []],
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "seed": cfg["seed"], "status": "started",
    })
    return directory


def finish_run(directory: Path, status: str = "completed", detail: str | None = None) -> None:
    path = directory / "run.json"
    receipt = read_json(path)
    receipt["outputs"] = {str(item.relative_to(directory)): sha256_file(item)
                          for item in sorted(directory.rglob("*"))
                          if item.is_file() and item != path and item.suffix != ".tmp"}
    receipt.update(status=status, detail=detail, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, receipt)
