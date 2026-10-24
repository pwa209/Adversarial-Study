"""Connect source-qualified EEG to the unchanged native instrument and discovery.

Run only in scheduled allocations. Stages preserve actual outputs and pointers;
qualification gates concern data/runtime integrity, never effect direction.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from adversarial_study import cli, core, data, discovery, effective_dynamics


def load_context(config_path, pointer_path):
    config = core.load_config(config_path)
    pointer = core.read_json(pointer_path) if Path(pointer_path).exists() else {}
    return config, pointer


def prepare(config_path, pointer_path):
    cfg, pointer = load_context(config_path, pointer_path)
    if pointer:
        raise RuntimeError("Preparation pointer exists; retain prior run")
    prepared = cli.prepare(cfg, "development", True)
    pointer = {"config": str(Path(config_path).resolve()), "prepared": str(prepared)}
    # First qualified export in source order, chosen before any clinical score.
    jobs = core.read_csv(prepared / "native/native_jobs.csv")
    features = {row["episode_id"]: row for row in core.read_csv(prepared / "features.csv")}
    recordings = {row["recording_id"]: row for row in core.read_csv(core.resolve_path(cfg, "recordings"))}
    source_receipts = {row["recording_id"]: row for row in core.read_json(prepared / "recordings.json")["recordings"]}
    job = jobs[0]
    episode = features[job["episode_id"]]
    source = source_receipts[episode["recording_id"]]
    fs = float(job["sfreq_hz"])
    reference = {"reference_kind": "empirical_eeg", "synthetic": False,
        "reference_id": job["job_id"], "participant_id": job["participant_id"],
        "source_id": job["source_id"], "channel": job["channel_name"],
        "source_recording_path": source["path"], "source_recording_sha256": source["sha256"],
        "signal_sha256": job["signal_sha256"], "start_sample": round(float(episode["start_s"]) * fs),
        "stop_sample": round(float(episode["stop_s"]) * fs), "sfreq_hz": fs,
        "continuity_verified": True, "signal_path": job["input_path"],
        "source_manifest_row": recordings[episode["recording_id"]],
        "selection_rule": "first source-order episode passing outcome-independent continuity/amplitude checks"}
    reference_path = Path(pointer_path).with_name(Path(pointer_path).stem + ".reference.json")
    core.write_json(reference_path, reference)
    pointer.update(reference=str(reference_path), jobs=str(prepared / "native/native_jobs.csv"))
    core.write_json(pointer_path, pointer)


def detector(config_path, pointer_path, mode, scores):
    cfg, pointer = load_context(config_path, pointer_path)
    release = core.resolve_path(cfg, "release_dir", True)
    qualification = core.resolve_path(cfg, "qualification")
    command = [sys.executable, "-m", "native_tools.detector_worker", "--mode", mode,
               "--release-dir", str(release), "--timeout",
               "42600" if any(word in Path(pointer_path).stem for word in ("cohort", "paris", "dream", "tactile"))
               else "13800" if "tactile" in Path(pointer_path).stem else "6600"]
    if mode == "qualify":
        reference = core.read_json(pointer["reference"])
        command += ["--signal-csv", reference["signal_path"], "--sfreq", str(reference["sfreq_hz"]),
                    "--reference-json", pointer["reference"], "--output-json", str(qualification)]
    else:
        command += ["--jobs-csv", pointer["jobs"], "--output-csv", str(scores),
                    "--qualification-json", str(qualification)]
    return subprocess.run(command, check=False).returncode


def discover(config_path, pointer_path, scores, output_dir):
    cfg, pointer = load_context(config_path, pointer_path)
    imported = cli.import_native(cfg, Path(pointer["prepared"]), Path(scores), "development")
    rows = cli._load_analysis_table(cfg, imported / "analysis.csv", "development")
    output = Path(output_dir)
    if output.exists():
        raise RuntimeError("Discovery output exists; preserve it")
    output.mkdir(parents=True)
    registry = discovery.discovery_registry(rows, cfg.get("discovery", {}))
    core.write_json(output / "discovery.json", {"evidence_status": "exploratory_empirical_candidates",
        "independent_participants": len({row["participant_id"] for row in rows}),
        "registry": registry, "analysis_table": str(imported / "analysis.csv"),
        "interpretation": "Natural associations and candidate dissociations; no human mechanism or whole-window awareness established."})
    pointer.update(imported=str(imported), discovery=str(output))
    core.write_json(Path(pointer_path).with_name(Path(pointer_path).stem + ".analysis.json"), pointer)


def dynamics(config_path, pointer_path, output_dir):
    cfg, pointer = load_context(config_path, pointer_path)
    prepared = Path(pointer["prepared"])
    cli._validate_parent(cfg, prepared / "features.csv", "development")
    windows = core.read_csv(prepared / "features.csv")
    receipts = core.read_json(prepared / "recordings.json")["recordings"]
    output = Path(output_dir)
    if output.exists():
        raise RuntimeError("Dynamics output exists; preserve it")
    output.mkdir(parents=True)
    features, exclusions = [], []
    for receipt in receipts:
        recording = data.read_recording(receipt["path"], "npz")
        fs = recording["sfreq_hz"]
        selected = [row for row in windows if row["recording_id"] == receipt["recording_id"]]
        settings = {"channel_selection": cfg.get("instrument", {}).get("channel_names", ["Cz", "Pz", "Fz", "Oz"]), "max_channels": 8,
                    "forecast_steps": sorted({1, round(.05 * fs), round(.1 * fs)})}
        result = effective_dynamics.analyze_recording_windows(recording, selected, settings=settings,
                                                              recording_id=receipt["recording_id"])
        features.extend(result.pop("features")); exclusions.extend(result["exclusions"])
        core.write_json(output / (receipt["recording_id"] + ".json"), result)
    write_dynamics_tables(output, features, exclusions)
    core.write_json(output / "receipt.json", {"prepared_features_sha256": core.sha256_file(prepared / "features.csv"),
        "fitted_episodes": len(features), "excluded_episodes": len(exclusions),
        "evidence_status": "WITHIN_WINDOW_MEASUREMENT_CONSTRUCTION",
        "interpretation": effective_dynamics.INTERPRETATION})


def write_dynamics_tables(output, features, exclusions):
    """Retain metadata-rich exclusions and a valid empty measurement table."""
    metadata = {"episode_id", "participant_id", "source_id", "cohort_id", "recording_id", "role", "start_s", "stop_s"}
    feature_fields = sorted(metadata | {key for row in features for key in row})
    exclusion_fields = sorted({"episode_id", "reason"} | {key for row in exclusions for key in row})
    core.write_csv(Path(output) / "episode_features.csv", features, feature_fields)
    core.write_csv(Path(output) / "exclusions.csv", exclusions, exclusion_fields)


def synthesize(config_path, pointer_path, dynamics_dir, output_dir):
    cfg, pointer = load_context(config_path, pointer_path)
    analysis_pointer = Path(pointer_path).with_name(Path(pointer_path).stem + ".analysis.json")
    imported = Path(core.read_json(analysis_pointer)["imported"])
    rows = cli._load_analysis_table(cfg, imported / "analysis.csv", "development")
    dynamics_dir = Path(dynamics_dir)
    receipt = core.read_json(dynamics_dir / "receipt.json")
    if receipt["prepared_features_sha256"] != core.sha256_file(Path(pointer["prepared"]) / "features.csv"):
        raise ValueError("Dynamics and native scores come from different prepared episodes")
    dynamic_rows = core.read_csv(dynamics_dir / "episode_features.csv")
    dynamic = {row["episode_id"]: row for row in dynamic_rows}
    if len(dynamic) != len(dynamic_rows) or set(dynamic) - {row["episode_id"] for row in rows}:
        raise ValueError("Duplicate or unknown dynamics episode")
    columns = sorted({key for row in dynamic_rows for key in row if key.startswith("effective_")})
    augmented = []
    for row in rows:
        extra = dynamic.get(row["episode_id"], {})
        if extra and any(extra[key] != row[key] for key in ("participant_id", "source_id", "recording_id")):
            raise ValueError("Dynamics identity differs from scored participant/source/recording")
        augmented.append({**row, **{key: extra.get(key, "") for key in columns},
                          "effective_dynamics_available": bool(extra)})
    output = Path(output_dir)
    if output.exists(): raise RuntimeError("Synthesis output exists; preserve it")
    output.mkdir(parents=True)
    core.write_csv(output / "scored_dynamics.csv", augmented)
    measured = list(dict.fromkeys(cfg.get("discovery", {}).get("relationship_columns", ["capacity_value"])
                                + ["effective_rank", "acf_100ms"] + columns))
    result = discovery.unselected_relationships(augmented, {"relationship_columns": measured})
    core.write_json(output / "relationships.json", {"evidence_status": "exploratory_empirical_development",
        "result": result, "analysis_sha256": core.sha256_file(imported / "analysis.csv"),
        "dynamics_features_sha256": core.sha256_file(dynamics_dir / "episode_features.csv"),
        "score_and_feature_dependence": "Clinical score and neural features derive from the same EEG; association or conditional matching does not establish a human mechanism.",
        "ordinal_rating_interpretation": "Numeric self-rating associations are descriptive coding-dependent summaries; no interval-scale or awareness-gold-standard assumption is made."})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "qualify", "batch", "discover", "dynamics", "synthesize"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pointer", required=True, type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dynamics-dir", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Requires scheduled allocation")
    if args.stage == "prepare": prepare(args.config, args.pointer)
    elif args.stage in {"qualify", "batch"}: return detector(args.config, args.pointer, args.stage, args.scores)
    elif args.stage == "discover": discover(args.config, args.pointer, args.scores, args.output_dir)
    elif args.stage == "dynamics": dynamics(args.config, args.pointer, args.output_dir)
    elif args.stage == "synthesize": synthesize(args.config, args.pointer, args.dynamics_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
