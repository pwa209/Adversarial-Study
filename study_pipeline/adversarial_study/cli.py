"""Staged driver. Empirical native inference is deliberately a separate stage."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import core, data, discovery, mechanisms, models, native, signals, simulation, statistics
from .core import StudyError

EPISODE_COLUMNS = ("episode_id", "recording_id", "start_s", "stop_s", "report_start_s",
                   "report_stop_s", "report_category", "capacity_value")


def _role_gate(cfg: dict, role: str) -> Path:
    manifest = core.resolve_path(cfg, "recordings", True)
    if role in {"evaluation", "diagnostic"}:
        if cfg["mode"] != "empirical":
            raise StudyError("Protected roles require an empirical configuration")
        core.verify_freeze(cfg, manifest, core.resolve_path(cfg, "freeze", True))
    return manifest


def _validate_parent(cfg: dict, table: Path, role: str, visited: set[Path] | None = None) -> None:
    visited = visited if visited is not None else set()
    directory = table.parent.resolve()
    if directory in visited:
        return
    visited.add(directory)
    record = core.read_json(table.parent / "run.json")
    if record.get("status") != "completed" or record.get("role") != role:
        raise StudyError("Table must belong to a completed run of the requested role")
    if record.get("config_sha256") != cfg["_config_sha256"] or record.get("code_sha256") != core.code_hashes():
        raise StudyError("Prepared inputs belong to a different configuration or code version")
    for relative, expected in record.get("outputs", {}).items():
        if core.sha256_file(directory / relative) != expected:
            raise StudyError(f"Run output changed: {relative}")
    for item in record.get("inputs", []):
        path = Path(item["path"])
        if core.sha256_file(path) != item["sha256"]:
            raise StudyError(f"Recorded input changed: {path.name}")
        if path.name in {"features.csv", "analysis.csv"} and (path.parent / "run.json").is_file():
            _validate_parent(cfg, path, role, visited)
    if record.get("stage") == "prepare":
        for item in core.read_json(directory / "recordings.json").get("recordings", []):
            if core.sha256_file(item["path"]) != item["sha256"]:
                raise StudyError("Raw normalized recording changed after preparation")


def prepare(cfg: dict, role: str = "development", export_native: bool = True) -> Path:
    manifest = _role_gate(cfg, role)
    recordings = core.select_recordings(core.read_csv(manifest, core.RECORDING_COLUMNS), role, cfg["mode"])
    episode_path = core.resolve_path(cfg, f"episodes_{role}", True)
    episodes = core.read_csv(episode_path, EPISODE_COLUMNS)
    by_recording = {row["recording_id"]: row for row in recordings}
    seen: set[str] = set()
    for row in episodes:
        if not row["episode_id"] or row["episode_id"] in seen:
            raise StudyError("Episode IDs must be nonempty and unique within their role")
        seen.add(row["episode_id"])
        if row["recording_id"] not in by_recording:
            raise StudyError("Role-specific episode file contains a recording outside that role")
    selected_channels = cfg.get("instrument", {}).get("channel_names", [])
    if cfg["mode"] == "empirical" and export_native and not selected_channels:
        raise StudyError("Specify outcome-independent instrument.channel_names before empirical scoring")
    if len(set(selected_channels)) != len(selected_channels):
        raise StudyError("Instrument channel names must be unique")
    run = core.new_run(cfg, "prepare", role, [manifest, episode_path])
    exclusions: list[dict] = []
    features: list[dict] = []
    windows: list[dict] = []
    recording_receipts: list[dict] = []
    root = core.resolve_path(cfg, "data_root", True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in episodes:
        grouped[row["recording_id"]].append(row)
    try:
        for rid, assigned_episodes in grouped.items():
            meta = by_recording[rid]
            if meta["modality"].upper() != "EEG" or meta["units"] != "uV":
                raise StudyError(f"{rid}: current empirical adapter requires EEG normalized to uV")
            signal_path = (root / meta["signal_path"]).resolve()
            if not signal_path.is_relative_to(root):
                raise StudyError(f"{rid}: signal_path escapes declared data_root")
            recording = data.read_recording(signal_path, meta["signal_format"])
            fs = recording["sfreq_hz"]
            if abs(fs - float(meta["sfreq_hz"])) > 1e-6:
                raise StudyError(f"{rid}: sampling rate differs from manifest")
            if len(recording["channel_names"]) != int(float(meta["channel_count"])):
                raise StudyError(f"{rid}: channel count differs from manifest")
            if abs(recording["data"].shape[1] / fs - float(meta["duration_s"])) > 1 / fs:
                raise StudyError(f"{rid}: recording duration differs from manifest")
            channels = selected_channels or recording["channel_names"]
            if set(channels) - set(recording["channel_names"]):
                raise StudyError(f"{rid}: selected detector channel is missing")
            channel_indices = [recording["channel_names"].index(name) for name in channels]
            recording_receipts.append({"recording_id": rid, "path": str(signal_path),
                                       "sha256": core.sha256_file(signal_path),
                                       "reference": recording.get("reference", "as_stored")})
            for episode in assigned_episodes:
                start = core.number(episode["start_s"], "start_s")
                stop = core.number(episode["stop_s"], "stop_s")
                try:
                    window = data.extract_window(recording, start, stop,
                        duration_s=float(cfg.get("window", {}).get("duration_s", 10)),
                        tolerance_s=float(cfg.get("window", {}).get("tolerance_s", .01)))
                except ValueError as exc:
                    exclusions.append({"episode_id": episode["episode_id"], "recording_id": rid,
                                       "reason": str(exc)})
                    continue
                source_channel_policy = cfg.get("channel_quality", {})
                quality_window = window[channel_indices] if source_channel_policy.get("enabled") else window
                quality = signals.quality_metrics(quality_window, fs, **cfg.get("quality", {}))
                if not quality["qualified"]:
                    exclusions.append({"episode_id": episode["episode_id"], "recording_id": rid,
                                       "reason": ";".join(quality["reasons"])})
                    continue
                report_start, report_stop, overlap, coverage = None, None, None, None
                if bool(episode["report_start_s"]) != bool(episode["report_stop_s"]):
                    raise StudyError("Both report interval endpoints must be supplied together")
                if episode["report_start_s"]:
                    report_start = core.number(episode["report_start_s"], "report_start_s")
                    report_stop = core.number(episode["report_stop_s"], "report_stop_s")
                    if report_start > report_stop:
                        raise StudyError("Report interval endpoints are reversed")
                    overlap = max(0.0, min(stop, report_stop) - max(start, report_start))
                    coverage = overlap / (stop - start)
                usable, channel_audit = signals.channel_quality_selection(window, fs, recording["channel_names"], source_channel_policy)
                minimum = int(source_channel_policy.get("minimum_channels", 1))
                if len(usable) < minimum:
                    exclusions.append({"episode_id": episode["episode_id"], "recording_id": rid,
                                       "reason": "insufficient_signal_qualified_scalp_channels"})
                    continue
                result = signals.compute_features(window[usable], fs, cfg.get("features", {}))
                features.append({**episode, **result, "participant_id": meta["participant_id"],
                    "source_id": meta["source_id"], "cohort_id": meta["cohort_id"],
                    "site_id": meta["site_id"], "session_id": meta["session_id"],
                    "role": role, "synthetic": core.flag(meta["synthetic"]),
                    "report_interval_overlap_s": overlap, "report_window_coverage": coverage,
                    "report_covers_entire_window": coverage is not None and coverage >= 1 - 1e-8,
                    "quality_max_abs_uv": quality["max_abs_uv"],
                    "quality_min_std_uv": quality["min_std_uv"], "score_origin": "not_scored"})
                features[-1].update(quality_amplitude_reference=quality["amplitude_reference"],
                    scalp_qualified_channel_count=len(usable), scalp_source_channel_count=window.shape[0],
                    scalp_qualified_channel_names=";".join(recording["channel_names"][i] for i in usable))
                if channel_audit:
                    quality_root = run / "channel_quality"
                    quality_root.mkdir(exist_ok=True)
                    core.write_json(quality_root / (episode["episode_id"] + ".json"), {"episode_id": episode["episode_id"], "policy": source_channel_policy, "channels": channel_audit})
                windows.append({"episode_id": episode["episode_id"], "participant_id": meta["participant_id"],
                    "source_id": meta["source_id"], "channel_names": channels,
                    "data": window[channel_indices], "sfreq_hz": fs})
        if not features:
            raise StudyError("No qualified episodes; exclusions will be preserved")
        core.write_csv(run / "features.csv", features)
        core.write_csv(run / "exclusions.csv", exclusions, ["episode_id", "recording_id", "reason"])
        core.write_json(run / "recordings.json", {"recordings": recording_receipts})
        job_path = native.export_jobs(windows, run / "native") if export_native else None
        contract = cfg.get("analysis", {})
        counts: dict[str, Counter] = defaultdict(Counter)
        for row in features:
            counts[row["participant_id"]][str(row.get(contract.get("condition_column", "report_category"), ""))] += 1
        target, comparator = contract.get("target_value"), contract.get("comparator_value")
        minimum = int(contract.get("minimum_episodes_per_condition", 1))
        informative = sum(c[str(target)] >= minimum and c[str(comparator)] >= minimum for c in counts.values()) if target is not None and comparator is not None else None
        core.write_json(run / "prepared.json", {"status": "qualified_window_preparation",
            "role": role, "synthetic": cfg["mode"] == "development", "eligible_recording_participants": len({r["participant_id"] for r in recordings}),
            "participants_with_qualified_windows": len(counts), "qualified_episodes": len(features),
            "excluded_episodes": len(exclusions), "informative_participants_for_configured_conditions": informative,
            "native_jobs": str(job_path) if job_path else None,
            "features_sha256": core.sha256_file(run / "features.csv"),
            "report_semantics": "Original categories retained; overlap does not certify experience for the whole window."})
        core.finish_run(run)
        return run
    except Exception as exc:
        core.write_csv(run / "exclusions.csv", exclusions, ["episode_id", "recording_id", "reason"])
        core.finish_run(run, "failed", str(exc))
        raise


def import_native(cfg: dict, prepared_dir: Path, scores: Path, role: str) -> Path:
    _role_gate(cfg, role)
    prepared_dir = prepared_dir.resolve()
    feature_path = prepared_dir / "features.csv"
    _validate_parent(cfg, feature_path, role)
    prep = core.read_json(prepared_dir / "prepared.json")
    if prep.get("features_sha256") != core.sha256_file(feature_path):
        raise StudyError("Prepared feature table changed")
    if cfg["mode"] != "empirical":
        raise StudyError("Native score import is an empirical command")
    native.verify_release(core.resolve_path(cfg, "release_dir", True),
                          cfg.get("instrument", {}).get("expected_checkpoint_sha256"))
    qualification = core.resolve_path(cfg, "qualification", True)
    jobs = prepared_dir / "native" / "native_jobs.csv"
    values = native.import_scores(scores, jobs, qualification)
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in values:
        by_episode[row["episode_id"]].append(row)
    features = core.read_csv(feature_path, ("episode_id", "participant_id", "source_id", "role", "synthetic"))
    if set(by_episode) != {r["episode_id"] for r in features}:
        raise StudyError("Scores do not match prepared episodes")
    aggregation = cfg.get("instrument", {}).get("channel_aggregation", "median")
    if aggregation not in {"median", "mean"}:
        raise StudyError("instrument.channel_aggregation must be median or mean")
    operation = np.median if aggregation == "median" else np.mean
    for feature in features:
        entries = by_episode[feature["episode_id"]]
        if any(v["participant_id"] != feature["participant_id"] or v["source_id"] != feature["source_id"] for v in entries):
            raise StudyError("Scoring identity differs from prepared participant/source")
        gcs = float(operation([v["gcs"] for v in entries]))
        preclip = [v["preclip_gcs"] for v in entries]
        feature.update(gcs=gcs, score=(gcs - 3) / 12,
            preclip_gcs=float(operation(preclip)) if all(v is not None for v in preclip) else None,
            saturated_fraction=float(np.mean([v["saturated"] for v in entries])),
            scored_channels=len(entries), channel_aggregation=aggregation, score_origin="native_frozen_v2")
    run = core.new_run(cfg, "import_scores", role, [feature_path, jobs, scores, qualification, Path(str(scores) + ".receipt.json")])
    core.write_csv(run / "analysis.csv", features)
    core.write_json(run / "analysis_table.json", {"analysis_sha256": core.sha256_file(run / "analysis.csv"),
                                                "prepared_run": str(prepared_dir), "native_receipts_verified": True})
    core.finish_run(run)
    return run


def _load_analysis_table(cfg: dict, table: Path, role: str) -> list[dict]:
    _role_gate(cfg, role)
    _validate_parent(cfg, table, role)
    table_receipt = core.read_json(table.parent / "analysis_table.json")
    if table_receipt.get("analysis_sha256") != core.sha256_file(table):
        raise StudyError("Analysis table changed after construction")
    rows = core.read_csv(table, ("episode_id", "participant_id", "source_id", "role", "synthetic", "score_origin"))
    if not rows or any(r["role"] != role for r in rows):
        raise StudyError("Analysis rows must have the requested role")
    if cfg["mode"] == "empirical" and (any(core.flag(r["synthetic"]) for r in rows) or
        any(r["score_origin"] != "native_frozen_v2" for r in rows) or not table_receipt.get("native_receipts_verified")):
        raise StudyError("Empirical study analysis requires verified native scores and nonsynthetic rows")
    return rows


def analyze(cfg: dict, table: Path, role: str = "development", predict: bool = False) -> Path:
    core.validate_analysis_contract(cfg)
    rows = _load_analysis_table(cfg, table, role)
    analysis = cfg["analysis"]
    run = core.new_run(cfg, "prediction" if predict else "contrast", role, [table])
    try:
        if predict:
            if role != "development":
                raise StudyError("Grouped exploratory refitting is development-only; freeze and transport a fitted model for protected prediction")
            result = models.grouped_prediction(rows, analysis["outcome_column"], analysis["baseline_columns"],
                analysis["dynamic_columns"], folds=int(analysis.get("folds", 5)), seed=cfg["seed"],
                ridge=float(analysis.get("ridge", 1)))
        else:
            result = statistics.paired_contrast(rows, analysis["value_column"], analysis["condition_column"],
                str(analysis["target_value"]), str(analysis["comparator_value"]),
                bootstrap_reps=int(analysis.get("bootstrap_reps", 2000)), seed=cfg["seed"],
                min_episodes=int(analysis.get("minimum_episodes_per_condition", 1)),
                equivalence_margin=analysis.get("equivalence_margin"), confidence=float(analysis.get("confidence", .95)))
        report = {"study_id": cfg["study_id"], "synthetic": cfg["mode"] == "development",
                  "evidence_status": "synthetic_software_check_only" if cfg["mode"] == "development" else "empirical_analysis",
                  "claim_id": analysis["claim_id"], "construct": analysis["construct"],
                  "timing_definition": analysis["timing_definition"], "result": result,
                  "interpretation": "This operation does not establish a biological mechanism or absent consciousness."}
        core.write_json(run / "result.json", report)
        core.finish_run(run)
        return run
    except Exception as exc:
        core.finish_run(run, "failed", str(exc))
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "configs" / "study.template.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("census", help="Metadata-only cohort and exposure audit")
    commands.add_parser("verify-instrument", help="Verify pinned release hashes without executing MATLAB")
    prep = commands.add_parser("prepare", help="Qualify genuine continuous windows, compute descriptive features and export native jobs")
    prep.add_argument("--role", choices=["development", "evaluation", "diagnostic"], default="development")
    prep.add_argument("--features-only", action="store_true", help="Prepare features without native job export; not complete study scoring")
    commands.add_parser("freeze", help="Record specification before opening protected cohorts")
    imp = commands.add_parser("import-scores", help="Verify native batch receipts and construct analysis table")
    imp.add_argument("--prepared", type=Path, required=True)
    imp.add_argument("--scores", type=Path, required=True)
    imp.add_argument("--role", choices=["development", "evaluation", "diagnostic"], default="development")
    for name in ("contrast", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--table", type=Path, required=True)
        command.add_argument("--role", choices=["development", "evaluation", "diagnostic"], default="development")
    discover_command = commands.add_parser("discover", help="Full distribution and exploratory within-person score/capacity matching")
    discover_command.add_argument("--table", type=Path, required=True)
    fit_command = commands.add_parser("fit-predictor", help="Fit development models once for later protected transport")
    fit_command.add_argument("--table", type=Path, required=True)
    evaluate_command = commands.add_parser("evaluate-predictor", help="Apply frozen prediction artifact to new participants without refitting")
    evaluate_command.add_argument("--table", type=Path, required=True)
    mechanism_fit = commands.add_parser("fit-mechanisms", help="Compare qualified generative observation-space candidates in development")
    mechanism_fit.add_argument("--table", type=Path, required=True)
    mechanism_fit.add_argument("--bank", type=Path, required=True)
    mechanism_evaluate = commands.add_parser("evaluate-mechanisms", help="Transport frozen candidate predictions to independent people")
    mechanism_evaluate.add_argument("--table", type=Path, required=True)
    commands.add_parser("precision", help="Assumption-based hierarchical recruitment/precision simulations")
    recovery = commands.add_parser("recovery", help="Linear effective-model recovery diagnostic; not biological NFTsim inference")
    recovery.add_argument("--settings", type=Path)
    bank = commands.add_parser("nftsim-bank", help="Create bounded native configuration candidates without execution")
    bank.add_argument("--template", type=Path, required=True)
    bank.add_argument("--parameters", type=Path, required=True)
    diagnostic = commands.add_parser("diagnostic", help="Analyze randomized paired outcomes with explicit missingness")
    diagnostic.add_argument("--trials", type=Path, required=True)
    diagnostic.add_argument("--outcome", required=True)
    diagnostic.add_argument("--active")
    diagnostic.add_argument("--control")
    demo = commands.add_parser("demo", help="Build and run an explicitly synthetic software check")
    demo.add_argument("--output", type=Path, default=Path(__file__).parent.parent / "examples" / "synthetic_demo")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import run_demo
            print(json.dumps(run_demo(args.output), indent=2))
            return 0
        cfg = core.load_config(args.config)
        if args.command == "census":
            manifest = core.resolve_path(cfg, "recordings", True)
            result = core.census(core.read_csv(manifest, core.RECORDING_COLUMNS))
            run = core.new_run(cfg, "census", inputs=[manifest])
            core.write_json(run / "census.json", result)
            core.finish_run(run)
        elif args.command == "verify-instrument":
            result = native.verify_release(core.resolve_path(cfg, "release_dir", True),
                cfg.get("instrument", {}).get("expected_checkpoint_sha256"))
            run = core.new_run(cfg, "verify_instrument")
            core.write_json(run / "release.json", result)
            core.finish_run(run)
        elif args.command == "prepare":
            run = prepare(cfg, args.role, not args.features_only)
        elif args.command == "freeze":
            result = core.freeze_analysis(cfg, core.resolve_path(cfg, "recordings", True), core.resolve_path(cfg, "freeze"))
            print(json.dumps(result, indent=2)); return 0
        elif args.command == "import-scores":
            run = import_native(cfg, args.prepared, args.scores, args.role)
        elif args.command in {"contrast", "predict"}:
            run = analyze(cfg, args.table.resolve(), args.role, args.command == "predict")
        elif args.command == "discover":
            table = args.table.resolve()
            rows = _load_analysis_table(cfg, table, "development")
            run = core.new_run(cfg, "discover", inputs=[table])
            registry = discovery.discovery_registry(rows, cfg.get("discovery", {}))
            pairs = registry["same_score_different_capacity"]
            core.write_json(run / "discovery.json", {"evidence_status": "exploratory_empirical_candidates" if cfg["mode"] == "empirical" else "synthetic_software_check_only", "distribution": registry["score_distribution"], "matching": pairs, "registry": registry})
            core.write_csv(run / "candidate_pairs.csv", pairs["pairs"], ["participant_id", "source_id", "recording_id", "episode_a", "episode_b", "score_a", "score_b", "score_difference_abs", "capacity_a", "capacity_b", "capacity_difference_abs", "report_a", "report_b"])
            core.finish_run(run)
        elif args.command == "fit-predictor":
            core.validate_analysis_contract(cfg)
            table = args.table.resolve()
            rows = _load_analysis_table(cfg, table, "development")
            analysis = cfg["analysis"]
            artifact_path = core.resolve_path(cfg, "prediction_model")
            if artifact_path.exists():
                raise StudyError("Prediction artifact already exists; preserve it and version the model/configuration")
            artifact = models.fit_prediction_model(rows, analysis["outcome_column"], analysis["baseline_columns"], analysis["dynamic_columns"], ridge=float(analysis.get("ridge", 1)))
            artifact["training_table_sha256"] = core.sha256_file(table)
            artifact["study_id"] = cfg["study_id"]
            core.write_json(artifact_path, artifact)
            run = core.new_run(cfg, "fit_predictor", inputs=[table, artifact_path])
            core.write_json(run / "artifact.json", {"path": str(artifact_path), "sha256": core.sha256_file(artifact_path), "status": "fitted_on_development_not_evaluated_independently"})
            core.finish_run(run)
        elif args.command == "evaluate-predictor":
            if cfg["mode"] != "empirical" or not cfg["analysis"].get("enable_prediction"):
                raise StudyError("Protected prediction requires empirical mode and a frozen enable_prediction=true contract")
            table = args.table.resolve()
            rows = _load_analysis_table(cfg, table, "evaluation")
            artifact_path = core.resolve_path(cfg, "prediction_model", True)
            artifact = core.read_json(artifact_path)
            if artifact.get("study_id") != cfg["study_id"]:
                raise StudyError("Prediction artifact belongs to another study")
            result = models.evaluate_prediction_model(rows, artifact)
            run = core.new_run(cfg, "evaluate_predictor", "evaluation", [table, artifact_path])
            core.write_json(run / "result.json", {"evidence_status": "protected_prediction_without_refitting", "result": result})
            core.finish_run(run)
        elif args.command in {"fit-mechanisms", "evaluate-mechanisms"}:
            core.validate_analysis_contract(cfg)
            settings = cfg.get("mechanisms", {})
            if not settings.get("enabled"):
                raise StudyError("Enable and specify the mechanism comparison in configuration")
            condition_column = settings.get("phenotype_condition_column")
            condition_value = settings.get("phenotype_condition_value")
            if not condition_column or condition_value is None:
                raise StudyError("Specify which independently measured condition defines the mechanism phenotype")
            role = "development" if args.command == "fit-mechanisms" else "evaluation"
            table = args.table.resolve()
            all_rows = _load_analysis_table(cfg, table, role)
            if any(condition_column not in row for row in all_rows):
                raise StudyError("Mechanism phenotype condition is absent from the observation table")
            rows = [row for row in all_rows if row[condition_column] == str(condition_value)]
            artifact_path = core.resolve_path(cfg, "mechanism_model")
            if args.command == "fit-mechanisms":
                if artifact_path.exists():
                    raise StudyError("Mechanism artifact exists; retain it and version any revised explanation")
                bank = core.read_csv(args.bank, ("candidate_id", "mechanism_family", "parameter_values", "mapping_status", "mapping_reference"))
                artifact = mechanisms.fit_candidate_bank(rows, bank, settings["feature_columns"], settings)
                artifact.update(study_id=cfg["study_id"], training_table_sha256=core.sha256_file(table),
                    candidate_table_sha256=core.sha256_file(args.bank),
                    phenotype_definition={"condition_column": condition_column, "condition_value": condition_value})
                artifact["training_participant_ids"] = sorted({row["participant_id"] for row in all_rows})
                core.write_json(artifact_path, artifact)
                run = core.new_run(cfg, "fit_mechanisms", inputs=[table, args.bank, artifact_path])
                core.write_json(run / "artifact.json", {"path": str(artifact_path), "sha256": core.sha256_file(artifact_path), "status": "candidate_explanations_fitted_not_human_mechanism_established"})
            else:
                if cfg["mode"] != "empirical":
                    raise StudyError("Protected mechanism evaluation requires empirical mode")
                artifact = core.read_json(artifact_path)
                if artifact.get("study_id") != cfg["study_id"]:
                    raise StudyError("Mechanism artifact belongs to another study")
                result = mechanisms.evaluate_candidate_bank(rows, artifact)
                run = core.new_run(cfg, "evaluate_mechanisms", "evaluation", [table, artifact_path])
                core.write_json(run / "result.json", {"evidence_status": "protected_candidate_prediction_without_refitting", "result": result})
            core.finish_run(run)
        elif args.command in {"precision", "recovery"}:
            settings = cfg.get("precision", {}) if args.command == "precision" else (core.read_json(args.settings) if args.settings else {})
            result = statistics.simulate_precision(settings) if args.command == "precision" else models.effective_recovery(settings)
            run = core.new_run(cfg, args.command)
            core.write_json(run / "result.json", {"evidence_status": "assumption_based_simulation_not_empirical_findings", "result": result})
            core.finish_run(run)
        elif args.command == "nftsim-bank":
            parameters = core.read_json(args.parameters)
            run = core.new_run(cfg, "nftsim_bank", inputs=[args.template, args.parameters])
            simulation.prepare_nftsim_bank(args.template, parameters["parameter_sets"], run / "bank", parameters.get("bounds"))
            core.finish_run(run)
        elif args.command == "diagnostic":
            if cfg["mode"] == "empirical":
                _role_gate(cfg, "diagnostic")
            core.validate_analysis_contract(cfg)
            diagnostic = cfg.get("diagnostic", {})
            treatment_column = diagnostic.get("treatment_column", "treatment")
            active = args.active or diagnostic.get("active_value", "active")
            control = args.control or diagnostic.get("control_value", "control")
            rows = core.read_csv(args.trials, ("participant_id", "episode_id", "eligible", "randomized", treatment_column, args.outcome))
            allowed = {r["participant_id"] for r in core.select_recordings(core.read_csv(core.resolve_path(cfg, "recordings", True), core.RECORDING_COLUMNS), "diagnostic" if cfg["mode"] == "empirical" else "development", cfg["mode"])}
            if any(r["participant_id"] not in allowed for r in rows):
                raise StudyError("Diagnostic trial contains participant outside assigned cohort")
            if cfg["mode"] == "empirical":
                if diagnostic.get("outcome_column") != args.outcome or active != diagnostic.get("active_value") or control != diagnostic.get("control_value"):
                    raise StudyError("Diagnostic outcome and treatment arms must match the frozen contract")
                for row in rows:
                    if core.flag(row["eligible"]) and core.flag(row["randomized"]):
                        times = {key: core.number(row.get(key), key) for key in
                            ("eligibility_start_s", "eligibility_stop_s", "assignment_s", "intervention_s", "post_window_start_s", "post_window_stop_s")}
                        if abs(times["eligibility_stop_s"] - times["eligibility_start_s"] - 10) > .01:
                            raise StudyError("Diagnostic eligibility requires a preceding complete ten-second window")
                        if not (times["eligibility_stop_s"] <= times["assignment_s"] <= times["intervention_s"] <= times["post_window_start_s"] < times["post_window_stop_s"]):
                            raise StudyError("Diagnostic eligibility/assignment/intervention/outcome timing is invalid")
                        if row[treatment_column] not in {active, control}:
                            raise StudyError("Randomized eligible trial has an unknown treatment arm")
            result = statistics.randomized_contrast(rows, args.outcome, treatment_column=treatment_column, active_value=active,
                control_value=control, bootstrap_reps=int(cfg["analysis"].get("bootstrap_reps", 2000)), seed=cfg["seed"])
            run = core.new_run(cfg, "diagnostic", "diagnostic" if cfg["mode"] == "empirical" else "development", [args.trials])
            core.write_json(run / "result.json", {"evidence_status": "empirical_completed_outcome_contrast" if cfg["mode"] == "empirical" else "synthetic_software_check_only", "result": result})
            core.finish_run(run)
        else:
            raise StudyError("Unknown stage")
        print(json.dumps({"stage": args.command, "run_dir": str(run), "status": "completed"}, indent=2))
        return 0
    except (StudyError, ValueError, OSError, KeyError, ImportError) as exc:
        print(f"Study stage stopped: {exc}", file=sys.stderr)
        return 2
