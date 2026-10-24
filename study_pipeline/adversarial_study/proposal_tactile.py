"""Source-pinned threshold-task EEG coverage with no invented perception labels.

The original threshold files contain stim-thr events and a stimulus staircase,
but their exported report/SDT/confidence fields can be unavailable. They support
neural/context comparisons, not observed experience-discordance evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from . import core, source_adapters as adapter


def _export_missing(value):
    """Source absence is distinct from an observed but unqualified report."""
    return value is None or str(value).strip().lower() in ("", "n/a", "na", "nan")


def threshold_episodes(events, *, recording_id, sfreq_hz, duration_s, capacity="detection"):
    required = {"onset", "trial_type", "event_sample", "stimamp", "stimon", "sdt", "confidence"}
    if not events or any(not required <= set(row) for row in events):
        raise ValueError("Original threshold event schema is incomplete")
    if not math.isfinite(sfreq_hz) or sfreq_hz <= 0 or abs(10 * sfreq_hz - round(10 * sfreq_hz)) > 1e-8:
        raise ValueError("Invalid threshold sampling grid")
    times = [adapter._number(row["onset"]) for row in events]
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("Original threshold events are unsorted")
    episodes, census = [], []
    for index, event in enumerate(events):
        if event["trial_type"] != "stim-thr":
            census.append({"source_event_index": index, "status": "non_stimulus_event_retained_in_original"})
            continue
        point, sample = adapter._number(event["onset"]), adapter._number(event["event_sample"])
        if sample != int(sample) or abs(sample / sfreq_hz - point) > 1.01 / sfreq_hz:
            raise ValueError("Threshold event sample and seconds disagree")
        eid = recording_id + "_threshold_" + str(index).zfill(5)
        disposition = {"episode_id": eid, "source_event_index": index, "source_point_s": point,
                       "status": "included", "original_trial_type": event["trial_type"]}
        if point < 10 or point > duration_s:
            disposition["status"] = "outside_full_continuous_extent"
            census.append(disposition)
            continue
        intensity = adapter._number(event["stimamp"], missing=True)
        report_status = ("missing_original_report_fields" if all(_export_missing(event[k])
                         for k in ("sdt", "confidence")) else "observed_fields_without_qualified_report_mapping")
        disposition["source_report_export_status"] = report_status
        episodes.append({"episode_id": eid, "recording_id": recording_id,
            "start_s": (int(sample) - round(10 * sfreq_hz)) / sfreq_hz,
            "stop_s": int(sample) / sfreq_hz,
            "report_category": "unqualified_threshold_report_export", "report_start_s": "", "report_stop_s": "",
            "report_point_s": "", "capacity_value": "", "capacity_construct": "", "capacity_units": "",
            "confidence": "", "detection": "", "stim_intensity": intensity,
            "source_trial_type": "stim-thr", "source_stimulus_intensity": intensity,
            "source_stimon": adapter._number(event["stimon"], missing=True),
            "source_original_sdt": event["sdt"], "source_original_confidence": event["confidence"],
            "source_sdt_export_status": "missing_original" if _export_missing(event["sdt"]) else "observed_unqualified",
            "source_confidence_export_status": "missing_original" if _export_missing(event["confidence"]) else "observed_unqualified",
            "source_report_export_status": report_status,
            "source_event_index": index, "analysis_window_basis": "10s_before_original_threshold_marker",
            "eeg_anchor_basis": "original_stim_thr_marker_physical_delivery_alignment_unqualified",
            "report_timing_basis": "no_qualified_experience_or_capacity_report_in_threshold_export"})
        census.append(disposition)
    return episodes, census


@contextmanager
def qualified_converter(helper, serialization_helper, serialization_helper_sha256):
    """Use the already-qualified SET/FDT integrity implementation in a new process.

    The helper is bundled byte-for-byte and pinned by the additive source
    manifest. It shares the original adapter's imports and measurement helpers.
    Event interpretation and the strictly qualified scalar-serialization writer
    are installed in this conversion process and restored on exit.
    """
    original = {key: getattr(adapter, key, None) for key in (
        "_tactile_one", "_tactile_sidecar", "_tactile_binary_integrity", "tactile_trial_episodes", "write_json")}
    original_keys = set(adapter.__dict__)
    try:
        payload = Path(helper).read_bytes()
        exec(compile(payload, str(helper), "exec"), adapter.__dict__)
        if not callable(getattr(adapter, "_tactile_binary_integrity", None)):
            raise ValueError("Full pinned binary-integrity converter is required")
        serialization_payload = Path(serialization_helper).read_bytes()
        serialization_sha = hashlib.sha256(serialization_payload).hexdigest()
        if serialization_sha != serialization_helper_sha256:
            raise ValueError("Qualified scalar-serialization helper differs from its manifest pin")
        namespace = {"__name__": "qualified_threshold_serialization"}
        exec(compile(serialization_payload, str(serialization_helper), "exec"), namespace)
        if not callable(namespace.get("install")) or not callable(namespace.get("_normalize")):
            raise ValueError("Qualified scalar-serialization implementation is required")
        namespace["install"](adapter)
        adapter.tactile_trial_episodes = threshold_episodes
        yield hashlib.sha256(payload).hexdigest(), serialization_sha, namespace["_normalize"]
    finally:
        for key, value in original.items():
            if key not in original_keys:
                adapter.__dict__.pop(key, None)
            else:
                setattr(adapter, key, value)


def convert(source_root, manifest, task, output, release_dir, qualification, integrity_helper,
            serialization_helper, serialization_helper_sha256):
    if task not in ("thrdown", "thrup"):
        raise ValueError("Only actual acquired threshold tasks are admitted")
    with qualified_converter(integrity_helper, serialization_helper, serialization_helper_sha256) as (helper_sha, serialization_sha, normalize):
        result = adapter.convert_sources("tactile", source_root, manifest, output,
            task=task, capacity="detection", release_dir=release_dir, qualification=qualification)
        # The returned report still contains original NumPy scalars even though
        # its first on-disk serialization succeeded; retain exact scalar values.
        result = normalize(result)
    root = Path(output)
    config = core.read_json(root / "study.config.json")
    config["study_id"] += "-" + task
    config["analysis"].update(construct="threshold_staircase_neural_context_without_reported_experience",
        timing_definition="10s_before_original_threshold_marker", enable_prediction=False)
    config["discovery"]["relationship_columns"] = ["stim_intensity"]
    config["source_conversion"].update(task=task, threshold_reports_qualified=False,
        integrity_helper_sha256=helper_sha, serialization_helper_sha256=serialization_sha,
        additive_source_sha256=core.sha256_file(__file__))
    core.write_json(root / "study.config.json", config)
    result["output_tables_sha256"]["study.config.json"] = core.sha256_file(root / "study.config.json")
    result.update(threshold_reports_qualified=False, task=task,
        analysis_scope="Original stimulus staircase EEG only; no consciousness/experience labels inferred",
        integrity_helper_sha256=helper_sha, serialization_helper_sha256=serialization_sha,
        additive_source_sha256=core.sha256_file(__file__))
    core.write_json(root / "source_qualification.json", result)
    return result


def staircase_analysis(rows, *, bootstrap_samples=1000, seed=20261008):
    """Equal-person within-recording association, exposing temporal confounding."""
    groups, missing = defaultdict(list), defaultdict(int)
    for row in rows:
        identity = row["source_id"], row["participant_id"], row["recording_id"]
        score, intensity, time = (adapter._number(row.get(k), missing=True) for k in ("score", "stim_intensity", "stop_s"))
        if any(value is None for value in (score, intensity, time)):
            missing[identity] += 1
        else:
            groups[identity].append((score, intensity, time))
    people = []
    for identity in sorted(set(groups) | set(missing)):
        values = np.asarray(groups[identity], dtype=float).reshape(-1, 3)
        record = dict(zip(("source_id", "participant_id", "recording_id"), identity))
        record.update(eligible_windows=len(values), missing_windows=missing[identity],
            status="insufficient_windows_or_no_intensity_variation", slope_per_source_intensity=None,
            partial_slope_given_time=None, staircase_time_intensity_correlation=None)
        if len(values) >= 3 and np.var(values[:, 1]) > 0:
            s, x, t = values.T
            record.update(status="descriptive_within_recording",
                slope_per_source_intensity=float(np.dot(x - x.mean(), s - s.mean()) / np.dot(x - x.mean(), x - x.mean())))
            if np.var(t) > 0:
                corr = float(np.corrcoef(x, t)[0, 1])
                record["staircase_time_intensity_correlation"] = corr
                # A deterministic staircase cannot identify intensity separately
                # from time. Do not report a pseudoinverse coefficient as causal.
                design = np.column_stack((np.ones(len(t)), x, t))
                if np.linalg.matrix_rank(design) == 3 and abs(corr) < .99:
                    record["partial_slope_given_time"] = float(np.linalg.lstsq(design, s, rcond=None)[0][1])
                else:
                    record["status"] = "descriptive_intensity_time_confounding"
        people.append(record)
    summaries = []
    rng = np.random.default_rng(seed)
    for source in sorted({r["source_id"] for r in people}):
        by_person = defaultdict(list)
        for row in people:
            if row["source_id"] == source and row["slope_per_source_intensity"] is not None:
                by_person[row["participant_id"]].append(row["slope_per_source_intensity"])
        values = np.asarray([np.mean(v) for v in by_person.values()])
        draws = [float(np.mean(rng.choice(values, len(values), replace=True))) for _ in range(bootstrap_samples)] if len(values) >= 2 else []
        summaries.append({"source_id": source, "informative_people": len(values),
            "mean_within_person_slope": float(values.mean()) if len(values) else None,
            "participant_bootstrap_ci": np.quantile(draws, [.025, .975]).tolist() if draws else None,
            "inference": "Descriptive staircase/context association; no randomized intensity effect or experiential endpoint"})
    return {"participant_recordings": people, "source_summaries": summaries,
        "all_input_windows": len(rows), "no_result_direction_gate": True,
        "threshold_experience_or_capacity_inferred": False}


def analyze(discovery, output):
    from . import model_guided
    source = Path(discovery).resolve()
    model_guided._verify_output(source)
    rows = core.read_csv(source / "all_scored_observations.csv")
    answer = staircase_analysis(rows)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    core.write_json(out / "staircase_analysis.json", answer)
    core.write_csv(out / "participant_staircases.csv", answer["participant_recordings"],
        ["source_id", "participant_id", "recording_id", "eligible_windows", "missing_windows", "status",
         "slope_per_source_intensity", "partial_slope_given_time", "staircase_time_intensity_correlation"])
    core.write_json(out / "receipt.json", {"status": "threshold_neural_context_analysis_completed",
        "inputs": [{"path": str(source / name), "sha256": core.sha256_file(source / name)}
                   for name in ("receipt.json", "all_scored_observations.csv")],
        "source_code_sha256": core.sha256_file(__file__), "human_experience_mapping_qualified": False,
        "outputs": {name: core.sha256_file(out / name) for name in ("staircase_analysis.json", "participant_staircases.csv")}})
    return answer


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("convert")
    for key in ("source-root", "manifest", "task", "output", "release-dir", "qualification", "integrity-helper",
                "serialization-helper", "serialization-helper-sha256"):
        c.add_argument("--" + key, required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--discovery", required=True); a.add_argument("--output", required=True)
    args = p.parse_args(argv)
    result = convert(args.source_root, args.manifest, args.task, args.output, args.release_dir,
                     args.qualification, args.integrity_helper, args.serialization_helper,
                     args.serialization_helper_sha256) if args.command == "convert" else analyze(args.discovery, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("recordings", "source_eligibility", "participant_recordings")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
