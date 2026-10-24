"""Frozen clinical-model-guided human discovery and generative challenges.

Human EEG and reports are observations. NFTsim outputs are native model-space
trajectories, not reconstructed scalp EEG. A one-node cortical proxy can be
challenged against scale-free temporal measurements without establishing an
anatomical forward map. Weak fits, out-of-domain observations, failed native
simulations and absent human-outcome mappings remain explicit scientific results.
No Python detector or artificial human observations are used by these stages.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import cli, core, data, discovery, native, signals
from .core import StudyError

FEATURE_DEFINITION = "guided_univariate_v1"
TEMPORAL_FEATURES = ("feature_acf_100ms", "feature_acf_500ms",
                     "feature_spectral_centroid_hz", "feature_spectral_entropy",
                     "feature_hjorth_mobility")
DYNAMIC_FEATURES = TEMPORAL_FEATURES[:2]
CONVENTIONAL_FEATURES = TEMPORAL_FEATURES[2:4]
MAPPING_INTERPRETATION = (
    "Native Pop.1.V is a single-node effective cortical proxy. Shared scale-free "
    "temporal summaries are a testable observation assumption; scalp mixing, "
    "anatomical coupling and biological identity remain unqualified. Neither a "
    "small residual nor a native clinical score qualifies that mapping."
)


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def univariate_features(signal, sfreq_hz):
    """Numerically qualified shared human/native temporal measurements."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = _univariate_features(signal, sfreq_hz)
    except (FloatingPointError, OverflowError) as exc:
        raise StudyError("Finite samples exceeded numeric temporal-measurement precision") from exc
    if any(not math.isfinite(value) for value in result.values()):
        raise StudyError("Temporal measurement produced a nonfinite value")
    return result


def _univariate_features(signal, sfreq_hz):
    """Identical measurement definition for actual human and simulator signals.

    ACF uses the centered full-series energy denominator (biased estimator).
    Spectral summaries use a Hann periodogram over 0.5--45 Hz. Absolute amplitude
    is recorded but excluded from cross-domain fitting because units do not map.
    These measurements do not apply or imitate the clinical MATLAB preprocessing.
    """
    x = np.asarray(signal, dtype=float)
    fs = float(sfreq_hz)
    if x.ndim != 1 or len(x) < 100 or not np.isfinite(x).all() or not math.isfinite(fs) or fs <= 90:
        raise StudyError("Temporal features need finite univariate samples above 90 Hz")
    x = x - x.mean()
    energy = float(x @ x)
    if not math.isfinite(energy) or energy <= np.finfo(float).tiny:
        raise StudyError("Constant signal has undefined temporal measurements")
    result = {}
    for label, duration in (("100ms", .1), ("500ms", .5)):
        lag = round(fs * duration)
        if lag >= len(x):
            raise StudyError("Signal is shorter than the requested temporal lag")
        result["feature_acf_" + label] = float(x[:-lag] @ x[lag:] / energy)
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    frequency = np.fft.rfftfreq(len(x), 1 / fs)
    selected = (frequency >= .5) & (frequency <= 45)
    p = power[selected]
    total = float(p.sum())
    if not math.isfinite(total) or total <= np.finfo(float).tiny or len(p) < 2:
        raise StudyError("Insufficient nonzero spectral energy for temporal features")
    probabilities = p / total
    nonzero = probabilities > 0
    result.update(feature_spectral_centroid_hz=float(probabilities @ frequency[selected]),
                  feature_spectral_entropy=float(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])) / np.log(len(p))),
                  feature_hjorth_mobility=float(np.sqrt(np.var(np.diff(x) * fs) / np.var(x))),
                  feature_std_amplitude=float(np.std(x, ddof=1)))
    return result


def multichannel_trajectory(window, sfreq_hz, *, maximum_states=400, threshold=.75, compute_rank=True):
    """Measure actual channel-state recurrence and persistence with bounded RAM.

    Channel z-scoring and nonoverlapping bin means define states; no latent-state
    labels or synthetic padding are created. Recurrence excludes separations
    below 100 ms and uses a fixed RMS standardized-distance threshold. Persistence
    refers to local signal-state changes, never persistence of conscious experience.
    """
    x = np.asarray(window, dtype=float)
    fs = float(sfreq_hz)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise StudyError("Trajectory requires finite multichannel samples")
    scales = np.std(x, axis=1, ddof=1)
    if np.any(scales <= np.finfo(float).eps):
        raise StudyError("Constant scalp channel makes trajectory scaling undefined")
    standardized = (x - x.mean(axis=1, keepdims=True)) / scales[:, None]
    width = max(1, math.ceil(x.shape[1] / maximum_states))
    count = x.shape[1] // width
    states = standardized[:, :count * width].reshape(x.shape[0], count, width).mean(axis=2).T
    norm = np.sum(states * states, axis=1)
    squared = np.maximum(0, (norm[:, None] + norm[None, :] - 2 * states @ states.T) / x.shape[0])
    distance = np.sqrt(squared)
    dt = width / fs
    separation = np.abs(np.arange(count)[:, None] - np.arange(count)[None, :]) * dt
    eligible = separation >= .1 - 1e-12
    recurrent = (distance <= threshold) & eligible
    changes = np.sqrt(np.mean(np.diff(states, axis=0) ** 2, axis=1))
    # A dwell ends when the adjacent standardized RMS change exceeds threshold.
    boundaries = np.flatnonzero(changes > threshold) + 1
    lengths = np.diff(np.r_[0, boundaries, count]) * dt
    result = {"trajectory_channels": x.shape[0], "trajectory_states": count,
            "trajectory_bin_s": dt, "trajectory_recurrence_threshold": threshold,
            "trajectory_recurrence_fraction": float(recurrent.sum() / eligible.sum()),
            "trajectory_local_dwell_median_s": float(np.median(lengths)),
            "trajectory_local_dwell_max_s": float(lengths.max()),
            "trajectory_transition_rate_hz": float(len(boundaries) / (count * dt)),
            "trajectory_path_length_rms": float(changes.sum()),
            "trajectory_covered_duration_s": count * dt}
    if compute_rank:
        covariance = np.cov(standardized)
        eigenvalues = np.maximum(0, np.linalg.eigvalsh(covariance))
        probabilities = eigenvalues[eigenvalues > 0] / eigenvalues.sum()
        result["trajectory_effective_rank"] = float(np.exp(-probabilities @ np.log(probabilities)))
        result["trajectory_rank_fraction"] = result["trajectory_effective_rank"]/x.shape[0]
    return result


def _human(pointer_path, scores_path, qualification_path):
    pointer = core.read_json(pointer_path)
    cfg = core.load_config(pointer["config"])
    if cfg["mode"] != "empirical":
        raise StudyError("Model-guided human stages require empirical source inputs")
    prepared = Path(pointer["prepared"])
    features_path = prepared / "features.csv"
    cli._validate_parent(cfg, features_path, "development")
    qualification_path = Path(qualification_path).resolve()
    if qualification_path != core.resolve_path(cfg, "qualification", True):
        raise StudyError("Qualification differs from the prepared instrument configuration")
    native.verify_release(core.resolve_path(cfg, "release_dir", True))
    jobs_path = prepared / "native/native_jobs.csv"
    if pointer.get("jobs") and Path(pointer["jobs"]).resolve() != jobs_path.resolve():
        raise StudyError("Pointer jobs differ from the prepared native export")
    imported = native.import_scores(Path(scores_path), jobs_path, qualification_path)
    by_episode = defaultdict(list)
    for row in imported:
        by_episode[row["episode_id"]].append(row)
    features = core.read_csv(features_path)
    if set(by_episode) != {row["episode_id"] for row in features}:
        raise StudyError("Native scores and human episodes differ")
    aggregation = cfg.get("instrument", {}).get("channel_aggregation", "median")
    if aggregation not in ("median", "mean"):
        raise StudyError("Unqualified channel aggregation")
    operation = np.median if aggregation == "median" else np.mean
    for row in features:
        if row["role"] != "development" or core.flag(row["synthetic"]):
            raise StudyError("Protected or synthetic rows cannot enter human development")
        entries = by_episode[row["episode_id"]]
        if any(item["participant_id"] != row["participant_id"] or item["source_id"] != row["source_id"] for item in entries):
            raise StudyError("Native jobs differ from human identities")
        preclip = [item["preclip_gcs"] for item in entries]
        row.update(score=float(operation([item["score"] for item in entries])),
                   preclip_gcs=float(operation(preclip)) if all(v is not None for v in preclip) else None,
                   saturated_fraction=float(np.mean([item["saturated"] for item in entries])),
                   native_channels=";".join(item["channel_name"] for item in entries),
                   score_origin="native_frozen_v2")
    inputs = [Path(pointer_path), features_path, Path(scores_path), Path(str(scores_path) + ".receipt.json"), jobs_path, qualification_path]
    return cfg, prepared, features, inputs


def _new_output(path):
    output = Path(path).resolve()
    if output.exists():
        raise StudyError("Output exists; preserve the previous model-guided record")
    output.mkdir(parents=True)
    return output


def _finish(output, inputs, status, **extra):
    record = {"status": status, "feature_definition": FEATURE_DEFINITION,
              "inputs": [{"path": str(Path(p).resolve()), "sha256": core.sha256_file(p)} for p in inputs],
              "outputs": {str(p.relative_to(output)): core.sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}, **extra}
    core.write_json(output / "receipt.json", record)
    return record


def _verify_output(directory):
    root = Path(directory).resolve()
    receipt = core.read_json(root / "receipt.json")
    for name, expected in receipt.get("outputs", {}).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or core.sha256_file(path) != expected:
            raise StudyError("Model-guided output was changed after completion")
    for item in receipt.get("inputs", []):
        if core.sha256_file(item["path"]) != item["sha256"]:
            raise StudyError("Model-guided source input was changed")
    if receipt.get("feature_definition") != FEATURE_DEFINITION:
        raise StudyError("Temporal measurement definitions differ")
    return receipt


def natural_pairs(rows, *, score_tolerance=.02, capacity_tolerance=0., maximum_pairs=20000):
    """Enumerate outcome-linked, disjoint within-person comparisons, retaining census.

    Score tolerances are exploratory resolution choices, not clinical equivalence.
    Saturated ties are retained and explicitly classified; never equated silently.
    """
    people = defaultdict(list)
    for row in rows:
        people[(row["participant_id"], row["source_id"], row["recording_id"])].append(row)
    pairs, counts = [], Counter()
    informative = defaultdict(set)
    for identity, observations in sorted(people.items()):
        observations = sorted(observations, key=lambda r: (float(r["start_s"]), r["episode_id"]))
        for i, a in enumerate(observations):
            for b in observations[i+1:]:
                counts["all_within_recording_pairs"] += 1
                if float(a["stop_s"]) > float(b["start_s"]) + 1e-9:
                    counts["overlapping_pairs"] += 1
                    continue
                counts["disjoint_pairs"] += 1
                difference = abs(float(a["score"]) - float(b["score"]))
                ca, cb = _finite(a.get("capacity_value")), _finite(b.get("capacity_value"))
                same_construct = bool(a.get("capacity_construct")) and a.get("capacity_construct") == b.get("capacity_construct") and a.get("capacity_units") == b.get("capacity_units")
                categories = []
                if difference <= score_tolerance:
                    if a.get("report_category") and b.get("report_category") and a["report_category"] != b["report_category"]:
                        categories.append("similar_clinical_score_different_original_reports")
                    if ca is not None and cb is not None and same_construct and abs(ca-cb) > capacity_tolerance:
                        categories.append("similar_clinical_score_different_measured_capacity")
                elif ca is not None and cb is not None and same_construct and abs(ca-cb) <= capacity_tolerance:
                    categories.append("different_clinical_score_comparable_measured_capacity")
                for category in categories:
                    counts[category] += 1
                    informative[category].add(identity[0])
                    if len(pairs) < maximum_pairs:
                        pair = {"pair_class": category, "participant_id": identity[0], "source_id": identity[1],
                                "recording_id": identity[2], "episode_a": a["episode_id"], "episode_b": b["episode_id"],
                                "score_a": float(a["score"]), "score_b": float(b["score"]), "score_difference": difference,
                                "report_a": a.get("report_category"), "report_b": b.get("report_category"),
                                "capacity_a": ca, "capacity_b": cb, "capacity_construct": a.get("capacity_construct"),
                                "gap_s": float(b["start_s"])-float(a["stop_s"]),
                                "saturation_a": a.get("saturated_fraction"), "saturation_b": b.get("saturated_fraction"),
                                "clipping_equivalence": "not_established" if (_finite(a.get("saturated_fraction")) or _finite(b.get("saturated_fraction"))) else "unsaturated_bounded_match"}
                        pairs.append(pair)
    return pairs, {"counts": dict(counts), "informative_participants": {key: len(value) for key, value in informative.items()},
                   "retained_pairs": len(pairs), "pair_output_limit": maximum_pairs,
                   "truncated": sum(counts[key] for key in informative) > len(pairs),
                   "score_tolerance": score_tolerance, "capacity_tolerance": capacity_tolerance,
                   "denominator": "All scored human development episodes; every compared pair shares a person and recording and has disjoint EEG intervals."}


def discover(pointer, scores, qualification, output):
    cfg, prepared, rows, inputs = _human(pointer, scores, qualification)
    out = _new_output(output)
    pairs, census = natural_pairs(rows)
    core.write_csv(out / "all_scored_observations.csv", rows)
    core.write_csv(out / "natural_adversarial_pairs.csv", pairs, sorted({key for row in pairs for key in row} | {"pair_class", "participant_id", "episode_a", "episode_b"}))
    registry = discovery.discovery_registry(rows, cfg.get("discovery", {}))
    core.write_json(out / "discovery.json", {"research_role": "Clinical model drives human phenomenon discovery before biological fitting",
                    "status": "candidates_for_explanation" if pairs else "no_candidate_under_recorded_resolution",
                    "registry": registry, "pair_census": census,
                    "participants": len({r["participant_id"] for r in rows}), "episodes": len(rows),
                    "retrospective_reports": "Original categories remain independent evidence; no-recall and mind blanking do not certify unconsciousness.",
                    "clinical_readout": "Fixed released GCS prediction; healthy-population clinical/awareness calibration remains unqualified."})
    return _finish(out, inputs, "model_guided_discovery_completed", pair_census=census)


def _event_response(recording, row, selected, quality_settings=None):
    missing = {"input_response_status": "missing_no_qualified_source_input", "input_response_rms_peak": None,
               "input_recovery_time_s": None, "input_recovery_censored": None}
    if row.get("source_trial_type") != "stim-adapt":
        return missing
    event_s = float(row["stop_s"])
    try:
        pre = data.extract_window(recording, event_s-1, event_s, duration_s=1.)[selected]
        post = data.extract_window(recording, event_s, event_s+3, duration_s=3.)[selected]
    except ValueError as exc:
        return {**missing, "input_response_status": "missing_source_window:"+str(exc)}
    fs = recording["sfreq_hz"]
    if quality_settings is not None:
        checks = [signals.quality_metrics(segment, fs, **quality_settings) for segment in (pre, post)]
        if any(not check["qualified"] for check in checks):
            return {**missing, "input_response_status": "missing_pre_or_post_event_signal_quality",
                    "input_response_quality_reasons": ";".join(reason for check in checks for reason in check["reasons"])}
    baseline = np.std(pre, axis=1, ddof=1)
    if np.any(baseline <= np.finfo(float).eps):
        return {**missing, "input_response_status": "missing_constant_source_baseline"}
    width = max(1, round(fs*.05))
    blocks = post.shape[1]//width
    normalized = (post-pre.mean(axis=1, keepdims=True))/baseline[:, None]
    rms = np.sqrt(np.mean(normalized[:, :blocks*width].reshape(len(selected), blocks, width)**2, axis=(0, 2)))
    peak = int(np.argmax(rms[:max(1, round(.5*fs/width))]))
    recovered = None
    detected = bool(rms[peak]>1.5)
    if detected:
        for i in range(peak+1, len(rms)-4):
            if np.all(rms[i:i+5] <= 1.5):
                recovered = i*width/fs
                break
    # Source trigger alignment is explicitly unqualified in ds001785; values are
    # descriptive response to the annotated event, not qualified sensory latency.
    return {"input_response_status": "source_annotated_tactile_event_physical_trigger_alignment_unqualified",
            "input_response_rms_peak": float(rms[peak]), "input_recovery_time_s": recovered,
            "input_response_detected_at_recorded_resolution":detected,
            "input_recovery_censored": detected and recovered is None,
            "input_recovery_status":"no_detectable_response" if not detected else ("right_censored_at3s" if recovered is None else "returned_within_recorded_baseline_envelope"),
            "input_response_onset_s": event_s, "input_response_intensity": _finite(row.get("stim_intensity")),
            "input_recovery_definition": "post-event normalized50msRMS <=1.5 for250ms; first3s observation, right censored if absent"}


def trajectories(pointer, scores, qualification, output):
    cfg, prepared, rows, inputs = _human(pointer, scores, qualification)
    out = _new_output(output)
    receipts = core.read_json(prepared / "recordings.json")["recordings"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["recording_id"]].append(row)
    observations, exclusions, transitions, controls = [], [], [], []
    for receipt in receipts:
        if core.sha256_file(receipt["path"]) != receipt["sha256"]:
            raise StudyError("Human trajectory recording changed")
        recording = data.read_recording(receipt["path"], "npz")
        names = recording["channel_names"]
        # Source adapters export audited scalp EEG. All available channels up to
        # the largest declared dense montage (257) are fixed before outcomes.
        fs = recording["sfreq_hz"]
        native_names = cfg.get("instrument", {}).get("channel_names", ["Cz", "Pz", "Fz", "Oz"])
        index = [names.index(name) for name in native_names]
        participant_rows = []
        for row in sorted(grouped[receipt["recording_id"]], key=lambda r: float(r["start_s"])):
            try:
                window = data.extract_window(recording, float(row["start_s"]), float(row["stop_s"]), duration_s=10.)
                policy = cfg.get("channel_quality", {})
                chosen, channel_audits = signals.channel_quality_selection(window, fs, names, policy)
                chosen = chosen[:257]
                audit_id = hashlib.sha256(row["episode_id"].encode()).hexdigest()[:24]
                core.write_json(out/"channel_quality"/(audit_id+".json"), {"episode_id":row["episode_id"], "policy":policy,
                                "channels":channel_audits, "selected_channel_names":[names[i] for i in chosen],
                                "source_channels":len(names), "maximum_channels":257,
                                "decision_basis":"Signal-only rule, fixed before clinical scores or human outcomes"})
                if len(chosen)<max(2, int(policy.get("minimum_channels",2))):
                    raise StudyError("insufficient_signal_qualified_scalp_channels_for_trajectory")
                prepared_channels = row.get("scalp_qualified_channel_names")
                if prepared_channels and prepared_channels != ";".join(names[i] for i in chosen):
                    raise StudyError("Prepared and trajectory scalp channel qualification differs")
                per_channel = [univariate_features(window[i], fs) for i in index]
                temporal = {key: float(np.median([features[key] for features in per_channel])) for key in per_channel[0]}
                measured = {**row, **temporal, **multichannel_trajectory(window[chosen], fs),
                            **_event_response(recording, row, chosen, cfg.get("quality", {})),
                            "trajectory_channel_names": ";".join(names[i] for i in chosen),
                            "temporal_channel_aggregation": "median_across_fixed_native_channels",
                            "trajectory_source_channel_count":len(names), "trajectory_quality_excluded_channels":len(names)-len(chosen),
                            "trajectory_channel_quality_audit":str(out/"channel_quality"/(audit_id+".json")),
                            "feature_definition": FEATURE_DEFINITION, "measurement_status": "actual_multichannel_eeg",
                            "human_outcome_mapping_qualified": False}
                observations.append(measured)
                participant_rows.append(measured)
                seed = int(hashlib.sha256(row["episode_id"].encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                source_window = window[chosen]
                variants = {"shared_time_shuffle": source_window[:, rng.permutation(source_window.shape[1])],
                            "shared_phase": discovery.phase_surrogate(source_window, seed, shared_phases=True),
                            "independent_phase": discovery.phase_surrogate(source_window, seed+1, shared_phases=False)}
                for label, variant in variants.items():
                    controls.append({"episode_id": row["episode_id"], "participant_id": row["participant_id"],
                                     "source_id": row["source_id"], "control": label, "seed": seed,
                                     "control_status": "measurement_control_not_human_intervention_not_clinically_scored",
                                     **multichannel_trajectory(variant, fs, compute_rank=False)})
            except (ValueError, IndexError) as exc:
                exclusions.append({"episode_id": row["episode_id"], "participant_id": row["participant_id"],
                                   "source_id": row["source_id"], "recording_id": row["recording_id"], "reason": str(exc)})
        for a, b in zip(participant_rows, participant_rows[1:]):
            if float(a["stop_s"]) > float(b["start_s"])+1e-9:
                continue
            transitions.append({"participant_id": a["participant_id"], "source_id": a["source_id"],
                                "episode_a": a["episode_id"], "episode_b": b["episode_id"],
                                "gap_s": float(b["start_s"])-float(a["stop_s"]),
                                "clinical_score_change": b["score"]-a["score"],
                                "original_report_a": a.get("report_category"), "original_report_b": b.get("report_category"),
                                "observed_continuous_between_windows": False,
                                **{key+"_change": b[key]-a[key] for key in TEMPORAL_FEATURES}})
        del recording
    fields = sorted({key for row in observations for key in row} | set(TEMPORAL_FEATURES) | {"episode_id", "participant_id", "source_id", "site_id", "recording_id", "score"})
    core.write_csv(out / "observations.csv", observations, fields)
    core.write_csv(out / "measurement_exclusions.csv", exclusions, ["episode_id", "participant_id", "source_id", "recording_id", "reason"])
    core.write_csv(out / "successive_window_changes.csv", transitions, sorted({key for row in transitions for key in row} | {"participant_id", "episode_a", "episode_b", "gap_s"}))
    core.write_csv(out / "trajectory_controls.csv", controls, sorted({key for row in controls for key in row}|{"episode_id", "participant_id", "source_id", "control"}))
    core.write_json(out / "trajectory_definition.json", {"multichannel": "Up to257source scalp channels passing fixed signal-only per-window quality; raw samples retained, no interpolation; per-channel audit retained. Channel z-score; disjoint bin means<=400states; fixed RMS standardized recurrence threshold.75; exclude <100ms separations",
                    "temporal": FEATURE_DEFINITION, "clinical_readout": "Actual unchanged native score drives the episode comparisons",
                    "scope": "Windows are separate observations. Successive differences do not imply a measured continuous transition through the gap.",
                    "controls": "Shared time shuffle preserves contemporaneous channel vectors/covariance, shared phase preserves spectra/cross spectra, independent phase preserves individual spectra and alters interchannel organization. Controls are not clinical model observations or human interventions.",
                    "channel_count_comparisons": "Per-window channel masks/counts and rank fractions are retained; changes in full-scalp geometry require matched-channel sensitivity analysis, not an unadjusted brain-dimensionality interpretation.",
                    "recovery": "Only source-supported tactile annotations yield exploratory measured response/recovery; physical trigger alignment remains unqualified."})
    return _finish(out, inputs, "model_guided_trajectories_completed", denominator_episodes=len(rows), measured_episodes=len(observations),
                   excluded_episodes=len(exclusions), independent_participants=len({r["participant_id"] for r in observations}))


def _verified_bank(inventory_path, jobs_path, scores_path, qualification_path, receipt_path):
    receipt = core.read_json(receipt_path)
    if receipt.get("status") != "native_simulation_bank_completed" or receipt.get("feature_definition") != FEATURE_DEFINITION:
        raise StudyError("Bank lacks completed native simulation/measurement provenance")
    for key, path in (("inventory_sha256", inventory_path), ("jobs_sha256", jobs_path)):
        if receipt.get(key) != core.sha256_file(path):
            raise StudyError("Native simulation bank table or export changed")
    from native_tools.nftsim_runtime import verify_release as verify_nftsim_release
    release = Path(receipt["release_dir"])
    if receipt.get("source_manifest_sha256") != core.sha256_file(release/"SOURCE_MANIFEST.json"):
        raise StudyError("Native simulation published source manifest changed")
    verify_nftsim_release(release)
    variants = receipt.get("variants", [])
    if not variants:
        raise StudyError("Native bank lacks actual executable/source provenance")
    for variant in variants:
        executable = Path(variant["executable"])
        if core.sha256_file(executable) != variant["executable_sha256"]:
            raise StudyError("Native simulation executable changed")
        for name, expected in variant.get("source_hashes", {}).items():
            path = (executable.parent/"src"/name).resolve()
            if not path.is_relative_to((executable.parent/"src").resolve()) or core.sha256_file(path) != expected:
                raise StudyError("Native simulation implementation source changed")
    for item in receipt.get("native_outputs", []):
        if core.sha256_file(item["path"]) != item["sha256"]:
            raise StudyError("Native simulated trajectory changed")
    inventory = core.read_csv(inventory_path, ("simulation_id", "account_class", "status", "config_sha256"))
    if len({row["simulation_id"] for row in inventory}) != len(inventory):
        raise StudyError("Duplicate simulation identity")
    imported = native.import_scores(Path(scores_path), Path(jobs_path), Path(qualification_path))
    by_id = defaultdict(list)
    for row in imported:
        by_id[row["episode_id"]].append(row)
    actual_ids = {item.get("simulation_id") for item in receipt.get("native_outputs", [])}
    actual_by_id = {item.get("simulation_id"):item for item in receipt.get("native_outputs", [])}
    if len(actual_by_id) != len(receipt.get("native_outputs", [])):
        raise StudyError("Duplicate native trajectory receipt identity")
    complete, failures = [], []
    for row in inventory:
        simulation = row["simulation_id"]
        if simulation not in by_id:
            failures.append({**row, "fit_status": "not_scored_or_native_simulation_failed"})
            continue
        if simulation not in actual_ids:
            raise StudyError("Scored simulation has no native trajectory hash receipt")
        values = by_id.pop(simulation)
        actual = actual_by_id[simulation]
        if row.get("status") != "finite":
            raise StudyError("Non-finite/failed native simulation was exported for scoring")
        if row.get("source_output_path") != actual["path"] or row.get("source_output_sha256") != actual["sha256"]:
            raise StudyError("Native simulation inventory and output receipt differ")
        config_path = Path(actual["path"]).parent/"configuration.conf"
        if core.sha256_file(config_path) != row["config_sha256"]:
            raise StudyError("Executed native simulation configuration changed")
        if any(_finite(row.get(key)) is None for key in TEMPORAL_FEATURES):
            failures.append({**row, "fit_status": "missing_temporal_measurement"})
            continue
        row.update(score=float(np.median([v["score"] for v in values])),
                   preclip_gcs=float(np.median([v["preclip_gcs"] for v in values])) if all(v["preclip_gcs"] is not None for v in values) else None,
                   saturated_fraction=float(np.mean([v["saturated"] for v in values])), score_origin="native_frozen_v2",
                   fit_status="finite_native_simulation_proxy")
        complete.append(row)
    if by_id:
        raise StudyError("Native simulation scores refer to absent inventory identities")
    return complete, failures, receipt


def _matrix(rows, columns):
    return np.asarray([[float(row[key]) for key in columns] for row in rows], dtype=float)


def _scales(matrix):
    std = np.std(matrix, axis=0, ddof=1) if len(matrix) > 1 else np.ones(matrix.shape[1])
    return np.where(np.isfinite(std) & (std > 1e-9), std, 1.)


def _nearest(human, bank, scales, columns=None):
    columns = np.arange(human.shape[1]) if columns is None else np.asarray(columns)
    indices, distances = [], []
    for row in human:
        distance = np.mean(((bank[:, columns]-row[columns])/scales[columns])**2, axis=1)
        chosen = int(np.argmin(distance))
        indices.append(chosen); distances.append(float(np.sqrt(distance[chosen])))
    return np.asarray(indices), np.asarray(distances)


def _ridge(train_x, train_y, test_x):
    means = train_x.mean(axis=0)
    scale = _scales(train_x)
    a = np.c_[np.ones(len(train_x)), (train_x-means)/scale]
    b = np.c_[np.ones(len(test_x)), (test_x-means)/scale]
    penalty = np.eye(a.shape[1]); penalty[0, 0] = 0
    beta = np.linalg.solve(a.T@a/len(a)+penalty, a.T@train_y/len(a))
    return b@beta


def grouped_outcome_prediction(rows, reconstructed, *, group_column="participant_id", bank_matrix=None):
    """Development-only grouped transport of generative proxies to human outcomes.

    Accounts use outcomes only after matching measured EEG summaries. Clinical-only
    and conventional-EEG comparators face the same held-out participants/sites.
    All outcomes stay distinct. Self-ratings are numerical descriptive codes.
    """
    output = []
    for target in ("capacity_value", "awareness_rating_raw", "detection", "confidence"):
        selected = [i for i, row in enumerate(rows) if _finite(row.get(target)) is not None]
        groups = sorted({rows[i].get(group_column, "") for i in selected} - {""})
        if len(groups) < (3 if group_column == "participant_id" else 2):
            output.append({"target": target, "grouping": group_column, "status": "insufficient_independent_groups", "groups": len(groups)})
            continue
        # Different sources/constructs/units never silently form one response scale.
        strata = defaultdict(list)
        for i in selected:
            row = rows[i]
            strata[(row["source_id"], row.get("capacity_construct", "") if target == "capacity_value" else target,
                    row.get("capacity_units", "") if target == "capacity_value" else "source_numeric_code")].append(i)
        for stratum, indices in sorted(strata.items()):
            actual_groups = sorted({rows[i].get(group_column, "") for i in indices} - {""})
            if len(actual_groups) < (3 if group_column == "participant_id" else 2):
                output.append({"target": target, "grouping": group_column, "source_id": stratum[0], "status": "insufficient_independent_groups", "groups": len(actual_groups)})
                continue
            y = np.asarray([float(rows[i][target]) for i in indices])
            measured = _matrix([rows[i] for i in indices], TEMPORAL_FEATURES)
            scores = np.asarray([[float(rows[i]["score"])] for i in indices])
            proposed = reconstructed[indices]
            feature_sets = {"clinical_only": scores, "clinical_plus_conventional_eeg": np.c_[scores, measured[:, 2:4]],
                            "clinical_plus_all_measured_temporal_eeg": np.c_[scores, measured],
                            "clinical_plus_generative_temporal_proxy": np.c_[scores, proposed]}
            for label, x in feature_sets.items():
                participants = defaultdict(list)
                folds = []
                for group in actual_groups:
                    test = np.asarray([rows[i].get(group_column) == group for i in indices])
                    train = ~test
                    if train.sum() < 3 or not np.isfinite(x[train]).all() or not np.isfinite(x[test]).all():
                        folds.append({"held_out_group": group, "status": "insufficient_finite_training_data"}); continue
                    fold_x = x
                    if label == "clinical_plus_generative_temporal_proxy" and bank_matrix is not None:
                        # Refit the measurement matching operator using only
                        # training-group scales, before opening held-out outcomes.
                        bank_indices, _ = _nearest(measured, bank_matrix, _scales(measured[train]))
                        fold_x = np.c_[scores, bank_matrix[bank_indices]]
                    prediction = _ridge(fold_x[train], y[train], fold_x[test])
                    for local, pred in zip(np.flatnonzero(test), prediction):
                        participants[rows[indices[local]]["participant_id"]].append((float(pred)-y[local])**2)
                    folds.append({"held_out_group": group, "training_groups": sorted({rows[indices[i]].get(group_column) for i in np.flatnonzero(train)}),
                                  "held_out_observations": int(test.sum()), "mse": float(np.mean((prediction-y[test])**2))})
                output.append({"target": target, "source_id": stratum[0], "construct": stratum[1], "units": stratum[2],
                               "grouping": group_column, "predictor": label, "status": "development_grouped_prediction" if participants else "no_valid_fold",
                               "independent_participants": len(participants), "equal_participant_mse": float(np.mean([np.mean(v) for v in participants.values()])) if participants else None,
                               "folds": folds, "interpretation": "Exploratory prediction in development cohorts; unqualified biological identity and causal human outcome mapping."})
    # Original reports are nominal, including no recall. They receive a separate
    # categorical transport test, never a fabricated linear consciousness scale.
    report_sources = defaultdict(list)
    for i, row in enumerate(rows):
        if row.get("report_category"):
            report_sources[row["source_id"]].append(i)
    for source, indices in sorted(report_sources.items()):
        groups = sorted({rows[i].get(group_column, "") for i in indices}-{ "" })
        labels = sorted({rows[i]["report_category"] for i in indices})
        if len(groups)<(3 if group_column=="participant_id" else 2) or len(labels)<2:
            output.append({"target":"original_nominal_report", "source_id":source, "grouping":group_column,
                           "status":"insufficient_groups_or_report_variation", "groups":len(groups), "original_categories":labels})
            continue
        measured = _matrix([rows[i] for i in indices], TEMPORAL_FEATURES)
        scores = np.asarray([[float(rows[i]["score"])] for i in indices])
        category = np.asarray([labels.index(rows[i]["report_category"]) for i in indices])
        onehot = np.eye(len(labels))[category]
        proposed = reconstructed[indices]
        sets = {"clinical_only":scores, "clinical_plus_conventional_eeg":np.c_[scores, measured[:,2:4]],
                "clinical_plus_all_measured_temporal_eeg":np.c_[scores,measured],
                "clinical_plus_generative_temporal_proxy":np.c_[scores,proposed]}
        for label,x in sets.items():
            person_errors, baseline_errors, folds = defaultdict(list), defaultdict(list), []
            class_counts = Counter()
            for group in groups:
                test = np.asarray([rows[i].get(group_column)==group for i in indices]); train=~test
                if train.sum()<3:
                    continue
                fold_x=x
                if label=="clinical_plus_generative_temporal_proxy" and bank_matrix is not None:
                    chosen,_=_nearest(measured,bank_matrix,_scales(measured[train]))
                    fold_x=np.c_[scores,bank_matrix[chosen]]
                prediction=np.argmax(_ridge(fold_x[train],onehot[train],fold_x[test]),axis=1)
                baseline=int(np.argmax(onehot[train].mean(axis=0)))
                for local,pred in zip(np.flatnonzero(test),prediction):
                    person=rows[indices[local]]["participant_id"]
                    person_errors[person].append(int(pred!=category[local]))
                    baseline_errors[person].append(int(baseline!=category[local]))
                    class_counts[labels[category[local]]]+=1
                folds.append({"held_out_group":group,"training_groups":sorted(set(groups)-{group}),
                              "held_out_observations":int(test.sum()),
                              "held_out_original_categories":sorted({labels[v] for v in category[test]}),
                              "unseen_training_categories":sorted({labels[v] for v in category[test]}-{labels[v] for v in category[train]})})
            output.append({"target":"original_nominal_report", "source_id":source,"grouping":group_column,"predictor":label,
                           "status":"development_grouped_nominal_report_prediction" if person_errors else "no_valid_fold",
                           "original_categories":labels,"held_out_category_counts":dict(class_counts),"independent_participants":len(person_errors),
                           "equal_participant_misclassification_rate":float(np.mean([np.mean(v) for v in person_errors.values()])) if person_errors else None,
                           "training_majority_equal_participant_misclassification_rate":float(np.mean([np.mean(v) for v in baseline_errors.values()])) if baseline_errors else None,
                           "folds":folds,"interpretation":"Development nominal report prediction; no-recall remains a category and mind blanking is not equated with unconsciousness. Discriminant outputs are not calibrated awareness probabilities."})
    return output


def fit(observations, bank, bank_jobs, bank_scores, qualification, bank_receipt, output):
    rows, inputs, denominator = [], [], []
    for directory in observations:
        receipt = _verify_output(directory)
        if receipt.get("status") != "model_guided_trajectories_completed":
            raise StudyError("Fit requires completed genuine human trajectory measurements")
        source_rows = core.read_csv(Path(directory)/"observations.csv")
        denominator.append({"path": str(directory), "all_scored_episodes": receipt["denominator_episodes"],
                            "measured_episodes": len(source_rows), "exclusions": receipt["excluded_episodes"]})
        rows.extend(source_rows); inputs.extend([Path(directory)/"receipt.json", Path(directory)/"observations.csv"])
    seen = set()
    for row in rows:
        identity = row["participant_id"], row["episode_id"]
        if identity in seen:
            raise StudyError("Duplicate human observations across trajectory sources")
        seen.add(identity)
        if row.get("measurement_status") != "actual_multichannel_eeg" or row.get("score_origin") != "native_frozen_v2" or core.flag(row.get("synthetic")):
            raise StudyError("Generative fitting requires genuine EEG and native clinical scores")
    inventory, failed, receipt = _verified_bank(bank, bank_jobs, bank_scores, qualification, bank_receipt)
    inputs += [Path(bank), Path(bank_jobs), Path(bank_scores), Path(str(bank_scores)+".receipt.json"), Path(qualification), Path(bank_receipt)]
    out = _new_output(output)
    valid = [row for row in rows if all(_finite(row.get(key)) is not None for key in TEMPORAL_FEATURES)]
    x = _matrix(valid, TEMPORAL_FEATURES) if valid else np.empty((0, len(TEMPORAL_FEATURES)))
    global_scales = _scales(x) if valid else np.ones(len(TEMPORAL_FEATURES))
    accounts = defaultdict(list)
    for row in inventory:
        accounts[(row["account_class"], row.get("variant", "unspecified"))].append(row)
    fits, summaries, predictions, recovery = [], [], [], []
    people = sorted({row["participant_id"] for row in valid})
    for (account, variant), candidates in sorted(accounts.items()):
        b = _matrix(candidates, TEMPORAL_FEATURES)
        if not valid:
            summaries.append({"account_class": account, "variant": variant, "status": "no_finite_human_measurements", "native_candidates": len(candidates)})
            continue
        nearest, distance = _nearest(x, b, global_scales)
        recon = b[nearest]
        for i, row in enumerate(valid):
            chosen = candidates[int(nearest[i])]
            outside = [key for j, key in enumerate(TEMPORAL_FEATURES) if x[i,j] < b[:,j].min()-global_scales[j]*.25 or x[i,j] > b[:,j].max()+global_scales[j]*.25]
            fits.append({"account_class": account, "variant": variant, "participant_id": row["participant_id"], "source_id": row["source_id"],
                         "episode_id": row["episode_id"], "simulation_id": chosen["simulation_id"], "standardized_summary_distance": float(distance[i]),
                         "out_of_bank_features": ";".join(outside), "fit_status": "out_of_bank_domain" if outside else "within_summary_envelope",
                         "human_clinical_score": float(row["score"]), "simulation_clinical_score": float(chosen["score"]),
                         **{key: chosen.get(key) for key in chosen if key.startswith("param_")}})
        held_errors, null_errors, fold_records = defaultdict(list), defaultdict(list), []
        # Fit four summaries and predict the fifth. Held-out people determine
        # scale; held target is never used in the simulation selection.
        for person in people:
            test = np.asarray([row["participant_id"] == person for row in valid])
            train = ~test
            if train.sum() < 2:
                continue
            scale = _scales(x[train])
            for held in range(len(TEMPORAL_FEATURES)):
                included = [j for j in range(len(TEMPORAL_FEATURES)) if j != held]
                idx, distances = _nearest(x[test], b, scale, included)
                predicted = b[idx, held]
                error = (predicted-x[test, held])/scale[held]
                null_prediction = float(x[train,held].mean())
                null_error = (null_prediction-x[test,held])/scale[held]
                held_errors[person].extend((error**2).tolist())
                null_errors[person].extend((null_error**2).tolist())
                fold_records.append({"held_out_participant": person, "held_out_measurement": TEMPORAL_FEATURES[held],
                                     "training_participants": sorted(set(people)-{person}), "held_out_episodes": int(test.sum()),
                                     "standardized_mse": float(np.mean(error**2)), "training_mean_standardized_mse": float(np.mean(null_error**2)),
                                     "selection_features": [TEMPORAL_FEATURES[j] for j in included]})
        equal_mse = float(np.mean([np.mean(v) for v in held_errors.values()])) if held_errors else None
        null_mse = float(np.mean([np.mean(v) for v in null_errors.values()])) if null_errors else None
        adequate = equal_mse is not None and null_mse is not None and equal_mse<1 and equal_mse<.95*null_mse
        status = "insufficient_participant_generalization" if len(held_errors)<3 else ("candidate_effective_proxy_predictive_adequacy" if adequate else "weak_proxy_predictive_adequacy")
        summaries.append({"account_class": account, "variant": variant, "status": status, "native_candidates": len(candidates),
                          "participants": len(people), "participants_with_held_feature_prediction": len(held_errors),
                          "equal_participant_standardized_held_feature_mse": equal_mse,
                          "training_mean_equal_participant_standardized_mse": null_mse,
                          "candidate_adequacy_resolution": "At least3held people; standardized heldfeature MSE<1 and >5% improvement over training-mean null. Exploratory resolution, not confirmatory biological evidence.",
                          "held_feature_folds": fold_records, "measurement_mapping": MAPPING_INTERPRETATION,
                          "anatomical_identification": "not_established", "human_outcome_mapping": "unqualified_development_prediction_only"})
        predictions.extend({"account_class": account, "variant": variant, **record} for group in ("participant_id", "site_id") for record in grouped_outcome_prediction(valid, recon, group_column=group, bank_matrix=b))
        # Recovery on model-generated summaries identifies ambiguity under the
        # sampled measurement operator; not recovery of human biology.
        if len(candidates) > 1:
            for i, candidate in enumerate(candidates):
                ds = np.mean(((b-b[i])/global_scales)**2, axis=1); ds[i] = np.inf
                j = int(np.argmin(ds))
                recovery.append({"account_class": account, "variant": variant, "simulation_id": candidate["simulation_id"],
                                 "nearest_distinct_simulation": candidates[j]["simulation_id"], "summary_distance": float(np.sqrt(ds[j])),
                                 "ambiguous_at_resolution_0_1": bool(ds[j] <= .01),
                                 "parameter_recovery_status": "sampled_bank_identifiability_only_not_human_parameter_recovery"})
    core.write_csv(out/"human_observations.csv", rows, sorted({key for row in rows for key in row}|set(TEMPORAL_FEATURES)|{"episode_id", "participant_id", "source_id", "score"}))
    core.write_csv(out/"scored_bank.csv", inventory, sorted({key for row in inventory for key in row}|set(TEMPORAL_FEATURES)|{"simulation_id", "account_class", "variant", "score"}))
    core.write_csv(out/"retained_simulation_failures.csv", failed, sorted({key for row in failed for key in row}|{"simulation_id", "status", "fit_status"}))
    core.write_csv(out/"empirical_proxy_fits.csv", fits, sorted({key for row in fits for key in row}|{"episode_id", "simulation_id", "account_class", "fit_status"}))
    core.write_csv(out/"bank_identifiability.csv", recovery, sorted({key for row in recovery for key in row}|{"simulation_id", "parameter_recovery_status"}))
    equivalence = []
    for i, a in enumerate(summaries):
        for b in summaries[i+1:]:
            ma = a.get("equal_participant_standardized_held_feature_mse")
            mb = b.get("equal_participant_standardized_held_feature_mse")
            if ma is None or mb is None:
                continue
            ratio = abs(ma-mb)/max(ma, mb, 1e-9)
            equivalence.append({"account_a": [a["account_class"], a["variant"]], "account_b": [b["account_class"], b["variant"]],
                                "relative_held_feature_error_difference": ratio,
                                "status": "similar_summary_predictive_adequacy_at_exploratory_resolution" if ratio<=.25 else "different_summary_predictive_adequacy",
                                "interpretation": "Descriptive adequacy class; no statistical or biological equivalence claim"})
    core.write_json(out/"accounts.json", {"status": "competing_proxy_accounts_evaluated" if accounts and valid else "no_evaluable_account",
                    "feature_columns": TEMPORAL_FEATURES, "training_scale": global_scales.tolist(), "accounts": summaries,
                    "exploratory_adequacy_equivalence_classes": equivalence,
                    "human_denominator": denominator, "native_failure_rows": len(failed), "feature_mapping": MAPPING_INTERPRETATION,
                    "absolute_amplitude_excluded": "EEG microvolts and native cortical voltage lack an identified scalp/unit observation map",
                    "clinical_score_role": "Human natural comparisons and simulated adversarial contrasts are driven by unchanged native judgments; fitting also must explain actual dynamics.",
                    "evidence_boundary": "All current human fitting and cross-validation are development; no protected human replication or causal identification is certified."})
    core.write_json(out/"human_outcome_predictions.json", {"results": predictions, "report_categories": "Distinct nominal original experience reports remain in discovery; no ordinal clinical-consciousness conversion is made.",
                    "interpretation": "Numerical self-ratings/confidence preserve source codes. These exploratory predictions cannot substitute for a qualified model-to-function mechanism or independent cohort."})
    return _finish(out, inputs, "model_guided_fit_completed", valid_human_episodes=len(valid), independent_participants=len(people),
                   native_complete_candidates=len(inventory), retained_simulation_failures=len(failed), account_classes=len(accounts),
                   human_data_fitting_executed=bool(valid and fits),
                   human_adaptation_kind="generative_parameter_fitting_to_actual_human_temporal_summaries",
                   human_outcome_mapping_qualified=False, anatomical_forward_mapping_qualified=False)


def adversarial_pairs(bank, scales, *, score_tolerance=.02, dynamic_difference=.5, conventional_tolerance=.5, limit=20000):
    """Search only actual natively scored finite bank trajectories, with full census."""
    rows = sorted(bank, key=lambda r: (float(r["score"]), r["simulation_id"]))
    candidates, counts = [], Counter()
    feature_index = {key:i for i, key in enumerate(TEMPORAL_FEATURES)}
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            score_difference = abs(float(a["score"])-float(b["score"]))
            if score_difference > score_tolerance:
                break
            counts["similar_score_bank_pairs"] += 1
            conventional = max(abs(float(a[key])-float(b[key]))/scales[feature_index[key]] for key in CONVENTIONAL_FEATURES)
            dynamics = max(abs(float(a[key])-float(b[key]))/scales[feature_index[key]] for key in DYNAMIC_FEATURES)
            cross_account = a["account_class"] != b["account_class"]
            saturated = (_finite(a.get("saturated_fraction")) or 0)>0 or (_finite(b.get("saturated_fraction")) or 0)>0
            category = "similar_clinical_score_different_temporal_dynamics" if dynamics>=dynamic_difference else "similar_score_similar_temporal_summary"
            counts[category] += 1
            if dynamics < dynamic_difference:
                continue
            if cross_account and conventional<=conventional_tolerance:
                counts["cross_account_conventional_matched_diagnostic_candidates"] += 1
            if len(candidates)<limit:
                candidates.append({"candidate_class": category, "simulation_a": a["simulation_id"], "simulation_b": b["simulation_id"],
                                   "account_a": a["account_class"], "account_b": b["account_class"],
                                   "variant_a": a.get("variant"), "variant_b": b.get("variant"),
                                   "score_a": float(a["score"]), "score_b": float(b["score"]), "score_difference": score_difference,
                                   "max_standardized_dynamic_difference": dynamics, "max_standardized_conventional_difference": conventional,
                                   "cross_account": cross_account, "conventional_matched": conventional<=conventional_tolerance,
                                   "bounded_tie_involves_saturation": saturated,
                                   "prediction_status": "model_space_candidate_human_mapping_unqualified",
                                   **{key+"_difference": float(b[key])-float(a[key]) for key in TEMPORAL_FEATURES}})
    # Explicitly paired interventions alter native scores while testing retention
    # of temporal properties. Condition pairing uses measured parameter identity.
    groups = defaultdict(list)
    parameters = sorted({key for row in rows for key in row if key.startswith("param_")})
    for row in rows:
        key = (row["account_class"], row.get("variant"), row.get("published_config_id"), *(row.get(p, "") for p in parameters))
        groups[key].append(row)
    for group in groups.values():
        for i, a in enumerate(group):
            for b in group[i+1:]:
                if a.get("input_amplitude") == b.get("input_amplitude"):
                    continue
                counts["input_paired_native_conditions"] += 1
                difference = abs(float(a["score"])-float(b["score"]))
                dynamic = max(abs(float(a[key])-float(b[key]))/scales[feature_index[key]] for key in DYNAMIC_FEATURES)
                if difference > score_tolerance and dynamic<dynamic_difference:
                    counts["clinical_score_change_preserved_temporal_property"] += 1
                    if len(candidates)<limit:
                        candidates.append({"candidate_class": "clinical_score_change_preserved_temporal_property", "simulation_a": a["simulation_id"], "simulation_b": b["simulation_id"],
                                           "account_a": a["account_class"], "account_b": b["account_class"], "score_a": float(a["score"]), "score_b": float(b["score"]),
                                           "score_difference": difference, "max_standardized_dynamic_difference": dynamic,
                                           "prediction_status": "model_space_candidate_human_mapping_unqualified"})
    return candidates, {"counts": dict(counts), "native_bank_denominator": len(rows), "score_tolerance": score_tolerance,
                        "dynamic_difference_threshold": dynamic_difference, "conventional_match_threshold": conventional_tolerance,
                        "retained_candidates": len(candidates), "candidate_output_limit": limit,
                        "human_functional_prediction": "not_qualified_by_bank_search"}


def search(fit_dir, output):
    _verify_output(fit_dir)
    fit_dir = Path(fit_dir)
    accounts = core.read_json(fit_dir/"accounts.json")
    rows = core.read_csv(fit_dir/"scored_bank.csv")
    scale = np.asarray(accounts["training_scale"], dtype=float)
    if scale.shape != (len(TEMPORAL_FEATURES),) or not np.isfinite(scale).all() or np.any(scale<=0):
        raise StudyError("Invalid development feature scale")
    pairs, census = adversarial_pairs(rows, scale)
    fit_rows = core.read_csv(fit_dir/"empirical_proxy_fits.csv")
    adequate = {(a["account_class"], a["variant"]) for a in accounts["accounts"] if a["status"] == "candidate_effective_proxy_predictive_adequacy"}
    support = defaultdict(set)
    for row in fit_rows:
        if row["fit_status"] == "within_summary_envelope" and float(row["standardized_summary_distance"]) <= 1.:
            support[(row["account_class"], row["variant"], row["simulation_id"])].add(row["participant_id"])
    bank_by_id = {row["simulation_id"]:row for row in rows}
    constrained = []
    for pair in pairs:
        a, b = bank_by_id[pair["simulation_a"]], bank_by_id[pair["simulation_b"]]
        ka, kb = (a["account_class"], a.get("variant", "unspecified")), (b["account_class"], b.get("variant", "unspecified"))
        sa, sb = support[(*ka, a["simulation_id"])], support[(*kb, b["simulation_id"])]
        eligible = ka in adequate and kb in adequate and bool(sa) and bool(sb) and len(sa|sb)>=2
        pair.update(account_a_empirical_predictive_adequacy=ka in adequate, account_b_empirical_predictive_adequacy=kb in adequate,
                    simulation_a_supporting_participants=len(sa), simulation_b_supporting_participants=len(sb),
                    independent_supporting_participants=len(sa|sb), empirical_constraint_status="supported_effective_proxy_comparison" if eligible else "unsupported_or_weak_empirical_mapping")
        if eligible:
            constrained.append(pair)
    census.update(empirically_constrained_candidates=len(constrained), empirically_adequate_account_implementations=len(adequate),
                  empirical_support_definition="Both simulation summaries have within-domain actual human fits at standardized distance<=1; accounts have held-person held-feature MSE<1 and >5% improvement over training-mean null with>=3people; union support>=2people. This qualifies only provisional effective-summary adequacy.")
    out = _new_output(output)
    core.write_csv(out/"adversarial_candidates.csv", pairs, sorted({key for row in pairs for key in row}|{"candidate_class", "simulation_a", "simulation_b", "prediction_status"}))
    core.write_csv(out/"empirically_constrained_candidates.csv", constrained, sorted({key for row in pairs for key in row}|{"candidate_class", "simulation_a", "simulation_b", "empirical_constraint_status"}))
    core.write_json(out/"search.json", {"status": "empirically_supported_candidate_explanatory_comparisons" if constrained else "no_empirically_supported_account_comparison",
                    "search_census": census, "account_adequacy": [{k:a.get(k) for k in ("account_class", "variant", "status", "participants", "equal_participant_standardized_held_feature_mse")} for a in accounts["accounts"]],
                    "retained_failures": str(fit_dir/"retained_simulation_failures.csv"), "retained_ood_fits": str(fit_dir/"empirical_proxy_fits.csv"),
                    "biological_plausibility": "Finite actual native configurations within the recorded published/operator bounds; this is computational plausibility, not validated human physiology.",
                    "prediction_boundary": "Simulated temporal/input-response differences require qualified scalp and human-function mappings before a diagnostic human contrast is committed."})
    return _finish(out, [fit_dir/"receipt.json", fit_dir/"scored_bank.csv", fit_dir/"accounts.json", fit_dir/"empirical_proxy_fits.csv"], "model_guided_search_completed", **census)


def diagnostic(search_dir, output):
    _verify_output(search_dir)
    source = Path(search_dir)
    pairs = core.read_csv(source/"empirically_constrained_candidates.csv")
    search_record = core.read_json(source/"search.json")
    preferred = [p for p in pairs if core.flag(p.get("cross_account")) and core.flag(p.get("conventional_matched")) and not core.flag(p.get("bounded_tie_involves_saturation"))]
    preferred.sort(key=lambda row: (-float(row["max_standardized_dynamic_difference"]), row["simulation_a"], row["simulation_b"]))
    candidate = preferred[0] if preferred else None
    out = _new_output(output)
    artifact = {"status": "candidate_comparison_requires_mapping_qualification" if candidate else "no_qualified_diagnostic_contrast_available",
                "research_question": "Can interrogation of the fixed clinical DoC model reveal competing generating mechanisms and predictions that discriminate them in humans?",
                "selected_model_comparison": candidate, "selection_rule": "Largest recorded temporal separation among unsaturated cross-account candidates matched on clinical score and conventional spectral summaries; deterministic IDs resolve ties",
                "source_search_sha256": core.sha256_file(source/"search.json"), "source_candidates_sha256": core.sha256_file(source/"empirically_constrained_candidates.csv"),
                "qualification_before_human_commitment": ["Demonstrate sensor-level observation mapping adequacy and parameter identifiability for the selected accounts",
                    "Establish a human assay linking the predicted dynamical difference to independent experience and task performance",
                    "Specify an ethically feasible manipulation/input for which the strongest accounts make distinguishable predictions",
                    "Fix source timing, outcome definitions, missingness/eligibility, assignment probabilities and precision targets before independent outcomes"],
                "protected_replication": {"provisional_informative_participant_target": 100, "acquisition_sources_at_least": 2,
                    "eligibility": "Enough prospectively defined target/comparator episodes; participant denominator and all missing/excluded outcomes retained",
                    "sample_calibration": "Pilot yield, reliability and precision determine final eligible recruitment; 100 is provisional"},
                "human_diagnostic_experiment": {"provisional_informative_target": [100, 120], "sessions_per_participant": 2, "sites_at_least": 2,
                    "randomization": "Within-person randomized eligible active/comparator trials with stored assignment probabilities and counterbalanced sessions",
                    "paired_measurements": ["EEG", "fixed clinical model judgments and native representations", "timed original experience reports", "separate task performance", "relevant physiology"],
                    "controls": ["Matched spontaneous recovery", "input/control and expectation/reporting controls relevant to the selected mechanism", "practice, session/order and physiological artifacts"],
                    "diagnostic_estimand": "Difference between prospectively fixed competing-account predictions on neural response/recovery and independently measured human outcomes",
                    "manipulation": "Pending selected-account mapping qualification; no fixed sensory/memory/reporting/stimulation intervention is imposed beforehand"},
                "current_human_experiment_status": "design_candidate_not_recruited_or_executed",
                "no_result_direction_gate": True, "observed_search_status": search_record["status"]}
    core.write_json(out/"frozen_candidate_predictions.json", artifact)
    core.write_json(out/"freeze_receipt.json", {"artifact_sha256": core.sha256_file(out/"frozen_candidate_predictions.json"),
                    "meaning": "Immutable computational candidate record; does not certify prospective human preregistration, clinical validity or independent evaluation"})
    return _finish(out, [source/"receipt.json", source/"search.json", source/"empirically_constrained_candidates.csv"], "model_guided_diagnostic_artifact_completed", scientific_status=artifact["status"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("discover", "trajectories"):
        command = commands.add_parser(name)
        for key in ("pointer", "scores", "qualification", "output"):
            command.add_argument("--"+key, required=True, type=Path)
    command = commands.add_parser("fit")
    command.add_argument("--observations", required=True, action="append", type=Path)
    for key in ("bank", "bank-jobs", "bank-scores", "qualification", "bank-receipt", "output"):
        command.add_argument("--"+key, required=True, type=Path)
    for name, argument in (("search", "fit"), ("diagnostic", "search")):
        command = commands.add_parser(name)
        command.add_argument("--"+argument, required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
    args = vars(parser.parse_args(argv)); command = args.pop("command")
    if command == "search":
        args["fit_dir"] = args.pop("fit")
    elif command == "diagnostic":
        args["search_dir"] = args.pop("search")
    try:
        globals()[command](**args)
    except (StudyError, ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"model-guided stage failed: {exc}\n")


if __name__ == "__main__":
    main()
