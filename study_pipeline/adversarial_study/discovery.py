"""Descriptive full-score audit and within-person adversarial pair discovery.

Tolerance and capacity thresholds are exploratory operational definitions, not
clinical cutoffs. Matching conditions on a detector output derived from EEG;
selected feature differences do not establish natural states or mechanisms.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .core import StudyError, flag


def _finite(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def score_distribution(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    people = defaultdict(list)
    excluded = 0
    for row in rows:
        score = _finite(row.get("score"))
        if score is None or not 0 <= score <= 1:
            excluded += 1
            continue
        if not row.get("participant_id"):
            raise StudyError("Global participant_id is required")
        groups[(row.get("source_id", "unspecified"), row.get("report_category", "unspecified"))].append(row)
        people[row["participant_id"]].append(score)
    summary = []
    for (source, report), entries in sorted(groups.items(), key=lambda item: str(item[0])):
        scores = [float(r["score"]) for r in entries]
        participant_means = defaultdict(list)
        for row in entries:
            participant_means[row["participant_id"]].append(float(row["score"]))
        known_saturation = [_finite(r.get("saturated_fraction")) for r in entries]
        known_saturation = [v for v in known_saturation if v is not None and 0 <= v <= 1]
        preclip = [_finite(r.get("preclip_gcs")) for r in entries]
        summary.append({"source_id": source, "original_report_category": report, "episodes": len(entries),
            "participants": len(participant_means), "score_quantiles_episode_weighted": np.quantile(scores, [0, .25, .5, .75, 1]).tolist(),
            "mean_score_participant_weighted": float(np.mean([np.mean(v) for v in participant_means.values()])),
            "mean_saturated_channel_fraction": float(np.mean(known_saturation)) if known_saturation else None,
            "rows_with_known_saturation": len(known_saturation),
            "rows_with_unknown_saturation": len(entries) - len(known_saturation),
            "rows_with_preclip_readout": sum(v is not None for v in preclip)})
    return {"analysis": "unselected_score_distribution", "input_rows": len(rows), "excluded_invalid_scores": excluded,
            "participants": len(people), "groups": summary,
            "readout": "Native bounded remapping (GCS - 3) / 12; preclip availability is recorded separately, never inferred from a tied bounded score.",
            "inference": "Descriptive report associations; neither awareness calibration nor a clinical error rate."}


def unselected_relationships(rows: list[dict], settings: dict | None = None) -> dict:
    """Describe within-person relationships using every finite available score.

    Sources stay separate because capacity units and report definitions can
    differ. Correlations are exploratory summaries, with equal person weight,
    no significance claim and no assumption about their direction. Saturated
    observations remain present, with their status exposed.
    """
    settings = settings or {}
    columns = settings.get("relationship_columns", [settings.get("capacity_column", "capacity_value")])
    if not isinstance(columns, list) or not columns or any(not isinstance(c, str) or not c for c in columns):
        raise StudyError("relationship_columns must be a nonempty list of named measured columns")
    if len(set(columns)) != len(columns):
        raise StudyError("relationship_columns must be distinct")
    minimum = settings.get("minimum_relationship_episodes", 3)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 3:
        raise StudyError("minimum_relationship_episodes must be an integer >= 3")
    by_source, seen = defaultdict(list), set()
    for row in rows:
        if not row.get("participant_id"):
            raise StudyError("Global participant_id is required")
        key = row["participant_id"], row.get("episode_id")
        if not key[1] or key in seen:
            raise StudyError("Relationship rows require unique participant/episode keys")
        seen.add(key)
        by_source[row.get("source_id", "unspecified")].append(row)
    summaries = []
    for source, entries in sorted(by_source.items(), key=lambda item: str(item[0])):
        for column in columns:
            people, excluded = defaultdict(list), Counter()
            for row in entries:
                score, measured = _finite(row.get("score")), _finite(row.get(column))
                if score is None or not 0 <= score <= 1:
                    excluded["invalid_or_missing_score"] += 1
                elif measured is None:
                    excluded["missing_measured_value"] += 1
                else:
                    people[row["participant_id"]].append((score, measured, row))
            records = []
            for person, observations in sorted(people.items(), key=lambda item: str(item[0])):
                values = np.asarray([(s, v) for s, v, _ in observations])
                score_centered = values[:, 0] - values[:, 0].mean()
                outcome_centered = values[:, 1] - values[:, 1].mean()
                score_ss = float(score_centered @ score_centered)
                outcome_ss = float(outcome_centered @ outcome_centered)
                cross = float(score_centered @ outcome_centered)
                estimable = (len(values) >= minimum and score_ss > 0 and outcome_ss > 0
                             and not np.all(values[:, 0] == values[0, 0])
                             and not np.all(values[:, 1] == values[0, 1]))
                records.append({"participant_id": person, "episodes_with_both_values": len(values),
                    "within_person_correlation": float(np.clip(cross / math.sqrt(score_ss * outcome_ss), -1, 1)) if estimable else None,
                    "within_person_slope_per_score_unit": cross / score_ss if estimable else None,
                    "status": "descriptive_available" if estimable else "insufficient_episodes_or_no_variation",
                    "saturation_status_counts": dict(Counter(_saturation_status(r) for _, _, r in observations))})
            correlations = [r["within_person_correlation"] for r in records if r["within_person_correlation"] is not None]
            summaries.append({"source_id": source, "measured_column": column, "input_rows": len(entries),
                "input_participants": len({r["participant_id"] for r in entries}),
                "participants_with_paired_values": len(records), "participants_with_estimable_relationship": len(correlations),
                "mean_within_person_correlation_equal_participant_weight": float(np.mean(correlations)) if correlations else None,
                "exclusions": dict(excluded), "participant_relationships": records})
    return {"analysis": "exploratory_unselected_within_person_relationships", "settings": settings,
        "groups": summaries, "selection": "All finite observed bounded scores, including saturation; no score-tail or candidate-pair selection.",
        "inference": "Descriptive associations conditional on observed recordings and complete measured values. Repeated episodes are not independent people; no causal or awareness-calibration inference, significance test, or capacity equivalence is performed."}


def _saturation_status(row: dict) -> str:
    value = _finite(row.get("saturated_fraction"))
    if value is None or not 0 <= value <= 1:
        return "unknown"
    return "at_least_one_saturated_channel" if value > 0 else "no_saturated_channels"


def _timed_report(row: dict, report_column: str) -> dict | None:
    """Retain supplied report timing without certifying an experiential state."""
    start, stop = _finite(row.get("report_start_s")), _finite(row.get("report_stop_s"))
    window_start, window_stop = _finite(row.get("start_s")), _finite(row.get("stop_s"))
    category = row.get(report_column)
    if not category or start is None or stop is None or stop <= start or window_start is None or window_stop is None:
        return None
    overlap = max(0.0, min(stop, window_stop) - max(start, window_start))
    if overlap <= 0:
        return None
    return {"original_category": str(category), "start_s": start, "stop_s": stop,
            "overlap_with_score_window_s": overlap, "timing_basis": row.get("report_timing_basis") or "unspecified"}


def match_score_pairs(rows: list[dict], settings: dict) -> dict:
    tolerance = float(settings.get("same_score_tolerance", .05))
    capacity_delta = float(settings.get("capacity_difference_min", .2))
    separation = float(settings.get("min_time_separation_s", 10))
    capacity_column = settings.get("capacity_column", "capacity_value")
    exclude_saturated = settings.get("exclude_saturated", True)
    if any(not math.isfinite(v) or v < 0 for v in (tolerance, capacity_delta, separation)):
        raise StudyError("Discovery tolerances must be finite and nonnegative")
    if not isinstance(exclude_saturated, bool):
        raise StudyError("exclude_saturated must be boolean")
    groups, seen = defaultdict(list), set()
    rejected = 0
    for row in rows:
        key = (row.get("participant_id"), row.get("episode_id"))
        if not all(key) or key in seen:
            raise StudyError("Discovery rows require unique participant/episode keys")
        seen.add(key)
        if "synthetic" in row:
            flag(row["synthetic"])
        score, capacity = _finite(row.get("score")), _finite(row.get(capacity_column))
        start, stop = _finite(row.get("start_s")), _finite(row.get("stop_s"))
        saturation = _finite(row.get("saturated_fraction"))
        if not row.get("recording_id") or score is None or not 0 <= score <= 1 or capacity is None or start is None or stop is None or stop <= start:
            rejected += 1; continue
        if exclude_saturated and (saturation is None or saturation > 0):
            rejected += 1; continue
        # Absolute timing is comparable only within the SAME recording. Session
        # labels alone can include disconnected recordings with separate clocks.
        groups[(row["participant_id"], row.get("source_id"), row.get("recording_id"))].append((row, score, capacity, start, stop))
    pairs = []
    for group, entries in sorted(groups.items(), key=lambda item: str(item[0])):
        candidates = []
        for i, left in enumerate(entries):
            for j in range(i + 1, len(entries)):
                right = entries[j]
                score_delta = abs(left[1] - right[1])
                gap = max(left[3], right[3]) - min(left[4], right[4])
                if score_delta <= tolerance and abs(left[2] - right[2]) >= capacity_delta and gap >= separation:
                    candidates.append((score_delta, left[0]["episode_id"], right[0]["episode_id"], i, j))
        used = set()
        for difference, _, _, i, j in sorted(candidates):
            if i in used or j in used:
                continue
            used.update((i, j))
            left, right = entries[i], entries[j]
            pairs.append({"participant_id": group[0], "source_id": group[1], "recording_id": group[2],
                "episode_a": left[0]["episode_id"], "episode_b": right[0]["episode_id"],
                "score_a": left[1], "score_b": right[1], "score_difference_abs": difference,
                "capacity_a": left[2], "capacity_b": right[2], "capacity_difference_abs": abs(left[2] - right[2]),
                "report_a": left[0].get("report_category"), "report_b": right[0].get("report_category")})
    return {"analysis": "exploratory_within_recording_score_matching", "settings": settings,
        "input_participants": len({r["participant_id"] for r in rows}), "input_episodes": len(rows),
        "excluded_missing_or_saturated_rows": rejected, "participants_with_pairs": len({p["participant_id"] for p in pairs}),
        "pair_count": len(pairs), "pairs": pairs,
        "algorithm": "Greedy smallest-score-difference matching, each episode used once, within participant/source/recording.",
        "inference": "Candidate contrasts only. Pair count is not independent human N or population prevalence."}


def _complementary_pairs(rows: list[dict], settings: dict, kind: str) -> dict:
    names = {"capacity": ("same_capacity_tolerance", .05), "report": ("same_score_tolerance", .05)}
    tolerance_name, default = names[kind]
    tolerance = float(settings.get(tolerance_name, default))
    score_delta_min = float(settings.get("score_difference_min", .2))
    separation = float(settings.get("min_time_separation_s", 10))
    if any(not math.isfinite(v) or v < 0 for v in (tolerance, score_delta_min, separation)):
        raise StudyError("Discovery tolerances must be finite and nonnegative")
    exclude_saturated = settings.get("exclude_saturated", True)
    if not isinstance(exclude_saturated, bool):
        raise StudyError("exclude_saturated must be boolean")
    capacity_column = settings.get("capacity_column", "capacity_value")
    report_column = settings.get("report_column", "report_category")
    if not isinstance(capacity_column, str) or not capacity_column or not isinstance(report_column, str) or not report_column:
        raise StudyError("Discovery measured columns must be named")
    groups, seen, excluded = defaultdict(list), set(), Counter()
    for row in rows:
        key = row.get("participant_id"), row.get("episode_id")
        if not all(key) or key in seen:
            raise StudyError("Discovery rows require unique participant/episode keys")
        seen.add(key)
        if "synthetic" in row:
            flag(row["synthetic"])
        score, capacity = _finite(row.get("score")), _finite(row.get(capacity_column))
        start, stop = _finite(row.get("start_s")), _finite(row.get("stop_s"))
        if not row.get("recording_id"):
            excluded["missing_continuous_recording_id"] += 1; continue
        if score is None or not 0 <= score <= 1 or start is None or stop is None or stop <= start:
            excluded["missing_invalid_score_or_timing"] += 1; continue
        if (kind == "capacity" and capacity is None) or (kind == "report" and not row.get(report_column)):
            excluded["missing_required_measured_value"] += 1; continue
        if exclude_saturated and _saturation_status(row) != "no_saturated_channels":
            excluded["saturated_or_unknown_saturation"] += 1; continue
        groups[(row["participant_id"], row.get("source_id"), row["recording_id"])].append((row, score, capacity, start, stop))
    pairs = []
    for group, entries in sorted(groups.items(), key=lambda item: str(item[0])):
        candidates = []
        for i, left in enumerate(entries):
            for j in range(i + 1, len(entries)):
                right = entries[j]
                gap = max(left[3], right[3]) - min(left[4], right[4])
                score_gap = abs(left[1] - right[1])
                capacity_gap = abs(left[2] - right[2]) if left[2] is not None and right[2] is not None else None
                if gap < separation:
                    continue
                selected = (capacity_gap <= tolerance and score_gap >= score_delta_min) if kind == "capacity" else (
                    score_gap <= tolerance and str(left[0][report_column]) != str(right[0][report_column]))
                if selected:
                    priority = capacity_gap if kind == "capacity" else score_gap
                    candidates.append((priority, str(left[0]["episode_id"]), str(right[0]["episode_id"]), i, j))
        used = set()
        for _, _, _, i, j in sorted(candidates):
            if i in used or j in used:
                continue
            used.update((i, j))
            left, right = entries[i], entries[j]
            capacity_gap = abs(left[2] - right[2]) if left[2] is not None and right[2] is not None else None
            pairs.append({"participant_id": group[0], "source_id": group[1], "recording_id": group[2],
                "episode_a": left[0]["episode_id"], "episode_b": right[0]["episode_id"],
                "score_a": left[1], "score_b": right[1], "score_difference_abs": abs(left[1] - right[1]),
                "capacity_a": left[2], "capacity_b": right[2], "capacity_difference_abs": capacity_gap,
                "report_a": left[0].get(report_column), "report_b": right[0].get(report_column),
                "report_support_a": _timed_report(left[0], report_column), "report_support_b": _timed_report(right[0], report_column),
                "saturation_status_a": _saturation_status(left[0]), "saturation_status_b": _saturation_status(right[0]),
                "preclip_gcs_a": _finite(left[0].get("preclip_gcs")), "preclip_gcs_b": _finite(right[0].get("preclip_gcs"))})
    return {"analysis": "exploratory_within_recording_" + ("capacity_matching" if kind == "capacity" else "score_matched_report_categories"),
        "settings": settings, "input_participants": len({r["participant_id"] for r in rows}), "input_episodes": len(rows),
        "excluded_by_reason": dict(excluded), "pair_count": len(pairs),
        "participants_with_pairs": len({p["participant_id"] for p in pairs}), "pairs": pairs,
        "algorithm": "Greedy smallest-matched-quantity difference, each episode used once within this registry, same participant/source/continuous recording.",
        "readout": "Matches use the bounded native remapping; available preclip GCS values are descriptive metadata. Ties with saturated or unknown channels are excluded by default.",
        "inference": "Candidate contrasts only. A numerical capacity tolerance does not establish preserved capacity or equivalence; distinct report categories do not certify absence of experience. Counts are not prevalence or independent human N. Registries may reuse the same observations."}


def match_capacity_pairs(rows: list[dict], settings: dict) -> dict:
    """Find similar measured capacity with differing bounded clinical scores."""
    return _complementary_pairs(rows, settings, "capacity")


def match_report_pairs(rows: list[dict], settings: dict) -> dict:
    """Retain distinct original report categories at similar bounded scores."""
    return _complementary_pairs(rows, settings, "report")


def temporal_registry(rows: list[dict], settings: dict | None = None) -> dict:
    """Compare consecutive sampled windows, without inventing state transitions.

    Only within-recording nonoverlapping intervals are comparable. Experience
    and capacity changes require their own supplied, disjoint timed support.
    Intervening unobserved states, transition times and dwell durations are
    unknown. Saturation is retained and exposed, not treated as invariance.
    """
    settings = settings or {}
    report_column = settings.get("report_column", "report_category")
    capacity_column = settings.get("capacity_column", "capacity_value")
    maximum_gap = settings.get("maximum_temporal_gap_s")
    if maximum_gap is not None:
        maximum_gap = float(maximum_gap)
        if not math.isfinite(maximum_gap) or maximum_gap < 0:
            raise StudyError("maximum_temporal_gap_s must be finite and nonnegative or unset")
    groups, seen, excluded = defaultdict(list), set(), Counter()
    for row in rows:
        key = row.get("participant_id"), row.get("episode_id")
        if not all(key) or key in seen:
            raise StudyError("Temporal rows require unique participant/episode keys")
        seen.add(key)
        start, stop = _finite(row.get("start_s")), _finite(row.get("stop_s"))
        if not row.get("recording_id") or start is None or stop is None or stop <= start:
            excluded["missing_continuous_recording_or_valid_window_timing"] += 1; continue
        groups[(row["participant_id"], row.get("source_id"), row["recording_id"])].append((start, stop, row))
    comparisons = []
    for group, entries in sorted(groups.items(), key=lambda item: str(item[0])):
        entries.sort(key=lambda entry: (entry[0], entry[1], str(entry[2]["episode_id"])))
        for left, right in zip(entries, entries[1:]):
            gap = right[0] - left[1]
            if gap < 0:
                excluded["overlapping_adjacent_score_windows"] += 1; continue
            if maximum_gap is not None and gap > maximum_gap:
                excluded["gap_exceeds_exploratory_limit"] += 1; continue
            a, b = left[2], right[2]
            score_a, score_b = _finite(a.get("score")), _finite(b.get("score"))
            if score_a is None or score_b is None or not (0 <= score_a <= 1 and 0 <= score_b <= 1):
                excluded["adjacent_pair_missing_valid_score"] += 1; continue
            report_a, report_b = _timed_report(a, report_column), _timed_report(b, report_column)
            reports_disjoint = report_a is not None and report_b is not None and report_a["stop_s"] <= report_b["start_s"]
            capacities = []
            for row in (a, b):
                first, last = _finite(row.get("capacity_start_s")), _finite(row.get("capacity_stop_s"))
                value = _finite(row.get(capacity_column))
                valid = (value is not None and first is not None and last is not None and last > first
                         and min(last, float(row["stop_s"])) > max(first, float(row["start_s"])))
                capacities.append({"value": value, "start_s": first, "stop_s": last,
                    "timing_basis": row.get("capacity_timing_basis") or "unspecified"} if valid else None)
            capacity_a, capacity_b = capacities
            capacities_disjoint = capacity_a is not None and capacity_b is not None and capacity_a["stop_s"] <= capacity_b["start_s"]
            comparisons.append({"participant_id": group[0], "source_id": group[1], "recording_id": group[2],
                "episode_a": a["episode_id"], "episode_b": b["episode_id"],
                "score_window_a": [left[0], left[1]], "score_window_b": [right[0], right[1]],
                "unobserved_gap_between_score_windows_s": gap, "score_a": score_a, "score_b": score_b,
                "bounded_score_change": score_b - score_a,
                "preclip_gcs_change": (_finite(b.get("preclip_gcs")) - _finite(a.get("preclip_gcs")))
                    if _finite(a.get("preclip_gcs")) is not None and _finite(b.get("preclip_gcs")) is not None else None,
                "saturation_status_a": _saturation_status(a), "saturation_status_b": _saturation_status(b),
                "report_support_a": report_a, "report_support_b": report_b,
                "sampled_report_category_changed": report_a["original_category"] != report_b["original_category"] if reports_disjoint else None,
                "unobserved_gap_between_report_intervals_s": report_b["start_s"] - report_a["stop_s"] if reports_disjoint else None,
                "capacity_support_a": capacity_a, "capacity_support_b": capacity_b,
                "measured_capacity_change": capacity_b["value"] - capacity_a["value"] if capacities_disjoint else None})
    return {"analysis": "exploratory_sampled_temporal_comparisons", "settings": settings,
        "input_episodes": len(rows), "comparison_count": len(comparisons),
        "participants_with_comparisons": len({r["participant_id"] for r in comparisons}),
        "comparisons": comparisons, "excluded_by_reason": dict(excluded),
        "inference": "Consecutive sampled observations only; shared episodes and overlapping comparisons are dependent. A report-category change does not locate an actual state transition or establish state duration. Supplied timing may be analyst assigned. A bounded score plateau, especially with saturation, is not biological invariance."}


def discovery_registry(rows: list[dict], settings: dict | None = None) -> dict:
    """Assemble complementary exploratory views without choosing a finding."""
    settings = settings or {}
    return {"analysis": "open_exploratory_discovery_registry", "settings": settings,
        "score_distribution": score_distribution(rows), "unselected_relationships": unselected_relationships(rows, settings),
        "same_score_different_capacity": match_score_pairs(rows, settings),
        "same_capacity_different_score": match_capacity_pairs(rows, settings),
        "same_score_different_reports": match_report_pairs(rows, settings),
        "temporal_comparisons": temporal_registry(rows, settings),
        "inference": "All views remain exploratory, including agreement, disagreement and unavailable contrasts. No significant result, biological direction, mechanism or future manipulation is required. Multiple registries can reuse observations; fresh participant/cohort evaluation follows discovery."}


def phase_surrogate(data: np.ndarray, seed: int, shared_phases: bool = False) -> np.ndarray:
    """Preserve each channel's Fourier magnitude, alter phase organization.

    Shared phases also preserve cross spectra. Independent phases generally
    destroy cross-channel phase relationships. Neither preserves amplitude
    distribution or event timing, and neither is a human neural intervention.
    """
    array = np.asarray(data, dtype=float)
    if array.ndim != 2 or array.shape[1] < 4 or not np.isfinite(array).all():
        raise StudyError("Surrogate input requires finite channels x samples")
    spectrum = np.fft.rfft(array, axis=1)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, size=(1 if shared_phases else array.shape[0], spectrum.shape[1]))
    phases[:, 0] = 0
    if array.shape[1] % 2 == 0:
        phases[:, -1] = 0
    return np.fft.irfft(spectrum * np.exp(1j * phases), n=array.shape[1], axis=1)
