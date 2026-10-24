"""Additive, source-bound discovery audit; the frozen legacy outputs stay intact.

Observed-report discordance excludes absent or unsupported labels. Sleep
comparisons pair disconnected awakenings within people, sources and stages,
without comparing their local clocks. Low-score analyses preserve every scored
person and expose report timing, saturation, recurrence and missing comparisons.
They are development analyses, not diagnoses or independent human validation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

from . import core


STATUS = "proposal_discovery_audit_completed"
PAIR_FIELDS = ["pair_id", "family", "participant_id", "source_id", "sleep_stage", "pairing_scope",
    "episode_a", "episode_b", "recording_a", "recording_b", "score_a", "score_b", "score_difference_abs",
    "report_a", "report_b", "capacity_a", "capacity_b", "capacity_construct", "capacity_difference_abs",
    "saturation_status_a", "saturation_status_b", "participant_weight", "retained_pairs_for_person_family",
    "all_eligible_pairs_for_person_family", "pair_output_truncated"]
DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "proposal_discovery.json"
MISSING_REPORTS = {"", "missing_original_response", "missing_first_order_response", "unknown", "nan", "n/a"}


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_config(path=DEFAULT_CONFIG):
    cfg = core.read_json(path)
    if cfg.get("schema_version") != 1:
        raise core.StudyError("Proposal discovery configuration schema must be 1")
    thresholds = cfg.get("low_score_thresholds")
    if not isinstance(thresholds, list) or not thresholds or len(set(thresholds)) != len(thresholds):
        raise core.StudyError("Provide distinct prespecified low-score thresholds")
    for value in thresholds + [cfg.get("main_low_score_threshold")]:
        if isinstance(value, bool) or _finite(value) is None or not 0 < value < 1:
            raise core.StudyError("Low-score thresholds must be finite numbers strictly between 0 and 1")
    if cfg["main_low_score_threshold"] not in thresholds:
        raise core.StudyError("The main threshold must occur in the sensitivity grid")
    for name in ("same_score_tolerance", "same_capacity_tolerance", "minimum_capacity_difference",
                 "minimum_score_difference", "minimum_time_separation_s"):
        if isinstance(cfg.get(name), bool) or _finite(cfg.get(name)) is None or cfg[name] < 0:
            raise core.StudyError("Invalid nonnegative discovery tolerance: " + name)
    for name in ("seed", "minimum_informative_cases", "minimum_informative_comparators", "maximum_input_rows",
                 "maximum_scanned_pairs", "maximum_pairs_per_participant_family", "maximum_output_pairs", "bootstrap_samples"):
        if isinstance(cfg.get(name), bool) or not isinstance(cfg.get(name), int) or cfg[name] < (0 if name == "seed" else 1):
            raise core.StudyError("Invalid integer discovery limit: " + name)
    if cfg["bootstrap_samples"] < 100 or not isinstance(cfg.get("exclude_saturated_pairs"), bool):
        raise core.StudyError("Require at least 100 bootstrap samples and an explicit saturation policy")
    if not isinstance(cfg.get("sources"), dict) or not cfg["sources"]:
        raise core.StudyError("Source-specific original report policies are required")
    for source, policy in cfg["sources"].items():
        if policy.get("context") not in {"waking", "sleep"} or not policy.get("support_timing"):
            raise core.StudyError("Invalid source context/timing: " + source)
        observed, supported = policy.get("observed_categories"), policy.get("supported_experience_categories")
        if not isinstance(observed, list) or len(set(observed)) != len(observed) or any(v.lower() in MISSING_REPORTS for v in observed):
            raise core.StudyError("Observed categories must be distinct nonmissing source labels")
        if not isinstance(supported, list) or not set(supported) <= set(observed):
            raise core.StudyError("Experience support must be a declared subset of observed labels")
        if policy["context"] == "sleep" and not policy.get("sleep_stages"):
            raise core.StudyError("Sleep sources require original stage vocabulary")
    return cfg


def qualify_rows(rows, cfg):
    """Retain missing reports/scores in the census; never silently infer support."""
    if len(rows) > cfg["maximum_input_rows"]:
        raise core.StudyError("Input row budget exceeded; declare a new budget rather than selecting outcomes")
    output, seen = [], set()
    for original in rows:
        row = dict(original)
        identity = tuple(row.get(k) for k in ("source_id", "participant_id", "episode_id"))
        if not all(identity) or not row.get("recording_id") or identity in seen:
            raise core.StudyError("Discovery observations require distinct source/person/episode identities and recordings")
        seen.add(identity)
        if "synthetic" in row and core.flag(row["synthetic"]):
            raise core.StudyError("Synthetic rows cannot enter human discovery")
        if row.get("role", "development") != "development":
            raise core.StudyError("Only public development rows may enter this additive analysis")
        if row.get("score_origin", "native_frozen_v2") != "native_frozen_v2":
            raise core.StudyError("Original frozen native clinical scores are required")
        score = _finite(row.get("score"))
        row["score_valid"] = score is not None and 0 <= score <= 1
        policy = cfg["sources"].get(row["source_id"])
        category = str(row.get("report_category") or "")
        if category.lower() in MISSING_REPORTS:
            eligibility = "missing_original_report"
        elif policy is None:
            eligibility = "source_report_policy_unqualified"
        elif category not in policy["observed_categories"]:
            eligibility = "unsupported_or_combined_original_report"
        else:
            eligibility = "observed_original_report"
        row["report_eligibility"] = eligibility
        row["report_observed"] = eligibility == "observed_original_report"
        row["supported_experience"] = bool(row["report_observed"] and category in policy["supported_experience_categories"])
        row["report_support_timing"] = policy["support_timing"] if policy else "unqualified"
        # Detection follows the EEG window and cannot establish its experience.
        row["support_is_subsequent_detection"] = bool(policy and policy["support_timing"].startswith("subsequent_"))
        row["experience_during_score_window_certified"] = False
        row["source_context"] = policy["context"] if policy else "unqualified"
        stage = str(row.get("sleep_stage") or "") if row["source_context"] == "sleep" else "not_applicable"
        row["comparison_sleep_stage"] = stage
        row["sleep_stage_qualified"] = row["source_context"] != "sleep" or stage in policy["sleep_stages"]
        saturation = _finite(row.get("saturated_fraction"))
        row["saturation_status"] = "unknown" if saturation is None or not 0 <= saturation <= 1 else (
            "at_least_one_saturated_channel" if saturation > 0 else "no_saturated_channels")
        row["low_score_main"] = bool(row["score_valid"] and score <= cfg["main_low_score_threshold"])
        row["low_score_supported_report_main"] = bool(row["low_score_main"] and row["supported_experience"])
        output.append(row)
    return sorted(output, key=lambda r: (r["source_id"], r["participant_id"], r["recording_id"], r["episode_id"]))


def _stratum(row):
    return row["source_id"], row["comparison_sleep_stage"]


def _within_support(a, b, cfg):
    if a["source_id"] != b["source_id"] or a["participant_id"] != b["participant_id"]:
        return False
    if a["source_context"] == "sleep":
        return (a["sleep_stage_qualified"] and b["sleep_stage_qualified"] and
                a["comparison_sleep_stage"] == b["comparison_sleep_stage"] and a["recording_id"] != b["recording_id"])
    if a["source_context"] != "waking" or a["recording_id"] != b["recording_id"]:
        return False
    span_a = [_finite(a.get(k)) for k in ("start_s", "stop_s")]
    span_b = [_finite(b.get(k)) for k in ("start_s", "stop_s")]
    if None in span_a + span_b or span_a[1] <= span_a[0] or span_b[1] <= span_b[0]:
        return False
    gap = max(span_a[0], span_b[0]) - min(span_a[1], span_b[1])
    return gap >= cfg["minimum_time_separation_s"]


def _pair(a, b, family):
    identity = [a["source_id"], a["participant_id"], a["episode_id"], b["episode_id"], family]
    ca, cb = _finite(a.get("capacity_value")), _finite(b.get("capacity_value"))
    return {"pair_id": hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest(),
        "family": family, "participant_id": a["participant_id"], "source_id": a["source_id"],
        "sleep_stage": a["comparison_sleep_stage"], "pairing_scope": "within_person_source_stage_disconnected_awakening" if a["source_context"] == "sleep" else "within_person_source_continuous_recording_disjoint_windows",
        "episode_a": a["episode_id"], "episode_b": b["episode_id"], "recording_a": a["recording_id"], "recording_b": b["recording_id"],
        "score_a": float(a["score"]), "score_b": float(b["score"]), "score_difference_abs": abs(float(a["score"])-float(b["score"])),
        "report_a": a.get("report_category"), "report_b": b.get("report_category"),
        "capacity_a": ca, "capacity_b": cb, "capacity_construct": a.get("capacity_construct") or "",
        "capacity_difference_abs": abs(ca-cb) if ca is not None and cb is not None else None,
        "saturation_status_a": a["saturation_status"], "saturation_status_b": b["saturation_status"]}


def natural_pairs(rows, cfg):
    """Bound outputs with deterministic hash sampling per person/family.

    Enumerate the complete supported comparison frame and report its counts;
    missing reports cannot produce report-discordance candidates. Repeated
    pairs never increase the independent person denominator.
    """
    groups = defaultdict(list)
    for row in rows:
        if row["score_valid"]:
            groups[(row["source_id"], row["participant_id"], row["comparison_sleep_stage"],
                    "sleep" if row["source_context"] == "sleep" else row["recording_id"])].append(row)
    counts, heaps, dispositions = Counter(), defaultdict(list), Counter()
    scanned = 0
    for _, entries in sorted(groups.items()):
        for a, b in itertools.combinations(entries, 2):
            scanned += 1
            if scanned > cfg["maximum_scanned_pairs"]:
                raise core.StudyError("Complete pair enumeration exceeds declared budget; no result-dependent truncation")
            if not _within_support(a, b, cfg):
                dispositions["unsupported_clock_stage_or_overlap"] += 1
                continue
            if cfg["exclude_saturated_pairs"] and any(r["saturation_status"] != "no_saturated_channels" for r in (a, b)):
                dispositions["saturated_or_unknown_saturation"] += 1
                continue
            delta = abs(float(a["score"])-float(b["score"]))
            ca, cb = _finite(a.get("capacity_value")), _finite(b.get("capacity_value"))
            same_capacity = (ca is not None and cb is not None and bool(a.get("capacity_construct")) and
                a.get("capacity_construct") == b.get("capacity_construct") and bool(a.get("capacity_units")) and a.get("capacity_units") == b.get("capacity_units"))
            families = []
            if delta <= cfg["same_score_tolerance"]:
                if a["report_observed"] and b["report_observed"] and a["report_category"] != b["report_category"]:
                    families.append("same_score_different_observed_reports")
                if same_capacity and abs(ca-cb) >= cfg["minimum_capacity_difference"] and ca != cb:
                    families.append("same_score_different_measured_capacity")
            if delta >= cfg["minimum_score_difference"] and same_capacity and abs(ca-cb) <= cfg["same_capacity_tolerance"]:
                families.append("different_score_comparable_measured_capacity")
            for family in families:
                pair = _pair(a, b, family)
                key = pair["source_id"], pair["sleep_stage"], pair["participant_id"], family
                counts[key] += 1
                priority = int(hashlib.sha256((str(cfg["seed"])+pair["pair_id"]).encode()).hexdigest(), 16)
                heap = heaps[key]
                entry = (-priority, pair["pair_id"], pair)
                if len(heap) < cfg["maximum_pairs_per_participant_family"]:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)
    # Refuse a globally insufficient budget instead of privileging early people.
    if sum(len(v) for v in heaps.values()) > cfg["maximum_output_pairs"]:
        raise core.StudyError("Participant-balanced pair export exceeds declared total budget")
    people_per_family = Counter((s, stage, family) for s, stage, _, family in heaps)
    output = []
    for key, heap in sorted(heaps.items()):
        for _, _, pair in sorted(heap, key=lambda value: value[1]):
            pair.update(retained_pairs_for_person_family=len(heap), all_eligible_pairs_for_person_family=counts[key],
                pair_output_truncated=counts[key] > len(heap), participant_weight=1/(people_per_family[(key[0], key[1], key[3])]*len(heap)))
            output.append(pair)
    census = {"scanned_comparisons": scanned, "excluded_comparisons": dict(dispositions),
        "all_eligible_pairs": sum(counts.values()), "retained_pairs": len(output),
        "truncated": any(counts[k] > len(heaps[k]) for k in counts),
        "family_census": [{"source_id": s, "sleep_stage": stage, "family": family,
            "participants": people_per_family[(s, stage, family)],
            "eligible_pairs": sum(v for (ss, st, _, ff), v in counts.items() if (ss, st, ff) == (s, stage, family))}
            for s, stage, family in sorted(people_per_family)],
        "sampling": "Complete census, seeded identity-hash sampling within each source/stage/person/family; equal total person weight within each source/stage/family"}
    return output, census


def _interval(values, cfg, salt):
    if len(values) < 2:
        return None
    seed = int(hashlib.sha256((str(cfg["seed"])+salt).encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(cfg["bootstrap_samples"])]
    return np.quantile(means, [.025, .975]).tolist()


def sleep_contrasts(rows, cfg):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["source_context"] == "sleep" and row["sleep_stage_qualified"] and row["report_observed"] and row["score_valid"]:
            grouped[(row["source_id"], row["comparison_sleep_stage"], row["participant_id"])][row["report_category"]].append(row)
    people, summaries = [], defaultdict(list)
    for (source, stage, person), categories in sorted(grouped.items()):
        for left, right in itertools.combinations(sorted(categories), 2):
            a, b = categories[left], categories[right]
            # A source-case clock is never compared with another case's clock.
            if not any(x["recording_id"] != y["recording_id"] for x in a for y in b):
                continue
            delta = float(np.mean([float(r["score"]) for r in a])-np.mean([float(r["score"]) for r in b]))
            people.append({"source_id": source, "sleep_stage": stage, "participant_id": person,
                "report_a": left, "report_b": right, "episodes_a": len(a), "episodes_b": len(b),
                "mean_score_a": float(np.mean([float(r["score"]) for r in a])), "mean_score_b": float(np.mean([float(r["score"]) for r in b])),
                "score_difference_a_minus_b": delta, "saturation_counts_a": json.dumps(dict(Counter(r["saturation_status"] for r in a)), sort_keys=True),
                "saturation_counts_b": json.dumps(dict(Counter(r["saturation_status"] for r in b)), sort_keys=True)})
            summaries[(source, stage, left, right)].append(delta)
    return people, [{"source_id": s, "sleep_stage": stage, "report_a": a, "report_b": b,
        "independent_participants": len(values), "mean_within_person_score_difference": float(np.mean(values)),
        "participant_bootstrap_95_interval": _interval(values, cfg, repr((s, stage, a, b))),
        "inference": "Exploratory within-source/stage report association, equal person weight; saturation retained and reported; not population occurrence or awareness calibration"}
        for (s, stage, a, b), values in sorted(summaries.items())]


def _disjoint_count(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["recording_id"]].append(row)
    count = 0
    for entries in grouped.values():
        spans = sorted([(_finite(r.get("start_s")), _finite(r.get("stop_s"))) for r in entries if
            _finite(r.get("start_s")) is not None and _finite(r.get("stop_s")) is not None and float(r["stop_s"]) > float(r["start_s"])], key=lambda v: v[1])
        stop = -math.inf
        for first, last in spans:
            if first >= stop:
                count += 1
                stop = last
    return count


def low_score_audit(rows, cfg):
    cases, yield_rows, reliability, comparators = [], [], [], []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source_id"], row["comparison_sleep_stage"], row["participant_id"])].append(row)
    for (source, stage, person), entries in sorted(grouped.items()):
        for threshold in sorted(cfg["low_score_thresholds"]):
            supported = [r for r in entries if r["supported_experience"] and r["score_valid"] and r["sleep_stage_qualified"]]
            low = [r for r in supported if float(r["score"]) <= threshold]
            high = [r for r in supported if float(r["score"]) > threshold]
            matched_high, comparator_count, low_with_comparator = set(), 0, []
            for a in low:
                possibilities = [b for b in high if a["report_category"] == b["report_category"] and _within_support(a, b, cfg)]
                matched_high.update(b["episode_id"] for b in possibilities)
                if possibilities:
                    # One comparator per case, selected by identity hash only.
                    b = min(possibilities, key=lambda r: hashlib.sha256((str(cfg["seed"])+a["episode_id"]+r["episode_id"]).encode()).hexdigest())
                    comparator_count += 1
                    low_with_comparator.append(a)
                    comparators.append({"source_id": source, "sleep_stage": stage, "participant_id": person,
                        "threshold": threshold, "main_threshold": threshold == cfg["main_low_score_threshold"],
                        "episode_low": a["episode_id"], "episode_comparator": b["episode_id"],
                        "recording_low": a["recording_id"], "recording_comparator": b["recording_id"],
                        "score_low": float(a["score"]), "score_comparator": float(b["score"]),
                        "original_report_category": a["report_category"], "report_support_timing": a["report_support_timing"],
                        "interpretation": "Exploratory same-original-report, within-person low/higher-score contrast; not preserved-capacity equivalence"})
                cases.append({"source_id": source, "sleep_stage": stage, "participant_id": person,
                    "episode_id": a["episode_id"], "recording_id": a["recording_id"], "threshold": threshold,
                    "main_threshold": threshold == cfg["main_low_score_threshold"], "score": float(a["score"]),
                    "original_report_category": a["report_category"], "saturation_status": a["saturation_status"],
                    "case_class": "low_score_with_subsequent_detection" if a["support_is_subsequent_detection"] else "low_score_with_supported_retrospective_experience_report",
                    "report_support_timing": a["report_support_timing"], "experience_during_score_window_certified": False,
                    "available_same_report_higher_score_comparators": len(possibilities)})
            disjoint = _disjoint_count(low)
            disjoint_comparator_low = _disjoint_count(low_with_comparator)
            disjoint_high = _disjoint_count([r for r in high if r["episode_id"] in matched_high])
            yield_rows.append({"source_id": source, "sleep_stage": stage, "participant_id": person, "threshold": threshold,
                "main_threshold": threshold == cfg["main_low_score_threshold"], "all_scored_observation_rows": len(entries),
                "valid_score_rows": sum(r["score_valid"] for r in entries), "observed_report_rows": sum(r["report_observed"] for r in entries),
                "missing_or_unsupported_report_rows": sum(not r["report_observed"] for r in entries), "supported_report_score_rows": len(supported),
                "low_supported_report_rows": len(low), "low_disjoint_windows_within_recordings": disjoint,
                "low_unsaturated_rows": sum(r["saturation_status"] == "no_saturated_channels" for r in low),
                "same_report_comparator_episodes": len(matched_high), "low_cases_with_comparator": comparator_count,
                "low_disjoint_cases_with_comparator": disjoint_comparator_low,
                "same_report_disjoint_comparator_windows": disjoint_high,
                "informative_under_declared_episode_budget": disjoint_comparator_low >= cfg["minimum_informative_cases"] and disjoint_high >= cfg["minimum_informative_comparators"],
                "missing_comparison_reason": "" if low and matched_high else "no_low_case" if not low else "no_supported_same_report_within_person_comparator"})
            reliability.append({"source_id": source, "sleep_stage": stage, "participant_id": person, "threshold": threshold,
                "main_threshold": threshold == cfg["main_low_score_threshold"], "low_case_rows": len(low),
                "low_disjoint_windows_within_recordings": disjoint, "recordings_with_low_supported_reports": len({r["recording_id"] for r in low}),
                "recordings_with_supported_reports": len({r["recording_id"] for r in supported}),
                "sampled_recurrence_available": disjoint >= 2,
                "independent_repeat_session_reliability_status": "not_qualified_by_legacy_point_reports_or_source_case_session_ids",
                "interpretation": "Sampled recurrence and observation availability only; disconnected recordings do not supply continuous dwell/transition times; no repeat-visit reliability established"})
    summaries = []
    keys = sorted({(r["source_id"], r["sleep_stage"], r["threshold"]) for r in yield_rows})
    for source, stage, threshold in keys:
        entries = [r for r in yield_rows if (r["source_id"], r["sleep_stage"], r["threshold"]) == (source, stage, threshold)]
        summaries.append({"source_id": source, "sleep_stage": stage, "threshold": threshold,
            "main_threshold": threshold == cfg["main_low_score_threshold"], "all_scored_participants": len(entries),
            "participants_with_supported_report_scores": sum(r["supported_report_score_rows"] > 0 for r in entries),
            "participants_with_any_low_supported_report": sum(r["low_supported_report_rows"] > 0 for r in entries),
            "participants_with_same_report_comparator": sum(r["low_cases_with_comparator"] > 0 for r in entries),
            "informative_participants": sum(r["informative_under_declared_episode_budget"] for r in entries),
            "informative_yield_among_all_scored_people": sum(r["informative_under_declared_episode_budget"] for r in entries)/len(entries),
            "denominator_scope": "All scored development people in this source/stage, including missing reports; excludes source/EEG qualification losses before legacy scoring; not recruited-population prevalence"})
    return cases, yield_rows, reliability, comparators, summaries


def analyse(rows, cfg):
    rows = qualify_rows(rows, cfg)
    pairs, pair_census = natural_pairs(rows, cfg)
    sleep_people, sleep_summary = sleep_contrasts(rows, cfg)
    low, yields, reliability, comparators, low_summary = low_score_audit(rows, cfg)
    by_source = defaultdict(list)
    for row in rows:
        by_source[_stratum(row)].append(row)
    audit = {"status": STATUS, "independent_unit": "participant", "all_observation_rows": len(rows),
        "all_scored_participants": len({r["participant_id"] for r in rows}), "pair_census": pair_census,
        "source_stage_census": [{"source_id": s, "sleep_stage": stage, "rows": len(entries),
            "participants": len({r["participant_id"] for r in entries}), "report_eligibility": dict(Counter(r["report_eligibility"] for r in entries)),
            "invalid_or_missing_score_rows": sum(not r["score_valid"] for r in entries),
            "saturation_status": dict(Counter(r["saturation_status"] for r in entries)),
            "sampling_frame": cfg["sources"].get(s, {}).get("sampling_frame", "qualified_scored_development_subset")}
            for (s, stage), entries in sorted(by_source.items())],
        "sleep_comparisons": sleep_summary, "low_score_yield": low_summary,
        "scientific_status": "exploratory_development_analysis_not_independent_replication",
        "missingness": "Missing/unsupported reports and invalid scores remain in the observation and person census; observed-report contrasts require genuine eligible source labels.",
        "low_score_interpretation": cfg["interpretation"],
        "reliability": "Sampled recurrence reported; independent repeat-session qualification is missing. Point-linked retrospective reports do not certify experience over the complete ten-second EEG window.",
        "result_direction_gate": False}
    return {"observations": rows, "natural_pairs": pairs, "sleep_participant_contrasts": sleep_people,
            "low_score_cases": low, "low_score_participant_yield": yields, "low_score_reliability": reliability,
            "low_score_comparators": comparators, "audit": audit}


def read_legacy(directories, cfg):
    rows, provenance, seen = [], [], set()
    for directory in directories:
        root = Path(directory).resolve()
        if not root.is_dir():
            raise core.StudyError("Legacy discovery directory is unavailable")
        if root in seen:
            raise core.StudyError("Duplicate legacy discovery directory")
        seen.add(root)
        receipt_path, table = root/"receipt.json", root/"all_scored_observations.csv"
        receipt_hash = core.sha256_file(receipt_path)
        receipt = core.read_json(receipt_path)
        table_hash = core.sha256_file(table)
        if receipt.get("status") != "model_guided_discovery_completed" or receipt.get("outputs", {}).get(table.name) != table_hash:
            raise core.StudyError("Require original completed discovery receipt and exact hash-bound scored observations")
        if not receipt.get("inputs") or any(not isinstance(item.get("path"), str) or len(item.get("sha256", "")) != 64 for item in receipt["inputs"]):
            raise core.StudyError("Legacy discovery source provenance is incomplete")
        entries = core.read_csv(table, ("source_id", "participant_id", "episode_id", "recording_id", "score", "report_category", "role", "synthetic", "score_origin"))
        if core.sha256_file(table) != table_hash or core.sha256_file(receipt_path) != receipt_hash:
            raise core.StudyError("Legacy discovery changed during observation loading")
        rows.extend(entries)
        if len(rows) > cfg["maximum_input_rows"]:
            raise core.StudyError("Combined source input exceeds declared row limit")
        provenance.append({"path": str(table), "sha256": table_hash, "rows": len(entries),
            "legacy_receipt": {"path": str(receipt_path), "sha256": receipt_hash},
            "legacy_source_inputs": receipt["inputs"], "legacy_source_input_validation": "Inherited immutable discovery provenance; this additive stage directly rehashes its actual scored CSV and completed receipt, not raw EEG or earlier measurements"})
    if not provenance:
        raise core.StudyError("Provide at least one source-bound legacy discovery directory")
    return rows, provenance


def verify_outputs(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise core.StudyError("Additive discovery output directory is unavailable")
    receipt = core.read_json(root/"receipt.json")
    if receipt.get("schema_version") != 1 or receipt.get("status") != STATUS:
        raise core.StudyError("Additive discovery has not completed")
    required = {"observations.csv", "natural_pairs.csv", "sleep_participant_contrasts.csv", "low_score_cases.csv",
                "low_score_participant_yield.csv", "low_score_reliability.csv", "low_score_comparators.csv", "audit.json"}
    if set(receipt.get("outputs", {})) != required:
        raise core.StudyError("Additive discovery output inventory differs")
    for name, expected in receipt["outputs"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or core.sha256_file(path) != expected:
            raise core.StudyError("Additive discovery output changed: " + name)
    if not receipt.get("inputs") or not receipt.get("sources") or not receipt.get("supporting_code"):
        raise core.StudyError("Additive discovery provenance inventory is incomplete")
    for item in receipt["inputs"] + [receipt["code"], receipt["config"]] + receipt["supporting_code"]:
        if core.sha256_file(item["path"]) != item["sha256"]:
            raise core.StudyError("Additive discovery input/code/config changed")
    for item in receipt.get("sources", []):
        if core.sha256_file(item["legacy_receipt"]["path"]) != item["legacy_receipt"]["sha256"]:
            raise core.StudyError("Legacy discovery receipt changed")
    return receipt


def run(directories, config_path, output):
    config_hash = core.sha256_file(config_path)
    code_hash = core.sha256_file(__file__)
    core_hash = core.sha256_file(core.__file__)
    cfg = load_config(config_path)
    rows, sources = read_legacy(directories, cfg)
    result = analyse(rows, cfg)
    if core.sha256_file(config_path) != config_hash or core.sha256_file(__file__) != code_hash or core.sha256_file(core.__file__) != core_hash:
        raise core.StudyError("Additive discovery code/config changed during analysis")
    for item in sources:
        if core.sha256_file(item["path"]) != item["sha256"] or core.sha256_file(item["legacy_receipt"]["path"]) != item["legacy_receipt"]["sha256"]:
            raise core.StudyError("Legacy source artifacts changed during additive analysis")
    target = Path(output).resolve()
    if target.exists():
        raise core.StudyError("Preserve prior additive output; use a new directory")
    target.mkdir(parents=True)
    # Explicit minimal schemas preserve empty/null branches as completed outputs.
    empty_fields = {
        "observations": ["source_id", "participant_id", "episode_id", "recording_id", "score", "report_eligibility", "low_score_main"],
        "natural_pairs": PAIR_FIELDS,
        "sleep_participant_contrasts": ["source_id", "sleep_stage", "participant_id", "report_a", "report_b", "score_difference_a_minus_b"],
        "low_score_cases": ["source_id", "sleep_stage", "participant_id", "episode_id", "threshold", "main_threshold", "score", "case_class"],
        "low_score_participant_yield": ["source_id", "sleep_stage", "participant_id", "threshold", "main_threshold", "all_scored_observation_rows", "low_supported_report_rows", "informative_under_declared_episode_budget"],
        "low_score_reliability": ["source_id", "sleep_stage", "participant_id", "threshold", "main_threshold", "low_case_rows", "independent_repeat_session_reliability_status"],
        "low_score_comparators": ["source_id", "sleep_stage", "participant_id", "threshold", "main_threshold", "episode_low", "episode_comparator"]}
    for name, fields in empty_fields.items():
        core.write_csv(target/(name+".csv"), result[name], sorted({k for row in result[name] for k in row}) or fields)
    core.write_json(target/"audit.json", result["audit"])
    receipt = {"schema_version": 1, "status": STATUS, "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [{"path": item["path"], "sha256": item["sha256"]} for item in sources], "sources": sources,
        "code": {"path": str(Path(__file__).resolve()), "sha256": code_hash},
        "supporting_code": [{"path": str(Path(core.__file__).resolve()), "sha256": core_hash}],
        "config": {"path": str(Path(config_path).resolve()), "sha256": config_hash},
        "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "configuration": cfg, "outputs": {p.name: core.sha256_file(p) for p in sorted(target.iterdir()) if p.is_file()},
        "all_scored_participants": result["audit"]["all_scored_participants"],
        "result_direction_gate": False, "scientific_status": result["audit"]["scientific_status"]}
    core.write_json(target/"receipt.json", receipt)
    verify_outputs(target)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True, help="Legacy model-guided discovery directory; repeat for each source")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(args.source, args.config, args.output)
    print(json.dumps({"status": receipt["status"], "all_scored_participants": receipt["all_scored_participants"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
