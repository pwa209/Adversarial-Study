"""Participant-based contrasts and explicitly synthetic recruitment scenarios.

These reference analyses average windows within episodes, episodes within a
condition, and condition differences within a global participant. They do not
equate a clinical prediction with experience. Confidence intervals are
percentile participant bootstrap intervals, not exact finite-sample tests.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from statistics import NormalDist
from typing import Any

import numpy as np


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _flag(value: Any, field: str) -> bool:
    """Accept CSV booleans without treating the string 'false' as truthy."""
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "True", "TRUE"):
        return True
    if value in (0, "0", "false", "False", "FALSE"):
        return False
    raise ValueError(f"{field} must be an explicit boolean, got {value!r}")


def _participant(row: dict) -> str:
    value = row.get("participant_id")
    if value is None or not str(value).strip():
        raise ValueError("Every row requires a global participant_id")
    return str(value)


def _source(row: dict) -> str:
    return str(row.get("source_id") or row.get("acquisition_source") or "unspecified")


def _bootstrap(differences: np.ndarray, strata: list[str], reps: int,
               seed: int) -> np.ndarray | None:
    if differences.size < 2:
        return None
    rng = np.random.default_rng(seed)
    result = np.zeros(reps, dtype=float)
    # Keeping stratum sizes fixed estimates the observed source composition.
    for stratum in sorted(set(strata)):
        values = differences[np.asarray([s == stratum for s in strata])]
        samples = rng.integers(0, len(values), size=(reps, len(values)))
        result += values[samples].sum(axis=1)
    return result / len(differences)


def _interval(values: np.ndarray | None, confidence: float) -> list[float] | None:
    if values is None:
        return None
    alpha = (1.0 - confidence) / 2.0
    return [float(v) for v in np.quantile(values, [alpha, 1.0 - alpha])]


def paired_contrast(rows: list[dict], value_column: str, condition_column: str,
                    target_value: str, comparator_value: str,
                    bootstrap_reps: int = 2000, seed: int = 20261004,
                    min_episodes: int = 1,
                    equivalence_margin: float | None = None,
                    confidence: float = .95) -> dict:
    """Return an equal-participant paired mean difference and source bootstrap.

    A row is eligible unless eligible=false or qc_pass=false. No filtering uses
    the outcome sign. Missing episode_id means a distinct episode per row and
    is counted explicitly. Missing outcomes remain in the accounting; they
    are not imputed. Source-stratified resampling is conditional on observed
    source composition, and does not estimate generalisation to new sites.
    """
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be between .5 and 1")
    if not isinstance(bootstrap_reps, int) or bootstrap_reps < 2:
        raise ValueError("bootstrap_reps must be an integer >= 2")
    if not isinstance(min_episodes, int) or min_episodes < 1:
        raise ValueError("min_episodes must be a positive integer")
    if target_value == comparator_value:
        raise ValueError("Target and comparator conditions must differ")
    if equivalence_margin is not None and (
            not math.isfinite(equivalence_margin) or equivalence_margin <= 0):
        raise ValueError("equivalence_margin must be finite and positive")

    exclusions: Counter = Counter()
    all_people: set[str] = set()
    eligible_people: set[str] = set()
    sources: dict[str, set[str]] = defaultdict(set)
    episode_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    eligible_episode_keys: set[tuple[str, str, str]] = set()
    rows_by_condition: Counter = Counter()
    valid_rows_by_condition: Counter = Counter()
    no_episode_id = 0
    for index, row in enumerate(rows):
        person = _participant(row)
        all_people.add(person)
        if "eligible" in row and not _flag(row["eligible"], "eligible"):
            exclusions["ineligible_rows"] += 1
            continue
        if "qc_pass" in row and not _flag(row["qc_pass"], "qc_pass"):
            exclusions["quality_failed_rows"] += 1
            continue
        eligible_people.add(person)
        sources[person].add(_source(row))
        condition = str(row.get(condition_column, ""))
        if condition not in (str(target_value), str(comparator_value)):
            exclusions["outside_contrast_rows"] += 1
            continue
        rows_by_condition[condition] += 1
        episode = row.get("episode_id")
        if episode is None or not str(episode).strip():
            episode = f"__input_row_{index}"
            no_episode_id += 1
        key = person, str(episode), condition
        eligible_episode_keys.add(key)
        value = _finite(row.get(value_column))
        if value is None:
            exclusions["missing_or_nonfinite_outcome_rows"] += 1
            continue
        valid_rows_by_condition[condition] += 1
        episode_values[key].append(value)

    per_person: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (person, _episode, condition), values in episode_values.items():
        per_person[person][condition].append(float(np.mean(values)))
    participant_records = []
    noninformative = []
    for person in sorted(eligible_people):
        target = per_person[person].get(str(target_value), [])
        comparator = per_person[person].get(str(comparator_value), [])
        record = {
            "participant_id": person,
            "source_ids": sorted(sources[person]),
            "target_episodes": len(target), "comparator_episodes": len(comparator),
        }
        if len(target) < min_episodes or len(comparator) < min_episodes:
            record["reason"] = "insufficient_completed_target_or_comparator_episodes"
            noninformative.append(record)
            continue
        target_mean, comparator_mean = float(np.mean(target)), float(np.mean(comparator))
        participant_records.append({**record, "target_mean": target_mean,
                                    "comparator_mean": comparator_mean,
                                    "difference": target_mean - comparator_mean})

    differences = np.asarray([r["difference"] for r in participant_records], dtype=float)
    # A cross-source global participant remains ONE resampled unit.
    strata = [" + ".join(r["source_ids"]) for r in participant_records]
    # A singleton source stratum contributes no estimated within-source variance.
    singleton_strata = [s for s, n in Counter(strata).items() if n == 1]
    # Do not manufacture a precise interval by treating singleton strata as
    # known population constants. Their uncertainty cannot be estimated here.
    boot = None if singleton_strata else _bootstrap(differences, strata, bootstrap_reps, seed)
    interval = _interval(boot, confidence)
    result = {
        "analysis": "paired_participant_contrast",
        "value_column": value_column, "condition_column": condition_column,
        "target": str(target_value), "comparator": str(comparator_value),
        "estimand": "Equal-participant mean of target-minus-comparator episode means among participants with the required completed episodes; conditional on observed source composition.",
        "independent_unit": "global participant_id",
        "aggregation": "Within-episode row mean, within-condition episode mean, then equal-participant paired difference.",
        "estimate": float(np.mean(differences)) if len(differences) else None,
        "confidence": confidence, "confidence_interval": interval,
        "uncertainty": {
            "method": "source-stratified percentile participant bootstrap",
            "bootstrap_reps": bootstrap_reps, "seed": seed,
            "inference_status": ("insufficient_independent_participants" if len(differences) < 2
                                 else "insufficient_participants_within_source_strata" if singleton_strata
                                 else "available"),
            "singleton_source_strata": singleton_strata,
            "scope": "Observed source composition; site-superpopulation variation is not estimated.",
        },
        "counts": {
            "input_rows": len(rows), "input_participants": len(all_people),
            "eligible_participants": len(eligible_people),
            "informative_participants": len(participant_records),
            "noninformative_participants": len(noninformative),
            "eligible_contrast_rows": sum(rows_by_condition.values()),
            "valid_contrast_rows": sum(valid_rows_by_condition.values()),
            "eligible_contrast_episodes": len(eligible_episode_keys),
            "completed_contrast_episodes": len(episode_values),
            "rows_without_episode_id": no_episode_id,
            "rows_by_condition": dict(rows_by_condition),
            "valid_rows_by_condition": dict(valid_rows_by_condition),
            "informative_participants_by_source_stratum": dict(Counter(strata)),
        },
        "exclusions": dict(exclusions), "participant_contrasts": participant_records,
        "noninformative_participants": noninformative,
        "equivalence": None,
    }
    if equivalence_margin is not None:
        # At alpha=.05, equivalence is supported by a 90% interval contained
        # STRICTLY inside the margins. The main estimate retains its 95% CI.
        equivalence_confidence = 2.0 * confidence - 1.0
        eq_ci = _interval(boot, equivalence_confidence)
        decision = None if eq_ci is None else (
            eq_ci[0] > -equivalence_margin and eq_ci[1] < equivalence_margin)
        result["equivalence"] = {
            "margin": equivalence_margin, "confidence": equivalence_confidence,
            "interval": eq_ci, "supported": decision,
            "method": "approximate bootstrap interval equivalence assessment",
            "interpretation": "The interval criterion is dual to two one-sided tests when the interval and tests are valid; this percentile bootstrap procedure is approximate and is not an exact TOST or an exact finite-sample p-value.",
        }
    return result


def randomized_contrast(rows: list[dict], outcome_column: str,
                        treatment_column: str = "treatment",
                        active_value: str = "active", control_value: str = "control",
                        bootstrap_reps: int = 2000, seed: int = 20261004) -> dict:
    """Analyze all explicitly randomized, eligible trials, retaining missingness.

    The estimate is a participant-paired COMPLETED-outcome contrast; it is not
    a full intention-to-treat estimate. Eligibility must be determined before
    assignment/delivery, independently of the post-intervention outcome.
    trial_id (preferred) or episode_id identifies repeated rows of one trial.
    """
    trials: dict[tuple[str, str], dict] = {}
    input_people: set[str] = set()
    excluded: Counter = Counter()
    rows_without_trial_id = 0
    for index, row in enumerate(rows):
        person = _participant(row)
        input_people.add(person)
        if "randomized" not in row or "eligible" not in row:
            raise ValueError("Randomized analysis requires explicit randomized and eligible flags on every row")
        if not _flag(row["randomized"], "randomized"):
            excluded["not_randomized_rows"] += 1
            continue
        if not _flag(row["eligible"], "eligible"):
            excluded["not_eligible_rows"] += 1
            continue
        if row.get("eligibility_basis") in ("post_outcome", "post_intervention", "successful_outcome"):
            raise ValueError("Eligibility cannot be selected from post-intervention outcomes")
        trial_id = row.get("trial_id") or row.get("episode_id")
        if trial_id is None:
            trial_id = f"__input_row_{index}"
            rows_without_trial_id += 1
        key = person, str(trial_id)
        arm = str(row.get(treatment_column, ""))
        if arm not in (str(active_value), str(control_value)):
            raise ValueError(f"Randomized eligible trial has an unknown treatment: {arm!r}")
        if key in trials and trials[key]["arm"] != arm:
            raise ValueError(f"Conflicting treatment assignments within trial {key}")
        if key in trials and trials[key]["source"] != _source(row):
            raise ValueError(f"Conflicting acquisition sources within trial {key}")
        trial = trials.setdefault(key, {"person": person, "trial_id": str(trial_id),
                                       "arm": arm, "source": _source(row), "values": [], "input_rows": 0,
                                       "missing_rows": 0})
        trial["input_rows"] += 1
        value = _finite(row.get(outcome_column))
        if value is None:
            trial["missing_rows"] += 1
        else:
            trial["values"].append(value)

    collapsed = []
    by_arm: dict[str, Counter] = defaultdict(Counter)
    for trial in trials.values():
        count = by_arm[trial["arm"]]
        count["randomized_eligible_trials"] += 1
        count["input_rows"] += trial["input_rows"]
        count["missing_outcome_rows"] += trial["missing_rows"]
        count["completed_trials"] += bool(trial["values"])
        count["missing_outcome_trials"] += not bool(trial["values"])
        count["partially_observed_trials"] += bool(trial["values"]) and trial["missing_rows"] > 0
        collapsed.append({"participant_id": trial["person"], "episode_id": trial["trial_id"],
                          "source_id": trial["source"], "eligible": True,
                          treatment_column: trial["arm"],
                          outcome_column: float(np.mean(trial["values"])) if trial["values"] else None})
    result = paired_contrast(collapsed, outcome_column, treatment_column,
                             active_value, control_value, bootstrap_reps, seed)
    result["analysis"] = "randomized_completed_outcome_participant_contrast"
    result["estimand"] = "Equal-participant active-minus-control difference among participants with completed outcomes in both arms; this is a completed-outcome estimand, not a full intention-to-treat effect. Outcome-related missingness can bias the contrast."
    result["randomization_accounting"] = {
        "input_rows": len(rows), "input_participants": len(input_people),
        "randomized_eligible_trials": len(trials),
        "randomized_eligible_participants": len({t["person"] for t in trials.values()}),
        "rows_without_trial_id": rows_without_trial_id,
        "excluded_input_rows": dict(excluded), "by_arm": {k: dict(v) for k, v in by_arm.items()},
        "selection_rule": "Explicit pre-intervention eligible and randomized flags; no post-outcome selection.",
        "missingness_handling": "Missing outcomes counted; no imputation. A trial with at least one finite outcome row uses their mean; partially observed trials are counted. All randomized eligible trials appear in the accounting, and completed paired participants define the estimate.",
    }
    return result


def simulate_precision(settings: dict) -> dict:
    """Synthetic hierarchical scenarios, never empirical findings or achieved power.

    Participant and site intercepts, participant/site treatment deviations,
    stationary AR(1) episode noise, independent informative yield, and MCAR
    episode missingness are explicit assumptions. Intervals use a normal
    approximation to the participant variance conditional on the sampled
    sites. This economical design aid must be rerun with pilot estimates.
    """
    defaults = {"eligible_n_grid": [100, 200, 500], "replicates": 200,
                "seed": 20261004, "informative_yield": .35,
                "outcome_missingness": .1, "episodes_per_condition": 10,
                "min_episodes": 1, "effect": .25, "participant_sd": .5,
                "participant_effect_sd": .5, "site_sd": .2,
                "site_effect_sd": .1, "episode_sd": 1.,
                "autocorrelation": .5, "sites": 2, "confidence": .95}
    config = {**defaults, **settings}
    grid = config["eligible_n_grid"]
    if not isinstance(grid, list) or not grid or any(
            not isinstance(n, int) or n < 2 for n in grid):
        raise ValueError("eligible_n_grid must contain integers >= 2")
    for name in ("replicates", "episodes_per_condition", "min_episodes", "sites"):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["replicates"] < 2:
        raise ValueError("At least two simulation replicates are required")
    if config["min_episodes"] > config["episodes_per_condition"]:
        raise ValueError("min_episodes exceeds episodes_per_condition")
    for name in ("informative_yield", "outcome_missingness"):
        if not 0 <= config[name] <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if not -1 < config["autocorrelation"] < 1:
        raise ValueError("autocorrelation must be in (-1, 1)")
    if not .5 < config["confidence"] < 1:
        raise ValueError("confidence must be in (.5, 1)")
    for name in ("participant_sd", "participant_effect_sd", "site_sd", "site_effect_sd", "episode_sd"):
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"{name} must be a finite nonnegative standard deviation")
    if not math.isfinite(config["effect"]):
        raise ValueError("effect must be finite")
    rng = np.random.default_rng(config["seed"])
    z = NormalDist().inv_cdf((1 + config["confidence"]) / 2)
    scenarios = []
    for n in grid:
        counts, widths, estimates, detections, conditional_coverage, nominal_coverage = [], [], [], [], [], []
        for _ in range(config["replicates"]):
            site = np.arange(n) % config["sites"]
            rng.shuffle(site)
            informative = rng.random(n) < config["informative_yield"]
            site_intercept = rng.normal(0, config["site_sd"], config["sites"])
            site_effect = rng.normal(0, config["site_effect_sd"], config["sites"])
            intercept = rng.normal(0, config["participant_sd"], n) + site_intercept[site]
            true_effect = (config["effect"] + site_effect[site]
                           + rng.normal(0, config["participant_effect_sd"], n))
            episodes = config["episodes_per_condition"]
            noise = np.empty((n, 2, episodes))
            noise[:, :, 0] = rng.normal(0, config["episode_sd"], (n, 2))
            innovation_sd = config["episode_sd"] * math.sqrt(1 - config["autocorrelation"] ** 2)
            for e in range(1, episodes):
                noise[:, :, e] = (config["autocorrelation"] * noise[:, :, e - 1]
                                  + rng.normal(0, innovation_sd, (n, 2)))
            observations = noise + intercept[:, None, None]
            observations[:, 1, :] += true_effect[:, None]
            completed = rng.random((n, 2, episodes)) >= config["outcome_missingness"]
            episode_counts = completed.sum(axis=2)
            paired = informative & (episode_counts >= config["min_episodes"]).all(axis=1)
            means = (observations * completed).sum(axis=2) / np.maximum(episode_counts, 1)
            differences = (means[:, 1] - means[:, 0])[paired]
            paired_sites = site[paired]
            m = len(differences)
            counts.append(m)
            if m < 2:
                continue
            # At least two contributors are required in each contributing site
            # to estimate conditional within-site variance.
            site_sizes = Counter(paired_sites.tolist())
            if any(size < 2 for size in site_sizes.values()):
                continue
            variance = sum((size / m) ** 2 * float(np.var(differences[paired_sites == s], ddof=1)) / size
                           for s, size in site_sizes.items())
            estimate = float(np.mean(differences))
            halfwidth = z * math.sqrt(variance)
            low, high = estimate - halfwidth, estimate + halfwidth
            conditional_truth = config["effect"] + float(np.mean(site_effect[paired_sites]))
            widths.append(2 * halfwidth)
            estimates.append(estimate)
            detections.append((low > 0) if config["effect"] >= 0 else (high < 0))
            conditional_coverage.append(low <= conditional_truth <= high)
            nominal_coverage.append(low <= config["effect"] <= high)
        scenarios.append({
            "eligible_participants": n, "replicates": config["replicates"],
            "replicates_with_estimable_interval": len(widths),
            "mean_informative_participants": float(np.mean(counts)),
            "informative_participant_quantiles_05_50_95": [float(v) for v in np.quantile(counts, [.05, .5, .95])],
            "mean_estimate": float(np.mean(estimates)) if estimates else None,
            "median_confidence_interval_width": float(np.median(widths)) if widths else None,
            "synthetic_directional_detection_rate": float(np.mean(detections)) if detections else None,
            "synthetic_conditional_estimand_coverage": float(np.mean(conditional_coverage)) if conditional_coverage else None,
            "synthetic_overall_assumed_effect_coverage": float(np.mean(nominal_coverage)) if nominal_coverage else None,
        })
    return {"analysis": "synthetic_precision_scenarios", "evidence_status": "ASSUMPTIONS_ONLY_SYNTHETIC",
            "assumptions": config, "independent_unit": "participant, conditional on sampled fixed sites",
            "uncertainty_method": "Normal approximation from within-site participant contrast variance; no site-superpopulation uncertainty.",
            "missingness_assumption": "Informative yield independent of effects; independent MCAR episode outcomes. Outcome-dependent selection is not represented.",
            "interpretation": "Design scenarios under declared assumptions; neither empirical findings nor established power. Pilot estimates and the actual diagnostic contrast must replace these assumptions.",
            "scenarios": scenarios}
