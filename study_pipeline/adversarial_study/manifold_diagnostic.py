"""Freeze a verified manifold-mechanism comparison and analyse fresh randomization.

Development candidates are computational predictions. Human intervention and
assay qualification are separate requirements. Evaluation requires a registered
protocol, new people, source-bound outcomes and the full eligible randomized
denominator. Missing outcomes produce bounds, not complete-case causal claims.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from . import core
from .core import StudyError


def _utc(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise StudyError("Explicit ISO acquisition/registration times are required") from exc
    if parsed.tzinfo is None:
        raise StudyError("Times require explicit timezone offsets")
    return parsed.astimezone(timezone.utc)


def _pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": core.sha256_file(path)}


def _verify_pins(pins):
    if not isinstance(pins, list) or not pins:
        raise StudyError("Required provenance is missing")
    for item in pins:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise StudyError("Malformed artifact binding")
        if core.sha256_file(Path(item["path"])) != item["sha256"]:
            raise StudyError("Frozen source or outcome artifact changed")


def freeze(states, representations, bridge, config, output):
    from .neural_state_space import verify_state_space
    from .representation_bridge import verify_representation
    from .manifold_mechanisms import verify_bridge
    states, representations, bridge = map(lambda p: Path(p).resolve(), (states, representations, bridge))
    verify_state_space(states)
    verify_representation(representations)
    verify_bridge(bridge)
    rows = core.read_csv(states / "rows.csv")
    development_people = sorted({r["participant_id"] for r in rows})
    diagnostic = core.read_json(bridge / "diagnostic.json")
    endpoint = diagnostic.get("primary_neural_endpoint")
    human_ready = diagnostic.get("human_intervention_qualified") is True and bool(endpoint)
    artifact = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_diagnostic_requires_independent_human_data" if human_ready else "frozen_computational_candidates_require_human_mapping",
        "question": "Which generating mechanisms explain the geometry and dynamics of low-scoring healthy human episodes and predict distinguishing human responses?",
        "human_intervention_qualified": human_ready,
        "primary_neural_endpoint": endpoint if human_ready else None,
        "computational_diagnostic": diagnostic,
        "development_participant_ids": development_people,
        "inputs": [_pin(states / "receipt.json"), _pin(representations / "receipt.json"), _pin(bridge / "receipt.json"), _pin(config)],
        "parent_directories": {"states": str(states), "representations": str(representations), "bridge": str(bridge)},
        "protected_human_protocol": {
            "replication_target_informative_people": 100,
            "randomized_target_informative_people": [100, 120],
            "sessions_per_person": 2,
            "sites_at_least": 2,
            "targets_status": "Provisional; recruitment requires yield, reliability and precision planning",
            "randomization": "Record every eligible assignment, its probability and original outcome missingness; retain protocol deviations",
            "estimator": "Equal-participant mean of trial-level Horvitz-Thompson active-minus-control contributions",
            "uncertainty": "Participant bootstrap; predeclared bounded-outcome sensitivity for missing outcomes",
            "missing_outcomes": "No complete-case causal point estimate; report identification bounds over all eligible randomized trials",
            "paired_assays": ["EEG with frozen channel/measurement mapping", "adapted model predictions", "fixed clinical score", "original timed experience report", "named task response", "available genuine physiology"],
            "prospective_requirements": ["Locally/externally registered assay and physically feasible manipulation before first outcome", "Frozen account predictions, endpoint units and range", "New participant identities and raw/source-bound outcome evidence", "Qualified common assay for any neural–behavioural model comparison"],
        },
        "no_result_direction_or_significance_gate": True,
        "independent_humans_collected": False,
    }
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    core.write_json(out / "frozen_protocol.json", artifact)
    core.write_json(out / "receipt.json", {"status": "manifold_diagnostic_frozen", "inputs": artifact["inputs"],
        "source_code_sha256": core.sha256_file(__file__),
        "protocol_sha256": core.sha256_file(out / "frozen_protocol.json"), "human_intervention_qualified": human_ready})
    return artifact


def verify_freeze(directory):
    root = Path(directory).resolve()
    receipt = core.read_json(root / "receipt.json")
    if receipt.get("status") != "manifold_diagnostic_frozen" or receipt.get("protocol_sha256") != core.sha256_file(root / "frozen_protocol.json") or receipt.get("source_code_sha256") != core.sha256_file(__file__):
        raise StudyError("Diagnostic freeze is missing or changed")
    protocol = core.read_json(root / "frozen_protocol.json")
    _verify_pins(protocol["inputs"])
    if receipt.get("inputs") != protocol["inputs"]:
        raise StudyError("Freeze sources differ")
    from .neural_state_space import verify_state_space
    from .representation_bridge import verify_representation
    from .manifold_mechanisms import verify_bridge
    parents = protocol["parent_directories"]
    verify_state_space(parents["states"])
    verify_representation(parents["representations"])
    verify_bridge(parents["bridge"])
    original = core.read_json(Path(parents["bridge"]) / "diagnostic.json")
    if protocol["computational_diagnostic"] != original:
        raise StudyError("Frozen account prediction differs from verified bridge")
    human_ready = original.get("human_intervention_qualified") is True and bool(original.get("primary_neural_endpoint"))
    if protocol.get("human_intervention_qualified") != human_ready or receipt.get("human_intervention_qualified") != human_ready or protocol.get("primary_neural_endpoint") != (original.get("primary_neural_endpoint") if human_ready else None):
        raise StudyError("Computational candidate was relabelled as a qualified human manipulation")
    people = sorted({r["participant_id"] for r in core.read_csv(Path(parents["states"]) / "rows.csv")})
    if protocol["development_participant_ids"] != people:
        raise StudyError("Protected-human exclusion registry changed")
    _utc(protocol["created_utc"])
    return protocol


def randomized_estimate(rows, *, outcome_min, outcome_max, bootstrap_samples=2000, seed=20261004):
    """Design-based participant-balanced contrast, including missing-outcome bounds.

    Probabilities are the recorded design probabilities, never fitted propensities.
    Each participant contributes all eligible assignments. Windows/trials are not
    independent bootstrap units. A null or negative contrast is a valid result.
    """
    lo, hi = float(outcome_min), float(outcome_max)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise StudyError("A finite predeclared outcome range is required")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or not 100 <= bootstrap_samples <= 10000:
        raise StudyError("Participant bootstrap budget must be 100–10000")
    groups, missing, seen, census = defaultdict(list), 0, set(), defaultdict(int)
    for row in rows:
        person, trial = row.get("participant_id"), row.get("trial_id")
        if not person or not trial or (person, trial) in seen:
            raise StudyError("Randomized trial identities must be unique")
        seen.add((person, trial))
        if str(row.get("eligible", "")) not in ("0", "1", "False", "True", "false", "true"):
            raise StudyError("Eligibility must be an explicit original flag")
        census["all_rows"] += 1
        if not core.flag(row["eligible"]):
            census["ineligible_rows"] += 1
            continue
        arm = row.get("assigned_arm")
        if arm not in ("active", "control"):
            raise StudyError("Every eligible trial requires its original randomized arm")
        try:
            probability = float(row["assignment_probability_active"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StudyError("Original assignment probability is required") from exc
        if not np.isfinite(probability) or not 0 < probability < 1:
            raise StudyError("Assignment probabilities must be strictly between zero and one")
        value = row.get("observed_outcome", "")
        coefficient = 1 / probability if arm == "active" else -1 / (1 - probability)
        census["eligible_rows"] += 1
        census[arm + "_rows"] += 1
        if value in (None, "", "NA", "NaN", "nan"):
            if not row.get("missing_reason"):
                raise StudyError("Missing original outcome requires a disposition")
            missing += 1
            lower, upper = sorted((coefficient * lo, coefficient * hi))
            groups[person].append((None, lower, upper))
        else:
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise StudyError("Measured primary outcome must be numeric") from exc
            if not np.isfinite(value) or not lo <= value <= hi:
                raise StudyError("Observed outcome lies outside the registered range")
            contribution = coefficient * value
            groups[person].append((contribution, contribution, contribution))
    if len(groups) < 2:
        raise StudyError("At least two independent randomized people are required")
    estimates = []
    for person, values in sorted(groups.items()):
        estimates.append({"participant_id": person, "eligible_randomized_trials": len(values),
            "missing_outcomes": sum(v[0] is None for v in values),
            "point_estimate": None if any(v[0] is None for v in values) else float(np.mean([v[0] for v in values])),
            "lower_bound": float(np.mean([v[1] for v in values])), "upper_bound": float(np.mean([v[2] for v in values]))})
    lower = np.asarray([r["lower_bound"] for r in estimates])
    upper = np.asarray([r["upper_bound"] for r in estimates])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(estimates), size=(bootstrap_samples, len(estimates)))
    ci_lower = np.quantile(lower[indices].mean(axis=1), .025)
    ci_upper = np.quantile(upper[indices].mean(axis=1), .975)
    return {"status": "randomized_contrast_with_missing_outcome_bounds" if missing else "randomized_contrast_estimated",
        "point_estimate": None if missing else float(lower.mean()),
        "identification_interval": [float(lower.mean()), float(upper.mean())],
        "participant_bootstrap_interval_95": [float(ci_lower), float(ci_upper)],
        "independent_participants": len(groups), "missing_outcomes": missing, "denominators": dict(census),
        "participant_estimates": estimates, "estimator": "Equal-person mean of trial Horvitz-Thompson contributions; finite design-based arm probabilities",
        "outcome_bounds": [lo, hi], "bootstrap_seed": seed,
        "interpretation": "Randomized assay contrast under recorded design/source assumptions; not identification of a unique biological mechanism"}


def evaluate(frozen, observations, registration, data_receipt, output):
    protocol = verify_freeze(frozen)
    if protocol.get("human_intervention_qualified") is not True or not protocol.get("primary_neural_endpoint"):
        raise StudyError("A computational candidate cannot become a qualified human experiment")
    registered = core.read_json(registration)
    data = core.read_json(data_receipt)
    protocol_hash = core.sha256_file(Path(frozen) / "frozen_protocol.json")
    if registered.get("frozen_protocol_sha256") != protocol_hash or data.get("frozen_protocol_sha256") != protocol_hash:
        raise StudyError("Independent outcomes must bind the actual frozen protocol")
    if registered.get("primary_neural_endpoint") != protocol["primary_neural_endpoint"]:
        raise StudyError("Registered primary endpoint differs from frozen prediction")
    if data.get("observations_sha256") != core.sha256_file(observations) or data.get("registration_sha256") != core.sha256_file(registration):
        raise StudyError("Independent source/outcome correspondence changed")
    if data.get("human_data") is not True or data.get("synthetic") is not False:
        raise StudyError("Protected test requires actual new human outcomes")
    _verify_pins(data.get("raw_source_artifacts"))
    assignment_pin = data.get("assignment_registry")
    _verify_pins([assignment_pin])
    acquired = _utc(data["first_outcome_acquired_utc"])
    if acquired <= max(_utc(protocol["created_utc"]), _utc(registered["registered_utc"])):
        raise StudyError("Prediction and registration must precede independent outcomes")
    rows = core.read_csv(observations, ("participant_id", "trial_id", "eligible", "assigned_arm", "assignment_probability_active", "observed_outcome", "missing_reason"))
    assignment_fields = ("participant_id", "trial_id", "eligible", "assigned_arm", "assignment_probability_active")
    assigned = core.read_csv(assignment_pin["path"], assignment_fields)
    assignment_keys = [(r["participant_id"], r["trial_id"]) for r in assigned]
    observation_keys = [(r["participant_id"], r["trial_id"]) for r in rows]
    if len(set(assignment_keys)) != len(assigned) or len(set(observation_keys)) != len(rows) or set(assignment_keys) != set(observation_keys):
        raise StudyError("Outcomes must retain every original assignment, including missing and ineligible trials")
    original_assignment = dict(zip(assignment_keys, assigned))
    for row in rows:
        original = original_assignment[(row["participant_id"], row["trial_id"])]
        if any(str(row[key]) != str(original[key]) for key in assignment_fields):
            raise StudyError("Randomized arm, probability or eligibility differs from original assignment")
    people = {r["participant_id"] for r in rows}
    if people & set(protocol["development_participant_ids"]):
        raise StudyError("Previously analysed development people cannot supply independent confirmation")
    if set(data.get("participant_ids", [])) != people or data.get("complete_eligible_randomization_denominator") is not True:
        raise StudyError("Complete prospective randomized denominator/participant correspondence is required")
    answer = randomized_estimate(rows, outcome_min=registered["outcome_min"], outcome_max=registered["outcome_max"],
        bootstrap_samples=int(registered.get("bootstrap_samples", 2000)), seed=int(registered.get("seed", 20261004)))
    answer.update(primary_neural_endpoint=protocol["primary_neural_endpoint"],
        registration_status="Source-bound declared prospective registration; external registry/ethical approval is not certified by this software",
        observation_status="Source-bound measured assay table; neural measurement validity requires the registered frozen operator and acquisition QA",
        inputs=[_pin(Path(frozen) / "receipt.json"), _pin(observations), _pin(registration), _pin(data_receipt)])
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    core.write_json(out / "independent_randomized_evaluation.json", answer)
    core.write_csv(out / "participant_estimates.csv", answer["participant_estimates"])
    return answer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze")
    for name in ("states", "representations", "bridge", "config", "output"):
        frozen.add_argument("--" + name, required=True)
    test = commands.add_parser("evaluate")
    for name in ("frozen", "observations", "registration", "data-receipt", "output"):
        test.add_argument("--" + name, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    print(json.dumps((freeze if command == "freeze" else evaluate)(**args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
