"""Additive prospective qualification and independent human validation.

Frozen legacy computational candidates stay computational. This module can
qualify a *new* human protocol only from supplied source-bound calibration,
account execution and precision evidence. Hashes establish correspondence, not
the honesty of an external source or ethical authorization. A pilot, incomplete
sample or missing outcome remains visible and cannot become a completed test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import core
from .core import StudyError

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "proposal_validation.json"
PARTICIPANT_FIELDS = ("participant_id", "identity_fingerprint", "source_id", "site_id", "eligible", "eligibility_reason", "detector_exposure", "analyst_exposure")
COMMON_FIELDS = ("observation_id", "participant_id", "session_id", "source_id", "site_id", "condition", "sampling_mode", "eligible", "eligibility_reason", "pre_clinical_score", "pre_window_quality", "eligibility_window_start_utc", "eligibility_window_end_utc", "eligibility_decided_utc", "outcome_acquired_utc", "endpoint_value", "missing_reason")
EXPERIMENT_FIELDS = ("block_id", "stratum_id", "assignment_probability_active", "randomized_utc", "intervention_start_utc", "clinical_window_start_utc", "clinical_window_end_utc", "response_window_start_utc", "response_window_end_utc", "experience_report_utc", "clinical_score", "experience_report", "experience_missing_reason", "capacity_name", "capacity_value", "capacity_missing_reason", "physiology_name", "physiology_value", "physiology_missing_reason")
SESSION_FIELDS = ("participant_id", "session_id", "source_id", "site_id", "session_role", "acquisition_start_utc", "acquisition_end_utc", "qualified_baseline_observations", "raw_session_sha256")
OUTCOME_FIELDS = ("endpoint_value", "missing_reason", "outcome_acquired_utc", "clinical_score", "experience_report", "experience_missing_reason", "capacity_value", "capacity_missing_reason", "physiology_value", "physiology_missing_reason")


def _utc(value):
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        raise StudyError("Explicit timezone-bearing acquisition times required") from exc
    if out.tzinfo is None:
        raise StudyError("Naive acquisition/registration time is not admissible")
    return out.astimezone(timezone.utc)


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": core.sha256_file(path)}


def _check_pin(pin):
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
        raise StudyError("A source artifact requires exact path and SHA256")
    path = Path(pin["path"]).resolve()
    if not path.is_file() or core.sha256_file(path) != pin["sha256"]:
        raise StudyError("Source artifact absent or changed: " + str(path))
    return path


def _json(pin):
    return core.read_json(_check_pin(pin))


def _rows(pin, fields=()):
    return core.read_csv(_check_pin(pin), tuple(fields))


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StudyError(name + " must be an integer >= " + str(minimum))
    return value


def _flag(value):
    if value not in (True, False, "0", "1", "true", "false", "True", "False"):
        raise StudyError("Eligibility must be an explicit original binary flag")
    return core.flag(value)


def _unique(rows, key):
    values = [r[key] for r in rows]
    if not values or any(not v for v in values) or len(set(values)) != len(values):
        raise StudyError("Empty or duplicate " + key)
    return dict(zip(values, rows))


def _ids(pin):
    # Registry identity is globally deidentified, not an arbitrary dataset alias.
    return set(_unique(_rows(pin, ("participant_id", "identity_fingerprint")), "participant_id"))


def _config(path):
    value = core.read_json(path)
    if value.get("schema_version") != 1 or value.get("result_direction_gate") is not False:
        raise StudyError("Unsupported or result-gated validation configuration")
    _integer(value["bootstrap_samples"], "bootstrap_samples", 100)
    _integer(value["seed"], "seed", 0)
    if value["bootstrap_samples"] > 10000:
        raise StudyError("Bootstrap budget exceeds declared limit")
    return value


def contract(spec):
    """The registrable contract omits its own registration pin to avoid a cycle."""
    names = ("kind", "analysis_mode", "endpoint", "sampling", "informative", "precision_plan", "account_predictions", "qualification", "development_registry", "replication_registry", "computational_candidate", "configuration", "randomization_plan")
    return {name: spec.get(name) for name in names}


def _endpoint(spec):
    value = spec.get("endpoint", {})
    for name in ("name", "unit", "source_column", "operator_id", "construct", "report_timing_basis"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise StudyError("Specify an actual endpoint " + name)
    lo, hi = (core.number(value[k], k) for k in ("minimum", "maximum"))
    if lo >= hi:
        raise StudyError("Prospective finite outcome range required")
    for key in ("scale", "offset", "validation_absolute_tolerance"):
        core.number(value[key], key)
    if value["scale"] == 0 or value["validation_absolute_tolerance"] < 0:
        raise StudyError("Endpoint operator cannot discard all information")
    _check_pin(value["operator_source"])
    return value


def _precision(spec):
    plan = _json(spec["precision_plan"])
    if plan.get("status") != "finalized_before_outcomes" or plan.get("analysis_mode") != spec["analysis_mode"]:
        raise StudyError("A finalized prospective precision/yield plan is required")
    n = _integer(plan["informative_people"], "informative_people", 2)
    eligible = _integer(plan["eligible_recruitment_target"], "eligible_recruitment_target", n)
    width = core.number(plan["maximum_interval_width"], "maximum_interval_width")
    if width <= 0 or not plan.get("reliability_definition") or not plan.get("episode_schedule"):
        raise StudyError("Precision, reliability and episode schedule must be supplied")
    rows = _rows(plan["simulation_table"], ("scenario_id", "informative_people", "interval_width_95", "informative_yield", "missing_outcome_rate"))
    selected = [r for r in rows if int(r["informative_people"]) == n]
    if not selected or len({r["scenario_id"] for r in selected}) != len(selected):
        raise StudyError("Final N lacks unique supplied simulation scenarios")
    # All predeclared scenarios count, including null effects and adverse missingness.
    if not plan.get("required_scenarios") or set(plan["required_scenarios"]) != {r["scenario_id"] for r in selected}:
        raise StudyError("Precision selection silently omitted required scenarios")
    for row in selected:
        interval = core.number(row["interval_width_95"], "interval_width_95")
        yield_rate = core.number(row["informative_yield"], "informative_yield")
        missing = core.number(row["missing_outcome_rate"], "missing_outcome_rate")
        if interval < 0 or interval > width or not 0 < yield_rate <= 1 or not 0 <= missing <= 1:
            raise StudyError("Selected sample does not satisfy supplied precision/yield scenarios")
        if eligible < math.ceil(n / yield_rate):
            raise StudyError("Recruitment target fails the declared informative yield")
    for pin in plan.get("simulation_provenance", []):
        _check_pin(pin)
    if not plan.get("simulation_provenance"):
        raise StudyError("Precision simulation code/input provenance required")
    return plan


def _operator_qualification(spec):
    """Validate actual source-to-endpoint extraction; no success Boolean suffices."""
    q = _json(spec["qualification"]["endpoint_validation"])
    if q.get("assay_id") != spec["sampling"]["assay_id"] or q.get("unit") != spec["endpoint"]["unit"]:
        raise StudyError("Endpoint calibration uses a different human assay or unit")
    source = _rows(q["raw_calibration_table"], ("validation_id", "participant_id", spec["endpoint"]["source_column"]))
    original = _unique(source, "validation_id")
    measured = _rows(q["validation_table"], ("validation_id", "participant_id", "endpoint_value"))
    observed = _unique(measured, "validation_id")
    if set(original) != set(observed):
        raise StudyError("Endpoint calibration omitted raw measured validation cases")
    participants = set()
    for identity, row in observed.items():
        raw = original[identity]
        if row["participant_id"] != raw["participant_id"] or not row["participant_id"]:
            raise StudyError("Endpoint calibration participant correspondence differs")
        participants.add(row["participant_id"])
        expected = spec["endpoint"]["scale"] * core.number(raw[spec["endpoint"]["source_column"]], "source endpoint") + spec["endpoint"]["offset"]
        if abs(expected - core.number(row["endpoint_value"], "calibrated endpoint")) > spec["endpoint"]["validation_absolute_tolerance"]:
            raise StudyError("Actual measured endpoint does not follow the supplied operator")
    if len(participants) < _integer(q["minimum_validation_people"], "minimum_validation_people", 2):
        raise StudyError("Insufficient actual human operator calibration people")
    if not q.get("raw_source_artifacts"):
        raise StudyError("Actual human calibration source evidence missing")
    for pin in q["raw_source_artifacts"]:
        _check_pin(pin)
    return {"validation_cases": len(measured), "independent_validation_people": len(participants), "interpretation": "Numeric extraction/assay correspondence; not a unique physiological mechanism"}


def _prediction_qualification(spec):
    execution = _json(spec["qualification"]["account_execution"])
    if execution.get("assay_id") != spec["sampling"]["assay_id"] or execution.get("unit") != spec["endpoint"]["unit"]:
        raise StudyError("Account predictions require the identical qualified assay and unit")
    if execution.get("prediction_table_sha256") != spec["account_predictions"]["sha256"]:
        raise StudyError("Account predictions are not bound to actual execution evidence")
    prediction_fields = ("prediction_id", "account_id", "condition", "predicted_endpoint", "unit", "assay_id", "input_variable", "input_unit", "intended_input", "native_input", "native_input_variable", "native_input_unit")
    rows = _rows(spec["account_predictions"], prediction_fields)
    recorded_predictions = _unique(rows, "prediction_id")
    executed = _unique(_rows(execution["native_execution_table"], prediction_fields), "prediction_id")
    if set(recorded_predictions) != set(executed) or any(any(recorded_predictions[key][field] != executed[key][field] for field in prediction_fields) for key in executed):
        raise StudyError("Frozen predictions differ from the actual native account input/output execution rows")
    physical = _json(spec["qualification"]["physical_intervention"])
    deliveries = _rows(physical["prescribed_delivery_table"], ("condition", "intended", "unit"))
    prescribed_inputs = defaultdict(set)
    for delivery in deliveries:
        prescribed_inputs[delivery["condition"]].add(core.number(delivery["intended"], "prescribed physical input"))
    if any(len(v) != 1 for v in prescribed_inputs.values()):
        raise StudyError("Freeze a single specified physical input per diagnostic condition")
    grouped = defaultdict(dict)
    for row in rows:
        account, condition = row["account_id"], row["condition"]
        if not account or condition not in spec["sampling"]["conditions"] or condition in grouped[account]:
            raise StudyError("Account predictions omit or duplicate qualified conditions")
        if row["unit"] != spec["endpoint"]["unit"] or row["assay_id"] != spec["sampling"]["assay_id"]:
            raise StudyError("Predicted endpoint unit/assay differs from human assay")
        if row["input_variable"] != physical["physical_variable"] or row["input_unit"] != physical["unit"] or core.number(row["intended_input"], "predicted physical input") not in prescribed_inputs[row["condition"]]:
            raise StudyError("Account prediction input differs from the calibrated physical human intervention")
        grouped[account][condition] = core.number(row["predicted_endpoint"], "account prediction")
    if len(grouped) < 2 or any(set(value) != set(spec["sampling"]["conditions"]) for value in grouped.values()):
        raise StudyError("At least two genuine accounts need the full same diagnostic conditions")
    provenance = execution.get("account_artifacts", {})
    if set(provenance) != set(grouped):
        raise StudyError("Actual account lineage is missing")
    shared_mapping = None
    for account, evidence in provenance.items():
        for name in ("source_code", "adapted_parameters", "training_receipt", "human_assay_mapping", "physical_input_mapping"):
            _check_pin(evidence[name])
        training = _json(evidence["training_receipt"])
        if training.get("assay_id") != spec["sampling"]["assay_id"] or training.get("adapted_parameters_sha256") != evidence["adapted_parameters"]["sha256"]:
            raise StudyError("Account was not human adapted on the declared assay")
        actual_training = _rows(training["human_training_observations"], ("observation_id", "participant_id", "split", "observed"))
        _unique(actual_training, "observation_id")
        adapted_people = defaultdict(set)
        for row in actual_training:
            if row["split"] not in ("train", "validation") or not row["participant_id"] or not row["observed"]:
                raise StudyError("Actual adaptation observations require named training/validation people and original outcomes")
            adapted_people[row["split"]].add(row["participant_id"])
        if not adapted_people["train"] or not adapted_people["validation"] or adapted_people["train"] & adapted_people["validation"]:
            raise StudyError("Actual human adaptation partitions are absent or overlap")
        declared_split = training.get("split", {})
        if any(set(declared_split.get(role+"_participant_ids", [])) != adapted_people[role] for role in ("train", "validation")):
            raise StudyError("Human training receipt does not match its actual source-bound people")
        if training.get("adaptation_kind") == "neural_optimizer":
            if _integer(training.get("actual_optimizer_steps"), "actual_optimizer_steps") < 1 or training.get("original_weights_sha256") == training.get("adapted_weights_sha256") or training.get("adapted_weights_sha256") != evidence["adapted_parameters"]["sha256"]:
                raise StudyError("Neural account has no actual changed-weight human adaptation")
        elif training.get("adaptation_kind") == "human_parameter_fit":
            fit_rows = _rows(training["fit_observations"], ("participant_id", "split", "observed", "predicted"))
            if not fit_rows or not any(r["split"] == "train" for r in fit_rows):
                raise StudyError("Biological account has no genuine human parameter-fit observations")
            fit_people = defaultdict(set)
            for row in fit_rows:
                fit_people[row["split"]].add(row["participant_id"])
            if any(fit_people[role] != adapted_people[role] for role in ("train", "validation")) or fit_people.get("test"):
                raise StudyError("Biological parameter fitting used different people or consumed test outcomes")
        else:
            raise StudyError("Account adaptation kind is not qualified")
        mapping = _json(evidence["human_assay_mapping"])
        if mapping.get("assay_id") != spec["sampling"]["assay_id"] or mapping.get("unit") != spec["endpoint"]["unit"]:
            raise StudyError("Account-to-human mapping assay/units differ")
        validation = _rows(mapping["validation_table"], ("observation_id", "participant_id", "split", "observed", "predicted"))
        _unique(validation, "observation_id")
        partition_ids = defaultdict(set)
        errors = defaultdict(list)
        for row in validation:
            if row["split"] not in ("train", "validation", "test") or not row["participant_id"]:
                raise StudyError("Human assay map needs recorded train/validation/test identities")
            partition_ids[row["split"]].add(row["participant_id"])
            errors[row["participant_id"]].append((core.number(row["observed"], "mapping observation") - core.number(row["predicted"], "mapping prediction")) ** 2)
        if any(not partition_ids[k] for k in ("train", "validation", "test")) or any(partition_ids[a] & partition_ids[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise StudyError("Assay mapping people overlap or lack a validation partition")
        if any(partition_ids[role] != adapted_people[role] for role in ("train", "validation")) or partition_ids["test"] & (adapted_people["train"] | adapted_people["validation"]):
            raise StudyError("Assay mapping tests overlap actual human adaptation or use different frozen partitions")
        if set(declared_split.get("test_participant_ids", [])) != partition_ids["test"]:
            raise StudyError("Declared adapted-model test identities differ from mapping tests")
        if len(partition_ids["test"]) < _integer(mapping["minimum_test_people"], "minimum_test_people", 2):
            raise StudyError("Insufficient genuine human mapping evaluation people")
        mse = float(np.mean([np.mean(errors[p]) for p in sorted(partition_ids["test"])]))
        bound = core.number(mapping["predeclared_maximum_mse"], "predeclared_maximum_mse")
        if bound < 0 or mse > bound:
            raise StudyError("Human outcome map fails its predeclared numerical qualification")
        for name in ("operator_source", "raw_observation_table", "frozen_mapping"):
            _check_pin(mapping[name])
        raw = _rows(mapping["raw_observation_table"], ("observation_id", "participant_id", "split", "observed"))
        _unique(raw, "observation_id")
        # Exact row correspondence preserves all weak/null and missing evaluation cases.
        if [(r["observation_id"], r["participant_id"], r["split"], r["observed"]) for r in raw] != [(r["observation_id"], r["participant_id"], r["split"], r["observed"]) for r in validation]:
            raise StudyError("Assay-map evaluation changed actual human observations")
        mapping_identity = [(r["observation_id"], r["participant_id"], r["split"], r["observed"]) for r in raw]
        actual_training_identity = [(r["observation_id"], r["participant_id"], r["split"], r["observed"]) for r in actual_training]
        if [r for r in mapping_identity if r[2] in ("train", "validation")] != actual_training_identity:
            raise StudyError("Human assay map training/validation observations differ from actual model adaptation")
        if shared_mapping is not None and mapping_identity != shared_mapping:
            raise StudyError("Accounts were qualified on different people/outcomes or held-out splits")
        shared_mapping = mapping_identity
        if not set(r["participant_id"] for r in raw) <= _ids(spec["development_registry"]):
            raise StudyError("Human mapping/calibration people must be in the development exclusion registry")
        input_mapping = _input_mapping(evidence["physical_input_mapping"], physical, spec)
        for row in rows:
            if row["account_id"] != account:
                continue
            native = core.number(row["native_input"], "actual selected native input")
            if row["native_input_variable"] != input_mapping["model_variable"] or row["native_input_unit"] != input_mapping["model_unit"]:
                raise StudyError("Actual native prediction used a different input variable/unit from calibration")
            if not input_mapping["calibrated_native_minimum"] <= native <= input_mapping["calibrated_native_maximum"]:
                raise StudyError("Actual native input lies outside empirical physical mapping support")
            mapped = input_mapping["scale"]*native+input_mapping["offset"]
            if abs(mapped-core.number(row["intended_input"], "actual predicted human input")) > input_mapping["predeclared_absolute_mapping_tolerance"]:
                raise StudyError("Actual selected native prediction input does not map to the prescribed human intervention")
    precision = _precision(spec)
    contrasts = {a: value["active"]-value["control"] for a, value in grouped.items()}
    accounts = sorted(contrasts)
    separations = [{"account_A": a, "account_B": b, "predicted_active_minus_control_A": contrasts[a], "predicted_active_minus_control_B": contrasts[b], "absolute_predicted_contrast_separation": abs(contrasts[a]-contrasts[b]), "planned_maximum_interval_width": precision["maximum_interval_width"], "separated_at_planned_resolution": abs(contrasts[a]-contrasts[b]) > precision["maximum_interval_width"]} for i, a in enumerate(accounts) for b in accounts[i+1:]]
    diagnostic_ready = any(row["separated_at_planned_resolution"] for row in separations)
    return {"accounts": sorted(grouped), "predictions": dict(grouped), "pairwise_frozen_prediction_separations": separations, "mechanism_discrimination_design_qualified": diagnostic_ready,
            "observational_equivalence_status": "at_least_one_account_contrast_distinguishable_at_planned_resolution" if diagnostic_ready else "accounts_not_distinguishable_at_planned_precision",
            "observed_positive_result_required": False}


def _input_mapping(pin, physical, spec):
    mapping = _json(pin)
    if mapping.get("assay_id") != spec["sampling"]["assay_id"] or mapping.get("physical_variable") != physical["physical_variable"] or mapping.get("physical_unit") != physical["unit"] or not mapping.get("model_variable") or not mapping.get("model_unit"):
        raise StudyError("Native model input and physical human intervention require an explicit same-assay mapping")
    scale, offset = core.number(mapping["scale"], "physical input scale"), core.number(mapping["offset"], "physical input offset")
    tolerance = core.number(mapping["predeclared_absolute_mapping_tolerance"], "physical mapping tolerance")
    if scale == 0 or tolerance < 0:
        raise StudyError("Physical input mapping must preserve information with a finite calibration tolerance")
    paired = _rows(mapping["calibration_table"], ("calibration_id", "condition", "model_input", "measured_physical_input"))
    _unique(paired, "calibration_id")
    raw = _rows(mapping["raw_human_input_table"], ("calibration_id", "condition", "measured_physical_input"))
    if [(r["calibration_id"], r["condition"], r["measured_physical_input"]) for r in paired] != [(r["calibration_id"], r["condition"], r["measured_physical_input"]) for r in raw]:
        raise StudyError("Physical input mapping changed or omitted actual human input measurements")
    if {r["condition"] for r in paired} != set(spec["sampling"]["conditions"]):
        raise StudyError("Native-to-human physical input calibration lacks diagnostic condition coverage")
    for row in paired:
        predicted = scale*core.number(row["model_input"], "native model input")+offset
        if abs(predicted-core.number(row["measured_physical_input"], "genuine measured physical input")) > tolerance:
            raise StudyError("Native-to-human physical input calibration exceeds the predeclared tolerance")
    for name in ("mapping_operator_source", "native_input_execution", "raw_physical_source"):
        _check_pin(mapping[name])
    native = [core.number(r["model_input"], "calibrated native input") for r in paired]
    return mapping | {"calibrated_native_minimum": min(native), "calibrated_native_maximum": max(native)}


def _physical_qualification(spec):
    q = _json(spec["qualification"]["physical_intervention"])
    if q.get("assay_id") != spec["sampling"]["assay_id"] or not q.get("physical_variable") or not q.get("unit"):
        raise StudyError("Actual physical intervention variable and assay required")
    tolerance = core.number(q["predeclared_absolute_delivery_tolerance"], "delivery tolerance")
    if tolerance < 0:
        raise StudyError("Delivery tolerance cannot be negative")
    original = _unique(_rows(q["raw_delivery_table"], ("calibration_id", "condition", "delivered", "unit", "delivered_utc")), "calibration_id")
    prescribed = _unique(_rows(q["prescribed_delivery_table"], ("calibration_id", "condition", "intended", "unit")), "calibration_id")
    if set(original) != set(prescribed) or {r["condition"] for r in prescribed.values()} != {"active", "control"}:
        raise StudyError("Physical calibration must include all prescribed active and control deliveries")
    for identity, row in original.items():
        target = prescribed[identity]
        _utc(row["delivered_utc"])
        if row["condition"] != target["condition"] or row["unit"] != target["unit"] or row["unit"] != q["unit"]:
            raise StudyError("Actual physical delivery differs from intended condition or units")
        if abs(core.number(row["delivered"], "delivered") - core.number(target["intended"], "intended")) > tolerance:
            raise StudyError("Actual physical delivery exceeds its predeclared tolerance")
    for name in ("delivery_operator_source", "approved_acquisition_protocol", "ethics_authorization", "raw_delivery_source"):
        _check_pin(q[name])
    return {"physical_variable": q["physical_variable"], "unit": q["unit"], "actual_calibration_deliveries": len(original), "scope": "Actual delivered input calibration; source approval documents are retained, not legally certified by software"}


def _randomization_plan(spec):
    plan = _json(spec["randomization_plan"])
    if plan.get("method") != "fixed_block_randomization" or not plan.get("allocation_generator"):
        raise StudyError("Supply a real prospectively fixed block randomization operator")
    _check_pin(plan["allocation_generator"])
    strata = plan.get("strata")
    if not isinstance(strata, list) or not strata:
        raise StudyError("Prospectively specified randomization strata required")
    names = set()
    for row in strata:
        if not row.get("stratum_id") or row["stratum_id"] in names:
            raise StudyError("Unique named randomization strata required")
        names.add(row["stratum_id"])
        size = _integer(row["block_size"], "block_size", 2)
        active = _integer(row["active_per_block"], "active_per_block")
        if active >= size or core.number(row["probability_active"], "probability_active") != active/size:
            raise StudyError("Randomization probability differs from fixed block composition")
    return plan


def _eligibility_rules(spec):
    sampling = spec["sampling"]
    rule = sampling.get("eligibility_rule")
    if not isinstance(rule, dict) or not rule.get("allowed_pre_window_quality") or not isinstance(rule.get("window_seconds"), (float, int)) or rule["window_seconds"] <= 0:
        raise StudyError("A numeric pre-window quality/score eligibility operator is required")
    cutoff = rule.get("clinical_score_max_for_enrichment")
    if cutoff is not None:
        core.number(cutoff, "pre-score cutoff")
    if spec["kind"] == "replication":
        rules = sampling.get("condition_rules")
        if not isinstance(rules, dict) or set(rules) != {"target", "comparator"}:
            raise StudyError("Freeze explicit source-field target/comparator rules")
        for clauses in rules.values():
            if not isinstance(clauses, list) or not clauses:
                raise StudyError("Each replication condition needs a source-bound rule")
            for clause in clauses:
                if not clause.get("field") or clause.get("operator") not in ("equal", "at_most", "at_least") or "value" not in clause:
                    raise StudyError("Unsupported replication condition operator")
    return rule


def _matches(raw, clauses):
    for clause in clauses:
        value = raw.get(clause["field"])
        if value is None:
            raise StudyError("Original raw condition field missing: " + clause["field"])
        if clause["operator"] == "equal":
            passes = str(value) == str(clause["value"])
        else:
            observed, threshold = core.number(value, "condition field"), core.number(clause["value"], "condition cutoff")
            passes = observed <= threshold if clause["operator"] == "at_most" else observed >= threshold
        if not passes:
            return False
    return True


def freeze(spec, output):
    path = Path(spec).resolve()
    value = core.read_json(path)
    if value.get("schema_version") != 1 or value.get("kind") not in ("replication", "experiment") or value.get("analysis_mode") not in ("pilot", "confirmatory"):
        raise StudyError("An explicit prospective replication/experiment contract is required")
    config = _config(_check_pin(value["configuration"]))
    endpoint = _endpoint(value)
    sampling = value["sampling"]
    conditions = ["target", "comparator"] if value["kind"] == "replication" else ["active", "control"]
    if sampling.get("conditions") != conditions or not sampling.get("assay_id"):
        raise StudyError("Specify the exact diagnostic conditions and assay")
    if not sampling.get("sampling_rule") or not sampling.get("report_schedule"):
        raise StudyError("Prospective eligibility, sampling and timed reporting rules required")
    _eligibility_rules(value)
    minimum_sources = _integer(sampling["minimum_sources"], "minimum_sources")
    minimum_sites = _integer(sampling["minimum_sites"], "minimum_sites")
    sessions = _integer(sampling["minimum_sessions_per_person"], "minimum_sessions_per_person")
    if value["kind"] == "replication" and minimum_sources < max(2, config["minimum_replication_sources"]):
        raise StudyError("Independent replication requires at least two acquisition sources")
    if value["kind"] == "experiment" and (minimum_sites < max(2, config["minimum_experiment_sites"]) or sessions < max(2, config["minimum_experiment_sessions_per_person"])):
        raise StudyError("Experiment requires two sites and two sessions per informative person")
    for name in ("minimum_target_or_active_observations", "minimum_comparator_or_control_observations"):
        _integer(value["informative"][name], name)
    if not isinstance(value["informative"]["maximum_missing_fraction"], (int, float)) or not 0 <= value["informative"]["maximum_missing_fraction"] <= 1:
        raise StudyError("Predeclare informative missingness without using result direction")
    _ids(value["development_registry"])
    if value["kind"] == "experiment":
        _ids(value["replication_registry"])
    _check_pin(value["computational_candidate"])
    plan = _precision(value)
    qualifications = {"endpoint": _operator_qualification(value)}
    endpoint_qa = _json(value["qualification"]["endpoint_validation"])
    if not {r["participant_id"] for r in _rows(endpoint_qa["validation_table"])} <= _ids(value["development_registry"]):
        raise StudyError("Endpoint calibration people must be excluded from independent evaluation")
    if value["kind"] == "experiment":
        qualifications["physical"] = _physical_qualification(value)
        qualifications["accounts"] = _prediction_qualification(value)
        qualifications["randomization"] = _randomization_plan(value)
        _integer(sampling["minimum_baseline_observations"], "minimum_baseline_observations")
        baseline_rule = sampling.get("baseline_analysis", {})
        core.number(baseline_rule["low_clinical_score_max"], "baseline low-score cutoff")
        if not baseline_rule.get("reported_experience_categories") or not isinstance(baseline_rule["reported_experience_categories"], list):
            raise StudyError("Freeze source-specific original experience categories for baseline co-report summaries")
        _integer(baseline_rule["minimum_repeat_pairs_per_person"], "minimum_repeat_pairs_per_person")
    # Replication can freeze a neural relationship without claiming a human mechanism.
    registration = _json(value["registration"])
    if registration.get("analysis_contract_sha256") != _canonical(contract(value)):
        raise StudyError("Registration does not bind this exact prospective analysis contract")
    if not registration.get("registry_identifier"):
        raise StudyError("Source-bound registration identifier required")
    registered = _utc(registration["registered_utc"])
    now = datetime.now(timezone.utc)
    if registered > now:
        raise StudyError("Registration timestamp is in the future")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = {"schema_version": 1, "status": "independent_replication_contract_frozen" if value["kind"] == "replication" else "human_protocol_frozen_from_supplied_qualification", "kind": value["kind"], "analysis_mode": value["analysis_mode"], "created_utc": now.isoformat(),
              "specification": value, "configuration": config, "precision_plan": plan, "qualification": qualifications,
              "analysis_contract_sha256": _canonical(contract(value)), "independent_data_opened": False,
              "human_intervention_qualified": value["kind"] == "experiment", "no_result_direction_gate": True,
              "qualification_scope": "Source-bound technical delivery, endpoint operator and held-person assay mapping; external truth, ethics and a unique physiological mechanism are not certified"}
    core.write_json(out / "frozen_protocol.json", frozen)
    core.write_json(out / "receipt.json", {"status": "supplied_human_protocol_frozen", "kind": value["kind"], "source_code_sha256": core.sha256_file(__file__), "specification": _pin(path), "frozen_protocol": _pin(out / "frozen_protocol.json")})
    return frozen


def verify_freeze(directory):
    root = Path(directory).resolve()
    receipt = core.read_json(root / "receipt.json")
    if receipt.get("status") != "supplied_human_protocol_frozen" or receipt.get("source_code_sha256") != core.sha256_file(__file__):
        raise StudyError("Prospective validation freeze/code provenance missing")
    if _check_pin(receipt["frozen_protocol"]) != root / "frozen_protocol.json":
        raise StudyError("Frozen protocol belongs to a different artifact directory")
    value = core.read_json(root / "frozen_protocol.json")
    supplied = _json(receipt["specification"])
    if value["specification"] != supplied or value["analysis_contract_sha256"] != _canonical(contract(supplied)):
        raise StudyError("Frozen human scientific contract changed")
    if _config(_check_pin(supplied["configuration"])) != value["configuration"] or _precision(supplied) != value["precision_plan"]:
        raise StudyError("Frozen configuration/precision plan changed")
    _endpoint(supplied)
    _eligibility_rules(supplied)
    observed = {"endpoint": _operator_qualification(supplied)}
    if value["kind"] == "experiment":
        observed["physical"] = _physical_qualification(supplied)
        observed["accounts"] = _prediction_qualification(supplied)
        observed["randomization"] = _randomization_plan(supplied)
    if observed != value["qualification"]:
        raise StudyError("Frozen numeric human qualification evidence changed")
    if _json(supplied["registration"]).get("analysis_contract_sha256") != value["analysis_contract_sha256"]:
        raise StudyError("Registered prospective contract differs")
    _check_pin(supplied["computational_candidate"])
    return value


def _data(frozen, receipt):
    data = core.read_json(receipt)
    if data.get("frozen_protocol_sha256") != frozen["protocol_sha256"]:
        raise StudyError("Acquisition does not bind the frozen protocol")
    if data.get("human_data") is not True or data.get("synthetic") is not False:
        raise StudyError("Actual human acquisition evidence is required")
    for pin in data.get("raw_source_artifacts", []):
        _check_pin(pin)
    if not data.get("raw_source_artifacts"):
        raise StudyError("Actual acquisition source artifacts missing")
    participants = _unique(_rows(data["participant_registry"], PARTICIPANT_FIELDS), "participant_id")
    fingerprints = [r["identity_fingerprint"] for r in participants.values()]
    if any(not p for p in fingerprints) or len(set(fingerprints)) != len(fingerprints):
        raise StudyError("One genuine person may not reappear under multiple IDs")
    specification = frozen["specification"]
    prior = _rows(specification["development_registry"], ("participant_id", "identity_fingerprint"))
    if frozen["kind"] == "experiment":
        prior += _rows(specification["replication_registry"], ("participant_id", "identity_fingerprint"))
    if set(participants) & {r["participant_id"] for r in prior} or set(fingerprints) & {r["identity_fingerprint"] for r in prior}:
        raise StudyError("New human participants overlap development or independent replication")
    for person in participants.values():
        if person["detector_exposure"] != "no_known_exposure" or person["analyst_exposure"] != "unopened_before_freeze":
            raise StudyError("Unknown/affirmative pretrained or analyst exposure cannot certify independence")
        if not person["source_id"] or not person["site_id"] or not person["eligibility_reason"]:
            raise StudyError("Participant acquisition/eligibility provenance missing")
        _flag(person["eligible"])
    fields = COMMON_FIELDS + (EXPERIMENT_FIELDS if frozen["kind"] == "experiment" else ())
    rows = _rows(data["observations"], fields)
    observed = _unique(rows, "observation_id")
    planned = _unique(_rows(data["acquisition_registry"], tuple(k for k in fields if k not in OUTCOME_FIELDS)), "observation_id")
    if set(observed) != set(planned):
        raise StudyError("Every original planned/assigned observation, including missing and ineligible, must remain")
    raw = _unique(_rows(data["raw_endpoint_table"], ("observation_id", "participant_id", "session_id", "source_id", "site_id", specification["endpoint"]["source_column"])), "observation_id")
    if set(raw) != set(observed):
        raise StudyError("Raw endpoint table differs from complete acquisition denominator")
    first = min(_utc(row["outcome_acquired_utc"]) for row in rows)
    registered = _utc(_json(specification["registration"])["registered_utc"])
    if first <= max(_utc(frozen["created_utc"]), registered):
        raise StudyError("Freeze and registration must precede every independent outcome")
    sessions = []
    if frozen["kind"] == "experiment":
        sessions = _rows(data["session_registry"], SESSION_FIELDS)
        known_sessions = {}
        raw_sessions = {_check_pin(pin): pin["sha256"] for pin in data["raw_session_artifacts"]}
        raw_session_hashes = set(raw_sessions.values())
        baseline = _rows(data["baseline_observations"], ("observation_id", "participant_id", "session_id", "clinical_score", "experience_report", "experience_missing_reason", "capacity_name", "capacity_value", "capacity_missing_reason", "acquired_utc"))
        _unique(baseline, "observation_id")
        first = min([first] + [_utc(r["acquired_utc"]) for r in baseline])
        clinical = _json(data["clinical_scoring_receipt"])
        from .native import CHECKPOINT_SHA256
        if clinical.get("original_checkpoint_sha256") != CHECKPOINT_SHA256 or clinical.get("scored_observations_sha256") != data["raw_endpoint_table"]["sha256"] or clinical.get("baseline_observations_sha256") != data["baseline_observations"]["sha256"]:
            raise StudyError("Paired clinical scores must come from the original frozen discovery instrument")
        for pin in clinical.get("source_artifacts", []):
            _check_pin(pin)
        if not clinical.get("source_artifacts"):
            raise StudyError("Original clinical scoring requires native source evidence")
        for row in sessions:
            key = row["participant_id"], row["session_id"]
            person = participants.get(row["participant_id"])
            if key in known_sessions or person is None or any(row[k] != person[k] for k in ("source_id", "site_id")) or row["session_role"] not in ("baseline", "diagnostic"):
                raise StudyError("Original baseline/diagnostic session registry differs or duplicates")
            if row["raw_session_sha256"] not in raw_session_hashes or not _utc(row["acquisition_start_utc"]) < _utc(row["acquisition_end_utc"]):
                raise StudyError("Session lacks actual source evidence or acquisition clock")
            if _utc(row["acquisition_start_utc"]) <= max(_utc(frozen["created_utc"]), registered):
                raise StudyError("Both human sessions must follow prospective freezing/registration")
            known_sessions[key] = row
        baseline_counts = Counter()
        for row in baseline:
            session = known_sessions.get((row["participant_id"], row["session_id"]))
            if session is None or session["session_role"] != "baseline" or not _utc(session["acquisition_start_utc"]) <= _utc(row["acquired_utc"]) <= _utc(session["acquisition_end_utc"]):
                raise StudyError("Baseline outcome is absent from its source-bound first session")
            core.number(row["clinical_score"], "baseline clinical score")
            if not row["capacity_name"] or (not row["experience_report"] and not row["experience_missing_reason"]) or (not row["capacity_value"] and not row["capacity_missing_reason"]):
                raise StudyError("Baseline must retain experience and named capacity with missing dispositions")
            if row["capacity_value"]:
                core.number(row["capacity_value"], "baseline capacity")
            baseline_counts[row["participant_id"], row["session_id"]] += 1
        for key, session in known_sessions.items():
            if int(session["qualified_baseline_observations"]) != baseline_counts[key]:
                raise StudyError("Qualified baseline episode census differs from actual recorded observations")
        for person in participants:
            personal = [r for r in sessions if r["participant_id"] == person]
            baseline_sessions = [r for r in personal if r["session_role"] == "baseline"]
            diagnostic_sessions = [r for r in personal if r["session_role"] == "diagnostic"]
            if baseline_sessions and diagnostic_sessions and max(_utc(r["acquisition_end_utc"]) for r in baseline_sessions) >= min(_utc(r["acquisition_start_utc"]) for r in diagnostic_sessions):
                raise StudyError("Occurrence/reliability session must precede the diagnostic session")
        design = {r["stratum_id"]: r for r in frozen["qualification"]["randomization"]["strata"]}
        blocks = defaultdict(list)
        for row in rows:
            session = known_sessions.get((row["participant_id"], row["session_id"]))
            if session is None or session["session_role"] != "diagnostic":
                raise StudyError("Randomized observations require the separate diagnostic session")
            if not _utc(session["acquisition_start_utc"]) <= _utc(row["eligibility_window_start_utc"]) < _utc(row["outcome_acquired_utc"]) <= _utc(session["acquisition_end_utc"]):
                raise StudyError("Diagnostic trial clocks escape their actual session")
            if _flag(row["eligible"]):
                stratum = design.get(row["stratum_id"])
                if stratum is None or core.number(row["assignment_probability_active"], "assignment probability") != stratum["probability_active"]:
                    raise StudyError("Assignment probability/stratum differs from prospective block design")
                blocks[row["participant_id"], row["session_id"], row["stratum_id"], row["block_id"]].append(row)
        for key, members in blocks.items():
            stratum = design[key[2]]
            if len(members) != stratum["block_size"] or sum(r["condition"] == "active" for r in members) != stratum["active_per_block"]:
                raise StudyError("Complete original randomized blocks are missing or have changed composition")
    if _utc(data["first_outcome_acquired_utc"]) != first or first <= max(_utc(frozen["created_utc"]), registered):
        raise StudyError("Declared first-outcome time differs or independent outcomes preceded freezing")
    for row in rows:
        person = participants.get(row["participant_id"])
        if person is None:
            raise StudyError("Observation person absent from complete acquired registry")
        original = planned[row["observation_id"]]
        for key in original:
            if key not in OUTCOME_FIELDS and row.get(key) != original[key]:
                raise StudyError("Original acquisition/assignment/eligibility field changed: " + key)
        if any(row[key] != person[key] for key in ("source_id", "site_id")) or not row["session_id"]:
            raise StudyError("Source/site/session identity differs from acquisition registry")
        if (_flag(row["eligible"]) and row["condition"] not in specification["sampling"]["conditions"]) or row["sampling_mode"] not in ("random_probe", "score_triggered", "scheduled_comparator") or not row["eligibility_reason"]:
            raise StudyError("Sampling/condition/eligibility contract missing")
        start, end, decided, acquired = map(_utc, (row["eligibility_window_start_utc"], row["eligibility_window_end_utc"], row["eligibility_decided_utc"], row["outcome_acquired_utc"]))
        if not start < end <= decided < acquired:
            raise StudyError("Eligibility must use completed pre-outcome windows")
        if _flag(row["eligible"]) and not _flag(person["eligible"]):
            raise StudyError("Eligible episode belongs to an excluded participant")
        if frozen["kind"] == "experiment" and _flag(row["eligible"]):
            _experiment_clock(row)
        source = raw[row["observation_id"]]
        if any(source[key] != row[key] for key in ("participant_id", "session_id", "source_id", "site_id")):
            raise StudyError("Measured raw endpoint identity differs")
        if any(source.get(key) != row[key] for key in ("pre_clinical_score", "pre_window_quality")):
            raise StudyError("Eligibility changed genuine pre-window measurements")
        rule = specification["sampling"]["eligibility_rule"]
        expected_eligibility = _flag(person["eligible"]) and row["pre_window_quality"] in rule["allowed_pre_window_quality"] and abs((end-start).total_seconds()-rule["window_seconds"]) <= .01
        if row["sampling_mode"] == "score_triggered":
            if rule["clinical_score_max_for_enrichment"] is None:
                raise StudyError("Score enrichment was not prospectively qualified")
            expected_eligibility = expected_eligibility and core.number(row["pre_clinical_score"], "genuine pre-score") <= rule["clinical_score_max_for_enrichment"]
        if _flag(row["eligible"]) != bool(expected_eligibility):
            raise StudyError("Eligibility flag does not follow the frozen genuine pre-intervention rule")
        if frozen["kind"] == "replication" and _flag(row["eligible"]):
            matched = [c for c, clauses in specification["sampling"]["condition_rules"].items() if _matches(source, clauses)]
            if matched != [row["condition"]]:
                raise StudyError("Replication contrast membership does not follow frozen source-field rules")
        if frozen["kind"] == "experiment":
            paired = ("clinical_score", "experience_report", "experience_missing_reason", "capacity_name", "capacity_value", "capacity_missing_reason", "physiology_name", "physiology_value", "physiology_missing_reason")
            if any(source.get(key) != row[key] for key in paired):
                raise StudyError("Paired clinical, experience, capacity or physiology differs from original measured source")
        measurement = source[specification["endpoint"]["source_column"]]
        if measurement in ("", "NA", "NaN", "nan"):
            if row["endpoint_value"] not in ("", "NA", "NaN", "nan") or not row["missing_reason"]:
                raise StudyError("Missing raw human outcome cannot become an observed value")
        else:
            expected = specification["endpoint"]["scale"] * core.number(measurement, "raw endpoint") + specification["endpoint"]["offset"]
            if abs(expected - core.number(row["endpoint_value"], "measured endpoint")) > specification["endpoint"]["validation_absolute_tolerance"]:
                raise StudyError("Actual outcome does not follow the frozen source operator")
    return data, participants, rows, sessions


def _experiment_clock(row):
    decided, randomized, delivered, acquired = map(_utc, (row["eligibility_decided_utc"], row["randomized_utc"], row["intervention_start_utc"], row["outcome_acquired_utc"]))
    if not decided <= randomized < delivered <= acquired:
        raise StudyError("Eligibility decision must precede original randomization and physical intervention")
    if not row["block_id"] or not row["stratum_id"] or not row["capacity_name"]:
        raise StudyError("Original block/stratum and named paired capacity are required")
    core.number(row["clinical_score"], "clinical score")
    for value, disposition in (("experience_report", "experience_missing_reason"), ("capacity_value", "capacity_missing_reason"), ("physiology_value", "physiology_missing_reason")):
        if not row[value] and not row[disposition]:
            raise StudyError("Paired human measurement requires an actual value or missing disposition")
    if row["capacity_value"]:
        core.number(row["capacity_value"], "capacity value")
    if row["physiology_value"]:
        if not row["physiology_name"]:
            raise StudyError("Available genuine physiology requires its named measurement")
        core.number(row["physiology_value"], "physiology value")
    probability = core.number(row["assignment_probability_active"], "assignment probability")
    if not 0 < probability < 1:
        raise StudyError("Original randomized probability must be strictly between zero and one")
    cs, ce, rs, re, report = map(_utc, (row["clinical_window_start_utc"], row["clinical_window_end_utc"], row["response_window_start_utc"], row["response_window_end_utc"], row["experience_report_utc"]))
    if cs < delivered or abs((ce-cs).total_seconds()-10) > .01 or not delivered <= rs < re <= acquired or not ce <= acquired or report < delivered or report > acquired:
        raise StudyError("Subsequent 10-second clinical score, faster response and timed report must retain distinct valid clocks")


def _bounds(value, condition, conditions, lo, hi, *, probability=None):
    if probability is None:
        sign = 1 if condition == conditions[0] else -1
    else:
        sign = 1/probability if condition == "active" else -1/(1-probability)
    if value in ("", "NA", "NaN", "nan", None):
        return None, *sorted((sign*lo, sign*hi))
    value = core.number(value, "endpoint")
    if not lo <= value <= hi:
        raise StudyError("Measured endpoint is outside its prospectively registered range")
    return sign*value, sign*value, sign*value


def estimate(rows, participants, spec, config, sessions=()):
    """Complete eligible denominator; direction-blind informativeness and equal people."""
    conditions = spec["sampling"]["conditions"]
    lo, hi = spec["endpoint"]["minimum"], spec["endpoint"]["maximum"]
    groups = defaultdict(list)
    for row in rows:
        if _flag(row["eligible"]):
            groups[row["participant_id"]].append(row)
    people, census = [], Counter({"all_planned_observations": len(rows), "ineligible_observations": sum(not _flag(r["eligible"]) for r in rows)})
    for person, registry in sorted(participants.items()):
        observations = groups.get(person, [])
        counts = Counter(r["condition"] for r in observations)
        missing = sum(r["endpoint_value"] in ("", "NA", "NaN", "nan") for r in observations)
        enough = counts[conditions[0]] >= spec["informative"]["minimum_target_or_active_observations"] and counts[conditions[1]] >= spec["informative"]["minimum_comparator_or_control_observations"]
        personal_sessions = [r for r in sessions if r["participant_id"] == person]
        session_count = len(personal_sessions) if spec["kind"] == "experiment" else len({r["session_id"] for r in observations})
        baseline_ready = spec["kind"] != "experiment" or (any(r["session_role"] == "baseline" and int(r["qualified_baseline_observations"]) >= spec["sampling"]["minimum_baseline_observations"] for r in personal_sessions) and any(r["session_role"] == "diagnostic" for r in personal_sessions))
        informative = bool(_flag(registry["eligible"]) and enough and session_count >= spec["sampling"]["minimum_sessions_per_person"] and baseline_ready and missing / max(1, len(observations)) <= spec["informative"]["maximum_missing_fraction"])
        census["acquired_people"] += 1
        census["eligible_people"] += _flag(registry["eligible"])
        census["informative_people"] += informative
        census["eligible_observations"] += len(observations)
        census["missing_outcomes"] += missing
        point, lower, upper = None, None, None
        if spec["kind"] == "experiment" and observations:
            contributions = [_bounds(r["endpoint_value"], r["condition"], conditions, lo, hi, probability=core.number(r["assignment_probability_active"], "assignment probability")) for r in observations]
            lower, upper = (float(np.mean([r[i] for r in contributions])) for i in (1, 2))
            point = None if missing else lower
        elif spec["kind"] == "replication" and all(counts[c] for c in conditions):
            means = {}
            for condition in conditions:
                values = [_bounds(r["endpoint_value"], condition, conditions, lo, hi) for r in observations if r["condition"] == condition]
                means[condition] = [float(np.mean([r[i] for r in values])) for i in (1, 2)]
            lower, upper = (sum(means[c][i] for c in conditions) for i in (0, 1))
            point = None if missing else lower
        elif spec["kind"] == "replication" and _flag(registry["eligible"]):
            # Missing an entire condition must remain in the eligible denominator.
            lower, upper = lo-hi, hi-lo
        people.append({"participant_id": person, "source_id": registry["source_id"], "site_id": registry["site_id"], "eligible": _flag(registry["eligible"]), "informative": informative, "eligible_observations": len(observations), "sessions": session_count, "baseline_acquisition_complete": baseline_ready, "missing_outcomes": missing, "point_estimate": point, "lower_bound": lower, "upper_bound": upper})
    eligible = [r for r in people if r["eligible"]]
    estimable = [r for r in eligible if r["lower_bound"] is not None]
    unmeasured = len(eligible)-len(estimable)
    if spec["kind"] == "experiment" and unmeasured:
        # No assignment cannot be converted into a randomized treatment estimate.
        status = "incomplete_eligible_people_without_assignments"
    else:
        status = "independent_contrast_with_missing_bounds" if census["missing_outcomes"] or any(r["point_estimate"] is None for r in estimable) else "independent_contrast_estimated"
    interval, ci, point = None, None, None
    if len(estimable) >= 2 and not (spec["kind"] == "experiment" and unmeasured):
        lows = np.array([r["lower_bound"] for r in estimable])
        highs = np.array([r["upper_bound"] for r in estimable])
        interval = [float(lows.mean()), float(highs.mean())]
        if all(r["point_estimate"] is not None for r in estimable) and not unmeasured:
            point = float(np.mean([r["point_estimate"] for r in estimable]))
        rng = np.random.default_rng(config["seed"])
        indices = rng.integers(0, len(estimable), size=(config["bootstrap_samples"], len(estimable)))
        ci = [float(np.quantile(lows[indices].mean(axis=1), .025)), float(np.quantile(highs[indices].mean(axis=1), .975))]
    return {"status": status, "point_estimate": point, "identification_interval": interval, "participant_bootstrap_interval_95": ci, "census": dict(census), "participant_estimates": people, "eligible_people_without_assignments": unmeasured,
            "independent_unit": "participant", "estimator": "Equal-person target-minus-comparator condition means" if spec["kind"] == "replication" else "Equal-person mean of eligible trial Horvitz-Thompson contributions", "informative_rule_result_blind": True, "all_eligible_people_retained": True}


def baseline_summary(rows, participants, spec, config):
    """Source-coded co-report occurrence and adjacent-episode repeatability.

    A clinical-score/report association does not establish experience throughout
    the EEG window. Adjacent-episode agreement is descriptive repeatability, not
    a diagnosis, causal mechanism or automatically an across-session ICC.
    """
    rule = spec["sampling"]["baseline_analysis"]
    groups = defaultdict(list)
    for row in rows:
        groups[row["participant_id"]].append(row)
    summaries = []
    for person, registry in sorted(participants.items()):
        if not _flag(registry["eligible"]):
            continue
        observations = sorted(groups.get(person, []), key=lambda r: (_utc(r["acquired_utc"]), r["observation_id"]))
        low = [core.number(r["clinical_score"], "baseline original clinical score") <= rule["low_clinical_score_max"] for r in observations]
        reported = [r["experience_report"] in rule["reported_experience_categories"] if r["experience_report"] else None for r in observations]
        pairs = [(i, i+1) for i in range(0, len(observations)-1, 2)]
        repeatable_reports = [(i, j) for i, j in pairs if reported[i] is not None and reported[j] is not None]
        summary = {"participant_id": person, "baseline_episodes": len(observations), "report_missing": sum(v is None for v in reported), "low_score_episodes": sum(low),
                   "low_score_with_original_reported_experience": sum(a and b is True for a, b in zip(low, reported)),
                   "co_report_prevalence_lower": float(np.mean([a and b is True for a, b in zip(low, reported)])) if observations else 0.,
                   "co_report_prevalence_upper": float(np.mean([a and b is not False for a, b in zip(low, reported)])) if observations else 1.,
                   "adjacent_repeat_pairs": len(pairs), "report_repeat_pairs": len(repeatable_reports),
                   "paired_low_score_agreement": float(np.mean([low[i] == low[j] for i, j in pairs])) if pairs else None,
                   "paired_report_category_agreement": float(np.mean([observations[i]["experience_report"] == observations[j]["experience_report"] for i, j in repeatable_reports])) if repeatable_reports else None,
                   "paired_score_absolute_difference": float(np.mean([abs(float(observations[i]["clinical_score"])-float(observations[j]["clinical_score"])) for i, j in pairs])) if pairs else None,
                   "repeatability_measurements_sufficient": len(pairs) >= rule["minimum_repeat_pairs_per_person"] and len(repeatable_reports) >= rule["minimum_repeat_pairs_per_person"]}
        summaries.append(summary)
    aggregate = {}
    for key in ("co_report_prevalence_lower", "co_report_prevalence_upper", "paired_low_score_agreement", "paired_report_category_agreement", "paired_score_absolute_difference"):
        values = np.asarray([r[key] for r in summaries if r[key] is not None], dtype=float)
        if not len(values):
            aggregate[key] = {"equal_person_estimate": None, "participant_bootstrap_interval_95": None, "measured_people": 0}
            continue
        generator = np.random.default_rng(config["seed"])
        indices = generator.integers(0, len(values), size=(config["bootstrap_samples"], len(values)))
        aggregate[key] = {"equal_person_estimate": float(values.mean()), "participant_bootstrap_interval_95": [float(v) for v in np.quantile(values[indices].mean(axis=1), [.025, .975])], "measured_people": len(values)}
    return {"status": "source_coded_baseline_occurrence_and_repeatability_evaluated", "participant_summaries": summaries, "aggregate": aggregate,
            "minimum_repeat_pairs_per_person": rule["minimum_repeat_pairs_per_person"], "sufficient_repeatability_people": sum(r["repeatability_measurements_sufficient"] for r in summaries),
            "independent_unit": "participant", "pairing": "Nonoverlapping adjacent episodes by actual acquisition clock; rule frozen before evaluation",
            "interpretation": "Original timed low-score co-report occurrence and descriptive within-session repeatability. Missing reports retain prevalence bounds; no claim of experience throughout an EEG window or causal/anatomical identification."}


def evaluate(frozen, data_receipt, output, *, kind):
    root = Path(frozen).resolve()
    value = verify_freeze(root)
    value["protocol_sha256"] = core.sha256_file(root / "frozen_protocol.json")
    if value["kind"] != kind:
        raise StudyError("Replication and randomized experiment require separate frozen protocols")
    data, participants, rows, sessions = _data(value, data_receipt)
    result = estimate(rows, participants, value["specification"], value["configuration"], sessions)
    sampling = value["specification"]["sampling"]
    eligible = [r for r in result["participant_estimates"] if r["eligible"]]
    reasons = []
    if value["analysis_mode"] != "confirmatory":
        reasons.append("pilot_cannot_complete_confirmatory_programme")
    if result["census"].get("informative_people", 0) < value["precision_plan"]["informative_people"]:
        reasons.append("finalized_informative_sample_not_reached")
    if len({r["source_id"] for r in eligible}) < sampling["minimum_sources"]:
        reasons.append("independent_acquisition_sources_not_reached")
    if len({r["site_id"] for r in eligible}) < sampling["minimum_sites"]:
        reasons.append("acquisition_sites_not_reached")
    if any(r["sessions"] < sampling["minimum_sessions_per_person"] for r in eligible):
        reasons.append("prespecified_repeated_sessions_incomplete")
    if any(not r["baseline_acquisition_complete"] for r in eligible):
        reasons.append("first_session_baseline_acquisition_incomplete")
    baseline = None
    if kind == "experiment":
        baseline = baseline_summary(_rows(data["baseline_observations"]), participants, value["specification"], value["configuration"])
        if baseline["sufficient_repeatability_people"] < value["precision_plan"]["informative_people"]:
            reasons.append("prespecified_baseline_repeatability_sample_incomplete")
    if result["eligible_people_without_assignments"]:
        reasons.append("eligible_people_lack_full_acquisition")
    if result["identification_interval"] is None:
        reasons.append("fewer_than_two_estimable_independent_people")
    result.update(kind=kind, analysis_mode=value["analysis_mode"], scientific_completion=not reasons, incomplete_reasons=reasons,
                  all_outcome_directions_retained=True, independent_human_data_evaluated=True, frozen_protocol_sha256=value["protocol_sha256"],
                  inputs=[_pin(root / "receipt.json"), _pin(data_receipt)],
                  interpretation="Specified independent human contrast; null, negative and imprecise results remain valid. Sample/protocol completeness does not establish a unique physiological mechanism.")
    result["scientific_completion_scope"] = "The declared independent contrast and acquisition/measurement contract; completion of the entire original scientific programme is not certified"
    result["acquisition_protocol_completion"] = not reasons
    if kind == "experiment":
        prediction_qa = value["qualification"]["accounts"]
        result.update(mechanism_discrimination_design_qualified=prediction_qa["mechanism_discrimination_design_qualified"],
                      pairwise_frozen_prediction_separations=prediction_qa["pairwise_frozen_prediction_separations"],
                      observational_equivalence_status=prediction_qa["observational_equivalence_status"],
                      mechanism_discriminating_test_completed=not reasons and prediction_qa["mechanism_discrimination_design_qualified"])
        result["baseline_occurrence_repeatability"] = baseline
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    core.write_json(out / "evaluation.json", result)
    core.write_csv(out / "participant_estimates.csv", result["participant_estimates"])
    core.write_json(out / "receipt.json", {"status": "independent_human_evaluation_completed" if not reasons else "independent_human_evaluation_incomplete", "source_code_sha256": core.sha256_file(__file__), "kind": kind, "evaluation": _pin(out / "evaluation.json"), "participant_estimates": _pin(out / "participant_estimates.csv"), "inputs": result["inputs"], "scientific_completion": not reasons})
    return result


def prepare_template(kind, output):
    """An intentionally inadmissible template; supply real evidence before freezing."""
    if kind not in ("replication", "experiment"):
        raise StudyError("kind must be replication or experiment")
    pin = {"path": "REQUIRED_ACTUAL_SOURCE_FILE", "sha256": "REQUIRED_ACTUAL_SHA256"}
    spec = {"schema_version": 1, "kind": kind, "analysis_mode": "confirmatory", "configuration": _pin(DEFAULT_CONFIG),
            "computational_candidate": dict(pin), "development_registry": dict(pin), "replication_registry": dict(pin) if kind == "experiment" else None,
            "precision_plan": dict(pin), "registration": dict(pin), "account_predictions": dict(pin) if kind == "experiment" else None, "randomization_plan": dict(pin) if kind == "experiment" else None,
            "endpoint": {"name": "REQUIRED_MEASURED_ENDPOINT", "construct": "REQUIRED_HUMAN_CONSTRUCT", "unit": "REQUIRED_SOURCE_UNITS", "minimum": None, "maximum": None, "source_column": "REQUIRED_SOURCE_COLUMN", "scale": 1., "offset": 0., "operator_id": "REQUIRED_FROZEN_OPERATOR", "operator_source": dict(pin), "report_timing_basis": "REQUIRED_ORIGINAL_TIMED_REPORT_SUPPORT", "validation_absolute_tolerance": 1e-8},
            "sampling": {"assay_id": "REQUIRED_IDENTICAL_HUMAN_ASSAY", "conditions": ["target", "comparator"] if kind == "replication" else ["active", "control"], "minimum_sources": 2 if kind == "replication" else 1, "minimum_sites": 1 if kind == "replication" else 2, "minimum_sessions_per_person": 1 if kind == "replication" else 2, "minimum_baseline_observations": 2, "baseline_analysis": {"low_clinical_score_max": None, "reported_experience_categories": [], "minimum_repeat_pairs_per_person": 1} if kind == "experiment" else None, "eligibility_rule": {"allowed_pre_window_quality": ["qualified"], "window_seconds": 10., "clinical_score_max_for_enrichment": None}, "condition_rules": {"target": [{"field": "REQUIRED_ORIGINAL_SOURCE_CONTRAST_FIELD", "operator": "equal", "value": "REQUIRED_TARGET_CODING"}], "comparator": [{"field": "REQUIRED_ORIGINAL_SOURCE_CONTRAST_FIELD", "operator": "equal", "value": "REQUIRED_COMPARATOR_CODING"}]} if kind == "replication" else None, "sampling_rule": "REQUIRED_RANDOM_AND_ENRICHED_SCHEDULE", "report_schedule": "REQUIRED_EXPERIENCE_BEHAVIOUR_PHYSIOLOGY_TIMING"},
            "informative": {"minimum_target_or_active_observations": 2, "minimum_comparator_or_control_observations": 2, "maximum_missing_fraction": .2},
            "qualification": {"endpoint_validation": dict(pin), **({"account_execution": dict(pin), "physical_intervention": dict(pin)} if kind == "experiment" else {})}}
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    core.write_json(out / "specification.template.json", spec)
    templates = {
        "precision_plan.template.json": {"status": "REQUIRED_FINALIZED_BEFORE_OUTCOMES", "analysis_mode": "confirmatory", "informative_people": 100 if kind == "replication" else 110, "eligible_recruitment_target": None, "maximum_interval_width": None, "reliability_definition": "", "episode_schedule": "", "required_scenarios": ["null", "adverse_missingness"], "simulation_table": dict(pin), "simulation_provenance": [dict(pin)]},
        "endpoint_validation.template.json": {"assay_id": "", "unit": "", "minimum_validation_people": 2, "raw_calibration_table": dict(pin), "validation_table": dict(pin), "raw_source_artifacts": [dict(pin)]},
        "registration.template.json": {"analysis_contract_sha256": "Compute contract(specification) canonical SHA256 after supplying all evidence", "registered_utc": "", "registry_identifier": ""},
        "data_receipt.template.json": {"frozen_protocol_sha256": "", "human_data": True, "synthetic": False, "first_outcome_acquired_utc": "", "participant_registry": dict(pin), "observations": dict(pin), "acquisition_registry": dict(pin), "raw_endpoint_table": dict(pin), "raw_source_artifacts": [dict(pin)]},
    }
    if kind == "experiment":
        templates["physical_intervention.template.json"] = {"assay_id": "", "physical_variable": "", "unit": "", "predeclared_absolute_delivery_tolerance": None, **{name: dict(pin) for name in ("raw_delivery_table", "prescribed_delivery_table", "delivery_operator_source", "approved_acquisition_protocol", "ethics_authorization", "raw_delivery_source")}}
        templates["account_execution.template.json"] = {"assay_id": "", "unit": "", "prediction_table_sha256": "", "native_execution_table": dict(pin), "account_artifacts": {"REQUIRED_ACCOUNT_A_AND_B": {name: dict(pin) for name in ("source_code", "adapted_parameters", "training_receipt", "human_assay_mapping", "physical_input_mapping")}}}
        templates["physical_input_mapping.template.json"] = {"assay_id": "", "physical_variable": "", "physical_unit": "", "model_variable": "", "model_unit": "", "scale": None, "offset": None, "predeclared_absolute_mapping_tolerance": None, **{name: dict(pin) for name in ("calibration_table", "raw_human_input_table", "mapping_operator_source", "native_input_execution", "raw_physical_source")}}
        templates["human_assay_mapping.template.json"] = {"assay_id": "", "unit": "", "minimum_test_people": 2, "predeclared_maximum_mse": None, **{name: dict(pin) for name in ("validation_table", "operator_source", "raw_observation_table", "frozen_mapping")}}
        templates["randomization_plan.template.json"] = {"method": "fixed_block_randomization", "allocation_generator": dict(pin), "strata": [{"stratum_id": "REQUIRED_PREDECLARED_STRATUM", "block_size": 4, "active_per_block": 2, "probability_active": .5}]}
        templates["data_receipt.template.json"].update(session_registry=dict(pin), raw_session_artifacts=[dict(pin)], baseline_observations=dict(pin), clinical_scoring_receipt=dict(pin))
    for name, template in templates.items():
        core.write_json(out / name, template)
    core.write_csv(out / "participant_registry.template.csv", [], list(PARTICIPANT_FIELDS))
    core.write_csv(out / "observations.template.csv", [], list(COMMON_FIELDS + (EXPERIMENT_FIELDS if kind == "experiment" else ())))
    if kind == "experiment":
        core.write_csv(out / "session_registry.template.csv", [], list(SESSION_FIELDS))
    core.write_json(out / "template_status.json", {"status": "BLOCKED_MISSING_ACTUAL_HUMAN_EVIDENCE", "kind": kind, "human_intervention_qualified": False, "actual_humans_supplied": False, "scientific_completion": False})
    return {"status": "templates_prepared_actual_inputs_required", "output": str(out), "kind": kind}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("prepare-template")
    template.add_argument("--kind", choices=("replication", "experiment"), required=True)
    template.add_argument("--output", required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--spec", required=True)
    frozen.add_argument("--output", required=True)
    for name in ("evaluate-replication", "evaluate-experiment"):
        command = commands.add_parser(name)
        for key in ("frozen", "data-receipt", "output"):
            command.add_argument("--" + key, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "prepare-template":
            result = prepare_template(**args)
        elif command == "freeze":
            result = freeze(args["spec"], args["output"])
        else:
            result = evaluate(**args, kind="replication" if command == "evaluate-replication" else "experiment")
        print(json.dumps({k: result[k] for k in ("status", "kind", "scientific_completion", "incomplete_reasons") if k in result}, indent=2))
    except (StudyError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, "Prospective human validation blocked: " + str(exc) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
