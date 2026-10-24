"""Grouped prediction and a transparent synthetic effective-dynamics example.

The ridge comparison assesses predictive increment. The two-state linear
simulation illustrates recovery under ideal observation, and is neither the
released NFTsim physiology nor evidence for a human biological mechanism.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any

import numpy as np


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _flag(value: Any, name: str) -> bool:
    if value in (True, "true", "True", "TRUE", 1, "1"):
        return True
    if value in (False, "false", "False", "FALSE", 0, "0"):
        return False
    raise ValueError(f"{name} must be an explicit boolean")


def _fit_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
               ridge: float) -> dict:
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    scale[scale < 1e-12] = 1.
    standardized = (x - mean) / scale
    y_mean = float(np.average(y, weights=weights))
    lhs = standardized.T @ (weights[:, None] * standardized) + ridge * np.eye(x.shape[1])
    rhs = standardized.T @ (weights * (y - y_mean))
    # lstsq handles even an unregularized rank-deficient design explicitly.
    beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return {"feature_mean": mean, "feature_scale": scale,
            "outcome_mean": y_mean, "coefficient_standardized": beta}


def _predict(model: dict, x: np.ndarray) -> np.ndarray:
    return ((x - model["feature_mean"]) / model["feature_scale"]
            @ model["coefficient_standardized"] + model["outcome_mean"])


def _public_fit(model: dict) -> dict:
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in model.items()}


def _prediction_spec(outcome_column: str, baseline_columns: list[str],
                     dynamic_columns: list[str], ridge: float) -> list[str]:
    if not isinstance(outcome_column, str) or not outcome_column:
        raise ValueError("A nonempty outcome column is required")
    if not isinstance(baseline_columns, list) or not isinstance(dynamic_columns, list):
        raise ValueError("Feature column names must be lists")
    columns = baseline_columns + dynamic_columns
    if not baseline_columns or not dynamic_columns or any(not isinstance(c, str) or not c for c in columns):
        raise ValueError("Specify nonempty baseline and dynamic feature names")
    if len(set(columns)) != len(columns):
        raise ValueError("Baseline and dynamic columns must be distinct")
    if set(columns) & {outcome_column, "participant_id", "episode_id", "trial_id", "fold", "partition", "role"}:
        raise ValueError("Outcomes and group identifiers cannot be predictive features")
    if not isinstance(ridge, (int, float)) or not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    return columns


def _prediction_records(rows: list[dict], outcome_column: str, columns: list[str]) -> dict:
    if not rows:
        raise ValueError("Prediction requires a nonempty observation table")
    schema = set().union(*(set(row) for row in rows))
    absent = {outcome_column, *columns} - schema
    if absent:
        raise ValueError(f"Required prediction columns unavailable: {sorted(absent)}")
    people, sources, cohorts = set(), set(), set()
    records, source_by_person = [], {}
    excluded, missing = Counter(), Counter()
    for index, row in enumerate(rows):
        person = row.get("participant_id")
        if person is None or not str(person).strip():
            raise ValueError("Every row requires a global participant_id")
        person = str(person)
        people.add(person)
        source = str(row.get("source_id") or "unspecified")
        sources.add(source)
        source_by_person.setdefault(person, set()).add(source)
        if row.get("cohort_id"):
            cohorts.add(str(row["cohort_id"]))
        origin = row.get("outcome_label_source", row.get(f"{outcome_column}_source"))
        if origin in {"clinical_detector", "detector_selected", "detector_score", "synthetic_awareness"}:
            raise ValueError("Detector-derived labels cannot supply independent awareness/experience outcomes")
        for flag in ("eligible", "qc_pass"):
            if flag in row and not _flag(row[flag], flag):
                excluded[f"{flag}_false_rows"] += 1
                break
        else:
            values = [_finite(row.get(c)) for c in [outcome_column] + columns]
            if any(value is None for value in values):
                excluded["incomplete_outcome_or_feature_rows"] += 1
                missing.update(c for c, v in zip([outcome_column] + columns, values) if v is None)
                continue
            records.append((index, person, values[0], values[1:]))
    return {"records": records, "input_participants": sorted(people),
            "source_ids": sorted(sources), "cohort_ids": sorted(cohorts),
            "source_by_person": source_by_person, "exclusions": dict(excluded), "missing_by_column": dict(missing)}


def fit_prediction_model(rows: list[dict], outcome_column: str,
                         baseline_columns: list[str], dynamic_columns: list[str],
                         ridge: float = 1.0) -> dict:
    """Fit development data once and return portable, finite JSON parameters.

    All inspected participant IDs are exposure IDs, including missing or
    excluded rows; they cannot subsequently enter protected model evaluation.
    No raw observations are stored in the artifact. This function creates a
    fitted artifact; the calling pipeline must hash/freeze it before accessing
    protected data and confirm cohort provenance through its manifest.
    """
    columns = _prediction_spec(outcome_column, baseline_columns, dynamic_columns, ridge)
    for row in rows:
        role = str(row.get("role") or row.get("partition") or "development").lower()
        if role in {"evaluation", "diagnostic", "protected", "holdout", "independent"}:
            raise ValueError("Fitting a transport model cannot use protected observations")
    prepared = _prediction_records(rows, outcome_column, columns)
    records = prepared["records"]
    people = sorted({r[1] for r in records})
    if len(people) < 2:
        raise ValueError("At least two complete-case independent training participants are required")
    counts = Counter(r[1] for r in records)
    weights = np.asarray([1. / counts[r[1]] for r in records])
    x = np.asarray([r[3] for r in records], dtype=float)
    y = np.asarray([r[2] for r in records], dtype=float)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            baseline = _fit_ridge(x[:, :len(baseline_columns)], y, weights, ridge)
            augmented = _fit_ridge(x, y, weights, ridge)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError("Training numerical scale is invalid; use qualified finite feature units") from exc
    artifact = {"artifact_schema_version": 1, "model_kind": "frozen_participant_weighted_ridge",
                "outcome_column": outcome_column, "baseline_columns": list(baseline_columns),
                "dynamic_columns": list(dynamic_columns), "ridge": float(ridge),
                "training_participant_ids": prepared["input_participants"],
                "fitted_participant_ids": people, "training_source_ids": prepared["source_ids"],
                "training_cohort_ids": prepared["cohort_ids"],
                "training_counts": {"input_rows": len(rows), "complete_case_rows": len(records),
                                    "input_participants": len(prepared["input_participants"]),
                                    "complete_case_participants": len(people)},
                "training_exclusions": prepared["exclusions"],
                "training_missing_by_column": prepared["missing_by_column"],
                "baseline_fit": _public_fit(baseline), "augmented_fit": _public_fit(augmented),
                "weighting": "Equal total weight per training participant; mean and scale fitted once to development data.",
                "provenance_scope": "Participant exposure IDs retained; source/cohort independence must additionally be verified by the calling pipeline's manifest. No raw observations stored."}
    # Validate every array and JSON finiteness now, not after protected access.
    _validate_prediction_artifact(artifact)
    json.dumps(artifact, allow_nan=False)
    return artifact


def _validate_prediction_artifact(artifact: dict) -> tuple[dict, dict, list[str]]:
    if artifact.get("artifact_schema_version") != 1 or artifact.get("model_kind") != "frozen_participant_weighted_ridge":
        raise ValueError("Unsupported frozen prediction artifact")
    required = {"outcome_column", "baseline_columns", "dynamic_columns", "ridge",
                "training_participant_ids", "training_source_ids", "baseline_fit", "augmented_fit"}
    absent = required - set(artifact)
    if absent:
        raise ValueError(f"Incomplete frozen prediction artifact: {sorted(absent)}")
    columns = _prediction_spec(artifact["outcome_column"], artifact["baseline_columns"],
                               artifact["dynamic_columns"], artifact["ridge"])
    if not isinstance(artifact["training_participant_ids"], list) or not artifact["training_participant_ids"]:
        raise ValueError("Artifact must record all training participant exposure IDs")
    if any(not isinstance(p, str) or not p.strip() for p in artifact["training_participant_ids"]):
        raise ValueError("Artifact training participant IDs must be nonempty strings")
    if not isinstance(artifact["training_source_ids"], list):
        raise ValueError("Artifact source IDs must be a list")
    models = []
    for name, width in [("baseline_fit", len(artifact["baseline_columns"])), ("augmented_fit", len(columns))]:
        fit = artifact[name]
        if not isinstance(fit, dict):
            raise ValueError(f"{name} must be an object")
        converted = {}
        for field in ("feature_mean", "feature_scale", "coefficient_standardized"):
            try:
                array = np.asarray(fit[field], dtype=float)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Missing or invalid {name}.{field}") from exc
            if array.shape != (width,) or not np.isfinite(array).all():
                raise ValueError(f"{name}.{field} must have {width} finite values")
            if field == "feature_scale" and np.any(array <= 0):
                raise ValueError("Frozen feature scales must be strictly positive")
            converted[field] = array
        converted["outcome_mean"] = _finite(fit.get("outcome_mean"))
        if converted["outcome_mean"] is None:
            raise ValueError(f"{name}.outcome_mean must be finite")
        models.append(converted)
    return models[0], models[1], columns


def evaluate_prediction_model(rows: list[dict], artifact: dict) -> dict:
    """Transport frozen parameters without refitting or evaluation-data scaling.

    Participant overlap is rejected before complete-case exclusions. Shared
    acquisition source is permitted and disclosed; independent cohorts and
    prior exposure require the calling pipeline's manifest checks. Intervals
    condition on this fixed trained artifact and observed source composition;
    they exclude training/model-selection and new-site uncertainty.
    """
    baseline, augmented, columns = _validate_prediction_artifact(artifact)
    prepared = _prediction_records(rows, artifact["outcome_column"], columns)
    overlap = set(prepared["input_participants"]) & set(artifact["training_participant_ids"])
    if overlap:
        raise ValueError(f"Protected evaluation overlaps inspected training participants: {sorted(overlap)}")
    records = prepared["records"]
    if not records:
        raise ValueError("No matched complete-case observations for frozen evaluation")
    x = np.asarray([r[3] for r in records], dtype=float)
    y = np.asarray([r[2] for r in records], dtype=float)
    participant = np.asarray([r[1] for r in records], dtype=object)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            baseline_predictions = _predict(baseline, x[:, :len(artifact["baseline_columns"])])
            augmented_predictions = _predict(augmented, x)
            baseline_loss = (y - baseline_predictions) ** 2
            augmented_loss = (y - augmented_predictions) ** 2
    except FloatingPointError as exc:
        raise ValueError("Evaluation numerical scale is invalid for the frozen model") from exc
    if not np.isfinite(baseline_loss).all() or not np.isfinite(augmented_loss).all():
        raise ValueError("Frozen evaluation produced non-finite predictions or losses")
    people = sorted(set(participant.tolist()))
    losses = []
    strata = []
    for person in people:
        mask = participant == person
        base_mse, augmented_mse = float(np.mean(baseline_loss[mask])), float(np.mean(augmented_loss[mask]))
        source_ids = sorted(prepared["source_by_person"][person])
        losses.append({"participant_id": person, "source_ids": source_ids, "rows": int(mask.sum()),
                       "baseline_mse": base_mse, "augmented_mse": augmented_mse,
                       "mse_improvement": base_mse - augmented_mse})
        strata.append(" + ".join(source_ids))
    intervals = {"baseline_mse": None, "augmented_mse": None, "incremental_mse_improvement": None}
    singleton = [source for source, count in Counter(strata).items() if count == 1]
    if len(people) >= 2 and not singleton:
        rng = np.random.default_rng(20261004)
        mean_losses = np.zeros((2000, 2))
        for source in sorted(set(strata)):
            values = np.asarray([[r["baseline_mse"], r["augmented_mse"]]
                                 for r, s in zip(losses, strata) if s == source])
            indices = rng.integers(0, len(values), (2000, len(values)))
            mean_losses += values[indices].sum(axis=1)
        mean_losses /= len(people)
        for name, values in [("baseline_mse", mean_losses[:, 0]), ("augmented_mse", mean_losses[:, 1]),
                             ("incremental_mse_improvement", mean_losses[:, 0] - mean_losses[:, 1])]:
            intervals[name] = [float(v) for v in np.quantile(values, [.025, .975])]
    base_mse = float(np.mean([r["baseline_mse"] for r in losses]))
    aug_mse = float(np.mean([r["augmented_mse"] for r in losses]))
    artifact_digest = hashlib.sha256(json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    return {"analysis": "frozen_prediction_transport", "artifact_content_sha256": artifact_digest,
            "outcome_column": artifact["outcome_column"], "baseline_columns": artifact["baseline_columns"],
            "dynamic_columns": artifact["dynamic_columns"], "ridge": artifact["ridge"],
            "counts": {"input_rows": len(rows), "complete_case_rows": len(records),
                       "input_participants": len(prepared["input_participants"]), "complete_case_participants": len(people)},
            "exclusions": prepared["exclusions"], "missing_by_column": prepared["missing_by_column"],
            "evaluation_participant_ids": prepared["input_participants"],
            "evaluation_source_ids": prepared["source_ids"], "evaluation_cohort_ids": prepared["cohort_ids"],
            "source_ids_shared_with_training": sorted(set(prepared["source_ids"]) & set(artifact["training_source_ids"])),
            "baseline_mse": base_mse, "augmented_mse": aug_mse,
            "incremental_mse_improvement": base_mse - aug_mse,
            "participant_losses": losses,
            "uncertainty": {"confidence": .95, "bootstrap_reps": 2000, "seed": 20261004,
                            "method": "source-stratified percentile bootstrap of paired participant prediction losses",
                            "intervals": intervals, "singleton_source_strata": singleton,
                            "inference_status": "available" if len(people) >= 2 and not singleton else "insufficient_independent_replication_within_strata",
                            "scope": "Conditional on frozen training model and observed source composition; excludes training/model-selection and site-superpopulation uncertainty."},
            "estimand": "Equal-participant mean squared prediction error and paired model-error difference on matched complete-case observations from unseen participants.",
            "operation": "Frozen training parameters and scaling applied directly; no refitting, imputation, calibration, or outcome-dependent selection.",
            "interpretation": "Transported predictive performance on participant-disjoint data. Cohort independence and outcome provenance need manifest confirmation; predictive increment alone does not establish a human mechanism.",
            "predictions": [{"input_row": r[0], "participant_id": r[1], "observed": r[2],
                             "baseline_prediction": float(b), "augmented_prediction": float(a)}
                            for r, b, a in zip(records, baseline_predictions, augmented_predictions)]}


def grouped_prediction(rows: list[dict], outcome_column: str,
                       baseline_columns: list[str], dynamic_columns: list[str],
                       folds: int = 5, seed: int = 20261004,
                       ridge: float = 1.0) -> dict:
    """Compare ridge predictions in balanced whole-participant held-out folds.

    Baseline and augmented models use the SAME complete-case observations.
    Training loss and scaling weight every participant equally, regardless of
    row count. MSE likewise averages row loss within participant first. Ridge
    is prespecified, not tuned against held-out outcomes. Explicit detector-
    derived awareness-label provenance is rejected. This cross-validation is
    not an external independent replication.
    """
    if not baseline_columns or not dynamic_columns:
        raise ValueError("Specify nonempty baseline_columns and dynamic_columns")
    columns = list(baseline_columns) + list(dynamic_columns)
    if len(set(columns)) != len(columns):
        raise ValueError("Baseline and dynamic feature columns must be distinct")
    forbidden = {outcome_column, "participant_id", "episode_id", "trial_id", "fold", "partition"}
    if set(columns) & forbidden:
        raise ValueError("Outcome/group identifiers cannot be predictive features")
    if not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer >= 2")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    forbidden_origins = {"clinical_detector", "detector_selected", "detector_score", "synthetic_awareness"}
    excluded: Counter = Counter()
    missing_columns: Counter = Counter()
    input_people = set()
    records = []
    partitions = set()
    for index, row in enumerate(rows):
        person = row.get("participant_id")
        if person is None or not str(person).strip():
            raise ValueError("Every row requires a global participant_id")
        person = str(person)
        input_people.add(person)
        origin = row.get("outcome_label_source", row.get(f"{outcome_column}_source"))
        if origin in forbidden_origins:
            raise ValueError("Detector-derived labels cannot supply independent awareness/experience outcomes")
        if row.get("partition"):
            partitions.add(str(row["partition"]))
        for flag in ("eligible", "qc_pass"):
            if flag in row and not _flag(row[flag], flag):
                excluded[f"{flag}_false_rows"] += 1
                break
        else:
            values = [_finite(row.get(c)) for c in [outcome_column] + columns]
            if any(value is None for value in values):
                excluded["incomplete_outcome_or_feature_rows"] += 1
                missing_columns.update(c for c, v in zip([outcome_column] + columns, values) if v is None)
                continue
            records.append((index, person, values[0], values[1:]))
    protected_names = {"protected", "evaluation", "holdout", "independent"}
    if len(partitions) > 1 and any(p.lower() in protected_names for p in partitions):
        raise ValueError("Do not mix protected evaluation observations with development data in cross-validation")
    people = sorted({r[1] for r in records})
    if len(people) < 2 * folds:
        raise ValueError("Every fold requires at least two independent test participants; reduce folds or obtain more participants")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(people, dtype=object)
    rng.shuffle(shuffled)
    assignment = {str(p): int(i % folds) for i, p in enumerate(shuffled)}
    participant = np.asarray([r[1] for r in records], dtype=object)
    x = np.asarray([r[3] for r in records], dtype=float)
    y = np.asarray([r[2] for r in records], dtype=float)
    predictions_baseline = np.empty(len(records))
    predictions_augmented = np.empty(len(records))
    fold_details = []
    for fold in range(folds):
        test = np.asarray([assignment[str(p)] == fold for p in participant])
        train = ~test
        training_people, test_people = sorted(set(participant[train])), sorted(set(participant[test]))
        if len(training_people) < 2 or len(test_people) < 2:
            raise ValueError("Undersampled training or test fold")
        counts = Counter(participant[train].tolist())
        weights = np.asarray([1. / counts[p] for p in participant[train]])
        baseline = _fit_ridge(x[train, :len(baseline_columns)], y[train], weights, ridge)
        augmented = _fit_ridge(x[train], y[train], weights, ridge)
        predictions_baseline[test] = _predict(baseline, x[test, :len(baseline_columns)])
        predictions_augmented[test] = _predict(augmented, x[test])
        fold_details.append({"fold": fold, "training_participants": training_people,
                             "test_participants": test_people, "training_rows": int(train.sum()),
                             "test_rows": int(test.sum()), "baseline_fit": _public_fit(baseline),
                             "augmented_fit": _public_fit(augmented)})
    losses = []
    for person in people:
        mask = participant == person
        baseline_mse = float(np.mean((y[mask] - predictions_baseline[mask]) ** 2))
        augmented_mse = float(np.mean((y[mask] - predictions_augmented[mask]) ** 2))
        losses.append({"participant_id": person, "rows": int(mask.sum()),
                       "baseline_mse": baseline_mse, "augmented_mse": augmented_mse,
                       "mse_improvement": baseline_mse - augmented_mse})
    baseline_mse = float(np.mean([p["baseline_mse"] for p in losses]))
    augmented_mse = float(np.mean([p["augmented_mse"] for p in losses]))
    predictions = [{"input_row": r[0], "participant_id": r[1], "fold": assignment[r[1]],
                    "observed": r[2], "baseline_prediction": float(b), "augmented_prediction": float(a)}
                   for r, b, a in zip(records, predictions_baseline, predictions_augmented)]
    return {"analysis": "grouped_ridge_prediction", "outcome_column": outcome_column,
            "baseline_columns": baseline_columns, "dynamic_columns": dynamic_columns,
            "folds": folds, "seed": seed, "ridge": ridge,
            "counts": {"input_rows": len(rows), "complete_case_rows": len(records),
                       "input_participants": len(input_people), "complete_case_participants": len(people)},
            "exclusions": dict(excluded), "missing_by_column": dict(missing_columns),
            "independent_unit": "global participant_id; all observations of one person held out together",
            "scaling": "Training-only mean and standard deviation with equal total weight per training participant",
            "estimand": "Equal-participant out-of-fold mean squared prediction error on matched complete-case observations",
            "baseline_mse": baseline_mse, "augmented_mse": augmented_mse,
            "incremental_mse_improvement": baseline_mse - augmented_mse,
            "interpretation": "Predictive increment under grouped cross-validation; no causal or biological mechanism claim, and no external independent replication. Outcome provenance must be verified before treating it as experience or capacity.",
            "participant_losses": losses, "fold_details": fold_details,
            "out_of_fold_predictions": predictions}


def _simulate_linear(parameters: dict, inputs: np.ndarray, noise_sd: float,
                     rng: np.random.Generator) -> np.ndarray:
    decay = math.exp(-parameters["dt"] / parameters["timescale"])
    coupling, gain = parameters["coupling"], parameters["gain"]
    matrix = np.asarray([[decay, coupling], [coupling, decay]])
    if max(abs(np.linalg.eigvals(matrix))) >= 1:
        raise ValueError("Effective transition matrix must be stable (spectral radius < 1)")
    states = np.zeros((len(inputs) + 1, 2))
    for t, stimulus in enumerate(inputs):
        states[t + 1] = matrix @ states[t] + np.asarray([gain * stimulus, 0.]) + rng.normal(0, noise_sd, 2)
    return states


def _fit_effective(states: np.ndarray, inputs: np.ndarray,
                   reference: dict, varying: list[str]) -> tuple[dict, float]:
    # Both state coordinates are observed without measurement error. This
    # is an idealised system-identification test, not scalp EEG inversion.
    design = np.column_stack([states[:-1].reshape(-1),
                              states[:-1, ::-1].reshape(-1),
                              np.column_stack([inputs, np.zeros_like(inputs)]).reshape(-1)])
    target = states[1:].reshape(-1)
    names = ["decay", "coupling", "gain"]
    beta = np.asarray([math.exp(-reference["dt"] / reference["timescale"]),
                       reference["coupling"], reference["gain"]])
    free = [names.index(n) for n in varying]
    fixed = [i for i in range(3) if i not in free]
    residual = target - design[:, fixed] @ beta[fixed]
    free_design = design[:, free]
    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(free_design, residual, rcond=None)
    beta[free] = coefficients
    identifiable = rank == len(free)
    stable = 0 < beta[0] < 1 and max(abs(beta[0] + beta[1]), abs(beta[0] - beta[1])) < 1
    parameters = {"dt": reference["dt"], "gain": float(beta[2]), "coupling": float(beta[1]),
                  "decay": float(beta[0]), "timescale": -reference["dt"] / math.log(beta[0]) if 0 < beta[0] < 1 else None,
                  "stable": bool(stable), "identifiable_under_ideal_observation": bool(identifiable),
                  "free_parameter_design_rank": int(rank), "free_parameters": varying,
                  "free_design_condition_number": (float(singular_values[0] / singular_values[-1])
                                                    if identifiable and len(singular_values) and singular_values[-1] > 0 else None)}
    return parameters, float(np.mean((target - design @ beta) ** 2))


def effective_recovery(settings: dict) -> dict:
    """Recover a declared synthetic two-state effective model and test holdout.

    Settings: true_parameters/reference_parameters dicts with gain, coupling,
    timescale, dt; train_steps, holdout_steps, process_noise_sd, replicates, seed.
    Alternative one-parameter families fix remaining quantities at reference.
    Full observed states and known inputs are deliberately ideal assumptions.
    """
    reference = {"gain": 1., "coupling": .04, "timescale": 5., "dt": 1.}
    reference.update(settings.get("reference_parameters", {}))
    truth = {**reference, **settings.get("true_parameters", {"gain": 1.2, "coupling": .08})}
    train_steps, holdout_steps = settings.get("train_steps", 300), settings.get("holdout_steps", 180)
    replicates, seed = settings.get("replicates", 20), settings.get("seed", 20261004)
    noise = settings.get("process_noise_sd", .03)
    for name, value in (("train_steps", train_steps), ("holdout_steps", holdout_steps), ("replicates", replicates)):
        if not isinstance(value, int) or value < (10 if name != "replicates" else 1):
            raise ValueError(f"Invalid {name}")
    if not math.isfinite(noise) or noise < 0:
        raise ValueError("process_noise_sd must be finite and nonnegative")
    for parameters in (truth, reference):
        if any(not math.isfinite(parameters[k]) for k in reference):
            raise ValueError("Effective parameters must be finite")
        if parameters["dt"] <= 0 or parameters["timescale"] <= 0:
            raise ValueError("dt and timescale must be positive")
    if truth["dt"] != reference["dt"]:
        raise ValueError("Truth and reference must use the same known sampling interval")
    rng = np.random.default_rng(seed)
    alternatives = {"gain_only": ["gain"], "coupling_only": ["coupling"],
                    "timescale_only": ["decay"], "joint_effective": ["decay", "coupling", "gain"]}
    reports = []
    recovered = []
    for replicate in range(replicates):
        train_input = (rng.random(train_steps) < .12) * rng.uniform(.5, 1.5, train_steps)
        holdout_input = np.zeros(holdout_steps)
        holdout_input[::17] = 1.
        holdout_input[5::29] += .5
        train = _simulate_linear(truth, train_input, noise, rng)
        holdout_truth = _simulate_linear(truth, holdout_input, 0., rng)
        for name, free in alternatives.items():
            fit, train_mse = _fit_effective(train, train_input, reference, free)
            holdout_mse = None
            if fit["stable"] and fit["identifiable_under_ideal_observation"]:
                predicted = _simulate_linear(fit, holdout_input, 0., rng)
                holdout_mse = float(np.mean((holdout_truth[1:] - predicted[1:]) ** 2))
            reports.append({"replicate": replicate, "family": name, "parameters": fit,
                            "training_one_step_mse": train_mse,
                            "heldout_noiseless_trajectory_mse": holdout_mse})
            if name == "joint_effective" and fit["identifiable_under_ideal_observation"]:
                recovered.append(fit)
    recovery = {}
    for parameter in ("gain", "coupling", "timescale"):
        estimates = np.asarray([r[parameter] for r in recovered if r[parameter] is not None])
        recovery[parameter] = {"true_value": truth[parameter], "valid_replicates": len(estimates),
                               "mean_estimate": float(np.mean(estimates)) if len(estimates) else None,
                               "bias": float(np.mean(estimates - truth[parameter])) if len(estimates) else None,
                               "rmse": float(np.sqrt(np.mean((estimates - truth[parameter]) ** 2))) if len(estimates) else None}
    summary = []
    for name in alternatives:
        current = [r for r in reports if r["family"] == name]
        holdouts = [r["heldout_noiseless_trajectory_mse"] for r in current if r["heldout_noiseless_trajectory_mse"] is not None]
        summary.append({"family": name, "stable_replicates": len(holdouts),
                        "mean_training_one_step_mse": float(np.mean([r["training_one_step_mse"] for r in current])),
                        "mean_heldout_noiseless_trajectory_mse": float(np.mean(holdouts)) if holdouts else None})
    return {"analysis": "synthetic_linear_effective_recovery", "evidence_status": "ASSUMPTIONS_ONLY_SYNTHETIC",
            "equation": "x[t+1] = [[exp(-dt/tau), c], [c, exp(-dt/tau)]] x[t] + [g*u[t], 0] + Gaussian process noise",
            "assumptions": {"true_parameters": truth, "reference_parameters": reference,
                            "train_steps": train_steps, "holdout_steps": holdout_steps,
                            "process_noise_sd": noise, "replicates": replicates, "seed": seed,
                            "observation": "Both latent states directly observed without measurement error; known input to node 1; stable symmetric linear transition."},
            "interpretation": "A numerical reference for recoverability and different input schedules under an ideal effective model. It is not NFTsim, a neural observation mapping, or a discovered human mechanism.",
            "recovery": recovery, "family_summary": summary, "replicate_fits": reports}
