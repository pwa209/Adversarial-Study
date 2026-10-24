"""Fit and transport an explicitly qualified observation-space candidate bank.

This operation compares fixed generative predictions with a participant-level
neural phenotype. Mapping qualification is an asserted upstream provenance
requirement, not a validation performed by this module. No simulator, neural
forward map, unique synapse, or causal human mechanism is established here.
"""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math

import numpy as np


NEURAL_FEATURES = {
    "relative_delta", "relative_theta", "relative_alpha", "relative_beta", "relative_gamma",
    "effective_rank", "acf_100ms", "acf_500ms", "spectral_entropy", "temporal_entropy",
    "line_length", "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
    "permutation_entropy", "sample_entropy", "recurrence", "persistence", "recovery_time",
}
NONNEURAL_FEATURES = {"gcs", "score", "preclip_gcs", "saturated_fraction", "capacity_value",
                     "reaction_time", "accuracy", "report_category", "awareness", "experience"}


def _finite(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _flag(value, name: str) -> bool:
    if value in (True, 1, "1", "true", "True", "TRUE"):
        return True
    if value in (False, 0, "0", "false", "False", "FALSE"):
        return False
    raise ValueError(f"{name} must be an explicit boolean")


def _spec(feature_columns: list[str], settings: dict) -> tuple[list[str], float, float]:
    if not isinstance(feature_columns, list) or not feature_columns or any(
            not isinstance(c, str) or not c for c in feature_columns):
        raise ValueError("Specify a nonempty list of observation-space feature columns")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Observation-space feature columns must be distinct")
    neural = settings.get("neural_feature_columns", [c for c in feature_columns if c in NEURAL_FEATURES])
    if not isinstance(neural, list) or not neural or set(neural) - set(feature_columns) or set(neural) & NONNEURAL_FEATURES:
        raise ValueError("At least one validated actual neural feature is required; detector-score-only fitting is prohibited")
    tolerance = _finite(settings.get("equivalence_tolerance", .05))
    extrapolation = _finite(settings.get("extrapolation_tolerance", 0.))
    if tolerance is None or tolerance < 0 or extrapolation is None or extrapolation < 0:
        raise ValueError("Equivalence and extrapolation tolerances must be finite and nonnegative")
    return list(neural), tolerance, extrapolation


def _observations(rows: list[dict], columns: list[str]) -> dict:
    if not rows:
        raise ValueError("An empirical observation table is required")
    missing_schema = set(columns) - set().union(*(set(row) for row in rows))
    if missing_schema:
        raise ValueError(f"Observation-space feature columns unavailable: {sorted(missing_schema)}")
    exposed, per_person, source_by_person, missing, exclusions = set(), {}, {}, Counter(), Counter()
    for row in rows:
        person = row.get("participant_id")
        if person is None or not str(person).strip():
            raise ValueError("Every empirical row requires a global participant_id")
        person = str(person)
        exposed.add(person)
        source_by_person.setdefault(person, set()).add(str(row.get("source_id") or "unspecified"))
        for field in ("eligible", "qc_pass"):
            if field in row and not _flag(row[field], field):
                exclusions[f"{field}_false_rows"] += 1
                break
        else:
            values = [_finite(row.get(column)) for column in columns]
            if any(value is None for value in values):
                exclusions["missing_or_nonfinite_feature_rows"] += 1
                missing.update(column for column, value in zip(columns, values) if value is None)
                continue
            per_person.setdefault(person, []).append(values)
    people = sorted(per_person)
    if not people:
        raise ValueError("No participants have complete observation-space features")
    with np.errstate(over="raise", invalid="raise"):
        try:
            values = np.asarray([np.mean(per_person[p], axis=0) for p in people], dtype=float)
        except FloatingPointError as exc:
            raise ValueError("Empirical feature units exceed numerical range") from exc
    return {"participant_ids": people, "exposed_participant_ids": sorted(exposed),
            "values": values, "source_by_person": source_by_person,
            "row_counts": {p: len(per_person[p]) for p in people},
            "input_rows": len(rows), "complete_case_rows": sum(len(v) for v in per_person.values()),
            "exclusions": dict(exclusions), "missing_by_column": dict(missing)}


def _candidates(rows: list[dict], columns: list[str]) -> list[dict]:
    if not rows:
        raise ValueError("A nonempty qualified candidate bank is required")
    result, ids = [], set()
    for row in rows:
        candidate_id, family = row.get("candidate_id"), row.get("mechanism_family")
        if not isinstance(candidate_id, str) or not candidate_id.strip() or candidate_id in ids:
            raise ValueError("Candidate IDs must be nonempty unique strings")
        if not isinstance(family, str) or not family.strip():
            raise ValueError("Every candidate requires a mechanism_family")
        ids.add(candidate_id)
        if row.get("mapping_status") != "qualified" or not isinstance(row.get("mapping_reference"), str) or not row["mapping_reference"].strip():
            raise ValueError("Every candidate requires mapping_status='qualified' and a nonempty mapping_reference")
        parameters = row.get("parameter_values")
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid parameter_values JSON for {candidate_id}") from exc
        if not isinstance(parameters, dict):
            raise ValueError("Candidate parameter_values must be an explicit dictionary or JSON object")
        try:
            json.dumps(parameters, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError("Candidate parameters must be finite JSON values") from exc
        predictions = [_finite(row.get(column)) for column in columns]
        if any(value is None for value in predictions):
            raise ValueError(f"Candidate {candidate_id} lacks finite predictions for all named features")
        result.append({"candidate_id": candidate_id, "mechanism_family": family,
                       "parameter_values": copy.deepcopy(parameters),
                       "feature_predictions": predictions, "mapping_status": "qualified",
                       "mapping_reference": row["mapping_reference"],
                       "simulator_status": str(row.get("simulator_status") or "not_verified_here"),
                       "simulator_reference": str(row.get("simulator_reference") or "unspecified")})
    return sorted(result, key=lambda row: row["candidate_id"])


def _envelope(values: np.ndarray, low: np.ndarray, high: np.ndarray,
              scale: np.ndarray, columns: list[str], people: list[str], tolerance: float) -> dict:
    distance = np.maximum((low - values) / scale, (values - high) / scale)
    outside = distance > tolerance
    return {"method": "Axis-aligned candidate feature range; this is not a test of joint manifold support.",
            "tolerance_in_training_scale_units": tolerance,
            "participants_beyond_envelope": int(outside.any(axis=1).sum()),
            "total_complete_case_participants": len(people),
            "counts_by_feature": {c: int(outside[:, i].sum()) for i, c in enumerate(columns)},
            "participant_flags": [{"participant_id": p,
                                   "outside_features": [c for i, c in enumerate(columns) if outside[j, i]],
                                   "maximum_exceedance_in_training_scale_units": float(max(0., distance[j].max()))}
                                  for j, p in enumerate(people)],
            "handling": "Out-of-envelope observations retained and disclosed; candidate ranges are not selection criteria."}


def fit_candidate_bank(empirical_rows: list[dict], candidate_rows: list[dict],
                       feature_columns: list[str], settings: dict | None = None) -> dict:
    """Select each family's development fit and retain explicit ambiguity sets.

    Windows/episodes average within participant first; each participant then
    receives equal weight. All inspected development people remain exposure
    IDs, including rows excluded from fitting. Predictions must already be in
    the same qualified observation space and units as the empirical features.
    """
    settings = dict(settings or {})
    neural, tolerance, extrapolation = _spec(feature_columns, settings)
    for row in empirical_rows:
        role = str(row.get("role") or row.get("partition") or "development").lower()
        if role in {"evaluation", "diagnostic", "protected", "holdout", "independent"}:
            raise ValueError("Development candidate selection cannot use protected observations")
    observations = _observations(empirical_rows, feature_columns)
    people, values = observations["participant_ids"], observations["values"]
    if len(people) < 2:
        raise ValueError("At least two complete-case independent development participants are required")
    candidates = _candidates(candidate_rows, feature_columns)
    prediction_matrix = np.asarray([candidate["feature_predictions"] for candidate in candidates])
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean = values.mean(axis=0)
            raw_scale = values.std(axis=0)
            scale = np.where(raw_scale < 1e-12, 1., raw_scale)
            errors = np.asarray([np.mean(((values - prediction) / scale) ** 2)
                                 for prediction in prediction_matrix])
            phenotype_errors = np.mean(((mean - prediction_matrix) / scale) ** 2, axis=1)
    except FloatingPointError as exc:
        raise ValueError("Candidate or empirical feature units exceed numerical range") from exc
    scored = [{**candidate, "development_normalized_mse": float(error),
               "development_group_mean_normalized_mse": float(phenotype_error)}
              for candidate, error, phenotype_error in zip(candidates, errors, phenotype_errors)]
    best = min(scored, key=lambda r: (r["development_normalized_mse"], r["candidate_id"]))
    equivalent = [r["candidate_id"] for r in scored
                  if r["development_normalized_mse"] <= best["development_normalized_mse"] + tolerance]
    families = []
    for name in sorted({r["mechanism_family"] for r in scored}):
        current = [r for r in scored if r["mechanism_family"] == name]
        representative = min(current, key=lambda r: (r["development_normalized_mse"], r["candidate_id"]))
        retained = [r["candidate_id"] for r in current
                    if r["development_normalized_mse"] <= representative["development_normalized_mse"] + tolerance]
        families.append({"mechanism_family": name, "selected_candidate_id": representative["candidate_id"],
                         "development_normalized_mse": representative["development_normalized_mse"],
                         "equivalent_candidate_ids": retained})
    low, high = prediction_matrix.min(axis=0), prediction_matrix.max(axis=0)
    artifact = {"artifact_schema_version": 1, "artifact_kind": "frozen_observation_space_candidate_bank",
                "feature_columns": list(feature_columns), "neural_feature_columns": neural,
                "training_participant_ids": observations["exposed_participant_ids"],
                "fitted_participant_ids": people,
                "training_source_ids": sorted(set().union(*observations["source_by_person"].values())),
                "training_scaling": {"feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
                                     "constant_features": [c for c, s in zip(feature_columns, raw_scale) if s < 1e-12]},
                "development_group_feature_mean": mean.tolist(),
                "equivalence_tolerance": tolerance,
                "equivalence_definition": "Candidates within an absolute normalized-MSE tolerance of the best fixed candidate; computational observational ambiguity, not statistical equivalence or parameter identifiability.",
                "extrapolation_tolerance": extrapolation,
                "candidate_envelope": {"feature_min": low.tolist(), "feature_max": high.tolist()},
                "selected_candidate_id": best["candidate_id"], "equivalent_candidate_ids": equivalent,
                "family_selections": families, "candidate_bank": scored,
                "training_counts": {"input_rows": len(empirical_rows), "complete_case_rows": observations["complete_case_rows"],
                                    "input_participants": len(observations["exposed_participant_ids"]), "complete_case_participants": len(people)},
                "training_exclusions": observations["exclusions"], "training_missing_by_column": observations["missing_by_column"],
                "development_extrapolation": _envelope(values, low, high, scale, feature_columns, people, extrapolation),
                "fit_definition": "Equal-participant mean of mean-squared observation-space prediction errors divided by squared development participant feature scales; fixed bank candidates only.",
                "mapping_provenance": "Qualified mapping statuses and references are required assertions supplied by the bank producer; their scientific validity is not established by this function.",
                "interpretation": "A constrained generative-bank comparison and retained observational ambiguity classes. No native NFTsim execution, unique synaptic cause, or causal human mechanism is established."}
    json.dumps(artifact, allow_nan=False)
    return artifact


def evaluate_candidate_bank(empirical_rows: list[dict], artifact: dict) -> dict:
    """Evaluate frozen family representatives and ambiguity sets without refit.

    The parent workflow must freeze the artifact hash and qualify independent
    cohorts. Protected errors cannot select new bank candidates or rescale
    features. Every family retains the same development representative.
    """
    if artifact.get("artifact_schema_version") != 1 or artifact.get("artifact_kind") != "frozen_observation_space_candidate_bank":
        raise ValueError("Unsupported frozen candidate-bank artifact")
    required = {"feature_columns", "neural_feature_columns", "equivalence_tolerance", "extrapolation_tolerance",
                "training_participant_ids", "training_scaling", "candidate_bank", "candidate_envelope", "family_selections"}
    if required - set(artifact):
        raise ValueError(f"Incomplete frozen candidate-bank artifact: {sorted(required - set(artifact))}")
    columns = artifact["feature_columns"]
    _neural, _tolerance, extrapolation = _spec(columns, artifact)
    if not isinstance(artifact["training_participant_ids"], list) or not artifact["training_participant_ids"]:
        raise ValueError("Frozen bank must retain training participant exposure IDs")
    observations = _observations(empirical_rows, columns)
    overlap = set(observations["exposed_participant_ids"]) & set(artifact["training_participant_ids"])
    if overlap:
        raise ValueError(f"Protected candidate evaluation overlaps inspected training participants: {sorted(overlap)}")
    scale = np.asarray(artifact["training_scaling"].get("feature_scale"), dtype=float)
    mean = np.asarray(artifact["training_scaling"].get("feature_mean"), dtype=float)
    low = np.asarray(artifact["candidate_envelope"].get("feature_min"), dtype=float)
    high = np.asarray(artifact["candidate_envelope"].get("feature_max"), dtype=float)
    if any(array.shape != (len(columns),) or not np.isfinite(array).all() for array in (scale, mean, low, high)) or np.any(scale <= 0) or np.any(low > high):
        raise ValueError("Frozen scaling and feature envelope must contain correctly sized finite arrays with positive scales")
    if any(not isinstance(row.get("feature_predictions"), list) or len(row["feature_predictions"]) != len(columns)
           for row in artifact["candidate_bank"]):
        raise ValueError("Frozen candidate predictions must match the named feature columns exactly")
    bank_rows = [{**row, **dict(zip(columns, row["feature_predictions"]))} for row in artifact["candidate_bank"]]
    candidates = _candidates(bank_rows, columns)
    lookup = {row["candidate_id"]: row for row in candidates}
    values, people = observations["values"], observations["participant_ids"]
    family_results = []
    for family in artifact["family_selections"]:
        name = family["mechanism_family"]
        selected = family["selected_candidate_id"]
        retained = family["equivalent_candidate_ids"]
        if selected not in retained or any(candidate not in lookup or lookup[candidate]["mechanism_family"] != name for candidate in retained):
            raise ValueError("Frozen family candidate selections are inconsistent with their bank")
        candidate_results = []
        for candidate_id in retained:
            candidate = lookup[candidate_id]
            predictions = np.asarray(candidate["feature_predictions"])
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    participant_feature_errors = ((values - predictions) / scale) ** 2
                    participant_errors = participant_feature_errors.mean(axis=1)
            except FloatingPointError as exc:
                raise ValueError("Protected feature units exceed numerical range") from exc
            candidate_results.append({"candidate_id": candidate_id, "parameter_values": candidate["parameter_values"],
                                      "feature_predictions": candidate["feature_predictions"],
                                      "heldout_normalized_mse": float(participant_errors.mean()),
                                      "heldout_error_by_feature": {c: float(v) for c, v in zip(columns, participant_feature_errors.mean(axis=0))},
                                      "participant_errors": [{"participant_id": p, "normalized_mse": float(error)}
                                                             for p, error in zip(people, participant_errors)]})
        representative = next(row for row in candidate_results if row["candidate_id"] == selected)
        family_results.append({"mechanism_family": name, "frozen_selected_candidate_id": selected,
                               "heldout_normalized_mse": representative["heldout_normalized_mse"],
                               "frozen_equivalent_candidate_ids": retained, "candidate_errors": candidate_results})
    artifact_digest = hashlib.sha256(json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    result = {"analysis": "frozen_generative_bank_evaluation", "artifact_content_sha256": artifact_digest,
              "feature_columns": columns, "independent_unit": "global participant_id",
              "evaluation_participant_ids": observations["exposed_participant_ids"],
              "counts": {"input_rows": len(empirical_rows), "complete_case_rows": observations["complete_case_rows"],
                         "input_participants": len(observations["exposed_participant_ids"]), "complete_case_participants": len(people)},
              "exclusions": observations["exclusions"], "missing_by_column": observations["missing_by_column"],
              "observed_group_feature_mean": {c: float(v) for c, v in zip(columns, values.mean(axis=0))},
              "observed_participant_feature_means": [{"participant_id": p, "complete_case_rows": observations["row_counts"][p],
                                                     "features": {c: float(v) for c, v in zip(columns, value)}}
                                                    for p, value in zip(people, values)],
              "family_evaluation": family_results,
              "extrapolation": _envelope(values, low, high, scale, columns, people, extrapolation),
              "operation": "Original family selections and development scaling retained; no protected refitting or candidate reselection.",
              "uncertainty": "Descriptive participant-level errors; no interval or parameter-recovery inference implemented. Candidate ambiguity is not statistical equivalence.",
              "interpretation": "Heldout comparison of qualified observation-space predictions. Participant nonoverlap is checked; cohort and mapping validity require upstream qualification. Predictive errors do not identify a unique synapse or establish causality."}
    json.dumps(result, allow_nan=False)
    return result
