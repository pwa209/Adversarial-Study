"""Fixed-split human adaptation of discrete, genuinely simulated biological banks.

Native configurations and their clinical scores remain unchanged. Training
people determine feature scaling, effective observation offsets and supported
discrete parameter configurations. Validation chooses an offset strength and a
preferred account before test measurements are evaluated. All accounts, weak
fits, domain failures and near-equivalent predictions are retained. These are
previously analysed development data, not independent human validation, a scalp
forward model, neural weight fine-tuning or physiological identification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import core, model_guided as guided
from .core import StudyError

ASSAY_ID = "human_temporal_dynamics_proxy_v1"
SYSTEM_ID = "biological_discrete_parameter_adaptation"
SEED = 20261004
ALPHAS = (0., .5, 1.)
FEATURES = guided.TEMPORAL_FEATURES
RULE = "SHA256(seed NUL participant_id) ordering; final ceil(15%) validation, preceding ceil(15%) test, remainder train; fewer than six people remain training-only"


def participant_split(people, seed=SEED):
    identities = sorted(set(people))
    if len(identities) != len(people) or any(not isinstance(p, str) or not p for p in identities):
        raise StudyError("Participant identities must be unique nonempty strings")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise StudyError("Split seed must be a nonnegative integer")
    ordered = sorted(identities, key=lambda p: hashlib.sha256((str(seed)+"\0"+p).encode()).hexdigest())
    n = math.ceil(len(ordered)*.15) if len(ordered) >= 6 else 0
    train, validation, test = (ordered[:-(2*n)], ordered[-n:], ordered[-2*n:-n]) if n else (ordered, [], [])
    return {"schema_version": 1, "seed": seed, "rule": RULE,
            "train_participant_ids": sorted(train), "validation_participant_ids": sorted(validation),
            "test_participant_ids": sorted(test), "status": "fixed_participant_disjoint" if n else "insufficient_participants_training_only",
            "interpretation": "Development heldout split of already analysed public human measurements; no claim of independent recruitment or unseen historical exposure"}


def _assignments(split):
    people = []
    assignments = {}
    for partition in ("train", "validation", "test"):
        values = split.get(partition+"_participant_ids")
        if not isinstance(values, list):
            raise StudyError("Malformed participant allocation")
        for value in values:
            if not isinstance(value, str) or not value or value in assignments:
                raise StudyError("Duplicate, missing or overlapping human split identity")
            assignments[value] = partition
        people.extend(values)
    if split != participant_split(people, split.get("seed")):
        raise StudyError("Human split is not the fixed outcome-blind SHA256 allocation")
    return assignments


def _read_partition(path, assignments, partition, *, identities_only=False):
    rows, exclusions, seen = [], [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"episode_id", "participant_id", "source_id", *FEATURES}
        if required-set(reader.fieldnames or []):
            raise StudyError("Human observations lack identity or shared measurement fields")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise StudyError("Malformed human observation row")
            person, episode = row["participant_id"], row["episode_id"]
            if not person or not episode or (person, episode) in seen:
                raise StudyError("Repeated or missing human observation identity")
            seen.add((person, episode))
            if identities_only:
                rows.append({"participant_id": person, "episode_id": episode, "source_id": row["source_id"]})
                continue
            if person not in assignments:
                raise StudyError("Human identity is absent from the frozen allocation")
            if assignments[person] != partition:
                continue
            if row.get("measurement_status") != "actual_multichannel_eeg" or row.get("score_origin") != "native_frozen_v2" or core.flag(row.get("synthetic")):
                raise StudyError("Parameter adaptation requires genuine scored development EEG")
            if row.get("role") and row["role"] != "development":
                raise StudyError("Protected human observations cannot enter development adaptation")
            if any(guided._finite(row.get(key)) is None for key in FEATURES):
                exclusions.append({"participant_id": person, "episode_id": episode, "source_id": row["source_id"],
                                   "split": partition, "reason": "missing_or_nonfinite_shared_temporal_measurement"})
                continue
            rows.append(row)
    return rows, exclusions


def verify_fit(directory):
    """Recheck parent hashes AND the actual native source/scoring chain."""
    root = Path(directory).resolve()
    receipt = guided._verify_output(root)
    if receipt.get("status") != "model_guided_fit_completed" or not receipt.get("inputs"):
        raise StudyError("Human adaptation requires a completed provenance-bearing mechanistic fit")
    for name in ("human_observations.csv", "scored_bank.csv", "retained_simulation_failures.csv"):
        if receipt.get("outputs", {}).get(name) != core.sha256_file(root/name):
            raise StudyError("Required fit artifact is not pinned by the source receipt")
    paths = sorted({Path(i["path"]).resolve() for i in receipt["inputs"]})
    json_inputs = []
    for path in paths:
        if path.suffix == ".json":
            json_inputs.append((path, core.read_json(path)))
    for path, value in json_inputs:
        if value.get("status") == "model_guided_trajectories_completed":
            guided._verify_output(path.parent)
    banks = [(p, r) for p, r in json_inputs if r.get("status") == "native_simulation_bank_completed"]
    if len(banks) != 1:
        raise StudyError("Parent fit must pin exactly one actual native bank receipt")
    bank_path, bank_receipt = banks[0]

    def by_hash(expected):
        matches = [p for p in paths if core.sha256_file(p) == expected]
        if len(matches) != 1:
            raise StudyError("Native bank provenance file is missing or ambiguous")
        return matches[0]

    inventory = by_hash(bank_receipt["inventory_sha256"])
    jobs = by_hash(bank_receipt["jobs_sha256"])
    score_receipts = [(p, r) for p, r in json_inputs if r.get("status") == "native_batch_completed" and r.get("jobs_sha256") == bank_receipt["jobs_sha256"]]
    if len(score_receipts) != 1:
        raise StudyError("Parent fit lacks one actual native simulation score receipt")
    score_receipt_path, scored = score_receipts[0]
    scores, qualification = by_hash(scored["scores_sha256"]), by_hash(scored["qualification_sha256"])
    if score_receipt_path != Path(str(scores)+".receipt.json"):
        raise StudyError("Native batch receipt is not paired with its score table")
    actual, failures, _ = guided._verified_bank(inventory, jobs, scores, qualification, bank_path)
    rows = core.read_csv(root/"scored_bank.csv", ("simulation_id", "account_class", "status", "score", *FEATURES))
    original = {r["simulation_id"]: r for r in actual}
    if len({r["simulation_id"] for r in rows}) != len(rows) or {r["simulation_id"] for r in rows} != set(original):
        raise StudyError("Fit candidates differ from the genuinely simulated scored bank")
    for row in rows:
        native = original[row["simulation_id"]]
        parameter_keys = {k for k in native if k.startswith("param_")}
        for key in ("account_class", "variant", "config_sha256", "source_output_path", "source_output_sha256", "score_origin", "published_config_id", *sorted(parameter_keys)):
            if row.get(key, "") != str(native.get(key, "")):
                raise StudyError("Fit candidate native identity or provenance differs")
        if row["status"] != "finite" or native["status"] != "finite":
            raise StudyError("Failed/constant native outputs cannot become fitted candidates")
        for key in (*FEATURES, "score"):
            if guided._finite(row[key]) is None or float(row[key]) != float(native[key]):
                raise StudyError("Fit candidate measurement or native clinical score differs")
    return receipt, rows, {"bank_receipt": {"path": str(bank_path), "sha256": core.sha256_file(bank_path)},
                           "score_receipt": {"path": str(score_receipt_path), "sha256": core.sha256_file(score_receipt_path)},
                           "retained_native_failures": len(failures)}


def _matrix(rows):
    return np.asarray([[float(r[k]) for k in FEATURES] for r in rows], dtype=float)


def _participant_mean(values, rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["participant_id"]].append(i)
    return np.mean([np.mean(values[indices], axis=0) for indices in groups.values()], axis=0)


def train_calibration(rows, candidates, alpha):
    """Fit offsets and supported *existing* native parameter configurations."""
    if not rows or not candidates or alpha not in ALPHAS:
        raise StudyError("Calibration needs finite training observations, actual native candidates and a fixed offset strength")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _train_calibration(rows, candidates, alpha)
    except (FloatingPointError, OverflowError) as exc:
        raise StudyError("Temporal calibration exceeded finite numerical precision") from exc


def _train_calibration(rows, candidates, alpha):
    x, bank = _matrix(rows), _matrix(candidates)
    if not np.isfinite(x).all() or not np.isfinite(bank).all():
        raise StudyError("Calibration received missing or nonfinite temporal measurements")
    mean = _participant_mean(x, rows)
    scales = np.sqrt(_participant_mean((x-mean)**2, rows))
    scales = np.where(scales > 1e-9, scales, 1.)
    offset = np.zeros(len(FEATURES))
    for _ in range(4):
        indices, _ = guided._nearest(x, bank+offset, scales)
        offset = alpha*_participant_mean(x-bank[indices], rows)
    indices, distances = guided._nearest(x, bank+offset, scales)
    supported = sorted(set(indices.tolist()), key=lambda i: candidates[i]["simulation_id"])
    counts = Counter(int(i) for i in indices)
    return {"offset_strength": alpha, "offset": offset.tolist(), "training_scale": scales.tolist(),
            "training_mean": mean.tolist(), "iterations": 4,
            "supported_parameters": [{"simulation_id": candidates[i]["simulation_id"],
                                      "training_episodes": counts[i], "config_sha256": candidates[i]["config_sha256"],
                                      "published_config_id": candidates[i].get("published_config_id"),
                                      "native_clinical_score": float(candidates[i]["score"]),
                                      "parameters": {k:v for k,v in candidates[i].items() if k.startswith("param_")}}
                                     for i in supported],
            "train_standardized_fit_distance": float(_participant_mean(distances, rows)),
            "fit_interpretation": "Discrete configurations and an effective measurement offset fitted to training temporal summaries; parameters are relative to their recorded published configuration, not identified human physiology"}


def predict(rows, candidates, fitted, account, variant, partition):
    """Predict each held feature from four other features; target never selects parameters."""
    if not rows:
        return [], {"status": "no_finite_partition_measurements", "participants": 0, "episodes": 0,
                    "equal_participant_standardized_mse": None, "training_mean_standardized_mse": None}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _predict(rows, candidates, fitted, account, variant, partition)
    except (FloatingPointError, OverflowError) as exc:
        raise StudyError("Heldout temporal prediction exceeded finite numerical precision") from exc


def _predict(rows, candidates, fitted, account, variant, partition):
    by_id = {r["simulation_id"]:r for r in candidates}
    support = [by_id[p["simulation_id"]] for p in fitted["supported_parameters"]]
    bank = _matrix(support)+np.asarray(fitted["offset"])
    x = _matrix(rows)
    scale, mean = np.asarray(fitted["training_scale"]), np.asarray(fitted["training_mean"])
    output, errors, null_errors = [], np.zeros_like(x), np.zeros_like(x)
    for held, feature in enumerate(FEATURES):
        included = [j for j in range(len(FEATURES)) if j != held]
        indices, distance = guided._nearest(x, bank, scale, included)
        for i, row in enumerate(rows):
            prediction = float(bank[indices[i], held])
            errors[i, held] = ((prediction-x[i, held])/scale[held])**2
            null_errors[i, held] = ((mean[held]-x[i, held])/scale[held])**2
            outside = [FEATURES[j] for j in included if x[i,j] < bank[:,j].min()-.25*scale[j] or x[i,j] > bank[:,j].max()+.25*scale[j]]
            output.append({"participant_id": row["participant_id"], "episode_id": row["episode_id"], "source_id": row["source_id"],
                           "site_id": row.get("site_id", ""), "split": partition, "account_class": account, "variant": variant,
                           "held_feature": feature, "selection_features": ";".join(FEATURES[j] for j in included),
                           "observed": float(x[i, held]), "predicted": prediction, "training_mean_prediction": float(mean[held]),
                           "standardized_squared_error": float(errors[i, held]), "training_mean_standardized_squared_error": float(null_errors[i, held]),
                           "simulation_id": support[int(indices[i])]["simulation_id"], "selection_summary_distance": float(distance[i]),
                           "domain_status": "out_of_training_supported_bank_envelope" if outside else "within_training_supported_bank_envelope",
                           "out_of_bank_selection_features": ";".join(outside)})
    mse = float(np.mean(_participant_mean(errors, rows)))
    null = float(np.mean(_participant_mean(null_errors, rows)))
    return output, {"status": "effective_proxy_improves_training_mean" if mse < null else "weak_or_equivalent_to_training_mean",
                    "participants": len({r["participant_id"] for r in rows}), "episodes": len(rows),
                    "equal_participant_standardized_mse": mse, "training_mean_standardized_mse": null,
                    "out_of_domain_predictions": sum(r["domain_status"].startswith("out_") for r in output),
                    "by_source": {source: {"predictions": len(group), "participants": len({r["participant_id"] for r in group}),
                                          "equal_participant_standardized_mse": float(np.mean([np.mean([r["standardized_squared_error"] for r in group if r["participant_id"] == p]) for p in sorted({r["participant_id"] for r in group})]))}
                                  for source in sorted({r["source_id"] for r in output})
                                  for group in [[r for r in output if r["source_id"] == source]]}}


def adapt(fit, output):
    source = Path(fit).resolve()
    parent, bank, provenance = verify_fit(source)
    census, _ = _read_partition(source/"human_observations.csv", {}, None, identities_only=True)
    split = participant_split(sorted({r["participant_id"] for r in census}))
    assignments = _assignments(split)
    train, train_excluded = _read_partition(source/"human_observations.csv", assignments, "train")
    validation, val_excluded = _read_partition(source/"human_observations.csv", assignments, "validation")
    accounts = defaultdict(list)
    for row in sorted(bank, key=lambda r:r["simulation_id"]):
        accounts[(row["account_class"], row.get("variant", "unspecified"))].append(row)
    fitted_accounts, validation_predictions = [], []
    if train:
        for (account, variant), candidates in sorted(accounts.items()):
            alternatives = []
            for alpha in ALPHAS:
                fitted = train_calibration(train, candidates, alpha)
                predictions, metrics = predict(validation, candidates, fitted, account, variant, "validation")
                alternatives.append({"parameters": fitted, "validation_metrics": metrics, "predictions": predictions})
            selected = min(alternatives, key=lambda a:(a["validation_metrics"]["equal_participant_standardized_mse"]
                           if a["validation_metrics"]["equal_participant_standardized_mse"] is not None else math.inf,
                           a["parameters"]["offset_strength"]))
            fitted_accounts.append({"account_class": account, "variant": variant, **selected["parameters"],
                                    "validation_metrics": selected["validation_metrics"],
                                    "validation_alternatives": [{"offset_strength": a["parameters"]["offset_strength"],
                                                                 "metrics": a["validation_metrics"]} for a in alternatives]})
            validation_predictions.extend(selected["predictions"])
    ranked = sorted(fitted_accounts, key=lambda a:(a["validation_metrics"]["equal_participant_standardized_mse"]
                    if a["validation_metrics"]["equal_participant_standardized_mse"] is not None else math.inf,
                    a["account_class"], a["variant"]))
    selected_account = [ranked[0]["account_class"], ranked[0]["variant"]] if ranked and validation else None
    equivalence = []
    for i, a in enumerate(ranked):
        for b in ranked[i+1:]:
            ma, mb = (r["validation_metrics"]["equal_participant_standardized_mse"] for r in (a,b))
            if ma is not None and mb is not None:
                relative = abs(ma-mb)/max(ma, mb, 1e-9)
                equivalence.append({"accounts": [[a["account_class"], a["variant"]], [b["account_class"], b["variant"]]],
                                    "relative_validation_error_difference": relative,
                                    "status": "descriptively_similar_at_25_percent_resolution" if relative<=.25 else "different_validation_errors",
                                    "statistical_equivalence_established": False})
    out = guided._new_output(output)
    fields = ["participant_id", "episode_id", "source_id", "site_id", *FEATURES]
    core.write_json(out/"dataset_split.json", split)
    core.write_csv(out/"train_examples.csv", [{k:r.get(k, "") for k in fields} for r in train], fields)
    core.write_csv(out/"validation_examples.csv", [{k:r.get(k, "") for k in fields} for r in validation], fields)
    core.write_csv(out/"validation_predictions.csv", validation_predictions, _prediction_fields())
    parameter_fit = {"schema_version": 1, "feature_definition": guided.FEATURE_DEFINITION,
                     "status": "human_discrete_parameter_adaptation_frozen" if fitted_accounts and validation else "insufficient_adaptation_evidence",
                     "accounts": fitted_accounts, "validation_selected_account": selected_account,
                     "descriptive_validation_equivalence": equivalence,
                     "training_participant_ids": split["train_participant_ids"], "validation_participant_ids": split["validation_participant_ids"],
                     "selection_rule": "Minimum equal-participant five-held-feature validation MSE; weaker offset then sorted account identities break ties; retain every account",
                     "scaling_and_calibration": "Training participants only; no validation refit or test-driven selection",
                     "observation_mapping": guided.MAPPING_INTERPRETATION,
                     "human_functional_mapping_qualified": False, "physiological_parameter_identification_established": False,
                     "absolute_amplitude_fitted": False, "native_bank_and_scores_modified": False}
    core.write_json(out/"parameter_fit.json", parameter_fit)
    completed = bool(train and validation and split["test_participant_ids"] and fitted_accounts)
    training = {"schema_version": 1, "status": "completed_actual_human_parameter_adaptation" if completed else "insufficient_human_parameter_adaptation",
                "system_id": SYSTEM_ID, "human_data": True, "assay_id": ASSAY_ID,
                "human_assay": {"assay_id": ASSAY_ID, "target": "Five scale-free temporal EEG measurements predicted individually from the other four",
                                "input_modality": "Source-selected human EEG summaries and actual single-node native cortical proxy summaries",
                                "human_functional_mapping_qualified": False},
                "split": split, "data_split_sha256": core.sha256_file(out/"dataset_split.json"),
                "train_examples_sha256": core.sha256_file(out/"train_examples.csv"),
                "validation_examples_sha256": core.sha256_file(out/"validation_examples.csv"),
                "validation_predictions_sha256": core.sha256_file(out/"validation_predictions.csv"),
                "trained_checkpoint": {"kind": "discrete_native_parameters_and_effective_observation_calibration",
                                       "path": str(out/"parameter_fit.json"), "sha256": core.sha256_file(out/"parameter_fit.json")},
                "source_fit": {"path": str(source), "receipt_sha256": core.sha256_file(source/"receipt.json")},
                "native_provenance": provenance, "example_counts_before_test": {"train": len(train), "validation": len(validation)},
                "native_training_executed": False, "human_parameter_fitting_executed": completed, "adaptation_completed": completed,
                "adaptation_kind": "constrained_discrete_parameter_adaptation", "optimizer_steps": 0,
                "pretrained_exposure": {"status": "unknown", "earlier_joint_fit_analysis": True,
                                       "interpretation": "Source people already contributed to the development joint fit; this split re-estimates adaptation from training people only and cannot certify independent historical exposure"},
                "frozen_before_test_evaluation_utc": datetime.now(timezone.utc).isoformat(),
                "test_access": "Test temporal values not used for this stage's scaling, calibration, support selection, validation or account selection; source CSV and earlier development analyses already contain these observations",
                "no_result_direction_or_significance_gate": True}
    core.write_json(out/"training_receipt.json", training)
    # This is the first partition-specific test measurement read. Its source is
    # already exposed development data; the freeze does not claim historical blinding.
    test, test_excluded = _read_partition(source/"human_observations.csv", assignments, "test")
    predictions, evaluation_accounts = [], []
    for fitted in fitted_accounts:
        key = (fitted["account_class"], fitted["variant"])
        rows, metrics = predict(test, accounts[key], fitted, *key, "test")
        predictions.extend(rows)
        evaluation_accounts.append({"account_class": key[0], "variant": key[1], "metrics": metrics,
                                    "validation_selected": list(key) == selected_account})
    core.write_csv(out/"heldout_predictions.csv", predictions, _prediction_fields())
    core.write_csv(out/"excluded_observations.csv", train_excluded+val_excluded+test_excluded,
                   ["participant_id", "episode_id", "source_id", "split", "reason"])
    evaluation = {"schema_version": 1, "status": "completed_development_heldout_parameter_prediction" if completed and test else "insufficient_heldout_parameter_evaluation",
                  "system_id": SYSTEM_ID, "assay_id": ASSAY_ID, "data_split_sha256": training["data_split_sha256"],
                  "training_receipt_sha256": core.sha256_file(out/"training_receipt.json"),
                  "parameter_fit_sha256": training["trained_checkpoint"]["sha256"],
                  "test_predictions": {"path": str(out/"heldout_predictions.csv"), "sha256": core.sha256_file(out/"heldout_predictions.csv")},
                  "test_participant_ids": sorted({r["participant_id"] for r in test}), "test_episodes": len(test), "accounts": evaluation_accounts,
                  "independent_human_validation": False, "human_functional_mapping_qualified": False,
                  "interpretation": "Fixed-split development prediction with equal-participant errors and training-mean comparator; no physiological identification, causal explanation, full scalp forward map or shared behavioural assay is established"}
    core.write_json(out/"evaluation_receipt.json", evaluation)
    return guided._finish(out, [source/"receipt.json", source/"human_observations.csv", source/"scored_bank.csv", source/"retained_simulation_failures.csv"],
                          "human_generative_adaptation_completed", assay_id=ASSAY_ID, human_data=True,
                          human_parameter_fitting_executed=completed, native_training_executed=False,
                          heldout_evaluation_executed=bool(completed and test), human_functional_mapping_qualified=False,
                          denominator_episodes=len(census), retained_native_failures=provenance["retained_native_failures"],
                          missing_measurement_episodes=len(train_excluded)+len(val_excluded)+len(test_excluded))


def _prediction_fields():
    return ["participant_id", "episode_id", "source_id", "site_id", "split", "account_class", "variant", "held_feature", "selection_features",
            "observed", "predicted", "training_mean_prediction", "standardized_squared_error", "training_mean_standardized_squared_error",
            "simulation_id", "selection_summary_distance", "domain_status", "out_of_bank_selection_features"]


def verify_adaptation(directory):
    """Verify actual adaptation artifacts without requiring favourable heldout results."""
    root = Path(directory).resolve()
    receipt = guided._verify_output(root)
    if receipt.get("status") != "human_generative_adaptation_completed":
        raise StudyError("Missing completed human parameter adaptation stage")
    training, evaluation = core.read_json(root/"training_receipt.json"), core.read_json(root/"evaluation_receipt.json")
    split = core.read_json(root/"dataset_split.json")
    assignments = _assignments(split)
    if training.get("status") != "completed_actual_human_parameter_adaptation" or training.get("human_parameter_fitting_executed") is not True or training.get("human_data") is not True or training.get("adaptation_completed") is not True:
        raise StudyError("No completed actual human parameter adaptation")
    if training.get("native_training_executed") is not False or training.get("assay_id") != ASSAY_ID or training.get("split") != split:
        raise StudyError("Biological adaptation kind, assay or split differs")
    for name, key in (("dataset_split.json", "data_split_sha256"), ("train_examples.csv", "train_examples_sha256"),
                      ("validation_examples.csv", "validation_examples_sha256"), ("validation_predictions.csv", "validation_predictions_sha256")):
        if training.get(key) != core.sha256_file(root/name):
            raise StudyError("Human adaptation input/validation artifact changed")
    checkpoint = training["trained_checkpoint"]
    if Path(checkpoint["path"]).resolve() != root/"parameter_fit.json" or checkpoint["sha256"] != core.sha256_file(root/"parameter_fit.json"):
        raise StudyError("Learned discrete parameter/calibration artifact changed")
    source = Path(training["source_fit"]["path"]).resolve()
    if training["source_fit"]["receipt_sha256"] != core.sha256_file(source/"receipt.json"):
        raise StudyError("Source joint fit changed")
    _, bank, provenance = verify_fit(source)
    if training.get("native_provenance") != provenance:
        raise StudyError("Native parameter/scoring provenance differs")
    people, _ = _read_partition(source/"human_observations.csv", {}, None, identities_only=True)
    if set(assignments) != {r["participant_id"] for r in people}:
        raise StudyError("Adaptation silently dropped human people before partitioning")
    parameters = core.read_json(root/"parameter_fit.json")
    if parameters.get("training_participant_ids") != split["train_participant_ids"] or parameters.get("validation_participant_ids") != split["validation_participant_ids"]:
        raise StudyError("Learned parameter allocation differs")
    bank_by_id = {r["simulation_id"]:r for r in bank}
    bank_accounts = defaultdict(list)
    for candidate in sorted(bank, key=lambda r:r["simulation_id"]):
        bank_accounts[(candidate["account_class"], candidate.get("variant", "unspecified"))].append(candidate)
    train, _ = _read_partition(source/"human_observations.csv", assignments, "train")
    validation, _ = _read_partition(source/"human_observations.csv", assignments, "validation")
    fields = ["participant_id", "episode_id", "source_id", "site_id", *FEATURES]
    for partition, actual_rows in (("train", train), ("validation", validation)):
        exported = core.read_csv(root/(partition+"_examples.csv"), tuple(fields))
        expected_export = [{k:str(r.get(k, "")) for k in fields} for r in actual_rows]
        if exported != expected_export or training.get("example_counts_before_test", {}).get(partition) != len(actual_rows):
            raise StudyError("Actual training/validation observations differ from adaptation inputs")
    account_keys = set()
    for account in parameters.get("accounts", []):
        key = account["account_class"], account["variant"]
        if key in account_keys:
            raise StudyError("Repeated adapted account")
        account_keys.add(key)
        for parameter in account["supported_parameters"]:
            candidate = bank_by_id.get(parameter["simulation_id"])
            if candidate is None or (candidate["account_class"], candidate.get("variant", "unspecified")) != key or parameter["config_sha256"] != candidate["config_sha256"] or parameter["training_episodes"] <= 0:
                raise StudyError("Adapted parameters lack genuine finite native training support")
            if parameter.get("parameters") != {k:v for k,v in candidate.items() if k.startswith("param_")} or parameter.get("published_config_id") != candidate.get("published_config_id") or parameter.get("native_clinical_score") != float(candidate["score"]):
                raise StudyError("Supported native parameter identity or unchanged clinical score differs")
        if len(account["training_scale"]) != len(FEATURES) or not all(math.isfinite(float(s)) and float(s)>0 for s in account["training_scale"]):
            raise StudyError("Invalid training-only temporal scale")
        # Re-estimate these small numerical artifacts from actual training people.
        # This establishes training use rather than accepting a completion flag.
        reproduced = train_calibration(train, bank_accounts[key], account["offset_strength"])
        if any(account.get(field) != value for field, value in reproduced.items()):
            raise StudyError("Discrete parameter/calibration artifact was not fitted to the declared human training people")
        alternatives = []
        for alpha in ALPHAS:
            fitted = train_calibration(train, bank_accounts[key], alpha)
            _, metrics = predict(validation, bank_accounts[key], fitted, *key, "validation")
            alternatives.append((metrics["equal_participant_standardized_mse"], alpha, metrics))
        best = min(alternatives, key=lambda a:(a[0] if a[0] is not None else math.inf, a[1]))
        if best[1] != account["offset_strength"] or best[2] != account["validation_metrics"]:
            raise StudyError("Human validation parameter selection differs from the frozen rule")
    if not account_keys:
        raise StudyError("No actual biological account was adapted")
    ranked = sorted(parameters["accounts"], key=lambda a:(a["validation_metrics"]["equal_participant_standardized_mse"]
                    if a["validation_metrics"]["equal_participant_standardized_mse"] is not None else math.inf,
                    a["account_class"], a["variant"]))
    if parameters.get("validation_selected_account") != [ranked[0]["account_class"], ranked[0]["variant"]]:
        raise StudyError("Preferred biological account was not selected by validation measurements")
    if evaluation.get("status") != "completed_development_heldout_parameter_prediction" or evaluation.get("assay_id") != ASSAY_ID or evaluation.get("data_split_sha256") != training["data_split_sha256"] or evaluation.get("training_receipt_sha256") != core.sha256_file(root/"training_receipt.json"):
        raise StudyError("Actual frozen-split heldout biological evaluation is missing")
    if evaluation.get("parameter_fit_sha256") != checkpoint["sha256"] or Path(evaluation["test_predictions"]["path"]).resolve() != root/"heldout_predictions.csv" or evaluation["test_predictions"]["sha256"] != core.sha256_file(root/"heldout_predictions.csv"):
        raise StudyError("Heldout biological predictions or their selected parameters changed")
    test, _ = _read_partition(source/"human_observations.csv", assignments, "test")
    predictions = core.read_csv(root/"heldout_predictions.csv", tuple(_prediction_fields()))
    expected = {(r["participant_id"], r["episode_id"], *key, feature) for r in test for key in account_keys for feature in FEATURES}
    observed = set()
    for row in predictions:
        identity = (row["participant_id"], row["episode_id"], row["account_class"], row["variant"], row["held_feature"])
        if identity in observed or assignments.get(row["participant_id"]) != "test" or row["split"] != "test" or set(row["selection_features"].split(";")) != set(FEATURES)-{row["held_feature"]}:
            raise StudyError("Heldout predictions overlap, leak held targets or cross participant partitions")
        observed.add(identity)
    if observed != expected or evaluation.get("test_episodes") != len(test):
        raise StudyError("Heldout evaluation silently omitted observations, accounts or features")
    reproduced_predictions, reproduced_metrics = [], []
    for account in parameters["accounts"]:
        key = account["account_class"], account["variant"]
        rows, metrics = predict(test, bank_accounts[key], account, *key, "test")
        reproduced_predictions.extend(rows)
        reproduced_metrics.append({"account_class": key[0], "variant": key[1], "metrics": metrics,
                                   "validation_selected": list(key) == parameters["validation_selected_account"]})
    if predictions != [{k:str(r.get(k, "")) for k in _prediction_fields()} for r in reproduced_predictions] or evaluation.get("accounts") != reproduced_metrics:
        raise StudyError("Heldout predictions are not generated by the frozen human-adapted native parameters")
    return training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = adapt(args.fit, args.output)
        print(json.dumps({"status": result["status"], "human_parameter_fitting_executed": result["human_parameter_fitting_executed"]}))
    except (StudyError, OSError, KeyError, OverflowError, FloatingPointError) as exc:
        parser.exit(2, "Human generative adaptation failed: "+str(exc)+"\n")


if __name__ == "__main__":
    main()
