"""Additional development analyses completing the proposal's mechanistic comparisons.

Legacy scientific artifacts remain immutable. Actual native configurations and
previously fitted effective observation maps support balanced adversarial search,
known-generator recovery with map refitting, corrected human discrepancy tests,
and the original clinical-score baseline. Native-to-human physical and functional
qualification is still separate work. Recovery observations are explicitly model
generated diagnostics, never additional human data or identified synapses.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from . import core, manifold_mechanisms as mm, model_guided as guided
from . import representation_bridge as rb

VERSION = "proposal_mechanisms_v1"
DEFAULTS = {
    "schema_version": 1, "seed": 20261008, "candidate_budget": 192,
    "score_tolerance": .02, "conventional_tolerance": .5, "dynamic_difference": .5,
    "geometry_preservation_tolerance": .1, "recovery_case_budget": 16,
    "recovery_replicates": 2, "noise_levels": [0., .05, .2],
    "noise_models": ["white", "colored_ar1"], "colored_noise_coefficient": .6,
    "misspecification_levels": [0., .2], "recovery_calibration_units": 6,
    "recovery_validation_units": 3, "recovery_equivalence_tolerance": .01,
    "recovery_ridge_grid": [.001, .01, .1], "bootstrap_samples": 500,
    "minimum_training_people": 3, "minimum_validation_people": 1,
    "minimum_test_people": 2, "require_scheduled_execution": True,
    "result_direction_gate": False,
}


def settings(path):
    supplied = core.read_json(path)
    if set(supplied) - set(DEFAULTS):
        raise core.StudyError("Unknown proposal mechanism options")
    cfg = DEFAULTS | supplied
    if cfg["schema_version"] != 1 or cfg["result_direction_gate"] is not False:
        raise core.StudyError("Unsupported design or result-direction gate")
    for key, low, high in (("candidate_budget", 1, 2000), ("recovery_case_budget", 1, 128),
                          ("recovery_replicates", 1, 10), ("recovery_calibration_units", 3, 20),
                          ("recovery_validation_units", 1, 10), ("bootstrap_samples", 20, 2000),
                          ("minimum_training_people", 2, 100), ("minimum_validation_people", 1, 100),
                          ("minimum_test_people", 2, 100), ("seed", 0, 2**32-1)):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or not low <= cfg[key] <= high:
            raise core.StudyError("Invalid bounded option " + key)
    for key in ("score_tolerance", "conventional_tolerance", "dynamic_difference",
                "geometry_preservation_tolerance", "recovery_equivalence_tolerance"):
        if guided._finite(cfg[key]) is None or not 0 < float(cfg[key]) <= 10:
            raise core.StudyError("Invalid positive tolerance " + key)
    for key in ("noise_levels", "misspecification_levels"):
        if not isinstance(cfg[key], list) or not cfg[key] or len(cfg[key]) > 6 or any(guided._finite(v) is None or not 0 <= v <= 2 for v in cfg[key]):
            raise core.StudyError("Invalid bounded synthetic diagnostic levels")
    if not cfg["noise_models"] or not set(cfg["noise_models"]) <= {"white", "colored_ar1"} or not 0 <= cfg["colored_noise_coefficient"] < .95:
        raise core.StudyError("Invalid declared synthetic noise model")
    if not cfg["recovery_ridge_grid"] or len(cfg["recovery_ridge_grid"]) > 6 or any(guided._finite(v) is None or not 0 < v <= 10 for v in cfg["recovery_ridge_grid"]):
        raise core.StudyError("Invalid recovery model-selection grid")
    if not isinstance(cfg["require_scheduled_execution"], bool):
        raise core.StudyError("Scheduled-execution option must be Boolean")
    return cfg


def _identity(value, seed):
    return hashlib.sha256((str(seed) + "\0" + json.dumps(value, sort_keys=True)).encode()).hexdigest()


def balanced_budget(proposals, budget, seed, stratum_fields):
    """Metadata-only round robin; no score, support, fit or outcome orders selection."""
    groups = defaultdict(list)
    for row in proposals:
        key = tuple(str(row[field]) for field in stratum_fields)
        groups[key].append(row)
    for key, rows in groups.items():
        rows.sort(key=lambda r: _identity(r["candidate_id"], seed))
    keys = sorted(groups, key=lambda k: _identity(k, seed))
    admitted = []
    for rank in range(max((len(v) for v in groups.values()), default=0)):
        for key in keys:
            if rank < len(groups[key]) and len(admitted) < budget:
                admitted.append(groups[key][rank])
    selected = {r["candidate_id"] for r in admitted}
    census = [{"stratum": json.dumps(key), "proposed": len(rows),
               "admitted": sum(r["candidate_id"] in selected for r in rows),
               "excluded_metadata_budget": sum(r["candidate_id"] not in selected for r in rows)}
              for key, rows in sorted(groups.items())]
    return admitted, census


def _csv(path, rows, minimum):
    core.write_csv(path, rows, sorted(set(minimum) | {k for r in rows for k in r}))


def _scales(rows):
    """Equal-person training scales, never using held-person outcomes."""
    people = defaultdict(list)
    for row in rows:
        if row.get("split") == "train" and all(guided._finite(row.get(k)) is not None for k in guided.TEMPORAL_FEATURES):
            people[row["participant_id"]].append([float(row[k]) for k in guided.TEMPORAL_FEATURES])
    if not people:
        return None
    mean = np.mean([np.mean(v, axis=0) for v in people.values()], axis=0)
    variance = np.mean([np.mean((np.asarray(v)-mean)**2, axis=0) for v in people.values()], axis=0)
    return dict(zip(guided.TEMPORAL_FEATURES, np.maximum(np.sqrt(variance), 1e-6)))


def _distance(a, b, fields, scales):
    if scales is None or any(guided._finite(a.get(k)) is None or guided._finite(b.get(k)) is None for k in fields):
        return None
    return max(abs(float(a[k])-float(b[k]))/scales[k] for k in fields)


def _unsaturated(row):
    value = guided._finite(row.get("saturated_fraction"))
    return value is not None and value == 0


def search_families(candidates, bank, source_scales, cfg):
    """Clinical judgments remain native; distinct search questions retain failures."""
    lookup = {r["simulation_id"]: r for r in bank}
    output, disposition = [], []
    for candidate in candidates:
        control, pulse = lookup[candidate["control_simulation_id"]], lookup[candidate["pulse_simulation_id"]]
        scales = source_scales.get(candidate["source_model_key"])
        dynamic = _distance(control, pulse, guided.DYNAMIC_FEATURES, scales)
        preservation = candidate.get("control_pulse_geometry_distance")
        score_change = abs(float(pulse["score"])-float(control["score"]))
        property_match = dynamic is not None and preservation is not None and score_change > cfg["score_tolerance"] and dynamic < cfg["dynamic_difference"] and preservation <= cfg["geometry_preservation_tolerance"]
        supported = candidate["status"] == "retained_supported_computational_challenge"
        accepted = property_match and supported
        record = {"family": "changed_score_preserved_specified_property", "candidate_a": candidate["candidate_id"],
                  "candidate_b": candidate["candidate_id"], "source_model_key": candidate["source_model_key"],
                  "clinical_score_difference": score_change, "standardized_dynamic_difference": dynamic,
                  "temporal_geometry_difference": preservation,
                  "property": "native_100ms_500ms_autocorrelation_and_mapped_temporal_Gram_geometry",
                  "status": "eligible_computational_contrast" if accepted else "retained_unsupported_property_contrast" if property_match else "does_not_meet_frozen_family_definition"}
        disposition.append(record)
        if accepted: output.append(record)
    for a, b in itertools.combinations(candidates, 2):
        if a["source_model_key"] != b["source_model_key"] or a["variant"] != b["variant"]:
            continue
        ca, cb = lookup[a["control_simulation_id"]], lookup[b["control_simulation_id"]]
        scale = source_scales.get(a["source_model_key"])
        conventional = _distance(ca, cb, guided.CONVENTIONAL_FEATURES, scale)
        dynamic = _distance(ca, cb, guided.DYNAMIC_FEATURES, scale)
        score = abs(float(ca["score"])-float(cb["score"]))
        base = {"candidate_a": a["candidate_id"], "candidate_b": b["candidate_id"],
                "source_model_key": a["source_model_key"], "account_a": a["account_class"], "account_b": b["account_class"],
                "clinical_score_difference": score, "standardized_conventional_difference": conventional,
                "standardized_dynamic_difference": dynamic, "unsaturated": _unsaturated(ca) and _unsaturated(cb),
                "response_endpoint_difference": float(a["latent_response_path_rms_v1"])-float(b["latent_response_path_rms_v1"]),
                "both_joint_support": a["status"] == b["status"] == "retained_supported_computational_challenge"}
        matched = score <= cfg["score_tolerance"] and conventional is not None and conventional <= cfg["conventional_tolerance"] and base["unsaturated"]
        if matched and dynamic is not None and dynamic >= cfg["dynamic_difference"]:
            record = base | {"family": "matched_clinical_score_conventional_features_different_dynamics", "status": "eligible_computational_contrast" if base["both_joint_support"] else "retained_unsupported_property_contrast"}
            disposition.append(record)
            if base["both_joint_support"]: output.append(record)
        matched_input = a["account_class"] != b["account_class"] and float(a["input_amplitude_model_units"]) == float(b["input_amplitude_model_units"]) and float(a["input_width_s"]) == float(b["input_width_s"])
        if matched_input:
            status = "eligible_computational_contrast" if matched and base["both_joint_support"] else "retained_unmatched_or_unsupported_account_disagreement"
            row = base | {"family": "matched_input_account_disagreement", "status": status}
            disposition.append(row)
            if status == "eligible_computational_contrast": output.append(row)
    return output, disposition


def load_maps(bridge):
    selection = core.read_json(Path(bridge)/"model_selection.json")
    with np.load(Path(bridge)/"observation_map.npz", allow_pickle=False) as archive:
        maps = []
        for item in selection["maps"]:
            m = dict(item); key = m["map_key"]
            for name in ("W", "bias", "baseline_A", "baseline_bias", "response_W", "bootstrap_W", "bootstrap_response_W"):
                m[name] = archive[key+"__"+name].copy() if key+"__"+name in archive.files else None
            m["times"] = archive[key+"__times_s"].copy()
            maps.append(m)
    return maps, selection["config"]


def balanced_candidates(maps, traces, pairs, state_rows, archive, mmcfg, cfg):
    proposals = []
    for m in maps:
        for identity, pair in pairs.items():
            native = pair["control"]["row"]
            if (native["account_class"], native.get("variant", "unspecified")) == (m["account_class"], m["variant"]):
                proposals.append({"candidate_id": m["map_key"]+"__"+native["pair_id"], "map_key": m["map_key"],
                                  "source_model_key": m["source_model_key"], "account_class": m["account_class"],
                                  "variant": m["variant"], "input_amplitude": pair["pulse"]["row"]["input_amplitude"],
                                  "control_id": identity})
    admitted, census = balanced_budget(proposals, cfg["candidate_budget"], cfg["seed"],
                                      ("source_model_key", "account_class", "variant", "input_amplitude"))
    bymap = {m["map_key"]: m for m in maps}; output = []; arrays = {}
    contexts = {}
    for source in sorted({m["source_model_key"] for m in maps}):
        train, _, _ = mm._source_examples(state_rows, archive, source, mmcfg, "train", next(m["times"] for m in maps if m["source_model_key"] == source))
        contexts[source] = {"grid": next(m["times"] for m in maps if m["source_model_key"] == source),
                            "support": mm.joint_support(train, mmcfg["support_quantile"], mmcfg["minimum_train"])}
    for row in admitted:
        m = bymap[row["map_key"]]; pair = pairs[row["control_id"]]
        measured, predicted, _, _ = mm._candidate_predictions([m], contexts, traces, {row["control_id"]: pair}, mmcfg | {"maximum_candidates": 1})
        for candidate in measured:
            lags = np.asarray(m["lags"])*mmcfg["delay_seconds_per_index"]
            control_state = mm._native_kernel(pair["control"], m["times"], lags)@m["W"]+m["bias"]
            pulse_state = mm._native_kernel(pair["pulse"], m["times"], lags, pair["control"])@m["W"]+m["bias"]
            candidate["control_pulse_geometry_distance"] = float(np.sqrt(np.mean((mm.rotation_geometry(control_state)-mm.rotation_geometry(pulse_state))**2)))
            output.append(candidate)
        arrays.update(predicted)
    return output, census, arrays


def discrepancy_evaluation(discovery_pairs, states, predictions, actual_arrays, predicted_arrays, model_grids=None):
    """Exact corrected human pairs, independently evaluated within frozen split."""
    lookup = {(r["participant_id"], r["source_id"], r["episode_id"]): r for r in states if r.get("status") == "included"}
    pred = defaultdict(dict)
    for row in predictions:
        pred[row["map_key"]][row["episode_id"]] = row
    results = []
    for pair in discovery_pairs:
        base = {k: pair.get(k, "") for k in ("pair_id", "family", "participant_id", "source_id", "episode_a", "episode_b", "sleep_stage")}
        a = lookup.get((pair["participant_id"], pair["source_id"], pair["episode_a"]))
        b = lookup.get((pair["participant_id"], pair["source_id"], pair["episode_b"]))
        if a is None or b is None or a["source_model_key"] != b["source_model_key"]:
            results.append(base | {"status": "missing_actual_common_state_domain"}); continue
        if a["split"] != b["split"]:
            raise core.StudyError("Corrected discrepancy crosses frozen human partitions")
        if a["split"] == "train":
            results.append(base | {"split": "train", "status": "training_case_not_held_person_explanation"}); continue
        mapped = False
        for key, entries in sorted(pred.items()):
            if pair["episode_a"] not in entries or pair["episode_b"] not in entries: continue
            pa, pb = entries[pair["episode_a"]], entries[pair["episode_b"]]
            if pa["source_model_key"] != a["source_model_key"] or pb["participant_id"] != pair["participant_id"]: continue
            ta, tb = a["state_token"], b["state_token"]
            ya, yb = actual_arrays[ta+"__state"], actual_arrays[tb+"__state"]
            za, zb = predicted_arrays[key+"__"+ta+"__prediction"], predicted_arrays[key+"__"+tb+"__prediction"]
            if model_grids is not None:
                grid = model_grids[key]
                times_a, times_b = actual_arrays[ta+"__times_s"], actual_arrays[tb+"__times_s"]
                ya = mm._interpolate(ya, times_a, times_a[-1]-9.998+grid)
                yb = mm._interpolate(yb, times_b, times_b[-1]-9.998+grid)
            cut_a, cut_b = int(pa.get("conditioning_states", 0)), int(pb.get("conditioning_states", 0))
            if cut_a != cut_b or ya.shape != za.shape or yb.shape != zb.shape:
                raise core.StudyError("Discrepancy comparison lacks common measured/predicted clocks")
            ya, yb, za, zb = ya[cut_a:], yb[cut_b:], za[cut_a:], zb[cut_b:]
            actual = mm.rotation_geometry(yb)-mm.rotation_geometry(ya)
            estimated = mm.rotation_geometry(zb)-mm.rotation_geometry(za)
            loss = float(np.mean((actual-estimated)**2)); zero = float(np.mean(actual**2))
            results.append(base | {"map_key": key, "split": a["split"], "source_model_key": a["source_model_key"],
                                  "account_class": pa["account_class"], "status": "development_discrepancy_prediction_evaluated",
                                  "geometry_contrast_mse": loss, "no_discrepancy_null_mse": zero,
                                  "loss_minus_null": loss-zero, "human_function_causal_mapping": False})
            mapped = True
        if not mapped: results.append(base | {"split": a["split"], "status": "no_actual_frozen_account_pair_predictions"})
    return results


def participant_summary(rows, value, cfg, strata):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        number = guided._finite(row.get(value))
        if number is not None:
            groups[tuple(row.get(k, "") for k in strata)][row["participant_id"]].append(number)
    output = []
    for key, people in sorted(groups.items()):
        values = [float(np.mean(v)) for _, v in sorted(people.items())]
        output.append(dict(zip(strata, key)) | {"participants": len(values), "equal_person_mean": float(np.mean(values)),
                                              "conditional_participant_bootstrap_95ci": rb._ci(values, cfg["seed"], cfg["bootstrap_samples"]),
                                              "independent_human_validation": False})
    return output


def original_score_baselines(pairs, state_rows, layers, cfg):
    """All four readouts face identical observed targets and complete original controls."""
    lookup = {(r["participant_id"], r["source_id"], r["episode_id"]): r for r in state_rows}
    augmented = []; census = []
    for pair in pairs:
        a, b = (lookup[pair["participant_id"], pair["source_id"], pair[e]] for e in ("episode_a", "episode_b"))
        original = {}; missing = []
        for field in ("score", "preclip_gcs", "saturated_fraction"):
            va, vb = guided._finite(a.get(field)), guided._finite(b.get(field))
            if va is None or vb is None: missing.append(field)
            else: original[field+"_difference"] = abs(vb-va)
        census.append({"participant_id": pair["participant_id"], "source_id": pair["source_id"], "episode_a": pair["episode_a"],
                       "episode_b": pair["episode_b"], "split": pair["split"], "status": "excluded_missing_original_controls" if missing else "common_case_included",
                       "missing_fields": ";".join(missing)})
        if "score" not in missing: augmented.append(pair | {"original_controls": original})
    predictions = []; selection = []; metrics = []
    groups = defaultdict(list)
    domains = defaultdict(list)
    for pair in augmented: domains[pair["source_id"], pair["source_model_key"]].append(pair)
    full_fields = {"score_difference", "preclip_gcs_difference", "saturated_fraction_difference"}
    for domain, records in domains.items():
        complete = [r for r in records if full_fields <= set(r["original_controls"])]
        groups[(*domain, "score_preclip_saturation_common_cases")] = complete
        if len(complete) != len(records):
            groups[(*domain, "score_available_common_cases")] = records
    for domain, records in sorted(groups.items()):
        for target in sorted({t for r in records for t in r["targets"]}):
            valid = [r for r in records if r["targets"][target][1] is not None]
            parts = {p: [r for r in valid if r["split"] == p] for p in ("train", "validation", "test")}
            if any(len({r["participant_id"] for r in parts[p]}) < cfg["minimum_"+p.replace("train", "training")+"_people"] for p in parts):
                metrics.append({"source_id": domain[0], "source_model_key": domain[1], "target": target, "common_case_panel": domain[2], "status": "insufficient_common_case_people"}); continue
            train, validation, test = (parts[p] for p in ("train", "validation", "test"))
            kind = valid[0]["targets"][target][0]
            truth = lambda rs: [r["targets"][target][1] for r in rs]
            classes = sorted(set(truth(train))) if kind == "nominal" else []
            if kind == "nominal" and len(classes) < 2: continue
            y = np.asarray(truth(train), float) if kind == "numeric" else np.asarray([[float(v == c) for c in classes] for v in truth(train)])
            by_predictor = {}
            models = (("frozen_clinical_score", ["score_difference"], False),
                    ("frozen_score_preclip_saturation", ["score_difference", "preclip_gcs_difference", "saturated_fraction_difference"], False),
                    ("frozen_score_plus_adapted_activation", ["score_difference"], True),
                    ("frozen_controls_plus_adapted_activation", ["score_difference", "preclip_gcs_difference", "saturated_fraction_difference"], True))
            for label, fields, activation in models:
                if domain[2] == "score_available_common_cases" and any(f != "score_difference" for f in fields): continue
                choices = []
                for layer in layers if activation else [None]:
                    matrix = lambda rs: np.asarray([[r["original_controls"][f] for f in fields] + ([r["activations"][layer]] if layer else []) for r in rs])
                    for alpha in cfg["recovery_ridge_grid"]:
                        fitted = rb.fit_ridge(matrix(train), y, [r["participant_id"] for r in train], alpha)
                        error, _ = rb._metric(validation, truth(validation), rb.predict_ridge(fitted, matrix(validation)), kind, classes)
                        choices.append((error, layer or "", alpha, fitted))
                error, layer, alpha, fitted = min(choices, key=lambda v: v[:3]); layer = layer or None
                matrix = lambda rs: np.asarray([[r["original_controls"][f] for f in fields] + ([r["activations"][layer]] if layer else []) for r in rs])
                estimate = rb.predict_ridge(fitted, matrix(test)); mean_loss, losses = rb._metric(test, truth(test), estimate, kind, classes)
                identity = _identity([domain, target, label], cfg["seed"])
                selection.append({"map_id": identity, "source_id": domain[0], "source_model_key": domain[1], "target": target,
                                  "common_case_panel": domain[2],
                                  "predictor": label, "layer": layer, "ridge": alpha, "training_people": sorted({r["participant_id"] for r in train}),
                                  "validation_people": sorted({r["participant_id"] for r in validation}), "test_used_for_selection": False,
                                  "means": fitted["means"].tolist(), "scale": fitted["scale"].tolist(), "beta": fitted["beta"].tolist(), "classes": classes})
                by_predictor[label] = losses
                for i, row in enumerate(test):
                    predictions.append({k: row[k] for k in ("participant_id", "source_id", "source_model_key", "episode_a", "episode_b", "split")} |
                                       {"target": target, "predictor": label, "common_case_panel": domain[2], "actual": truth(test)[i], "loss": float(losses[i]), "prediction": json.dumps(np.asarray(estimate[i]).tolist())})
                metrics.append({"source_id": domain[0], "source_model_key": domain[1], "target": target, "predictor": label,
                                "common_case_panel": domain[2],
                                "equal_person_loss": mean_loss, "people": len({r["participant_id"] for r in test}), "status": "development_original_instrument_comparison"})
            reference = by_predictor["frozen_clinical_score"]
            for label, losses in by_predictor.items():
                rows = [{"participant_id": r["participant_id"], "paired_loss_difference": float(v-b)} for r, v, b in zip(test, losses, reference)]
                for result in participant_summary(rows, "paired_loss_difference", cfg, []):
                    metrics.append(result | {"source_id": domain[0], "source_model_key": domain[1], "target": target, "predictor": label,
                                            "common_case_panel": domain[2],
                                            "comparison": "paired_loss_minus_frozen_clinical_score", "common_case_people": len({r["participant_id"] for r in test})})
    return predictions, selection, metrics, census


def apply_report_eligibility(pairs, observations):
    """Legacy representation tables cannot reintroduce missing-response categories."""
    output = []
    for pair in pairs:
        a, b = [observations.get((pair["participant_id"], pair["source_id"], pair[e]), {}) for e in ("episode_a", "episode_b")]
        observed = a.get("report_eligibility") == b.get("report_eligibility") == "observed_original_report"
        targets = dict(pair["targets"])
        for name in ("report_category", "experience_report"):
            if name in targets and not observed:
                kind, _, semantics, va, vb = targets[name]
                targets[name] = (kind, None, semantics, va, vb)
        output.append(pair | {"targets": targets, "nominal_report_pair_observed": observed})
    return output


def synthetic_noise(shape, level, model, rng, coefficient=.6):
    noise = rng.normal(size=shape)
    if model == "colored_ar1":
        for t in range(1, len(noise)): noise[t] = coefficient*noise[t-1]+math.sqrt(1-coefficient**2)*noise[t]
    elif model != "white": raise core.StudyError("Unknown declared noise model")
    return level*noise


def recover_generated(kernels, truths, W, bias, cfg, seed, noise_level=0., noise_model="white", misspecification=0., calibration_pool=None):
    """Known actual native kernels; jointly refit assignments and observation map.

    Truth labels are never passed to fitting. Frozen kernel lags/rank are the
    conditioning assumptions. A validation grid selects ridge/offset; all native
    accounts compete. Calibration blocks are synthetic units, not people.
    """
    rng = np.random.default_rng(seed); identities = sorted(kernels)
    known = {k: np.asarray(v, float)@W+bias for k, v in kernels.items()}
    n = len(next(iter(known.values()))); rank = W.shape[1]; cut = int(n*.65)
    true_pool = list(truths)
    if not true_pool or any(k not in known for k in true_pool): raise core.StudyError("Unknown native recovery generator")
    calibration_pool = true_pool if calibration_pool is None else list(calibration_pool)
    if not calibration_pool or any(k not in known for k in calibration_pool):
        raise core.StudyError("Unknown native calibration configuration")
    scale = np.maximum(np.std(np.concatenate([known[k] for k in true_pool]), axis=0), 1e-6)
    def observations(count, pool):
        generated = []; assigned = []
        for i in range(count):
            truth = pool[i % len(pool)]; actual = known[truth].copy()
            actual += synthetic_noise(actual.shape, noise_level, noise_model, rng, cfg["colored_noise_coefficient"])*scale
            if misspecification:
                # Explicit unmodelled latent input, never claimed as measured human physiology.
                t = np.linspace(0, 10, len(actual)); phase = rng.uniform(0, 2*np.pi)
                actual += misspecification*np.sin(2*np.pi*.731*t+phase)[:, None]*scale
            generated.append(actual); assigned.append(truth)
        return generated, assigned
    train, _ = observations(cfg["recovery_calibration_units"], calibration_pool)
    validation, _ = observations(cfg["recovery_validation_units"], calibration_pool)
    test, test_truth = observations(max(1, len(true_pool)), true_pool)
    families = sorted({k.split("\0", 1)[0] for k in identities})
    fits = []
    for family in families:
        candidate_ids = [k for k in identities if k.split("\0", 1)[0] == family]
        X = [kernels[k] for k in candidate_ids]
        for ridge, strength in itertools.product(cfg["recovery_ridge_grid"], (0., 1.)):
            # The initialization uses complete early temporal geometry, as the human bridge does.
            g = np.asarray([mm.rotation_geometry(x[:cut]) for x in X]); h = np.asarray([mm.rotation_geometry(y[:cut]) for y in train])
            chosen = np.argmin(((h[:, None, :]-g[None, :, :])**2).mean(axis=2), axis=1)
            weights = np.full(len(train), 1/len(train))
            for _ in range(3):
                fitted_W, fitted_b = mm._fit_weights([X[i][:cut] for i in chosen], [y[:cut] for y in train], weights, ridge, min(rank, W.shape[0]))
                predictions = np.asarray(X)@fitted_W+fitted_b
                example = [{"y": y, "early": cut} for y in train]
                chosen, _ = mm._nearest_trace(example, predictions, strength)
            fitted_W, fitted_b = mm._fit_weights([X[i][:cut] for i in chosen], [y[:cut] for y in train], weights, ridge, min(rank, W.shape[0]))
            predictions = np.asarray(X)@fitted_W+fitted_b
            picks, offsets = mm._nearest_trace([{"y": y, "early": cut} for y in validation], predictions, strength)
            losses = [float(np.mean((predictions[p]+offset-y)[cut:]**2)) for p, offset, y in zip(picks, offsets, validation)]
            fits.append((float(np.mean(losses)), family, ridge, strength, fitted_W, fitted_b, candidate_ids, predictions))
    if not fits: raise core.StudyError("Recovery requires actual native kernel candidates")
    best = min(fits, key=lambda v: v[:4]); loss, family, ridge, strength, fitted_W, fitted_b, ids, predicted = best
    profile = [{"account_class": f[1], "ridge": f[2], "offset_strength": f[3], "validation_mse": f[0]}
               for f in sorted(fits, key=lambda v: v[:4])]
    picks, offsets = mm._nearest_trace([{"y": y, "early": cut} for y in test], predicted, strength)
    rows = []
    for truth, actual, pick, offset in zip(test_truth, test, picks, offsets):
        reference = known[truth]; denominator = max(float(np.mean(reference**2)), 1e-9)
        equivalents = sorted(k for k, y in known.items() if math.sqrt(float(np.mean((reference-y)**2))/denominator) <= cfg["recovery_equivalence_tolerance"])
        recovered = ids[int(pick)]; err = float(np.mean((predicted[pick]+offset-actual)[cut:]**2))/denominator
        rows.append({"known_generator_id": truth, "recovered_generator_id": recovered, "correct_configuration": recovered == truth,
                     "generating_equivalence_class": json.dumps(equivalents), "equivalence_class_size": len(equivalents),
                     "recovered_in_equivalence_class": recovered in equivalents, "unique_configuration_identifiable_under_generating_map": len(equivalents) == 1,
                     "later_prediction_standardized_mse": err, "refitted_map_rmse": float(np.sqrt(np.mean((W-fitted_W)**2))),
                     "refitted_bias_rmse": float(np.sqrt(np.mean((bias-fitted_b)**2))), "selected_family": family,
                     "selected_ridge": ridge, "selected_offset_strength": strength, "validation_mse": loss,
                     "account_map_validation_profile": json.dumps(profile),
                     "noise_level": noise_level, "noise_model": noise_model, "misspecification_level": misspecification,
                     "calibration_configuration_count": len(calibration_pool),
                     "physiological_recovery_established": False, "evidence_status": "model_generated_native_inverse_diagnostic"})
    return rows


def native_recovery(maps, traces, pairs, mmcfg, cfg):
    proposals = [{"candidate_id": m["map_key"]+"__"+identity, "map_key": m["map_key"], "source_model_key": m["source_model_key"],
                  "account_class": m["account_class"], "variant": m["variant"], "input_amplitude": pair["pulse"]["row"]["input_amplitude"], "control_id": identity}
                 for m in maps for identity, pair in pairs.items()
                 if (pair["control"]["row"]["account_class"], pair["control"]["row"].get("variant", "unspecified")) == (m["account_class"], m["variant"])]
    admitted, census = balanced_budget(proposals, cfg["recovery_case_budget"], cfg["seed"], ("source_model_key", "account_class", "variant"))
    lookup = {m["map_key"]: m for m in maps}; rows = []
    for case in admitted:
        m = lookup[case["map_key"]]; lags = np.asarray(m["lags"])*mmcfg["delay_seconds_per_index"]
        kernels = {trace["row"]["account_class"]+"\0"+identity: mm._native_kernel(trace, m["times"], lags)
                   for identity, trace in traces.items() if identity in pairs and trace["row"].get("variant", "unspecified") == m["variant"]}
        truth = m["account_class"]+"\0"+case["control_id"]
        # Paired input diagnostics use actual pulse-minus-control native kernels.
        delta = {pairs[identity]["control"]["row"]["account_class"]+"\0"+identity:
                 mm._native_kernel(pair["pulse"], m["times"], lags, pair["control"])-mm._native_kernel(pair["control"], m["times"], lags)
                 for identity, pair in pairs.items() if pair["control"]["row"].get("variant", "unspecified") == m["variant"]}
        for observation, inputs, W, bias in (("native_control_trajectory", kernels, m["W"], m["bias"]),
                ("native_pulse_minus_control", delta, m["response_W"] if m["response_W"] is not None else m["W"], np.zeros(m["W"].shape[1]))):
            for level, noise, misspec, replicate in itertools.product(cfg["noise_levels"], cfg["noise_models"], cfg["misspecification_levels"], range(cfg["recovery_replicates"])):
                seed = int(_identity([case["candidate_id"], observation, level, noise, misspec, replicate], cfg["seed"])[:8], 16)
                calibration = sorted((k for k in inputs if k.split("\0", 1)[0] == m["account_class"]), key=lambda k: _identity(k, cfg["seed"]))
                # Diverse conditions identify the unknown effective map; the held
                # target condition has a separate noise draw and supplies no labels.
                results = recover_generated(inputs, [truth], W, bias, cfg, seed, level, noise, misspec, calibration)
                rows.extend(r | {"case_id": case["candidate_id"], "source_model_key": m["source_model_key"], "generating_account_class": m["account_class"],
                                      "observation": observation, "replicate": replicate, "source_config_sha256": traces[case["control_id"]]["row"]["config_sha256"],
                                      "native_input_physical_units_qualified": False} for r in results)
    return rows, census


def _pin(path):
    return {"path": str(Path(path).resolve()), "sha256": core.sha256_file(path)}


def run(states, representations, bridge, fit, discovery, config, output):
    cfg = settings(config)
    if cfg["require_scheduled_execution"]:
        from native_tools.nftsim_runtime import require_scheduled_job
        require_scheduled_job()
    root = Path(output).resolve()
    if root.exists(): raise core.StudyError("Preserve previous analysis; output must be new")
    states, representations, bridge, fit = map(lambda p: Path(p).resolve(), (states, representations, bridge, fit))
    from .neural_state_space import verify_state_space
    from .human_generative_adaptation import verify_fit
    from .proposal_discovery import verify_outputs
    state_receipt = verify_state_space(states); rep_receipt = rb.verify_representation(representations); mm.verify_bridge(bridge)
    _, bank, provenance = verify_fit(fit)
    bridge_receipt = core.read_json(bridge/"receipt.json")
    declared = bridge_receipt.get("input_paths", {})
    for key, path in (("states", states), ("representations", representations), ("fit", fit)):
        if Path(declared.get(key, "")).resolve() != path: raise core.StudyError("Bridge belongs to different declared parent " + key)
    corrected = []; low_cases = {}; input_pins = [_pin(p/"receipt.json") for p in (states, representations, bridge, fit)] + [_pin(config)]
    for directory in discovery:
        verify_outputs(directory); input_pins.append(_pin(Path(directory)/"receipt.json"))
        corrected.extend(core.read_csv(Path(directory)/"natural_pairs.csv"))
        for row in core.read_csv(Path(directory)/"observations.csv"):
            low_cases[row["participant_id"], row["source_id"], row["episode_id"]] = row
    if len({r["pair_id"] for r in corrected}) != len(corrected): raise core.StudyError("Repeated corrected discrepancy identity")
    rows = core.read_csv(states/"rows.csv"); maps, mmcfg = load_maps(bridge)
    traces, pairs, rejected = mm._native_bank(bank)
    root.mkdir(parents=True)
    with np.load(states/"states.npz", allow_pickle=False) as archive:
        candidates, census, candidate_arrays = balanced_candidates(maps, traces, pairs, rows, archive, mmcfg, cfg)
    scale = {source: _scales([r for r in rows if r.get("source_model_key") == source]) for source in {m["source_model_key"] for m in maps}}
    families, dispositions = search_families(candidates, bank, scale, cfg)
    _csv(root/"balanced_candidates.csv", candidates, ["candidate_id", "source_model_key", "account_class"])
    _csv(root/"candidate_coverage.csv", census, ["stratum", "proposed", "admitted"])
    _csv(root/"adversarial_families.csv", families, ["family", "candidate_a", "candidate_b", "status"])
    _csv(root/"contrast_dispositions.csv", dispositions, ["family", "candidate_a", "candidate_b", "status"])
    np.savez_compressed(root/"candidate_predictions.npz", **candidate_arrays)
    recovery, recovery_census = native_recovery(maps, traces, pairs, mmcfg, cfg)
    _csv(root/"native_recovery.csv", recovery, ["case_id", "known_generator_id", "recovered_generator_id", "evidence_status"])
    _csv(root/"recovery_coverage.csv", recovery_census, ["stratum", "proposed", "admitted"])
    # This is an additional development freeze; parent held-person data were already analysed.
    freeze = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "status": "strengthened_computational_candidates_require_human_mapping",
              "candidate_families": dict(Counter(r["family"] for r in families)), "inputs": input_pins,
              "candidate_artifacts": [_pin(root/p) for p in ("balanced_candidates.csv", "adversarial_families.csv", "candidate_predictions.npz", "native_recovery.csv")],
              "human_intervention_qualified": False, "human_function_mapping_qualified": False, "physiological_parameter_identification": False,
              "primary_human_endpoint": None, "selection_inputs": "Frozen train/validation maps, actual native trajectories and metadata-only balanced budgets; no additional human test outcomes",
              "historical_exposure": "Existing public development and parent held-person results already exposed; not protected independent confirmation",
              "recovery_scope": "Actual native configuration inversion with unknown assignment/observation-map refitting and validation-only ridge/offset selection, conditional on frozen kernel lags/rank; noise and misspecification are declared diagnostic assumptions"}
    core.write_json(root/"strengthened_computational_freeze.json", freeze)
    account_predictions = core.read_csv(bridge/"account_predictions.csv")
    with np.load(states/"states.npz", allow_pickle=False) as actual, np.load(bridge/"state_predictions.npz", allow_pickle=False) as predicted:
        discrepancy = discrepancy_evaluation(corrected, rows, account_predictions, actual, predicted, {m["map_key"]: m["times"] for m in maps})
    for row in discrepancy:
        for endpoint, field in (("a", "episode_a"), ("b", "episode_b")):
            source = low_cases.get((row["participant_id"], row["source_id"], row[field]), {})
            for name in ("low_score_main", "low_score_supported_report_main", "report_support_timing", "support_is_subsequent_detection", "experience_during_score_window_certified"):
                row[endpoint+"_"+name] = source.get(name, "")
    _csv(root/"discrepancy_evaluation.csv", discrepancy, ["pair_id", "family", "participant_id", "status"])
    summary = participant_summary(discrepancy, "loss_minus_null", cfg, ["source_id", "family", "split", "map_key"])
    core.write_json(root/"discrepancy_summary.json", {"results": summary, "unit": "participant", "prediction_scope": "Previously exposed development held-person cases; no independent human confirmation"})
    # Reuse the exact eligible adapted pairs; original outcomes/controls stay source-bound.
    original_config = core.read_json(rep_receipt["input_paths"]["config"])
    clinical = core.read_json(Path(rep_receipt["input_paths"]["clinical_trained"])/"training_receipt.json")
    from native_tools.human_finetune import validate_split
    roles = validate_split(core.read_json(states/"split.json")); clinical_roles = validate_split(clinical["split"])
    baseline_pairs, _, _, layers = rb._pair_data(states, rep_receipt["input_paths"]["adapted_geometry"], rows, roles, clinical_roles, rb.effective_config(original_config))
    baseline_pairs = apply_report_eligibility(baseline_pairs, low_cases)
    baseline_predictions, selections, metrics, baseline_census = original_score_baselines(baseline_pairs, rows, layers, cfg)
    _csv(root/"original_score_predictions.csv", baseline_predictions, ["participant_id", "target", "predictor", "loss"])
    _csv(root/"original_score_eligibility.csv", baseline_census, ["participant_id", "episode_a", "episode_b", "status"])
    core.write_json(root/"original_score_selection.json", {"maps": selections, "test_used_for_selection": False})
    core.write_json(root/"original_score_metrics.json", {"metrics": metrics, "paired_comparison": "Equal-person loss differences on identical eligible held-person pairs", "missing_original_controls": "Excluded explicitly; no substituted preclip or saturation values"})
    core.write_json(root/"analysis_summary.json", {"version": VERSION, "config": cfg, "actual_native_pairs": len(pairs), "retained_native_pair_failures": rejected,
                    "corrected_discrepancy_pairs": len(corrected), "discrepancy_dispositions": dict(Counter(r["status"] for r in discrepancy)),
                    "candidate_families": freeze["candidate_families"], "recovery_rows": len(recovery), "baseline_maps": len(selections),
                    "native_provenance": provenance, "human_finetuning_verified": True, "adapted_checkpoint_sha256": rep_receipt["adapted_checkpoint_sha256"],
                    "state_split_sha256": state_receipt["split_sha256"], "independent_human_validation": False, "no_result_direction_gate": True})
    receipt = {"schema_version": 1, "version": VERSION, "status": "proposal_mechanisms_completed", "inputs": input_pins,
               "input_paths": {"states": str(states), "representations": str(representations), "bridge": str(bridge), "fit": str(fit), "discovery": [str(Path(p).resolve()) for p in discovery], "config": str(Path(config).resolve())},
               "source_code_sha256": core.sha256_file(__file__), "outputs": {p.name: core.sha256_file(p) for p in sorted(root.iterdir()) if p.is_file()},
               "scientific_scope": "Additive previously exposed development comparisons and model-generated native inverse diagnostics; human diagnostic qualification remains external"}
    core.write_json(root/"receipt.json", receipt)
    return receipt


def verify_outputs(directory):
    root = Path(directory).resolve(); receipt = core.read_json(root/"receipt.json")
    if receipt.get("status") != "proposal_mechanisms_completed" or receipt.get("source_code_sha256") != core.sha256_file(__file__):
        raise core.StudyError("Proposal mechanism artifact/code mismatch")
    for item in receipt.get("inputs", []):
        if core.sha256_file(item["path"]) != item["sha256"]: raise core.StudyError("Proposal mechanism parent changed")
    for name, digest in receipt.get("outputs", {}).items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or core.sha256_file(path) != digest: raise core.StudyError("Proposal mechanism output changed")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("states", "representations", "bridge", "fit", "config", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--discovery", action="append", required=True, type=Path)
    args = vars(parser.parse_args(argv))
    try: print(json.dumps(run(**args), indent=2)); return 0
    except (core.StudyError, OSError, KeyError, ValueError, np.linalg.LinAlgError) as exc:
        parser.exit(2, "Proposal mechanism analysis failed: "+str(exc)+"\n")


if __name__ == "__main__": raise SystemExit(main())
