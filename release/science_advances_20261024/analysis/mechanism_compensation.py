"""Held-configuration tests of effective dynamics and parameter ambiguity.

This is a finite, deliberately sampled simulator bank.  Its configuration
bootstrap describes sensitivity to this bank, never uncertainty over people.
No observed human test outcome is an input to any fit in this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

VERSION = "cross_mechanism_compensation_v1"
FAMILIES = {"gain", "coupling", "history", "combined"}
FACTORS = ("param_gain_factor", "param_coupling_factor", "param_history_factor")
RIDGES = (.01, 1., 100.)
RESPONSE_NAMES = ("voltage_peak_rms", "rate_peak_rms", "voltage_integrated_rms",
                  "rate_integrated_rms", "voltage_mode_fraction", "rate_mode_fraction")
BOOTSTRAP_REPETITIONS = 2000


def require(value, text):
    if not value:
        raise ValueError("Mechanism compensation: " + text)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def _number(value):
    require(not isinstance(value, (bool, np.bool_)), "Boolean numeric field")
    result = float(value)
    require(math.isfinite(result), "Nonfinite numeric field")
    return result


def structural_key(trace):
    """Same physical configuration across pulse/input seeds has one role.

    Removing observed input design is appropriate for leakage grouping, not for
    qualifying pulse subtraction. Actual pairs are separately strictly checked.
    Output declarations are recorder settings, not dynamical parameters.
    """
    text = trace.get("config_text", trace.get("configuration_text", ""))
    require(isinstance(text, str) and text, "Actual configuration text required")
    clean = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", " ", text)
    clean = re.split(r"^\s*Output:", clean, maxsplit=1, flags=re.M)[0]
    # Limit pulse-field masking to the actual external population11 block.
    expression = r"(^\s*Population\s+11:\s*[^\n]*\n)([\s\S]*?)(?=^\s*(?:Population\s+\d+:|Propagator\s+\d+:|Coupling\s+\d+:)|\Z)"
    matches = list(re.finditer(expression, clean, re.M))
    require(len(matches) == 1, "External pulse block must identify one population")
    hit = matches[0]
    pulse = re.sub(r"\b(?:Onset|Amplitude|Width|Pulses):\s*[-+0-9.eE]+", "PULSE_DESIGN", hit[2])
    clean = clean[:hit.start(2)] + pulse + clean[hit.end(2):]
    clean = re.sub(r"\bWhiteSeed:\s*[-+0-9.eE]+", "WhiteSeed: OBSERVED_NOISE", clean)
    clean = " ".join(clean.split())
    return hashlib.sha256((str(trace["row"]["variant"]) + "\n" + clean).encode()).hexdigest()


def assign_roles(traces, seed):
    """Deterministic 60/20/20 split of whole physical configurations by family."""
    groups = defaultdict(list)
    for simulation_id, trace in sorted(traces.items()):
        require(simulation_id == trace["row"]["simulation_id"], "Trace identity differs")
        groups[structural_key(trace)].append(simulation_id)
    strata = defaultdict(list)
    for group, ids in groups.items():
        variants = {traces[s]["row"]["variant"] for s in ids}
        require(len(variants) == 1, "A physical group crosses implementations")
        family = "+".join(sorted({traces[s]["row"]["account_class"] for s in ids}))
        strata[(next(iter(variants)), family)].append(group)
    role_by_group = {}
    for stratum, groups_in_stratum in sorted(strata.items()):
        ordered = sorted(groups_in_stratum, key=lambda group: hashlib.sha256(
            (str(int(seed)) + "|" + group).encode()).hexdigest())
        count = len(ordered)
        n_test = max(1, int(round(.20 * count))) if count >= 3 else 0
        n_val = max(1, int(round(.20 * count))) if count >= 3 else 0
        require(count - n_test - n_val >= 1, "No training configuration remains")
        for index, group in enumerate(ordered):
            role_by_group[group] = "test" if index < n_test else "validation" if index < n_test + n_val else "train"
    rows = []
    for group, ids in sorted(groups.items()):
        for simulation_id in sorted(ids):
            row = traces[simulation_id]["row"]
            rows.append({"simulation_id": simulation_id, "configuration_group": group,
                         "variant": row["variant"], "account_class": row["account_class"],
                         "role": role_by_group[group], "pair_id": row.get("pair_id", ""),
                         "input_amplitude": _number(row["input_amplitude"])})
    return {row["simulation_id"]: row for row in rows}, rows


def _clock(trace):
    t = np.asarray(trace.get("times", trace.get("time")), float)
    require(t.ndim == 1 and len(t) >= 16 and np.isfinite(t).all(), "Invalid observed clock")
    dt = float(np.median(np.diff(t)))
    require(dt > 0 and np.allclose(np.diff(t), dt, rtol=0, atol=1e-9), "Nonuniform observed clock")
    return t, dt


def _values(trace):
    value = np.asarray(trace["values"], float)
    require(value.ndim == 2 and value.shape[1] == 18 and np.isfinite(value).all(),
            "All eighteen finite voltage/rate columns required")
    require(len(value) == len(_clock(trace)[0]), "Observed values/clock differ")
    return value


def _variance_floor(level):
    return np.maximum(1e-12, np.abs(np.asarray(level, float)) * 1e-12)


def reference_fit(traces):
    """Fit common units and mode axes using training zero-input traces only."""
    require(len(traces) >= 2, "At least two training configurations needed")
    covariances, means, within_variance = [], [], []
    for trace in traces:
        value = _values(trace)
        means.append(value.mean(0))
        covariances.append(np.cov(value, rowvar=False))
        within_variance.append(np.var(value, axis=0, ddof=1))
    mean = np.mean(means, axis=0)
    scale = np.maximum(np.sqrt(np.mean(within_variance, axis=0)), _variance_floor(mean))
    covariance = np.mean(covariances, axis=0) / scale[:, None] / scale[None, :]
    eigenvalues, axes = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, axes = np.maximum(eigenvalues[order], 0), axes[:, order]
    for column in range(axes.shape[1]):
        anchor = np.argmax(np.abs(axes[:, column]))
        if axes[anchor, column] < 0:
            axes[:, column] *= -1
    return {"mean": mean, "scale": scale, "axes": axes, "eigenvalues": eigenvalues,
            "training_simulation_ids": [t["row"]["simulation_id"] for t in traces]}


def _participation(eigenvalues):
    eig = np.maximum(np.asarray(eigenvalues, float), 0)
    if not np.any(eig):
        return 0.
    eig = eig / eig.max()
    return float(eig.sum() ** 2 / np.sum(eig ** 2))


def _autocorrelation(value, lag):
    x = np.asarray(value, float)
    if lag >= len(x) - 2:
        return 0.
    a, b = x[:-lag] - x[:-lag].mean(), x[lag:] - x[lag:].mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 0 else 0.


def _temporal(value, dt):
    x = np.asarray(value, float)
    x = x - x.mean()
    energy = float(x @ x)
    if energy <= 0:
        return {"acf_100ms": 0., "acf_500ms": 0., "ar1_slope": 0.,
                "spectral_centroid_hz": 0., "spectral_entropy": 0.}
    frequencies = np.fft.rfftfreq(len(x), dt)
    power = np.abs(np.fft.rfft(x)) ** 2
    power[0] = 0
    power_sum = float(power.sum())
    p = power[power > 0] / power_sum if power_sum > 0 else np.empty(0)
    denominator = float(x[:-1] @ x[:-1])
    return {"acf_100ms": _autocorrelation(x, max(1, round(.1 / dt))),
            "acf_500ms": _autocorrelation(x, max(1, round(.5 / dt))),
            "ar1_slope": float(x[:-1] @ x[1:] / denominator) if denominator > 0 else 0.,
            "spectral_centroid_hz": float(power @ frequencies / power_sum) if power_sum > 0 else 0.,
            "spectral_entropy": float(-np.sum(p * np.log(p)) / math.log(len(power))) if len(p) else 0.}


def trace_features(trace, reference, before=None):
    """Raw-unit and training-common-scaled summaries; no per-trace SD division."""
    t, dt = _clock(trace)
    value = _values(trace)
    if before is not None:
        value = value[t < float(before)]
    require(len(value) >= 16, "Insufficient pre-input observations")
    centered = value - value.mean(0)
    scaled = centered / reference["scale"]
    covariance = scaled.T @ scaled / (len(value) - 1)
    eig = np.maximum(np.linalg.eigvalsh(covariance), 0)
    common = scaled @ reference["axes"][:, 0]
    total = float(np.trace(covariance))
    features = {"common_scaled_participation": _participation(eig),
                "common_training_mode_fraction": float(np.var(common, ddof=1) / total) if total > 0 else 0.}
    features.update({"common_mode_" + key: value for key, value in _temporal(common, dt).items()})
    for column in range(18):
        prefix = ("V" if column % 2 == 0 else "Q") + str(column // 2 + 1)
        features[prefix + "_mean"] = float(value[:, column].mean())
        features[prefix + "_std"] = float(value[:, column].std(ddof=1))
    for quantity, offset in (("V", 0), ("Q", 1)):
        current = centered[:, offset::2]
        raw_eig = np.linalg.eigvalsh(current.T @ current / (len(value) - 1))
        features[quantity + "_raw_participation"] = _participation(raw_eig)
        features[quantity + "_raw_first_mode_fraction"] = float(max(0, raw_eig[-1]) / max(np.sum(raw_eig), 1e-300))
        features.update({quantity + "1_" + key: number for key, number in _temporal(current[:, 0], dt).items()})
    require(all(math.isfinite(v) for v in features.values()), "Nonfinite dynamical features")
    return features


def feature_columns(features, space):
    names = sorted(features)
    if space == "cortical":
        return [name for name in names if name.startswith(("V1_", "Q1_"))]
    if space == "effective_mode":
        return [name for name in names if name.startswith("common_")]
    if space == "full_population":
        return names
    raise ValueError("Unknown feature representation")


def ridge_fit(x, y, penalty):
    x, y = np.asarray(x, float), np.asarray(y, float)
    require(x.ndim == y.ndim == 2 and len(x) == len(y) >= 2 and np.isfinite(x).all() and np.isfinite(y).all(),
            "Invalid ridge training arrays")
    mean, sd = x.mean(0), x.std(0)
    # Constant columns stay zero; small physical voltage units are not mistaken
    # for constants. Only actual machine precision relative to level is floored.
    floor = np.finfo(float).eps * np.maximum(np.abs(mean), 1e-300) * 100
    sd = np.where(sd > floor, sd, 1.)
    z = (x - mean) / sd
    y_mean, y_sd = y.mean(0), y.std(0)
    y_sd = np.where(y_sd > np.finfo(float).eps * np.maximum(abs(y_mean), 1e-300) * 100, y_sd, 1.)
    normalized = (y - y_mean) / y_sd
    u, singular, vt = np.linalg.svd(z, full_matrices=False)
    weight = (vt.T * (singular / (singular ** 2 + float(penalty)))) @ u.T @ normalized
    return {"mean": mean, "sd": sd, "weight": weight, "y_mean": y_mean, "y_sd": y_sd,
            "penalty": float(penalty), "effective_df": float(np.sum(singular ** 2 / (singular ** 2 + penalty))) + 1}


def ridge_predict(model, x):
    x = np.asarray(x, float)
    require(x.ndim == 2 and x.shape[1] == len(model["mean"]) and np.isfinite(x).all(), "Invalid ridge prediction arrays")
    return ((x - model["mean"]) / model["sd"] @ model["weight"]) * model["y_sd"] + model["y_mean"]


def select_ridge(x, y, roles):
    roles = np.asarray(roles)
    train, validation = roles == "train", roles == "validation"
    require(train.sum() >= 3 and validation.sum() >= 1, "Training/validation coverage is insufficient")
    rows, fitted = [], []
    for penalty in RIDGES:
        model = ridge_fit(x[train], y[train], penalty)
        prediction = ridge_predict(model, x[validation])
        loss = float(np.mean(((prediction - y[validation]) / model["y_sd"]) ** 2))
        rows.append({"ridge": penalty, "validation_standardized_mse": loss, "effective_df": model["effective_df"]})
        fitted.append(model)
    index = min(range(len(rows)), key=lambda i: (rows[i]["validation_standardized_mse"], -rows[i]["ridge"]))
    return fitted[index], rows


def parameter_eligible(trace):
    row = trace["row"]
    # Published-bank rows carry nominal factors1 while numerous other native
    # parameters differ; those rows cannot be inverse-parameter ground truth.
    return row.get("account_class") in FAMILIES and _number(row["input_amplitude"]) == 0


def _base(trace, roles):
    row = trace["row"]
    return {"simulation_id": row["simulation_id"], "variant": row["variant"],
            "account_class": row["account_class"], "configuration_group": roles[row["simulation_id"]]["configuration_group"],
            "role": roles[row["simulation_id"]]["role"]}


def prediction_experiment(records, design_by_name, targets, target_names):
    """Fit/choose only train/validation; score each fixed test group once."""
    roles = [r["role"] for r in records]
    y = np.asarray(targets, float)
    train = np.asarray(roles) == "train"
    test_indices = np.flatnonzero(np.asarray(roles) == "test")
    if train.sum() < 3 or roles.count("validation") < 1 or len(test_indices) < 1:
        return [], {"status": "unsupported_configuration_role_coverage", "role_counts": dict(Counter(roles))}
    target_sd = y[train].std(0)
    target_sd = np.where(target_sd > np.finfo(float).eps * np.maximum(abs(y[train].mean(0)), 1e-300) * 100, target_sd, 1.)
    predictions, selections = [], {}
    for model_name, design in design_by_name.items():
        x = np.asarray(design, float)
        if x.shape[1] == 0:
            fitted = None
            prediction = np.tile(y[train].mean(0), (len(test_indices), 1))
            selections[model_name] = {"status": "training_mean_baseline", "effective_df": 1}
        else:
            fitted, grid = select_ridge(x, y, roles)
            prediction = ridge_predict(fitted, x[test_indices])
            selections[model_name] = {"status": "validation_selected", "ridge": fitted["penalty"],
                                      "effective_df": fitted["effective_df"], "validation_grid": grid,
                                      "training_rows": int(train.sum()), "input_columns": x.shape[1]}
        for index, estimate in zip(test_indices, prediction):
            row = dict(records[index]) | {"model": model_name,
                    "standardized_mse": float(np.mean(((estimate - y[index]) / target_sd) ** 2))}
            for j, name in enumerate(target_names):
                row["true_" + name] = float(y[index, j])
                row["predicted_" + name] = float(estimate[j])
            predictions.append(row)
    return predictions, {"status": "completed_held_configuration_prediction", "models": selections,
                         "role_counts": dict(Counter(roles)), "test_target_used_for_selection": False,
                         "target_scale_training_only": target_sd.tolist()}


def coverage(records, parameter_matrix):
    p = np.asarray(parameter_matrix, float)
    train = np.asarray([r["role"] == "train" for r in records])
    axes = {}
    for j, name in enumerate(("gain", "coupling", "history")):
        unique = sorted(set(float(x) for x in np.exp(p[:, j])))
        train_unique = sorted(set(float(x) for x in np.exp(p[train, j])))
        axes[name] = {"observed_factor_values": unique, "training_factor_values": train_unique,
                      "largest_log_gap": float(np.diff(np.log(unique)).max()) if len(unique) > 1 else None}
    singular = np.linalg.svd(p[train] - p[train].mean(0), compute_uv=False) if train.sum() else np.empty(0)
    rank = int(np.linalg.matrix_rank(p[train] - p[train].mean(0))) if train.sum() else 0
    return {"status": "finite_bank_parameter_coverage", "actual_eligible_configurations": len(records),
            "configuration_groups": len({r["configuration_group"] for r in records}),
            "role_counts": dict(Counter(r["role"] for r in records)), "factor_axes": axes,
            "training_parameter_design_singular_values": singular.tolist(),
            "training_parameter_design_rank": rank,
            "interpretation": "Design coverage and held-configuration recovery, not structural or human physiological identifiability; local hsyn and nu compensation does not imply exact whole-family equivalence."}


def parameter_geometry(records, parameter_matrix, designs):
    """Training-only finite-bank linear sensitivity, never a system Jacobian."""
    train = np.asarray([row["role"] == "train" for row in records])
    p = np.asarray(parameter_matrix, float)
    if train.sum() < 4 or np.linalg.matrix_rank(p[train] - p[train].mean(0)) < 3:
        return {"status": "unsupported_parameter_design_rank"}
    results = {}
    for name, design in designs.items():
        x = np.asarray(design, float)
        if x.shape[1] == 0:
            continue
        fitted = ridge_fit(p[train], x[train], 1.)
        # Every target feature was standardized on training configs by ridge_fit.
        # Dividing by sqrt(feature count) makes spectra comparable across views.
        u, singular, _ = np.linalg.svd(fitted["weight"] / math.sqrt(x.shape[1]), full_matrices=False)
        results[name] = {"singular_values": singular.tolist(),
                         "standardized_log_factor_directions_columns": u.tolist(),
                         "descriptive_rank_at_5pct_of_largest_singular_value": int(np.sum(singular > singular[0] * .05)) if len(singular) and singular[0] > 0 else 0,
                         "ridge": 1., "training_configuration_groups": int(train.sum()),
                         "meaning": "Regularized average sensitivity of sampled dynamical features to standardized log factors; not structural identifiability or the full delay-system Jacobian."}
    return {"status": "completed_training_bank_feature_sensitivity", "representations": results,
            "factor_order": ["log_gain", "log_coupling", "log_history"],
            "test_configurations_used": False}


def bootstrap_predictions(rows, seed, repetitions=BOOTSTRAP_REPETITIONS):
    """Stratified configuration-block bootstrap of fixed held-case losses."""
    require(type(repetitions) is int and 100 <= repetitions <= 10000, "Bootstrap repetitions outside100..10000")
    if not rows:
        return {"status": "unsupported_no_held_predictions"}
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["variant"], row["configuration_group"])].append(row)
    models = sorted({row["model"] for row in rows})
    for group in by_group.values():
        require(Counter(row["model"] for row in group) == Counter(models), "Each model must score each physical held configuration once")
        require(len({row["account_class"] for row in group}) == 1, "Configuration group crosses bootstrap families")
    result = []
    for variant in sorted({key[0] for key in by_group}):
        keys = sorted(key for key in by_group if key[0] == variant)
        strata = defaultdict(list)
        matrix = []
        for index, key in enumerate(keys):
            group = by_group[key]
            strata[group[0]["account_class"]].append(index)
            matrix.append([next(row["standardized_mse"] for row in group if row["model"] == name) for name in models])
        matrix = np.asarray(matrix, float)
        rng = np.random.default_rng(int(seed) + int(hashlib.sha256(variant.encode()).hexdigest()[:8], 16))
        draws = np.empty((repetitions, len(models)))
        for bootstrap in range(repetitions):
            sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata.values()])
            draws[bootstrap] = matrix[sampled].mean(0)
        baseline = "factors_and_observed_input" if "factors_and_observed_input" in models else "training_mean"
        baseline_index = models.index(baseline)
        for j, model in enumerate(models):
            delta = draws[:, j] - draws[:, baseline_index]
            result.append({"variant": variant, "model": model, "baseline": baseline,
                           "held_configuration_groups": len(keys), "mean_standardized_mse": float(matrix[:, j].mean()),
                           "bank_stability_mse_percentiles_2p5_97p5": np.quantile(draws[:, j], [.025, .975]).tolist(),
                           "loss_minus_baseline": float(matrix[:, j].mean() - matrix[:, baseline_index].mean()),
                           "bank_stability_difference_percentiles_2p5_97p5": np.quantile(delta, [.025, .975]).tolist(),
                           "family_strata_sizes": {str(key): len(value) for key, value in strata.items()},
                           "singleton_strata": [str(key) for key, value in strata.items() if len(value) == 1]})
    return {"status": "completed_conditional_finite_bank_configuration_bootstrap", "seed": int(seed),
            "repetitions": repetitions, "results": result,
            "inference": "Configuration blocks within family and variant; fixed train/validation-selected fits. Percentile ranges describe this deliberately sampled bank and omit fit uncertainty; they are not human confidence intervals or evidence of independent replication.",
            "human_participants_resampled": 0, "pseudo_replicated_timepoints": False}


def _response_targets(control, pulse):
    from native_population import qualify_pair
    proof = qualify_pair(control, pulse)
    t, _ = _clock(control)
    onset = float(control["config"]["input_onset_s"] - control["config"]["output_start_s"])
    width = float(control["config"]["input_width_s"])
    amplitude = _number(pulse["row"]["input_amplitude"])
    require(amplitude > 0 and width > 0, "Positive observed external input required")
    observed = np.asarray(pulse["stimulus"], float)
    require(np.max(abs(observed)) > 0, "Pulse must actually be observed")
    # Input amplitude is an observed design covariate. It is not divided out as
    # though the finite, nonlinear bank were a linear impulse-response design.
    observed_peak = float(np.max(abs(observed)))
    require(np.isclose(observed_peak, amplitude, rtol=1e-8, atol=1e-10), "Configured and observed pulse amplitude differ")
    post = (t >= onset) & (t <= onset + 2.)
    require(post.sum() >= 16, "Insufficient post-input window")
    difference = _values(pulse)[post] - _values(control)[post]
    target = {}
    for kind, offset, label in (("V", 0, "voltage"), ("Q", 1, "rate")):
        delta = difference[:, offset::2]
        rms = np.sqrt(np.mean(delta ** 2, axis=1))
        target[label + "_peak_rms"] = float(rms.max())
        target[label + "_integrated_rms"] = float(np.trapezoid(rms, t[post]))
        _, singular, _ = np.linalg.svd(delta, full_matrices=False)
        target[label + "_mode_fraction"] = float(singular[0] ** 2 / np.sum(singular ** 2)) if np.any(singular) else 0.
    require(all(math.isfinite(value) for value in target.values()), "Nonfinite finite-response target")
    return target, onset, [math.log(observed_peak), math.log(width)], proof


def _deduplicate(records, features):
    """Repeated runs of one configuration are one predictive unit."""
    kept, indices = [], []
    for index, row in enumerate(records):
        if row["configuration_group"] not in kept:
            kept.append(row["configuration_group"])
            indices.append(index)
    return [records[i] for i in indices], [features[i] for i in indices], indices


def analyze(bank, seed=20261024, repetitions=BOOTSTRAP_REPETITIONS, expected_zero=616, expected_pairs=48):
    traces = bank["traces"]
    roles, role_rows = assign_roles(traces, seed)
    zero = {s: trace for s, trace in sorted(traces.items()) if _number(trace["row"]["input_amplitude"]) == 0}
    if expected_zero is not None:
        require(len(zero) == expected_zero, "Unexpected zero-input configuration count")
    for trace in zero.values():
        require(np.all(np.asarray(trace["stimulus"]) == 0), "Declared spontaneous trace has observed pulse")
    references, feature_by_id, feature_rows = {}, {}, []
    for variant in sorted({trace["row"]["variant"] for trace in zero.values()}):
        training = [trace for s, trace in zero.items() if trace["row"]["variant"] == variant and roles[s]["role"] == "train"]
        # Repeated/noise runs of the same physical model are not overweighted.
        unique = {}
        for trace in training:
            unique.setdefault(structural_key(trace), trace)
        reference = reference_fit(list(unique.values()))
        references[variant] = reference
        for simulation_id, trace in zero.items():
            if trace["row"]["variant"] != variant:
                continue
            features = trace_features(trace, reference)
            feature_by_id[simulation_id] = features
            feature_rows.append(_base(trace, roles) | features)
    parameter_rows, parameter_models, parameter_coverages = [], {}, {}
    response_rows, response_models = [], {}
    neighbour_rows = []
    pair_members = defaultdict(list)
    for simulation_id, trace in traces.items():
        if trace["row"].get("pair_id"):
            pair_members[(trace["row"]["variant"], trace["row"]["pair_id"])].append(simulation_id)
    require(expected_pairs is None or len(pair_members) == expected_pairs, "Unexpected finite-pair count")
    for variant in sorted(references):
        eligible = [trace for trace in zero.values() if trace["row"]["variant"] == variant and parameter_eligible(trace)]
        records = [_base(trace, roles) for trace in eligible]
        features = [feature_by_id[trace["row"]["simulation_id"]] for trace in eligible]
        records, features, keep = _deduplicate(records, features)
        eligible = [eligible[i] for i in keep]
        if not records:
            parameter_coverages[variant] = {"status": "unsupported_no_actual_factor_sweep", "nominal_published_factors_excluded": True}
            continue
        factors = np.asarray([[math.log(_number(trace["row"][name])) for name in FACTORS] for trace in eligible])
        parameter_coverages[variant] = coverage(records, factors)
        designs = {"training_mean": np.empty((len(records), 0))}
        for space in ("cortical", "effective_mode", "full_population"):
            columns = feature_columns(features[0], space)
            designs[space] = np.asarray([[row[key] for key in columns] for row in features])
        parameter_coverages[variant]["training_feature_parameter_geometry"] = parameter_geometry(records, factors, designs)
        rows, selected = prediction_experiment(records, designs, factors, ("log_gain", "log_coupling", "log_history"))
        parameter_rows.extend(rows)
        parameter_models[variant] = selected
        train_indices = [i for i, row in enumerate(records) if row["role"] == "train"]
        test_indices = [i for i, row in enumerate(records) if row["role"] == "test"]
        for space in ("cortical", "effective_mode", "full_population"):
            x = designs[space]
            if not train_indices:
                continue
            mean, scale = x[train_indices].mean(0), x[train_indices].std(0)
            scale = np.where(scale > np.finfo(float).eps * np.maximum(abs(mean), 1e-300) * 100, scale, 1.)
            z = (x - mean) / scale
            for index in test_indices:
                for matching in ("any_training_family", "different_training_family"):
                    possible = [i for i in train_indices if matching == "any_training_family" or records[i]["account_class"] != records[index]["account_class"]]
                    if not possible:
                        continue
                    best = min(possible, key=lambda i: (float(np.mean((z[index] - z[i]) ** 2)), records[i]["simulation_id"]))
                    neighbour_rows.append(records[index] | {"representation": space, "matching": matching,
                        "training_neighbour": records[best]["simulation_id"], "neighbour_family": records[best]["account_class"],
                        "feature_rms_distance": float(np.sqrt(np.mean((z[index] - z[best]) ** 2))),
                        "log_factor_rms_distance": float(np.sqrt(np.mean((factors[index] - factors[best]) ** 2))),
                        "log_gain_plus_coupling_distance": float(abs(sum(factors[index, :2]) - sum(factors[best, :2]))),
                        "matched_signal_equivalence_established": False})
        # Response prediction is restricted to pre-input observed history.
        pair_records, pair_features, pair_factors, pair_inputs, targets = [], [], [], [], []
        for (pair_variant, pair_id), ids in sorted(pair_members.items()):
            if pair_variant != variant:
                continue
            controls = [traces[s] for s in ids if _number(traces[s]["row"]["input_amplitude"]) == 0]
            pulses = [traces[s] for s in ids if _number(traces[s]["row"]["input_amplitude"]) > 0]
            require(len(ids) == 2 and len(controls) == len(pulses) == 1, "Ambiguous pulse/control pair")
            control, pulse = controls[0], pulses[0]
            require(structural_key(control) == structural_key(pulse), "Pair members cross physical groups")
            require(parameter_eligible(control), "Paired simulation is outside actual factor families")
            target, onset, observed_input, proof = _response_targets(control, pulse)
            base = _base(control, roles) | {"pair_id": pair_id, "observed_input_amplitude": math.exp(observed_input[0])}
            pair_records.append(base)
            pair_features.append(trace_features(control, references[variant], before=onset - .010))
            pair_factors.append([math.log(_number(control["row"][key])) for key in FACTORS])
            pair_inputs.append(observed_input)
            targets.append([target[key] for key in RESPONSE_NAMES])
        if pair_records:
            # More than one amplitude/input on one config must remain one block;
            # this bank has exactly one, and duplicates would need multirow loss.
            require(len({row["configuration_group"] for row in pair_records}) == len(pair_records),
                    "Repeated physical configuration response needs explicit aggregation")
            input_covariates = np.asarray(pair_inputs, float)
            factors_and_input = np.column_stack((pair_factors, input_covariates))
            designs = {"training_mean": np.empty((len(pair_records), 0)), "observed_input_only": input_covariates,
                       "factors_and_observed_input": factors_and_input}
            for space in ("cortical", "effective_mode", "full_population"):
                names = feature_columns(pair_features[0], space)
                matrix = np.asarray([[row[key] for key in names] for row in pair_features])
                designs[space + "_and_observed_input"] = np.column_stack((matrix, input_covariates))
                if space == "effective_mode":
                    designs["factors_effective_mode_and_observed_input"] = np.column_stack((factors_and_input, matrix))
            rows, selected = prediction_experiment(pair_records, designs, targets, RESPONSE_NAMES)
            response_rows.extend(rows)
            response_models[variant] = selected
    modes = []
    for variant in sorted(references):
        for role in ("train", "validation", "test"):
            rows = [row for row in feature_rows if row["variant"] == variant and row["role"] == role]
            metrics = {}
            for key in ("V_raw_participation", "Q_raw_participation", "common_scaled_participation", "common_training_mode_fraction"):
                values = [row[key] for row in rows]
                metrics[key] = {"count": len(values), "median": float(np.median(values)) if values else None,
                                "range": [float(min(values)), float(max(values))] if values else []}
            modes.append({"variant": variant, "role": role, "summaries": metrics})
    bootstrap = {"parameter_recovery": bootstrap_predictions(parameter_rows, seed + 101, repetitions),
                 "finite_response_prediction": bootstrap_predictions(response_rows, seed + 202, repetitions)}
    return {"roles": role_rows, "features": feature_rows, "parameter_predictions": parameter_rows,
            "response_predictions": response_rows, "neighbours": neighbour_rows,
            "coverage": parameter_coverages, "parameter_models": parameter_models,
            "response_models": response_models, "mode_summaries": modes, "bootstrap": bootstrap,
            "references": references, "n_zero": len(zero), "n_pairs": len(pair_members)}


def _load_input_cache(inputs):
    from native_population import read_cache
    specs = inputs.get("files", inputs)
    cache = specs.get("population_cache", specs.get("native_cache"))
    require(cache is not None, "Missing population_cache input")
    cache_path = cache.get("path", cache.get("cache_dir")) if isinstance(cache, dict) else cache
    require(cache_path, "Missing population_cache path")
    receipt_path = Path(cache_path) / "receipt.json"
    if isinstance(cache, dict) and cache.get("receipt_sha256"):
        require(sha(receipt_path) == cache["receipt_sha256"], "Cache receipt differs from deployment pin")
    bank = read_cache(cache_path)
    require(bank["receipt"].get("schema") == "native_nine_population_cache_v2", "Repaired native clock cache required")
    return bank, Path(cache_path), specs


def qualification_probe(inputs, seed=20261024):
    """Real cache integrity, grouping and small actual-feature/input probe.

    Called only in the qualification compute allocation; it does not repeat the
    complete bank model fitting, bootstrap or any completed human analysis.
    """
    bank, cache_path, specs = _load_input_cache(inputs)
    traces = bank["traces"]
    require(len(traces) == 664, "Unexpected native cache size")
    roles, role_rows = assign_roles(traces, int(seed))
    zeros = {s: trace for s, trace in traces.items() if _number(trace["row"]["input_amplitude"]) == 0}
    require(len(zeros) == 616, "Unexpected zero-input count")
    probes = []
    for variant in sorted({trace["row"]["variant"] for trace in zeros.values()}):
        training = [trace for s, trace in sorted(zeros.items()) if trace["row"]["variant"] == variant and roles[s]["role"] == "train"]
        require(len(training) >= 2, "Insufficient actual training histories")
        reference = reference_fit(training[:2])
        held = next(trace for s, trace in sorted(zeros.items()) if trace["row"]["variant"] == variant and roles[s]["role"] == "test")
        features = trace_features(held, reference)
        probes.append({"variant": variant, "training_simulation_ids": reference["training_simulation_ids"],
                       "held_feature_probe_id": held["row"]["simulation_id"], "finite_feature_columns": len(features)})
    members = defaultdict(list)
    for simulation_id, trace in traces.items():
        if trace["row"].get("pair_id"):
            members[(trace["row"]["variant"], trace["row"]["pair_id"])].append(simulation_id)
    require(len(members) == 48, "Unexpected actual pair count")
    for ids in members.values():
        require(len(ids) == 2 and len({roles[s]["configuration_group"] for s in ids}) == 1,
                "Actual pulse/control members cross physical grouping")
    ids = members[sorted(members)[0]]
    control = next(traces[s] for s in ids if _number(traces[s]["row"]["input_amplitude"]) == 0)
    pulse = next(traces[s] for s in ids if _number(traces[s]["row"]["input_amplitude"]) > 0)
    target, onset, observed, proof = _response_targets(control, pulse)
    return {"status": "cross_mechanism_compensation_runtime_qualified", "version": VERSION,
            "native_cache_receipt_sha256": sha(cache_path / "receipt.json"), "native_traces": len(traces),
            "zero_input_configurations": len(zeros), "paired_groups": len(members),
            "clock_labels_preserved": True, "actual_feature_probes": probes,
            "actual_pair_probe": {"control_id": control["row"]["simulation_id"], "onset_relative_s": onset,
                                  "finite_target_columns": len(target), "matched_observed_noise": bool(proof["full_noise_stream_exactly_equal"])},
            "full_model_selection_repeated": False, "no_preferred_scientific_result_gate": True}


def run(inputs, output, seed=20261024):
    bank, cache_path, specs = _load_input_cache(inputs)
    receipt_path = cache_path / "receipt.json"
    repetitions = inputs.get("bootstrap_repetitions", specs.get("bootstrap_repetitions", BOOTSTRAP_REPETITIONS))
    result = analyze(bank, int(seed), int(repetitions))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for filename, key in (("configuration_roles.csv", "roles"), ("zero_input_features.csv", "features"),
                          ("parameter_predictions.csv", "parameter_predictions"),
                          ("finite_response_predictions.csv", "response_predictions"),
                          ("cross_family_neighbours.csv", "neighbours")):
        _csv(out / filename, result[key])
    _json(out / "parameter_coverage.json", result["coverage"])
    _json(out / "configuration_bootstrap.json", result["bootstrap"])
    references = {variant: {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in ref.items()}
                  for variant, ref in result["references"].items()}
    summary = {"status": "completed_cross_mechanism_configuration_analysis", "version": VERSION,
               "seed": int(seed), "zero_input_configurations": result["n_zero"], "qualified_finite_pairs": result["n_pairs"],
               "native_cache_receipt_sha256": sha(receipt_path), "parameter_recovery": result["parameter_models"],
               "finite_response_prediction": result["response_models"], "mode_summaries": result["mode_summaries"],
               "training_only_population_reference": references,
               "interpretation": "Model-bank effective dynamics, finite-input prediction and parameter ambiguity. Simulator variants separate; deliberately sampled configurations are not independent human observations.",
               "scope": {"brain_spatial_nodes": 1, "human_held_test_targets_used": False,
                         "new_simulations": False, "whole_family_exact_hsyn_nu_equivalence_asserted": False,
                         "selective_causal_pathway_identified": False, "full_delay_system_jacobian_estimated": False,
                         "bootstrap_units": "held physical model configurations within family and implementation",
                         "human_finetuning_gates": "enforced before this module by root driver", "behavioral_target": 42}}
    _json(out / "summary.json", summary)
    receipt = {"status": summary["status"], "version": VERSION, "seed": int(seed),
               "zero_input_configurations": result["n_zero"], "qualified_finite_pairs": result["n_pairs"],
               "native_cache_receipt_sha256": summary["native_cache_receipt_sha256"],
               "output_sha256": {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()},
               "human_data_refitted": False, "human_test_selection": False, "target_participants": 42}
    _json(out / "receipt.json", receipt)
    return receipt
