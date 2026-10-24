"""Frozen-person, common-clock full-population observation-bottleneck test.

Only NumPy and the standard library are required, plus the sibling native cache
loader. Native states are effective predictors of frozen human EEG coordinates;
neither fitted mixtures nor labels identify anatomical human populations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np


VERSION = "full_population_bridge_v2_observed_clock"
VIEWS = ("cortical_voltage", "cortical_delay18", "population_voltage9", "population_voltage_rate18")
PRIMARY = ("population_voltage_rate18", "cortical_delay18")
ROLES = ("mechanism_selection", "mechanism_training", "mechanism_evaluation", "account_predictions",
         "state_predictions", "observation_map", "state_rows", "state_arrays", "state_models",
         "state_selection", "state_split", "native_bank")
NOMINAL_STOP = 9.998


def require(value, message):
    if not value:
        raise ValueError("Population bridge: " + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def number(value):
    require(not isinstance(value, (bool, np.bool_)), "Boolean numeric value")
    value = float(value)
    require(math.isfinite(value), "Nonfinite numeric value")
    return value


def mean(values):
    values = list(values)
    require(bool(values), "Empty mean")
    return math.fsum(number(x) / len(values) for x in values)


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), "CSV fields differ")
        rows = list(reader)
    require(len(rows) <= 500000 and all(None not in r and all(v is not None for v in r.values()) for r in rows), "CSV budget/width differs")
    return rows


def write_json(path, value):
    with Path(path).open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(path, rows):
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["status"]
    with Path(path).open("x", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_inputs(inputs):
    specs = inputs.get("files", inputs)
    paths, pins = {}, {}
    for role in ROLES:
        require(role in specs, "Missing input " + role)
        spec = specs[role]
        path = Path(spec["path"])
        require(path.is_file() and not path.is_symlink() and re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256", "")), "Plain pinned input required: " + role)
        require(sha(path) == spec["sha256"], "Input hash differs: " + role)
        paths[role], pins[role] = path, spec["sha256"]
    return paths, pins


def array(archive, key, ndim):
    require(key in archive, "Missing actual array: " + key)
    value = np.asarray(archive[key], dtype=float)
    require(value.ndim == ndim and value.size and np.isfinite(value).all(), "Array rank/finite status differs: " + key)
    return value


def interpolate(values, times, queries):
    y, t, q = np.asarray(values, float), np.asarray(times, float), np.asarray(queries, float)
    require(y.ndim == 2 and t.ndim == q.ndim == 1 and len(t) == len(y) and len(t) >= 4
            and all(np.isfinite(x).all() for x in (y, t, q)) and np.all(np.diff(t) > 0)
            and len(q) > 0 and q.min() >= t[0] - 1e-8 and q.max() <= t[-1] + 1e-8,
            "Interpolation cannot extrapolate, pad, or join episodes")
    return np.column_stack([np.interp(q, t, y[:, j]) for j in range(y.shape[1])])


def validate_design(selection, training, state_selection, split, rows, account_rows, maps, expected_maps=10):
    require(training.get("test_outcomes_used_for_selection") is False
            and training.get("human_parameter_fitting_executed") is True, "Original adaptation/split provenance differs")
    roles = {r: set(training["split"][r]) for r in ("train", "validation", "test")}
    require(all(roles[r] <= set(split[r + "_participant_ids"]) for r in roles)
            and not any(roles[a] & roles[b] for a, b in itertools.combinations(roles, 2)), "Frozen person roles overlap or escape parent split")
    descriptions = selection["maps"]
    require(len(descriptions) == training["maps"] and len({m["map_key"] for m in descriptions}) == len(descriptions), "Original map inventory differs")
    by_source = defaultdict(list)
    for m in descriptions:
        by_source[m["source_model_key"]].append(m)
    neural = {m["source_model_key"]: m for m in state_selection["source_models"]}
    for source, group in by_source.items():
        require(len(group) == expected_maps and source in neural, "Complete account inventory required")
        require(set(neural[source]["training_participant_ids"]) <= roles["train"]
                and set(neural[source]["validation_participant_ids"]) <= roles["validation"], "Frozen neural basis was trained on other roles")
    row_index, person_roles = {}, defaultdict(set)
    for row in rows:
        if row.get("status") != "included":
            continue
        identity = (row["source_id"], row["participant_id"], row["episode_id"])
        require(identity not in row_index and row["split"] in roles and row["participant_id"] in roles[row["split"]], "State identity/role differs")
        row_index[identity] = row
        person_roles[row["participant_id"]].add(row["split"])
    require(all(len(v) == 1 for v in person_roles.values()), "Person crosses frozen partitions")
    panels = defaultdict(dict)
    info = {m["map_key"]: m for m in descriptions}
    for row in account_rows:
        key = row["map_key"]
        require(key in info and row["split"] in ("validation", "test"), "Original prediction identity differs")
        identity = (row["source_id"], row["participant_id"], row["episode_id"])
        require(identity in row_index and row_index[identity]["split"] == row["split"]
                and identity not in panels[(key, row["split"])], "Prediction panel duplicates or changes original roles")
        require(all(row[k] == info[key][k] for k in ("source_model_key", "account_class", "variant")), "Prediction account metadata differs")
        require(row_index[identity]["source_model_key"] == info[key]["source_model_key"], "Prediction target belongs to a different source basis")
        panels[(key, row["split"])][identity] = row
    clocks, cuts, census = {}, {}, []
    for source, group in sorted(by_source.items()):
        key0 = group[0]["map_key"]
        reference_grid = array(maps, key0 + "__times_s", 1)
        require(np.all(np.diff(reference_grid) > 0), "Frozen clock is not increasing")
        for role in ("validation", "test"):
            reference = set(panels[(key0, role)])
            require(reference, "No common original " + role + " panel")
            for m in group:
                key = m["map_key"]
                require(set(panels[(key, role)]) == reference, "Original account case sets differ; silent intersection forbidden")
                require(np.array_equal(array(maps, key + "__times_s", 1), reference_grid), "Exact frozen account clock differs")
                for row in panels[(key, role)].values():
                    cut = int(row["conditioning_states"])
                    require(4 <= cut < len(reference_grid) - 2 and int(row["grid_states"]) == len(reference_grid), "Conditioning clock differs")
                    cuts.setdefault(source, cut)
                    require(cut == cuts[source], "Conditioning stop differs across accounts/cases")
            eligible = {i for i, r in row_index.items() if r["source_model_key"] == source and r["split"] == role}
            census.append({"source_model_key": source, "split": role, "common_people": len({i[1] for i in reference}),
                           "common_episodes": len(reference), "eligible_episodes": len(eligible),
                           "unrepresented_episodes": [{"participant_id": i[1], "episode_id": i[2], "reason": "absent_from_original_common_panel"} for i in sorted(eligible - reference)]})
        clocks[source] = reference_grid
    return roles, by_source, row_index, panels, clocks, cuts, census


def examples_for(source, role, row_index, panels, descriptions, archive, grid, cut):
    if role == "train":
        identities = sorted(i for i, r in row_index.items() if r["source_model_key"] == source and r["split"] == role)
    else:
        identities = sorted(panels[(descriptions[0]["map_key"], role)])
    result = []
    for identity in identities:
        row = row_index[identity]
        token = row["state_token"]
        state = array(archive, token + "__state", 2)
        times = array(archive, token + "__times_s", 1)
        dt = float(np.median(np.diff(times)))
        require(len(times) == len(state) and np.allclose(np.diff(times), dt, rtol=1e-6, atol=1e-7)
                and math.isclose(dt, number(row["state_bin_s"]), rel_tol=1e-6, abs_tol=1e-7), "Original state sampling interval differs")
        queries = times[-1] - NOMINAL_STOP + grid
        support = row.get("baseline_support_stop_s", "")
        require(support == "" or number(support) <= queries[cut - 1] + 1e-8, "Parent nuisance centering reaches future forecast targets")
        y = interpolate(state, times, queries)
        if role != "train":
            require(int(panels[(descriptions[0]["map_key"], role)][identity]["latent_rank"]) == y.shape[1], "Original declared state rank differs")
        result.append({"row": row, "identity": identity, "y": y, "cut": cut, "grid": grid,
                       "dt": dt, "scale": max(float(np.mean(y[:cut] ** 2)), 1e-9)})
    require(result, "No actual " + role + " examples")
    return result


def temporal_geometry(y, points=12):
    positions = np.linspace(0, len(y) - 1, points)
    z = np.column_stack([np.interp(positions, np.arange(len(y)), y[:, j]) for j in range(y.shape[1])])
    z -= z.mean(axis=0)
    energy = float(np.mean(np.sum(z * z, axis=1)))
    if energy <= 1e-15:
        return np.zeros(points * (points + 1) // 2)
    return (z @ z.T / energy)[np.triu_indices(points)]


def available_history(grid, candidates, requested_max_delay):
    """Bound one shared delay range by observed clocks, without reading outcomes."""
    grid = np.asarray(grid, float)
    require(grid.ndim == 1 and len(grid) >= 4 and np.isfinite(grid).all()
            and np.all(np.diff(grid) > 0), "Frozen comparison clock is invalid")
    require(candidates and np.isfinite(requested_max_delay) and requested_max_delay > 0,
            "Positive requested cortical history and native candidates required")
    first, last = [], []
    for trace in candidates:
        times = np.asarray(trace["times"], float)
        require(times.ndim == 1 and len(times) >= 4 and np.isfinite(times).all()
                and np.all(np.diff(times) > 0), "Observed native clock is invalid")
        first.append(float(times[0])); last.append(float(times[-1]))
    latest_start, earliest_stop = max(first), min(last)
    history = float(grid[0] - latest_start)
    require(history > 0 and grid[-1] <= earliest_stop + 1e-8,
            "Positive shared observed cortical history on the frozen human clock required")
    used = min(float(requested_max_delay), history)
    return {"requested_max_delay_s": float(requested_max_delay), "used_max_delay_s": used,
            "earliest_frozen_forecast_time_s": float(grid[0]),
            "latest_native_first_sample_s": latest_start, "earliest_native_last_sample_s": earliest_stop,
            "candidate_configurations": len(candidates), "delay_columns": 18,
            "support_reason": "limited_by_observed_native_history" if used < requested_max_delay else "requested_history_fully_observed",
            "shared_across_all_account_candidates": True, "human_target_grid_changed": False,
            "native_timestamps_rebased": False, "human_test_targets_used": False}


def native_features(trace, grid, cut, view, max_delay=.2):
    require(view in VIEWS and isinstance(trace.get("columns"), list), "Undeclared native view")
    columns = trace["columns"]
    expected = ["Pop." + str(i) + "." + q for i in range(1, 10) for q in ("V", "Q")]
    require(columns == expected, "Exactly nine observed endogenous voltage/rate populations required")
    values = np.asarray(trace["values"], float)
    times = np.asarray(trace["times"], float)
    if view == "cortical_delay18":
        delays = np.linspace(0, max_delay, 18)
        raw = np.column_stack([interpolate(values[:, :1], times, grid - lag)[:, 0] for lag in delays])
    else:
        ids = [0] if view == "cortical_voltage" else list(range(0, 18, 2)) if view == "population_voltage9" else list(range(18))
        raw = interpolate(values[:, ids], times, grid)
    # Native prediction conditioning is known; future samples never set its origin or units.
    center = raw[:cut].mean(axis=0)
    scale = raw[:cut].std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    return (raw - center) / scale


def weights_for(examples):
    counts = Counter(e["row"]["participant_id"] for e in examples)
    return np.asarray([1. / (len(counts) * counts[e["row"]["participant_id"]]) for e in examples])


def weighted_basis(blocks, weights, rank):
    require(len(blocks) == len(weights) and sum(weights) > 0, "Training-only feature blocks required")
    x = np.concatenate(blocks)
    w = np.concatenate([np.full(len(v), a / len(v)) for v, a in zip(blocks, weights)])
    center = np.sum(x * w[:, None], axis=0)
    scale = np.sqrt(np.sum((x - center) ** 2 * w[:, None], axis=0))
    scale = np.where(scale > 1e-9, scale, 1.)
    normalized = (x - center) / scale
    covariance = (normalized * w[:, None]).T @ normalized
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    k = min(int(rank), normalized.shape[1])
    require(k >= 1 and np.max(eigenvalues) > 1e-15, "No varying training features")
    basis = eigenvectors[:, order[:k]]
    # Fix PCA sign without outcome information for deterministic serialization.
    for j in range(k):
        pivot = int(np.argmax(np.abs(basis[:, j])))
        if basis[pivot, j] < 0:
            basis[:, j] *= -1
    return {"center": center, "scale": scale, "basis": basis,
            "eigenvalues": eigenvalues[order[:k]], "actual_feature_rank": int(np.sum(eigenvalues > 1e-10))}


def transform(x, basis):
    return ((x - basis["center"]) / basis["scale"]) @ basis["basis"]


def fit_ridge(X, Y, weights, ridge):
    x, y = np.concatenate(X), np.concatenate(Y)
    w = np.concatenate([np.full(len(v), a / len(v)) for v, a in zip(X, weights)])
    mx, my = np.sum(x * w[:, None], axis=0), np.sum(y * w[:, None], axis=0)
    xx, yy = x - mx, y - my
    gram = (xx * w[:, None]).T @ xx
    W = np.linalg.solve(gram + ridge * np.eye(x.shape[1]), (xx * w[:, None]).T @ yy)
    bias = my - mx @ W
    eigen = np.maximum(np.linalg.eigvalsh(gram), 0.)
    df_per_output = float(np.sum(eigen / (eigen + ridge))) + 1.
    require(np.isfinite(W).all(), "Readout fit diverged")
    return W, bias, df_per_output


def choose_conditioning(examples, predictions, offset_strength):
    cut = examples[0]["cut"]
    require(all(e["cut"] == cut for e in examples), "Conditioning clocks differ")
    y = np.asarray([e["y"][:cut] for e in examples])
    p = predictions[:, :cut]
    a, b = y.reshape(len(y), -1), p.reshape(len(p), -1)
    distance = np.mean(a * a, axis=1)[:, None] + np.mean(b * b, axis=1)[None, :] - 2 * a @ b.T / a.shape[1]
    offsets = y.mean(axis=1)[:, None, :] - p.mean(axis=1)[None, :, :]
    distance -= (2 * offset_strength - offset_strength ** 2) * np.mean(offsets ** 2, axis=2)
    selected = np.argmin(distance, axis=1)
    return selected, offsets[np.arange(len(y)), selected] * offset_strength


def predict(examples, raw_kernels, fitted):
    ids = fitted["support_indices"]
    predictions = transform(raw_kernels[ids], fitted["feature_basis"]) @ fitted["W"] + fitted["bias"]
    chosen, offset = choose_conditioning(examples, predictions, fitted["offset_strength"])
    selected = np.asarray(ids)[chosen]
    return predictions[chosen] + offset[:, None, :], selected, offset


def loss(y, prediction, scale):
    require(y.shape == prediction.shape and np.isfinite(prediction).all(), "Prediction shape/finite status differs")
    return number(np.mean((y - prediction) ** 2) / scale)


def validation_loss(examples, predictions):
    people = defaultdict(list)
    for e, p in zip(examples, predictions):
        people[e["row"]["participant_id"]].append(loss(e["y"][e["cut"]:], p[e["cut"]:], e["scale"]))
    return mean(mean(v) for v in people.values())


def fit_observation(train, validation, raw_kernels, rank, ridge, offset_strength, iterations=2):
    weights = weights_for(train)
    cut = train[0]["cut"]
    native_geometry = np.asarray([temporal_geometry(x[:cut]) for x in raw_kernels])
    human_geometry = np.asarray([temporal_geometry(e["y"][:cut]) for e in train])
    distance = np.sum(human_geometry ** 2, axis=1)[:, None] + np.sum(native_geometry ** 2, axis=1)[None, :] - 2 * human_geometry @ native_geometry.T
    assignments = np.argmin(distance, axis=1)
    # Alternating assignment sees only training conditioning states.
    for step in range(iterations + 1):
        basis = weighted_basis([raw_kernels[j, :cut] for j in assignments], weights, rank)
        kernels = transform(raw_kernels, basis)
        W, bias, df = fit_ridge([kernels[j, :cut] for j in assignments], [e["y"][:cut] for e in train], weights, ridge)
        if step < iterations:
            assignments, _ = choose_conditioning(train, kernels @ W + bias, offset_strength)
    fitted = {"feature_basis": basis, "W": W, "bias": bias, "feature_rank": rank,
              "ridge": ridge, "offset_strength": offset_strength,
              "support_indices": sorted(set(assignments.tolist())), "train_assignments": assignments.tolist(),
              "effective_df_per_output": df, "effective_readout_df": df * train[0]["y"].shape[1],
              "nominal_readout_coefficients": (min(rank, raw_kernels.shape[-1]) + 1) * train[0]["y"].shape[1],
              "assignment_iterations": iterations, "training_people": sorted({e["row"]["participant_id"] for e in train}),
              "training_episode_ids": [e["row"]["episode_id"] for e in train]}
    val_prediction, val_assignment, _ = predict(validation, raw_kernels, fitted)
    fitted["validation_loss"] = validation_loss(validation, val_prediction)
    fitted["validation_assignments"] = val_assignment.tolist()
    return fitted


def recurse(initial, queries, A, bias, dt):
    require(dt > 0 and np.all(queries > 0) and np.all(np.diff(queries) > 0), "Forecast times differ")
    count = int(math.ceil(float(queries[-1]) / dt))
    require(count <= 100000, "Forecast recursion budget exceeded")
    values = [np.asarray(initial, float)]
    for _ in range(count):
        values.append(values[-1] @ A + bias)
    values = np.asarray(values)
    require(np.isfinite(values).all(), "Statistical dynamics diverged; no clipping permitted")
    return np.column_stack([np.interp(queries, np.arange(count + 1) * dt, values[:, j]) for j in range(values.shape[1])])


def fit_statistical(train, validation, ridges):
    weights = weights_for(train)
    alternatives = []
    for ridge in ridges:
        A, bias, df = fit_ridge([e["y"][:e["cut"] - 1] for e in train], [e["y"][1:e["cut"]] for e in train], weights, ridge)
        predictions = []
        for e in validation:
            cut, y = e["cut"], e["y"]
            future = recurse(y[cut - 1], np.arange(1, len(y) - cut + 1, dtype=float), A, bias, 1.)
            predictions.append(np.concatenate([y[:cut], future]))
        alternatives.append({"A": A, "bias": bias, "ridge": ridge, "validation_loss": validation_loss(validation, predictions), "effective_df_per_output": df})
    return min(alternatives, key=lambda x: (x["validation_loss"], x["ridge"]))


def paired_inference(differences, seed, bootstrap_replicates):
    ids = sorted(differences)
    x = np.asarray([number(differences[p]) for p in ids])
    n = len(x)
    require(1 <= n <= 20, "Participant inference budget differs")
    report = {"n_people": n, "person_ids": ids, "mean_difference": mean(x),
              "person_differences": [{"participant_id": p, "difference": float(v)} for p, v in zip(ids, x)],
              "conditional_on_fixed_models": True, "equivalence_established": False}
    if n < 2:
        return report | {"bootstrap_interval95": None, "exact_signflip_two_sided": None}
    rng = np.random.default_rng(seed)
    boot = np.mean(x[rng.integers(0, n, (bootstrap_replicates, n))], axis=1)
    observed = abs(float(x.mean()))
    tolerance = 16 * np.finfo(float).eps * max(float(np.max(np.abs(x))), observed)
    extreme = 0
    for start in range(0, 1 << n, 4096):
        masks = np.arange(start, min(start + 4096, 1 << n), dtype=np.uint64)
        signs = 1 - 2 * ((masks[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float)
        extreme += int(np.sum(np.abs(signs @ x / n) >= observed - tolerance))
    report.update(bootstrap_interval95=np.quantile(boot, [.025, .975]).tolist(),
                  bootstrap_replicates=bootstrap_replicates, exact_signflip_two_sided=extreme / (1 << n),
                  exact_reference_assumption="Independent participant differences symmetric about zero; no randomized human assignment",
                  leave_one_person_out=[{"omitted_participant_id": p, "mean_difference": mean(x[j] for j in range(n) if j != i)} for i, p in enumerate(ids)])
    return report


def holm(rows):
    ordered = sorted(((i, r["exact_signflip_two_sided"]) for i, r in enumerate(rows) if r["exact_signflip_two_sided"] is not None), key=lambda v: (v[1], v[0]))
    running = 0.
    for rank, (i, value) in enumerate(ordered):
        running = max(running, min(1., (len(ordered) - rank) * value))
        rows[i].update(holm_adjusted_reference=running, multiplicity_family_size=len(ordered))


def serialize_model(fitted):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value
    return convert(fitted)


def original_prediction(example, description, predictions, account_panel):
    key, token = description["map_key"], example["row"]["state_token"]
    value = array(predictions, key + "__" + token + "__prediction", 2)
    cut = example["cut"]
    computed = loss(example["y"][cut:], value[cut:], example["scale"])
    recorded = number(account_panel[example["identity"]]["later_state_standardized_mse"])
    require(math.isclose(computed, recorded, rel_tol=1e-7, abs_tol=1e-9), "Original scalar loss cannot be reconstructed")
    return value


def summarize_cases(cases, selected_accounts, seed, bootstrap_replicates):
    panel = defaultdict(dict)
    for row in cases:
        identity = (row["source_model_key"], row["account_class"], row["variant"], row["participant_id"], row["episode_id"])
        require(row["predictor"] not in panel[identity], "Duplicate predictor/case")
        panel[identity][row["predictor"]] = row
    comparisons = [("matched_population_voltage_rate18", "matched_cortical_delay18"),
                   ("selected_population_voltage_rate18", "selected_cortical_delay18"),
                   ("selected_population_voltage9", "selected_cortical_voltage"),
                   ("matched_population_voltage_rate18", "original_scalar_map"),
                   ("matched_population_voltage_rate18", "training_ar"),
                   ("matched_population_voltage_rate18", "frozen_neural_transition"),
                   ("matched_population_voltage_rate18", "persistence")]
    groups = defaultdict(list)
    for identity, values in panel.items():
        source, account, variant, person, episode = identity
        for a, b in comparisons:
            require(a in values and b in values, "Common predictor case set differs")
            groups[(source, account, variant, a, b, person)].append(values[a]["later_state_standardized_mse"] - values[b]["later_state_standardized_mse"])
    results = []
    for source, account, variant, a, b in sorted({k[:-1] for k in groups}):
        people = {k[-1]: mean(v) for k, v in groups.items() if k[:-1] == (source, account, variant, a, b)}
        result = {"source_model_key": source, "account_class": account, "variant": variant,
                  "predictor": a, "comparator": b, "difference_direction": "negative_favors_predictor",
                  "validation_selected_account": selected_accounts[source] == (account, variant),
                  **paired_inference(people, seed, bootstrap_replicates)}
        results.append(result)
    holm(results)
    selected = [dict(r) for r in results if r["validation_selected_account"]]
    holm(selected)
    return {"all_account_comparisons": results, "validation_selected_account_comparisons": selected,
            "multiplicity_scope": "Holm across all declared account-by-contrast tests; separately across selected-account contrasts",
            "interval_scope": "Exploratory percentile participant bootstrap, conditional on fixed maps and historical development exposure; not simultaneous or training-uncertainty intervals"}


def run(inputs, output, seed=20261024, bootstrap_replicates=20000, expected_maps_per_source=10,
        feature_ranks=(2, 4, 8), ridges=(.01, .1, 1.), offset_strengths=(0., 1.), assignment_iterations=2):
    """Fit only train people; freeze validation choices before reading test targets.

    Caller verifies both actual human-data fine-tunings before this run. Inputs
    are pinned original mechanism roles plus population_cache (loader output).
    """
    require(type(seed) is int and 0 <= seed < 2 ** 64, "Unsigned seed required")
    require(type(bootstrap_replicates) is int and 100 <= bootstrap_replicates <= 100000, "Bootstrap budget differs")
    require(tuple(feature_ranks) == (2, 4, 8) and tuple(ridges) == (.01, .1, 1.)
            and tuple(offset_strengths) == (0., 1.) and assignment_iterations == 2, "Predeclared matched search grid differs")
    output = Path(output)
    require(not output.exists(), "Fresh result directory required")
    paths, pins = read_inputs(inputs)
    selection, training, evaluation = (json_read(paths[x]) for x in ("mechanism_selection", "mechanism_training", "mechanism_evaluation"))
    state_selection, split = (json_read(paths[x]) for x in ("state_selection", "state_split"))
    require(evaluation["training_receipt_sha256"] == pins["mechanism_training"]
            and evaluation["observation_map_sha256"] == pins["observation_map"]
            and evaluation["account_predictions_sha256"] == pins["account_predictions"]
            and training["observation_map"]["sha256"] == pins["observation_map"]
            and state_selection["models_sha256"] == pins["state_models"]
            and state_selection["split_sha256"] == pins["state_split"], "Original metadata/array bindings differ")
    require("population_cache" in inputs, "Verified population cache is required")
    import native_population
    bank = native_population.read_cache(inputs["population_cache"])
    traces = bank["traces"]
    scored = {r["simulation_id"]: r for r in csv_read(paths["native_bank"])}
    require(traces and len(scored) == len(csv_read(paths["native_bank"])), "Duplicate or empty native bank")
    native_groups = defaultdict(list)
    for identity, trace in sorted(traces.items()):
        if identity not in scored:
            continue
        row = scored[identity]
        require(all(trace["row"].get(k) == row.get(k) for k in ("source_output_sha256", "config_sha256", "account_class", "variant")), "Cached native provenance differs from scored bank")
        if number(row.get("input_amplitude", 0)) == 0:
            native_groups[(row["account_class"], row["variant"])].append(trace)
    rows, account_rows = csv_read(paths["state_rows"]), csv_read(paths["account_predictions"])
    models, grid_records, frozen_groups, compatibility_groups, cases = {}, [], {}, [], []
    prediction_arrays, selected_accounts, census, delay_support = {}, {}, [], {}
    with np.load(paths["state_arrays"], allow_pickle=False) as states, \
            np.load(paths["state_predictions"], allow_pickle=False) as original_predictions, \
            np.load(paths["observation_map"], allow_pickle=False) as maps, \
            np.load(paths["state_models"], allow_pickle=False) as neural:
        roles, by_source, row_index, panels, clocks, cuts, census = validate_design(selection, training, state_selection, split, rows, account_rows, maps, expected_maps_per_source)
        for source, descriptions in sorted(by_source.items()):
            grid, cut = clocks[source], cuts[source]
            # Use every original account's candidates to freeze one available-history range.
            # Actual NFTsim labels start at .002s; their origin and human target clock stay fixed.
            common_candidates = {t["row"]["simulation_id"]: t for description in descriptions
                                 for t in native_groups[(description["account_class"], description["variant"]) ]}
            delay_support[source] = available_history(grid, list(common_candidates.values()),
                max(selection["config"]["lags"]) * selection["config"]["delay_seconds_per_index"])
            train = examples_for(source, "train", row_index, panels, descriptions, states, grid, cut)
            validation = examples_for(source, "validation", row_index, panels, descriptions, states, grid, cut)
            statistical = fit_statistical(train, validation, ridges)
            groups = []
            for description in sorted(descriptions, key=lambda m: m["map_key"]):
                account, variant = description["account_class"], description["variant"]
                candidates = native_groups[(account, variant)]
                require(candidates, "Missing finite spontaneous population bank for original account")
                # Identical candidates for every representation, never a feature-specific shortlist.
                kernels = {v: np.asarray([native_features(t, grid, cut, v, delay_support[source]["used_max_delay_s"]) for t in candidates]) for v in VIEWS}
                alternatives = {}
                for view in VIEWS:
                    ranks = (1,) if view == "cortical_voltage" else feature_ranks
                    for rank, ridge, offset in itertools.product(ranks, ridges, offset_strengths):
                        fitted = fit_observation(train, validation, kernels[view], rank, ridge, offset, assignment_iterations)
                        key = (view, rank, ridge, offset)
                        alternatives[key] = fitted
                        grid_records.append({"source_model_key": source, "account_class": account, "variant": variant,
                            "view": view, "rank": rank, "ridge": ridge, "offset_strength": offset,
                            "validation_loss": fitted["validation_loss"], "effective_readout_df": fitted["effective_readout_df"],
                            "nominal_readout_coefficients": fitted["nominal_readout_coefficients"],
                            "candidate_configurations": len(candidates), "training_supported_configurations": len(fitted["support_indices"]),
                            "cortical_delay_requested_max_s": delay_support[source]["requested_max_delay_s"],
                            "cortical_delay_used_max_s": delay_support[source]["used_max_delay_s"],
                            "cortical_delay_support_reason": delay_support[source]["support_reason"]})
                chosen = {}
                for view in VIEWS:
                    key = min((k for k in alternatives if k[0] == view), key=lambda k: (alternatives[k]["validation_loss"], k[1:]))
                    chosen["selected_" + view] = alternatives[key]
                common = list(itertools.product(feature_ranks, ridges, offset_strengths))
                matched = min(common, key=lambda hp: (mean(alternatives[(v,) + hp]["validation_loss"] for v in PRIMARY), hp))
                for view in PRIMARY:
                    chosen["matched_" + view] = alternatives[(view,) + matched]
                group_key = description["map_key"]
                for name, fitted in chosen.items():
                    model_key = group_key + "__" + name
                    fitted.update(model_key=model_key, source_model_key=source, account_class=account, variant=variant,
                                  view=name.removeprefix("selected_").removeprefix("matched_"),
                                  candidate_ids=[t["row"]["simulation_id"] for t in candidates],
                                  support_ids=[candidates[j]["row"]["simulation_id"] for j in fitted["support_indices"]])
                    models[model_key] = fitted
                    if name.startswith("matched_"):
                        support_ids = sorted(set(fitted["support_ids"]) | {candidates[j]["row"]["simulation_id"] for j in fitted["validation_assignments"]})
                        compatibility_groups.append({"source_model_key": source, "account_class": account, "variant": variant,
                            "view": fitted["view"], "model_key": model_key, "compatible_simulation_ids": support_ids,
                            "assigned_train_ids": fitted["support_ids"],
                            "validation_supported_ids": sorted({candidates[j]["row"]["simulation_id"] for j in fitted["validation_assignments"]}),
                            "rank": fitted["feature_rank"], "ridge": fitted["ridge"], "offset_strength": fitted["offset_strength"]})
                groups.append((description, candidates, kernels, chosen))
            # Account selection uses the full-population/delay joint validation criterion only.
            selected_group = min(groups, key=lambda g: (mean(g[3]["matched_" + v]["validation_loss"] for v in PRIMARY), g[0]["account_class"], g[0]["variant"]))
            selected_accounts[source] = (selected_group[0]["account_class"], selected_group[0]["variant"])
            frozen_groups[source] = {"descriptions": descriptions, "groups": groups, "statistical": statistical}
        output.mkdir(parents=True)
        # On-disk immutable choice record is written before any test target is loaded.
        freeze = {"schema_version": 1, "version": VERSION, "selection_uses_test_targets": False,
                  "feature_scaling_and_pca": "Assigned training-person conditioning states only",
                  "selection_rule": "Minimum equal-person validation future-state standardized MSE; fixed hyperparameter tie order",
                  "matched_selection_rule": "Joint mean validation loss for population_voltage_rate18 and cortical_delay18 at identical feature-rank/ridge/offset settings",
                  "cortical_delay_support": delay_support,
                  "selected_accounts": {k: list(v) for k, v in selected_accounts.items()},
                  "frozen_person_roles": {k: sorted(v) for k, v in roles.items()},
                  "models": {key: serialize_model(value) for key, value in models.items()},
                  "statistical_models": {s: serialize_model(g["statistical"]) for s, g in frozen_groups.items()},
                  "native_cache_receipt": bank["receipt"]}
        write_json(output / "training_validation_freeze.json", freeze)
        write_csv(output / "validation_grid.csv", grid_records)
        compatibility = {"schema_version": 1, "selection_uses_test_targets": False, "selection_unit": "training/validation",
            "criterion": "Configuration is retained by a matched-capacity map fitted on training conditioning observations; validation conditioning chooses only among that training support. No equivalence threshold is asserted.",
            "compatible_simulation_ids": sorted(set().union(*(set(g["compatible_simulation_ids"]) for g in compatibility_groups))),
            "groups": compatibility_groups, "physiological_equivalence_established": False,
            "training_validation_freeze_sha256": sha(output / "training_validation_freeze.json")}
        write_json(output / "compatibility.json", compatibility)
        for source, context in sorted(frozen_groups.items()):
            descriptions, grid, cut = context["descriptions"], clocks[source], cuts[source]
            test = examples_for(source, "test", row_index, panels, descriptions, states, grid, cut)
            for description, candidates, kernels, chosen in context["groups"]:
                account, variant = description["account_class"], description["variant"]
                controls = context["statistical"]
                future_queries = grid[cut:] - grid[cut - 1]
                actual = {name: predict(test, kernels[fitted["view"]], fitted) for name, fitted in chosen.items()}
                for i, e in enumerate(test):
                    row, y, token = e["row"], e["y"], e["row"]["state_token"]
                    predicted = {name: values[0][i] for name, values in actual.items()}
                    predicted["original_scalar_map"] = original_prediction(e, description, original_predictions, panels[(description["map_key"], "test")])
                    predicted["persistence"] = np.repeat(y[cut - 1][None, :], len(y), axis=0)
                    ar_future = recurse(y[cut - 1], np.arange(1, len(y) - cut + 1, dtype=float), controls["A"], controls["bias"], 1.)
                    predicted["training_ar"] = np.concatenate([y[:cut], ar_future])
                    neural_future = recurse(y[cut - 1], future_queries, array(neural, source + "__transition", 2), array(neural, source + "__intercept", 1), e["dt"])
                    predicted["frozen_neural_transition"] = np.concatenate([y[:cut], neural_future])
                    for name, prediction in predicted.items():
                        fitted = chosen.get(name)
                        trace_id = candidates[int(actual[name][1][i])]["row"]["simulation_id"] if fitted else ""
                        item = {"source_model_key": source, "source_id": row["source_id"], "account_class": account, "variant": variant,
                            "participant_id": row["participant_id"], "episode_id": row["episode_id"], "state_token": token,
                            "split": "test", "predictor": name, "simulation_id": trace_id,
                            "later_state_standardized_mse": loss(y[cut:], prediction[cut:], e["scale"]),
                            "conditioning_states": cut, "grid_states": len(grid), "latent_rank": y.shape[1],
                            "effective_readout_df": fitted["effective_readout_df"] if fitted else "",
                            "feature_rank": fitted["feature_rank"] if fitted else "",
                            "ridge": fitted["ridge"] if fitted else "", "offset_strength": fitted["offset_strength"] if fitted else ""}
                        cases.append(item)
                        prediction_arrays[description["map_key"] + "__" + name + "__" + token] = prediction
            prediction_arrays[source + "__times_s"] = grid
    require(sorted({r["participant_id"] for r in cases}) == sorted(evaluation["test_person_ids"]), "New evaluation changed original held-out people")
    require(sorted({r["episode_id"] for r in cases}) == sorted(evaluation["test_observation_ids"]), "New evaluation changed original held-out episodes")
    report = summarize_cases(cases, selected_accounts, seed, bootstrap_replicates)
    report.update(schema_version=1, version=VERSION, status="full_population_observation_bottleneck_evaluated",
        behavioral_target=42, behavioral_split=[26, 8, 8], behavioral_target_is_neural_sample_size=False,
        human_finetuning_gate="Caller verifies both actual matched human-data fine-tunings before comparison; no checkpoint updates in this branch",
        original_test_people=len({r["participant_id"] for r in cases}), original_test_episodes=len({r["episode_id"] for r in cases}),
        common_case_ladder=census, new_split_or_human_pca=False, human_future_data_used_for_conditioning=False,
        mapped_target="Original frozen training-derived multivariate human neural-state coordinates, same final10s clock",
        primary_interpretation="Information retained by full native population states beyond an equally searched cortical delay observation, conditional on effective observation maps",
        matched_capacity_scope="Identical nominal feature count18, PCA-rank/ridge/offset grid and joint-selected settings; effective degrees of freedom reported, not asserted identical",
        cortical_delay_support=delay_support,
        native_population_count=9, native_spatial_nodes=1, anatomical_human_mapping_qualified=False,
        mechanistic_identification_established=False, human_intervention_qualified=False, independent_human_validation=False,
        public_data_status="Previously exposed development data; original participant split retained",
        test_outcomes_used_for_selection=False, result_direction_used_as_execution_gate=False,
        candidate_dispositions=bank.get("rows", []), training_validation_freeze_sha256=sha(output / "training_validation_freeze.json"))
    require(all(sha(path) == pins[role] for role, path in paths.items()), "Pinned input changed during analysis")
    write_csv(output / "prediction_cases.csv", cases)
    np.savez_compressed(output / "predictions.npz", **prediction_arrays)
    write_json(output / "population_bridge.json", report)
    names = ("training_validation_freeze.json", "validation_grid.csv", "compatibility.json", "prediction_cases.csv", "predictions.npz", "population_bridge.json")
    receipt = {"schema_version": 1, "status": report["status"], "version": VERSION,
        "source_sha256": sha(__file__), "inputs": {r: {"path": str(p), "sha256": pins[r]} for r, p in paths.items()},
        "outputs": {name: sha(output / name) for name in names}, "test_people": report["original_test_people"],
        "test_episodes": report["original_test_episodes"], "models": len(models), "case_rows": len(cases),
        "original_outputs_modified": False, "target": 42, "both_human_data_finetunings_required_before_comparison": True}
    write_json(output / "receipt.json", receipt)
    return receipt
