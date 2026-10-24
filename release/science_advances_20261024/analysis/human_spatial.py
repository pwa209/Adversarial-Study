"""Fixed-ROI DREAM report comparisons; no sensor or analysis selection on results.

The input EEG is the genuine, hash-bound portable source recording.  This module
does not infer anatomical connectivity or equate dream recall with experience.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROLES = ("dream_observations", "dream_trajectories", "state_split", "state_receipt",
         "prepared_recordings", "electrode_coordinates")
CONTRASTS = {
    "experience_presence": ({"experience", "experience_without_recall"}, {"no_experience"}),
    "content_recall_given_experience": ({"experience"}, {"experience_without_recall"}),
}
RIDGES = (.01, .1, 1., 10.)
PANELS = ("primary_common_average", "as_stored_reference", "unsaturated_common_average",
          "high_frequency_adjusted_common_average")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(item):
    if not isinstance(item, dict) or not item.get("sha256"):
        raise ValueError("Every metadata input requires an explicit SHA256")
    path = Path(item["path"])
    if sha(path) != item["sha256"]:
        raise ValueError("Input hash mismatch: " + str(path))
    return path


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def identity(row):
    result = tuple(row.get(k, "") for k in ("source_id", "participant_id", "episode_id"))
    if not all(result):
        raise ValueError("Missing source/person/episode identity")
    return result


def unique(rows):
    result = {}
    for row in rows:
        key = identity(row)
        if key in result:
            raise ValueError("Duplicate source/person/episode")
        result[key] = row
    return result


def fixed_roles(split):
    roles = {}
    for name in ("train", "validation", "test"):
        for person in split[name + "_participant_ids"]:
            if person in roles:
                raise ValueError("Original participant roles overlap")
            roles[person] = name
    return roles


def sensor_rois(path):
    """Author EGI Cartesian coordinates: +Y anterior; +Z superior, in cm.

    Geometric sensor ROIs are fixed before outcomes.  They are not source-space
    cortical regions. Chan 257 is the author file's final Cz reference entry.
    """
    coordinates = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 4:
            raise ValueError("Malformed author electrode coordinate")
        token = fields[0]
        name = "Chan 257" if token == "Cz" else "Chan " + str(int(token))
        value = np.asarray([float(x) for x in fields[1:]])
        if name in coordinates or not np.isfinite(value).all():
            raise ValueError("Invalid/duplicate electrode coordinate")
        coordinates[name] = value
    if len(coordinates) != 257:
        raise ValueError("Expected original 257-channel EGI coordinate file")
    rois = {
        "global": sorted(k for k, (_, _, z) in coordinates.items() if z >= 0),
        "posterior": sorted(k for k, (_, y, z) in coordinates.items() if y <= -3 and z >= 0),
        "frontal": sorted(k for k, (_, y, z) in coordinates.items() if y >= 3 and z >= 0),
    }
    if min(map(len, rois.values())) < 8:
        raise ValueError("Author montage does not support fixed ROIs")
    return rois


def window_from_npz(path, start, stop):
    """Read exact contiguous source window without padding or silent unit change."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"data", "sfreq_hz", "channel_names"}
        if not required <= set(archive.files):
            raise ValueError("Missing portable EEG keys")
        frequency = np.asarray(archive["sfreq_hz"])
        if frequency.size != 1:
            raise ValueError("Sampling frequency must be one qualified scalar")
        fs = float(frequency.reshape(-1)[0])
        data = np.asarray(archive["data"])
        labels = np.asarray(archive["channel_names"])
        if (data.ndim != 2 or data.dtype.kind not in "fiu" or labels.ndim != 1
                or labels.dtype.kind != "U" or len(labels) != data.shape[0]
                or len(set(labels.tolist())) != len(labels) or any(not str(x).strip() for x in labels)
                or not np.isfinite(fs) or fs < 100):
            raise ValueError("Invalid portable EEG arrays/sampling rate")
        if "units" in archive and str(np.asarray(archive["units"]).reshape(-1)[0]) not in {"uV", "microvolts"}:
            raise ValueError("Unqualified EEG units")
        if not np.isfinite([start, stop]).all() or start < 0 or abs(stop-start-10.) > 1e-9:
            raise ValueError("Expected original exact ten-second window")
        samples = np.asarray([start, stop]) * fs
        indices = np.rint(samples).astype(int)
        if np.max(np.abs(samples-indices)) > 1e-6 or indices[1] > data.shape[1]:
            raise ValueError("Off-grid or unsupported source window")
        if "sample_times_s" in archive and "time_s" in archive:
            raise ValueError("Ambiguous source sample times")
        for key in ("sample_times_s", "time_s"):
            if key in archive:
                times = np.asarray(archive[key], float)
                expected = np.arange(indices[0], indices[1])/fs
                if times.shape != (data.shape[1],) or not np.allclose(times[indices[0]:indices[1]], expected, rtol=0, atol=1e-6/fs):
                    raise ValueError("Discontinuous source clock")
        if "gap_intervals_s" in archive:
            gaps = np.asarray(archive["gap_intervals_s"], float).reshape(-1, 2)
            if not np.isfinite(gaps).all() or np.any(gaps[:, 1] < gaps[:, 0]):
                raise ValueError("Malformed source gaps")
            if np.any((gaps[:, 0] < stop) & (gaps[:, 1] >= start)):
                raise ValueError("Window overlaps source gap/boundary")
        bad = set(np.asarray(archive["bad_channels"]).tolist()) if "bad_channels" in archive else set()
        if not bad <= set(labels.tolist()):
            raise ValueError("Invalid source bad-channel list")
        reference = str(np.asarray(archive["reference"]).reshape(-1)[0]) if "reference" in archive else "documented_NPZ_as_stored"
        window = np.asarray(data[:, indices[0]:indices[1]], float)
    return window, fs, labels.tolist(), bad, reference


def bandpowers(data, fs):
    """Two-second Hann Welch windows, 50% overlap, per-window mean removal."""
    data = np.asarray(data, float)
    n = int(round(2*fs))
    if data.ndim != 2 or data.shape[1] < n or not np.isfinite(data).all():
        raise ValueError("Invalid EEG for fixed spectral estimator")
    taper = np.hanning(n)
    frequencies = np.fft.rfftfreq(n, 1/fs)
    powers = []
    for start in range(0, data.shape[1]-n+1, n//2):
        segment = data[:, start:start+n]
        segment = (segment-segment.mean(axis=1, keepdims=True))*taper
        density = np.abs(np.fft.rfft(segment, axis=1))**2/(fs*np.sum(taper**2))
        density[:, 1:-1] *= 2
        powers.append(density)
    density = np.mean(powers, axis=0)
    df = fs/n
    return {name: density[:, (frequencies >= lo) & (frequencies <= hi)].sum(axis=1)*df
            for name, lo, hi in (("delta", 1., 4.), ("high_frequency", 20., 40.))}


def spatial_features(window, fs, labels, qualified, bad, rois, reference):
    available = set(labels) & set(qualified) - set(bad)
    chosen = {key: [name for name in values if name in available] for key, values in rois.items()}
    for key, values in chosen.items():
        if len(values)/len(rois[key]) < .8:
            raise ValueError("Insufficient fixed " + key + " ROI coverage (<80%)")
    names = chosen["global"]
    selected = np.asarray(window[[labels.index(name) for name in names]], float)
    if not np.isfinite(selected).all():
        raise ValueError("Nonfinite qualified ROI EEG")
    if reference == "common_average":
        selected = selected-selected.mean(axis=0, keepdims=True)
    elif reference != "as_stored":
        raise ValueError("Unknown fixed reference")
    powers = bandpowers(selected, fs)
    result = {"reference_operator": reference, "global_channels": len(names),
              "posterior_channels": len(chosen["posterior"]), "frontal_channels": len(chosen["frontal"]),
              "global_channel_names": ";".join(names), "posterior_channel_names": ";".join(chosen["posterior"]),
              "frontal_channel_names": ";".join(chosen["frontal"])}
    for band, values in powers.items():
        aggregate = {roi: float(values[[names.index(k) for k in chosen[roi]]].mean()) for roi in chosen}
        if min(aggregate.values()) <= 0:
            raise ValueError("Zero power in fixed source ROI")
        for roi, value in aggregate.items():
            result[roi + "_log_" + band] = math.log10(value)
        result["posterior_minus_global_" + band] = math.log10(aggregate["posterior"]/aggregate["global"])
        result["posterior_minus_frontal_" + band] = math.log10(aggregate["posterior"]/aggregate["frontal"])
    return result


def person_weights(rows):
    counts = Counter(r["participant_id"] for r in rows)
    return np.asarray([1/(len(counts)*counts[r["participant_id"]]) for r in rows])


def sigmoid(value):
    return 1/(1+np.exp(-np.clip(value, -40, 40)))


def fit_logistic(rows, features, labels, ridge):
    x = np.asarray([[float(r[k]) for k in features] for r in rows])
    y = np.asarray(labels, float)
    if x.ndim != 2 or not np.isfinite(x).all() or set(y.tolist()) != {0., 1.}:
        raise ValueError("Both original report classes needed in training")
    w = person_weights(rows)
    mean = w@x
    scale = np.sqrt(w@((x-mean)**2))
    scale = np.where(scale > 1e-9, scale, 1.)
    design = np.c_[np.ones(len(x)), (x-mean)/scale]
    coef = np.zeros(design.shape[1])
    coef[0] = math.log(float(w@y)/(1-float(w@y)))
    penalty = np.eye(len(coef))*ridge
    penalty[0, 0] = 0
    converged = False
    for iteration in range(100):
        prediction = sigmoid(design@coef)
        gradient = design.T@(w*(prediction-y))+penalty@coef
        hessian = design.T@((w*prediction*(1-prediction))[:, None]*design)+penalty
        step = np.linalg.solve(hessian+np.eye(len(coef))*1e-12, gradient)
        old_objective = float(np.sum(w*(np.logaddexp(0, design@coef)-y*(design@coef)))+.5*coef@penalty@coef)
        factor = 1.
        for _ in range(30):
            candidate = coef-factor*step
            eta = design@candidate
            objective = float(np.sum(w*(np.logaddexp(0, eta)-y*eta))+.5*candidate@penalty@candidate)
            if objective <= old_objective + 1e-12:
                break
            factor *= .5
        coef = candidate
        if np.max(np.abs(factor*step)) < 1e-8:
            converged = True
            break
    if not converged or not np.isfinite(coef).all():
        raise ValueError("Training-only logistic fit failed convergence")
    return {"features": list(features), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficient": coef.tolist(), "ridge": ridge, "iterations": iteration+1,
            "training_people": sorted({r["participant_id"] for r in rows})}


def predict(model, rows):
    x = np.asarray([[float(r[k]) for k in model["features"]] for r in rows])
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite predictor input")
    return sigmoid(np.c_[np.ones(len(x)), (x-np.asarray(model["mean"]))/np.asarray(model["scale"])]@np.asarray(model["coefficient"]))


def inference(values, seed, bootstrap_samples=10000):
    """People are resampled, never windows, channels or synthetic replications."""
    values = np.asarray(values, float)
    n = len(values)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected finite participant contrasts")
    result = {"people": n, "mean": float(values.mean()) if n else None,
              "bootstrap_unit": "participant", "bootstrap_samples": bootstrap_samples,
              "bootstrap_interpretation": "uncertainty of fixed-feature/fixed-prediction participant mean; not added sample size"}
    if n < 2:
        return {**result, "status": "insufficient_informative_people", "ci95": None, "p_signflip": None}
    rng = np.random.default_rng(seed)
    boot = []
    for start in range(0, bootstrap_samples, 500):
        index = rng.integers(n, size=(min(500, bootstrap_samples-start), n))
        boot.extend(values[index].mean(axis=1).tolist())
    threshold = abs(float(values.mean()))-1e-14
    if n <= 16:
        extreme = sum(abs(float(np.dot(values, signs)/n)) >= threshold
                      for signs in itertools.product((-1, 1), repeat=n))
        p = extreme/2**n
        method = "exact participant sign flip under independent-person symmetric-zero contrast assumption"
    else:
        extreme = 0
        for start in range(0, 19999, 500):
            signs = rng.choice((-1, 1), size=(min(500, 19999-start), n))
            extreme += int(np.sum(np.abs(signs@values/n) >= threshold))
        p = (extreme+1)/20000
        method = "19999 Monte Carlo participant sign flips under symmetric-zero contrast assumption"
    return {**result, "status": "estimated", "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "p_signflip": float(p), "p_method": method,
            "leave_one_person_out_means": [(float(values.sum()-v)/(n-1)) for v in values]}


def holm(records):
    valid = [r for r in records if r.get("p_signflip") is not None]
    running = 0.
    for index, row in enumerate(sorted(valid, key=lambda r: r["p_signflip"])):
        # Untestable declared contrasts retain their place in the family (p=1).
        running = max(running, min(1., (len(records)-index)*row["p_signflip"]))
        row["p_holm"] = running
    for row in records:
        row.setdefault("p_holm", None)


def outcome(row, contrast):
    positive, negative = CONTRASTS[contrast]
    label = row.get("report_category")
    return 1 if label in positive else (0 if label in negative else None)


def select_model(train, validation, contrast, features):
    if len({r["participant_id"] for r in train}) < 2 or not validation:
        raise ValueError("Insufficient original train/validation report cases")
    alternatives = []
    ytrain = [outcome(r, contrast) for r in train]
    yval = np.asarray([outcome(r, contrast) for r in validation])
    for ridge in RIDGES:
        model = fit_logistic(train, features, ytrain, ridge)
        loss = float(person_weights(validation)@((predict(model, validation)-yval)**2))
        alternatives.append((loss, ridge, model))
    selected = min(alternatives, key=lambda x: (x[0], x[1]))[2]
    selected["validation_grid"] = [{"ridge": x[1], "equal_person_brier": x[0]} for x in alternatives]
    selected["validation_people"] = sorted({r["participant_id"] for r in validation})
    return selected


def prediction_comparison(rows, contrast, panel, seed):
    available = [r for r in rows if outcome(r, contrast) is not None]
    split = {role: [r for r in available if r["split"] == role] for role in ("train", "validation", "test")}
    result = {"contrast": contrast, "panel": panel, "kind": "held_person_incremental_prediction",
              "case_counts": {role: len(group) for role, group in split.items()},
              "people_by_role": {role: sorted({r["participant_id"] for r in group}) for role, group in split.items()},
              "effect_definition": "posterior-augmented minus baseline person-mean Brier loss; negative favors spatial information",
              "uncertainty_scope": "conditional on frozen train/validation fitted models"}
    if not split["test"]:
        return {**result, "status": "no_original_test_cases", "p_signflip": None}, [], {}
    baseline = ["score", "global_log_delta"]
    if panel == "high_frequency_adjusted_common_average":
        baseline += ["global_log_high_frequency", "posterior_minus_global_high_frequency"]
    models = {}
    try:
        for name, features in (("baseline", baseline), ("spatial", baseline+["posterior_minus_global_delta"])):
            models[name] = select_model(split["train"], split["validation"], contrast, features)
    except ValueError as error:
        return {**result, "status": "unqualified_prediction", "reason": str(error), "p_signflip": None}, [], {}
    test = split["test"]
    observed = np.asarray([outcome(r, contrast) for r in test])
    predictions = {key: predict(model, test) for key, model in models.items()}
    cases = [{"source_id": row["source_id"], "participant_id": row["participant_id"], "episode_id": row["episode_id"],
              "split": "test", "contrast": contrast, "panel": panel, "observed": int(observed[i]),
              "baseline_probability": float(predictions["baseline"][i]), "spatial_probability": float(predictions["spatial"][i]),
              "baseline_brier": float((predictions["baseline"][i]-observed[i])**2),
              "spatial_brier": float((predictions["spatial"][i]-observed[i])**2)} for i, row in enumerate(test)]
    people = sorted({r["participant_id"] for r in test})
    values = [np.mean([r["spatial_brier"]-r["baseline_brier"] for r in cases if r["participant_id"] == person]) for person in people]
    result.update(inference(values, seed))
    result.update(baseline_brier=float(person_weights(test)@((predictions["baseline"]-observed)**2)),
                  spatial_brier=float(person_weights(test)@((predictions["spatial"]-observed)**2)),
                  participant_effects=[{"participant_id": p, "difference": float(v)} for p, v in zip(people, values)])
    return result, cases, models


def matched_contrast(rows, contrast, panel, seed):
    """Disjoint greedy matching uses score/global delta, never posterior feature."""
    groups = defaultdict(list)
    for row in rows:
        if row.get("sleep_stage") == "N2" and outcome(row, contrast) is not None:
            groups[(row["source_id"], row["participant_id"], row["sleep_stage"])].append(row)
    pairs = []
    for (_, person, _), group in sorted(groups.items()):
        candidates = []
        for positive in group:
            if outcome(positive, contrast) != 1:
                continue
            for negative in group:
                if outcome(negative, contrast) != 0:
                    continue
                score_distance = abs(float(positive["score"])-float(negative["score"]))
                global_distance = abs(float(positive["global_log_delta"])-float(negative["global_log_delta"]))
                if score_distance <= .05 and global_distance <= .1:
                    candidates.append((score_distance/.05+global_distance/.1, positive["episode_id"], negative["episode_id"], positive, negative))
        used = set()
        for distance, posid, negid, positive, negative in sorted(candidates, key=lambda x: x[:3]):
            if posid in used or negid in used:
                continue
            used.update((posid, negid))
            pairs.append({"source_id": positive["source_id"], "participant_id": person, "contrast": contrast, "panel": panel,
                          "positive_episode": posid, "negative_episode": negid, "matching_distance": distance,
                          "score_difference": float(positive["score"])-float(negative["score"]),
                          "global_log_delta_difference": float(positive["global_log_delta"])-float(negative["global_log_delta"]),
                          "posterior_relative_delta_difference": float(positive["posterior_minus_global_delta"])-float(negative["posterior_minus_global_delta"])})
    people = sorted({r["participant_id"] for r in pairs})
    values = [np.mean([r["posterior_relative_delta_difference"] for r in pairs if r["participant_id"] == person]) for person in people]
    result = {"contrast": contrast, "panel": panel, "kind": "all_people_retrospective_within_person_matched_contrast",
              "matching": "disjoint deterministic greedy matches; same source/person/N2; score <=.05; global log10(delta power) <=.1",
              "pairs": len(pairs), "posterior_feature_used_for_matching": False,
              "effect_definition": "positive report class minus negative class log10(posterior/global delta power)",
              **inference(values, seed), "participant_effects": [{"participant_id": p, "difference": float(v)} for p, v in zip(people, values)]}
    return result, pairs


def run(inputs, output, seed=20261024):
    files = inputs.get("files", inputs)
    paths = {role: checked(files[role]) for role in ROLES}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    observations = unique(read_csv(paths["dream_observations"]))
    trajectories = unique(read_csv(paths["dream_trajectories"]))
    if observations.keys() != trajectories.keys():
        raise ValueError("Scored and actual trajectory case sets differ")
    split = json.loads(paths["state_split"].read_text(encoding="utf-8-sig"))
    roles = fixed_roles(split)
    receipt = json.loads(paths["state_receipt"].read_text(encoding="utf-8-sig"))
    if receipt.get("status") != "neural_state_space_completed":
        raise ValueError("Expected completed original neural state receipt")
    pinned = {str(Path(item["path"])): item["sha256"] for item in receipt["inputs"]}
    if pinned.get(str(paths["prepared_recordings"])) != files["prepared_recordings"]["sha256"]:
        raise ValueError("Prepared recording inventory not bound to original state receipt")
    records = json.loads(paths["prepared_recordings"].read_text(encoding="utf-8-sig"))["recordings"]
    recording_map = {r["recording_id"]: r for r in records}
    if len(recording_map) != len(records):
        raise ValueError("Duplicate prepared recording identity")
    rois = sensor_rois(paths["electrode_coordinates"])
    features, dispositions, raw_provenance = [], [], []
    for key, observed in sorted(observations.items()):
        trajectory = trajectories[key]
        for field in ("recording_id", "start_s", "stop_s", "score", "report_category", "sleep_stage"):
            if observed.get(field) != trajectory.get(field):
                raise ValueError("Original scored/trajectory identity or measurement differs: " + field)
        if trajectory.get("measurement_status") != "actual_multichannel_eeg" or trajectory.get("score_origin") != "native_frozen_v2" or trajectory.get("synthetic", "").lower() in {"true", "1"}:
            raise ValueError("Input is not qualified actual source EEG")
        base = {name: observed.get(name, "") for name in ("source_id", "participant_id", "episode_id", "recording_id", "sleep_stage", "report_category", "score", "saturation_status")}
        base["split"] = roles.get(base["participant_id"], "unassigned")
        if base["split"] == "unassigned":
            raise ValueError("A source person has no preserved original role")
        if base["sleep_stage"] != "N2":
            dispositions.append({**base, "status": "excluded", "reason": "prespecified_N2_only"})
            continue
        if base["report_category"] not in set.union(*[set.union(*v) for v in CONTRASTS.values()]):
            dispositions.append({**base, "status": "excluded", "reason": "original_report_not_one_of_three_named_categories"})
            continue
        recording = recording_map[base["recording_id"]]
        path = Path(recording["path"])
        if path.suffix.lower() != ".npz" or pinned.get(str(path)) != recording["sha256"]:
            raise ValueError("Source EEG lacks original raw-byte state-receipt binding")
        if sha(path) != recording["sha256"]:
            raise ValueError("Source EEG changed: " + str(path))
        raw_provenance.append({"path": str(path), "sha256": recording["sha256"]})
        try:
            window, fs, labels, bad, source_reference = window_from_npz(path, float(observed["start_s"]), float(observed["stop_s"]))
            qualified = trajectory["trajectory_channel_names"].split(";")
            episode_features = []
            for reference in ("common_average", "as_stored"):
                result = spatial_features(window, fs, labels, qualified, bad, rois, reference)
                episode_features.append({**base, **result, "sfreq_hz": fs, "source_reference": source_reference,
                                         "recorded_reference_note": recording.get("reference", ""), "signal_sha256": recording["sha256"]})
            features.extend(episode_features)
            dispositions.append({**base, "status": "included", "reason": "qualified_fixed_ROIs_both_references"})
        except ValueError as error:
            dispositions.append({**base, "status": "excluded", "reason": str(error)})
        if sha(path) != recording["sha256"]:
            raise ValueError("Source EEG changed while reading")
    comparisons, predictions, matched, models = [], [], [], {}
    for panel_index, panel in enumerate(PANELS):
        reference = "as_stored" if panel == "as_stored_reference" else "common_average"
        selected = [r for r in features if r["reference_operator"] == reference]
        if panel == "unsaturated_common_average":
            selected = [r for r in selected if r["saturation_status"] == "no_saturated_channels"]
        for contrast_index, contrast in enumerate(CONTRASTS):
            result, cases, fitted = prediction_comparison(selected, contrast, panel, seed+100*panel_index+contrast_index)
            comparisons.append(result)
            predictions.extend(cases)
            models[panel+"/"+contrast] = fitted
            # HF adjustment is a predictive nuisance control; repeating identical matching adds no test.
            if panel != "high_frequency_adjusted_common_average":
                result, pairs = matched_contrast(selected, contrast, panel, seed+1000+100*panel_index+contrast_index)
                comparisons.append(result)
                matched.extend(pairs)
    primary = [r for r in comparisons if r["panel"] == "primary_common_average"]
    secondary = [r for r in comparisons if r["panel"] != "primary_common_average"]
    holm(primary)
    holm(secondary)
    for row in comparisons:
        row["multiplicity_family"] = "four_primary_prediction_and_matched_report_contrasts" if row in primary else "ten_fixed_sensitivity_contrasts"
    write_csv(out/"episode_features.csv", features)
    write_csv(out/"episode_dispositions.csv", dispositions)
    write_csv(out/"held_person_predictions.csv", predictions)
    write_csv(out/"matched_pairs.csv", matched)
    write_json(out/"models.json", models)
    summary = {"status": "human_spatial_completed", "seed": seed,
               "question": "Does fixed posterior sensor delta power add report information beyond global delta and clinical score?",
               "retrospective_development": True, "new_independent_people": False,
               "original_split_preserved": True, "bootstrap_adds_independent_people": False,
               "eligible_episodes": sum(r["status"] == "included" for r in dispositions),
               "eligible_people": len({r["participant_id"] for r in dispositions if r["status"] == "included"}),
               "disposition_counts": dict(Counter(r["reason"] for r in dispositions)),
               "fixed_roi_geometry": {"coordinate_units": "author_EGI_centimeters", "orientation": "+Y anterior, +Z superior", "global": "z>=0", "posterior": "y<=-3 and z>=0", "frontal": "y>=3 and z>=0", "minimum_coverage_fraction": .8},
               "roi_channels": rois, "comparisons": comparisons,
               "interpretation": "Sensor topographic association/prediction, not deep-pathway identification or independent replication. Posterior report effects may reproduce the original study in this public cohort. Bootstrap uncertainty and sign-flip references do not manufacture statistical significance."}
    write_json(out/"summary.json", summary)
    for role in ROLES:
        checked(files[role])
    result = {"status": "human_spatial_completed", "seed": seed,
              "inputs": {role: {"path": str(paths[role]), "sha256": files[role]["sha256"]} for role in ROLES},
              "raw_recordings": raw_provenance,
              "outputs": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"receipt.json", result)
    return result


def qualification_probe(inputs):
    """Bounded actual-input technical probe; no fitted/report-effect acceptance rule."""
    files = inputs.get("files", inputs)
    paths = {role: checked(files[role]) for role in ROLES}
    fixed_roles(json.loads(paths["state_split"].read_text(encoding="utf-8-sig")))
    original = json.loads(paths["state_receipt"].read_text(encoding="utf-8-sig"))
    if original.get("status") != "neural_state_space_completed":
        raise ValueError("Original state receipt not completed")
    pinned = {str(Path(item["path"])): item["sha256"] for item in original["inputs"]}
    if pinned.get(str(paths["prepared_recordings"])) != files["prepared_recordings"]["sha256"]:
        raise ValueError("Prepared recording metadata lacks original binding")
    records = json.loads(paths["prepared_recordings"].read_text(encoding="utf-8-sig"))["recordings"]
    recording_map = {r["recording_id"]: r for r in records}
    rois = sensor_rois(paths["electrode_coordinates"])
    observations = unique(read_csv(paths["dream_observations"]))
    trajectories = unique(read_csv(paths["dream_trajectories"]))
    if observations.keys() != trajectories.keys():
        raise ValueError("Original observation and trajectory identities differ")
    exclusions = []
    for key, row in sorted(trajectories.items()):
        if row.get("sleep_stage") != "N2":
            continue
        if row.get("measurement_status") != "actual_multichannel_eeg" or row.get("score_origin") != "native_frozen_v2" or row.get("synthetic", "").lower() in {"true", "1"}:
            raise ValueError("Technical probe requires genuine original EEG")
        original_row = observations[key]
        for field in ("recording_id", "start_s", "stop_s", "score", "report_category", "sleep_stage"):
            if row.get(field) != original_row.get(field):
                raise ValueError("Technical probe source measurement mismatch")
        record = recording_map[row["recording_id"]]
        path = Path(record["path"])
        if pinned.get(str(path)) != record["sha256"] or sha(path) != record["sha256"]:
            raise ValueError("Technical probe source EEG hash/binding mismatch")
        try:
            window, fs, names, bad, reference = window_from_npz(path, float(row["start_s"]), float(row["stop_s"]))
            operators = {}
            for operator in ("common_average", "as_stored"):
                measured = spatial_features(window, fs, names, row["trajectory_channel_names"].split(";"), bad, rois, operator)
                operators[operator] = {key: measured[key] for key in ("global_channels", "posterior_channels", "frontal_channels")}
        except ValueError as error:
            exclusions.append({"recording_id": row["recording_id"], "reason": str(error)})
            continue
        if sha(path) != record["sha256"]:
            raise ValueError("Technical probe source EEG changed while reading")
        return {"status": "actual_input_spatial_probe_passed", "recording_id": row["recording_id"],
                "signal_sha256": record["sha256"], "start_s": float(row["start_s"]), "stop_s": float(row["stop_s"]),
                "window_shape": list(window.shape), "sfreq_hz": fs, "source_reference": reference,
                "reference_roi_channels": operators, "earlier_technical_exclusions": exclusions,
                "report_effect_fitted_or_tested": False}
    raise ValueError("No technically qualified fixed-ROI original EEG window: " + str(exclusions[:5]))
