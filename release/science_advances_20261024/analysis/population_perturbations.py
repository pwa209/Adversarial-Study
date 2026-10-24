"""Finite-input responses of all nine native neural populations.

The existing bank is an enumerated model experiment, not a human sample.  Every
native control/pulse pair is qualified before subtraction.  No new simulation,
fit, human selection or clinical inference is performed by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

VERSION = "full_population_finite_perturbations_v1"
REQUIRED_ROLES = ("native_cache",)
OPTIONAL_ROLES = ("bridge_compatibility",)
POPULATIONS = ("excitatory_cortex", "inhibitory_cortex", "thalamic_reticular",
               "thalamic_relay", "striatal_D1", "striatal_D2", "GPi", "GPe", "STN")
# Declared before outcome inspection.  These are matching rules, not claims of
# clinical equivalence or parameter-identification uncertainty intervals.
MATCH_TOLERANCES = {"score": .025, "feature_acf_100ms": .05,
                    "feature_acf_500ms": .05,
                    "feature_spectral_centroid_hz": 1.,
                    "feature_spectral_entropy": .05,
                    "feature_hjorth_mobility_relative": .10,
                    "feature_std_amplitude_log_ratio": math.log(1.10)}
RECOVERY_FRACTION = .05
RECOVERY_HOLD_S = .250
RECRUITMENT_SD = 3.
RECRUITMENT_HOLD_S = .010


def require(value, message):
    if not value:
        raise ValueError("Full-population perturbations: " + message)


def number(value, name="value"):
    require(not isinstance(value, (bool, np.bool_)), "Boolean " + name)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("Full-population perturbations: invalid " + name) from None
    require(math.isfinite(result), "Nonfinite " + name)
    return result


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def _time(trace):
    t = np.asarray(trace.get("times", trace.get("time")), dtype=float)
    require(t.ndim == 1 and len(t) >= 4 and np.isfinite(t).all()
            and np.all(np.diff(t) > 0), "Invalid native clock")
    dt = float(np.median(np.diff(t)))
    require(np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-10), "Irregular native clock")
    return t, dt


def _matrix(trace, kind):
    keys = {"V": ("voltage", "V"), "Q": ("rate", "Q")}[kind]
    result = np.asarray(trace.get(keys[0], trace.get(keys[1])), dtype=float)
    require(result.ndim == 2 and result.shape[1] == 9 and np.isfinite(result).all(),
            "All nine finite " + kind + " populations required")
    return result


def sustained_crossing(mask, start, samples):
    """First fully observed run; a truncated final run does not qualify."""
    mask = np.asarray(mask, dtype=bool)
    require(mask.ndim == 1 and samples >= 1 and 0 <= start <= len(mask), "Invalid run search")
    if len(mask) - start < samples:
        return None
    running = np.convolve(mask[start:].astype(int), np.ones(samples, dtype=int), mode="valid")
    hits = np.flatnonzero(running == samples)
    return int(start + hits[0]) if len(hits) else None


def finite_energy(y, t):
    """Integrate squared response, rejecting overflow rather than clipping it."""
    y, t = np.asarray(y, float), np.asarray(t, float)
    require(y.ndim == t.ndim == 1 and len(y) == len(t) and len(t) >= 2
            and np.isfinite(y).all(), "Invalid response energy input")
    peak = float(np.max(np.abs(y)))
    if peak == 0:
        return 0.
    # Do the bounded integral first; this does not silently cap extreme energy.
    integral = float(np.trapezoid((y / peak) ** 2, t))
    result = (peak * integral) * peak
    require(math.isfinite(result), "Response energy exceeds finite numerical range")
    return result


def channel_response(control, pulse, t, onset_s, width_s, *, population, kind):
    a, b, t = np.asarray(control, float), np.asarray(pulse, float), np.asarray(t, float)
    require(a.shape == b.shape == t.shape and np.isfinite(a).all() and np.isfinite(b).all(),
            "Response trajectory shapes or values differ")
    before = t < onset_s - .010
    after = t >= onset_s
    require(before.sum() >= 10 and after.sum() >= 2, "Insufficient baseline or response clock")
    dt = float(np.median(np.diff(t)))
    with np.errstate(over="ignore", invalid="ignore"):
        delta = b - a
        baseline_sd = float(np.std(a[before], ddof=1))
    require(np.isfinite(delta).all() and math.isfinite(baseline_sd), "Nonfinite difference or baseline variance")
    numerical_floor = np.finfo(float).eps * max(1., float(np.max(np.abs(a[before])))) * 100
    scale_valid = baseline_sd > numerical_floor
    abs_delta = np.abs(delta)
    post = np.flatnonzero(after)
    peak_index = int(post[np.argmax(abs_delta[after])])
    peak = float(abs_delta[peak_index])
    energy = finite_energy(delta[after], t[after])
    response_detected = peak > numerical_floor
    active_after = int(np.searchsorted(t, onset_s, side="left"))
    crossing = None
    if scale_valid:
        crossing = sustained_crossing(abs_delta >= RECRUITMENT_SD * baseline_sd,
                                      active_after, max(1, int(math.ceil(RECRUITMENT_HOLD_S / dt))))
    recover = None
    if response_detected:
        recovery_start = max(peak_index + 1, int(np.searchsorted(t, onset_s + width_s, side="right")))
        recover = sustained_crossing(abs_delta <= RECOVERY_FRACTION * peak, recovery_start,
                                     max(1, int(math.ceil(RECOVERY_HOLD_S / dt))))
    status = ("no_detectable_response" if not response_detected else
              "right_censored_at_end" if recover is None else "returned_below_5pct_peak_for_250ms")
    result = {"population_index": population + 1, "population": POPULATIONS[population], "quantity": kind,
              "baseline_sd": baseline_sd, "baseline_scale_qualified": bool(scale_valid),
              "response_detected": bool(response_detected), "peak_absolute": peak,
              "signed_response_at_peak": float(delta[peak_index]),
              "peak_latency_s": float(t[peak_index] - onset_s), "finite_response_energy": energy,
              "peak_in_baseline_sd": peak / baseline_sd if scale_valid else None,
              "energy_in_baseline_sd_squared_s": energy / baseline_sd ** 2 if scale_valid else None,
              "recruited_3sd_for_10ms": bool(crossing is not None),
              "recruitment_latency_s": float(t[crossing] - onset_s) if crossing is not None else None,
              "recruited_1sd_for_10ms": bool(sustained_crossing(abs_delta >= baseline_sd, active_after,
                max(1, int(math.ceil(RECRUITMENT_HOLD_S / dt)))) is not None) if scale_valid else None,
              "recovery_s": float(t[recover] - onset_s) if recover is not None else None,
              "recovery_status": status, "observed_response_horizon_s": float(t[-1] - onset_s),
              "recovery_censoring_time_s": float(t[-1] - onset_s) if status == "right_censored_at_end" else None}
    require(all(v is None or not isinstance(v, float) or math.isfinite(v) for v in result.values()),
            "Nonfinite summary; no epsilon-based amplification allowed")
    return result, delta, baseline_sd if scale_valid else None


def system_response(delta, baseline_scales, t, onset_s, width_s, kind):
    """Population participation of an uncentered finite-response trajectory.

    Baseline-scaled V and Q are kept separate.  The participation ratio describes
    the recorded response, not a stability spectrum of the delayed simulator.
    """
    delta, t = np.asarray(delta, float), np.asarray(t, float)
    require(delta.shape == (len(t), 9), "Nine population differences required")
    valid = np.asarray([s is not None and s > 0 for s in baseline_scales], bool)
    after = t >= onset_s
    result = {"quantity": kind, "n_populations": 9,
              "n_baseline_scalable_populations": int(valid.sum()),
              "all_nine_baseline_scalable": bool(valid.all()),
              "response_mode_participation_ratio": None,
              "readout_population_energy_fraction": None,
              "other_population_energy_fraction": None,
              "system_recovery_s": None, "system_recovery_status": "baseline_scale_unqualified"}
    if not valid.any():
        return result, None
    z = delta[:, valid] / np.asarray([s for s in baseline_scales if s is not None])[None, :]
    require(np.isfinite(z).all(), "Nonfinite baseline-scaled system response")
    energy = np.asarray([finite_energy(z[after, j], t[after]) for j in range(z.shape[1])])
    singular = np.linalg.svd(z[after], compute_uv=False)
    eigen = singular ** 2
    total = float(energy.sum())
    norm = np.linalg.norm(z, axis=1)
    require(np.isfinite(eigen).all() and math.isfinite(total) and np.isfinite(norm).all(),
            "Nonfinite collective response")
    if eigen.sum() > 0:
        fractions = eigen / eigen.sum()
        result["response_mode_participation_ratio"] = float(1. / np.sum(fractions ** 2))
    if valid.all() and total > 0:
        result["readout_population_energy_fraction"] = float(energy[0] / total)
        result["other_population_energy_fraction"] = float(energy[1:].sum() / total)
    peak_index = int(np.flatnonzero(after)[np.argmax(norm[after])])
    peak = float(norm[peak_index])
    if peak <= np.finfo(float).eps * 100:
        result["system_recovery_status"] = "no_detectable_response"
    else:
        dt = float(np.median(np.diff(t)))
        start = max(peak_index + 1, int(np.searchsorted(t, onset_s + width_s, side="right")))
        recover = sustained_crossing(norm <= RECOVERY_FRACTION * peak, start,
                                     max(1, int(math.ceil(RECOVERY_HOLD_S / dt))))
        result["system_recovery_status"] = ("right_censored_at_end" if recover is None else
                                              "returned_below_5pct_peak_for_250ms")
        result["system_recovery_s"] = float(t[recover] - onset_s) if recover is not None else None
    # Only compare full nine-population signatures across configurations.
    return result, z[after] if valid.all() else None


def _factor_row(row):
    result = {}
    for key in ("param_gain_factor", "param_coupling_factor", "param_history_factor"):
        result[key] = number(row[key], key)
    return result


def _metadata(control, pulse):
    a, b = control["row"], pulse["row"]
    for key in ("account_class", "variant", "pair_id"):
        require(a.get(key) == b.get(key) and a.get(key) not in (None, ""), "Pair metadata differ: " + key)
    require(_factor_row(a) == _factor_row(b), "Pair mechanistic factors differ")
    require(number(a["input_amplitude"]) == 0 and number(b["input_amplitude"]) > 0,
            "Pair must contain one zero-input control and one positive native pulse")
    return {"pair_id": a["pair_id"], "control_simulation_id": a["simulation_id"],
            "pulse_simulation_id": b["simulation_id"], "account_class": a["account_class"],
            "variant": a["variant"], **_factor_row(a),
            "input_amplitude": number(b["input_amplitude"]),
            "input_onset_absolute_s": number(b["input_onset_s"]),
            "input_width_s": number(b["input_width_s"])}


def pair_groups(bank, expected_pairs=48):
    """Keep every declared pair; missing/duplicate members are fatal."""
    groups = defaultdict(list)
    for row in bank["rows"]:
        if row.get("pair_id"):
            groups[(row["variant"], row["pair_id"])].append(row)
    require(len(groups) == expected_pairs, f"Expected {expected_pairs} declared pairs, found {len(groups)}")
    pairs = []
    for key, rows in sorted(groups.items()):
        require(len(rows) == 2, "Missing or duplicate native pair members: " + str(key))
        ordered = sorted(rows, key=lambda r: number(r["input_amplitude"]))
        ids = [r["simulation_id"] for r in ordered]
        require(all(s in bank["traces"] for s in ids), "Declared pair lacks a qualified raw trajectory: " + str(key))
        control, pulse = [bank["traces"][s] for s in ids]
        _metadata(control, pulse)
        pairs.append((control, pulse))
    return pairs


def _score(row):
    # Only the frozen, already-scored native control is accepted.  Scores are not
    # recomputed from an adapted instrument or substituted with GCS/preclip GCS.
    status = row.get("score_status", row.get("detector_status", row.get("status")))
    require(status in ("ok", "finite", "finite_constant_not_standardizable", "included"),
            "Frozen control score not eligible")
    score = number(row["score"], "actual frozen control score")
    require(0 <= score <= 1, "Frozen clinical score outside [0,1]")
    return score


def clipping_stratum(a, b):
    a, b = _score(a), _score(b)
    if a == b == 0:
        return "both_floor"
    if a == b == 1:
        return "both_ceiling"
    return "mixed_clipping" if a in (0., 1.) or b in (0., 1.) else "neither_clipped"


def match_controls(a, b, *, spectra=False):
    """Fixed same-input, same-implementation matching, with optional EEG features."""
    ra, rb = a["control_row"], b["control_row"]
    if a["variant"] != b["variant"]:
        return False, "different_implementation"
    if any(a[k] != b[k] for k in ("input_amplitude", "input_onset_absolute_s", "input_width_s")):
        return False, "different_input"
    if a.get("noise_stream_sha256") != b.get("noise_stream_sha256") or not a.get("noise_stream_sha256"):
        return False, "different_or_unverified_background_noise_stream"
    if abs(_score(ra) - _score(rb)) > MATCH_TOLERANCES["score"]:
        return False, "score_outside_tolerance"
    if not spectra:
        return True, "score_matched"
    for key in ("feature_acf_100ms", "feature_acf_500ms", "feature_spectral_centroid_hz", "feature_spectral_entropy"):
        if abs(number(ra[key], key) - number(rb[key], key)) > MATCH_TOLERANCES[key]:
            return False, key + "_outside_tolerance"
    xa, xb = [number(r["feature_hjorth_mobility"], "Hjorth mobility") for r in (ra, rb)]
    if xa < 0 or xb < 0 or abs(xa - xb) / max((xa + xb) / 2., np.finfo(float).tiny) > .10:
        return False, "mobility_outside_tolerance"
    xa, xb = [number(r["feature_std_amplitude"], "cortical standard deviation") for r in (ra, rb)]
    if min(xa, xb) <= 0 or abs(math.log(xa / xb)) > MATCH_TOLERANCES["feature_std_amplitude_log_ratio"]:
        return False, "amplitude_outside_tolerance"
    return True, "score_and_conventional_features_matched"


def relative_distance(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), "Matched response clocks differ")
    # Normalize before norms so that finite large values do not overflow squares.
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if scale == 0:
        return 0.
    a, b = a / scale, b / scale
    denominator = (np.linalg.norm(a) + np.linalg.norm(b)) / 2.
    return float(np.linalg.norm(a - b) / denominator) if denominator > 0 else 0.


def response_contrasts(records, compatible_ids=None):
    rows, disposition = [], Counter()
    for a, b in itertools.combinations(records, 2):
        for matching in ("score", "score_and_conventional_features"):
            try:
                matched, reason = match_controls(a, b, spectra=matching != "score")
            except (KeyError, ValueError) as exc:
                disposition[matching + ":metadata_unavailable"] += 1
                continue
            disposition[matching + ":" + reason] += 1
            if not matched:
                continue
            for kind in ("V", "Q"):
                za, zb = a["signatures"][kind], b["signatures"][kind]
                row = {"matching": matching, "quantity": kind, "variant": a["variant"],
                       "input_amplitude": a["input_amplitude"], "pair_a": a["pair_id"], "pair_b": b["pair_id"],
                       "family_a": a["account_class"], "family_b": b["account_class"],
                       "control_a": a["control_simulation_id"], "control_b": b["control_simulation_id"],
                       "common_background_noise_stream_sha256": a["noise_stream_sha256"],
                       "control_score_a": _score(a["control_row"]), "control_score_b": _score(b["control_row"]),
                       "score_clipping_stratum": clipping_stratum(a["control_row"], b["control_row"]),
                       "both_human_training_validation_supported": (None if compatible_ids is None else
                         a["control_simulation_id"] in compatible_ids and b["control_simulation_id"] in compatible_ids),
                       "response_comparison_status": "all_nine_baseline_scalable" if za is not None and zb is not None else "baseline_scale_unqualified",
                       "cortical_readout_relative_response_distance": None,
                       "other_populations_relative_response_distance": None,
                       "full_population_relative_response_distance": None,
                       "other_minus_readout_response_distance": None}
                if za is not None and zb is not None:
                    row["cortical_readout_relative_response_distance"] = relative_distance(za[:, :1], zb[:, :1])
                    row["other_populations_relative_response_distance"] = relative_distance(za[:, 1:], zb[:, 1:])
                    row["full_population_relative_response_distance"] = relative_distance(za, zb)
                    row["other_minus_readout_response_distance"] = row["other_populations_relative_response_distance"] - row["cortical_readout_relative_response_distance"]
                rows.append(row)
    return rows, dict(disposition)


def read_compatibility(spec):
    if spec is None:
        return None, {"status": "not_supplied"}
    path = Path(spec["path"] if isinstance(spec, dict) else spec)
    if isinstance(spec, dict) and spec.get("sha256"):
        require(_sha(path) == spec["sha256"], "Bridge compatibility SHA256 differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("selection_uses_test_targets") is False and value.get("selection_unit") == "training/validation",
            "Bridge support must be selected exclusively from human training/validation")
    ids = value.get("compatible_simulation_ids")
    require(isinstance(ids, list) and all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids)),
            "Unique explicit bridge-supported simulation IDs required")
    return set(ids), {"status": "training_validation_support_only", "sha256": _sha(path),
                      "n_supported_configurations": len(ids), "not_physiological_equivalence": True,
                      "support_means_fit_support_not_observational_adequacy": True}


def evaluate_bank(bank, *, expected_pairs=48, compatibility=None, qualifier=None):
    if qualifier is None:
        from native_population import qualify_pair
        qualifier = qualify_pair
    pairs = pair_groups(bank, expected_pairs)
    population_rows, system_rows, proof_rows, records = [], [], [], []
    for control, pulse in pairs:
        meta = _metadata(control, pulse)
        proof = qualifier(control, pulse)
        require(isinstance(proof, dict), "Native pair qualification must return its proof")
        t, dt = _time(control)
        require(np.array_equal(t, _time(pulse)[0]), "Pulse and control clocks differ")
        onset = number(proof.get("onset_relative_s", proof.get("onset_s", 2.)), "qualified relative onset")
        width = meta["input_width_s"]
        require(abs(onset - 2.) <= 1e-10 and abs(width - .02) <= 1e-10,
                "Existing bank pulse clock changed: require onset2s and width20ms")
        proof_rows.append(meta | {"status": "qualified", "input_onset_relative_s": onset,
                                  "proof_json": json.dumps(proof, sort_keys=True, allow_nan=False)})
        signatures = {}
        for kind in ("V", "Q"):
            a, b = _matrix(control, kind), _matrix(pulse, kind)
            require(len(a) == len(t) == len(b), "Response matrix clock differs")
            deltas, scales = [], []
            for j in range(9):
                metrics, delta, scale = channel_response(a[:, j], b[:, j], t, onset, width, population=j, kind=kind)
                metrics["finite_peak_to_native_input_amplitude_ratio"] = metrics["peak_absolute"] / meta["input_amplitude"]
                metrics["finite_energy_to_native_input_amplitude_squared_ratio"] = metrics["finite_response_energy"] / meta["input_amplitude"] ** 2
                population_rows.append(meta | metrics)
                deltas.append(delta)
                scales.append(scale)
            metrics, signature = system_response(np.column_stack(deltas), scales, t, onset, width, kind)
            channels = [r for r in population_rows[-9:] if r["quantity"] == kind]
            metrics["n_recruited_3sd_for_10ms"] = sum(r["recruited_3sd_for_10ms"] for r in channels)
            metrics["n_recruited_1sd_for_10ms"] = sum(r["recruited_1sd_for_10ms"] is True for r in channels)
            metrics["recovery_right_censored_populations"] = sum(r["recovery_status"] == "right_censored_at_end" for r in channels)
            system_rows.append(meta | metrics)
            signatures[kind] = signature
        try:
            score = _score(control["row"])
        except (KeyError, ValueError):
            score = None
        records.append(meta | {"control_row": control["row"], "control_score": score,
                               "noise_stream_sha256": hashlib.sha256(np.asarray(control["noise"], dtype="<f8").tobytes()).hexdigest(),
                               "signatures": signatures})
    matched_rows, matched_disposition = response_contrasts(records, compatibility)
    return population_rows, system_rows, proof_rows, matched_rows, matched_disposition


def _finite_summary(values):
    a = np.asarray([v for v in values if v is not None], float)
    return {"n": len(a), "minimum": float(a.min()) if len(a) else None,
            "median": float(np.median(a)) if len(a) else None, "maximum": float(a.max()) if len(a) else None}


def summarize(population_rows, system_rows, proof_rows, contrasts, dispositions, compatibility_receipt):
    strata = []
    groups = defaultdict(list)
    for row in system_rows:
        groups[(row["variant"], row["account_class"], row["input_amplitude"], row["quantity"])].append(row)
    for (variant, family, amplitude, kind), rows in sorted(groups.items()):
        strata.append({"variant": variant, "account_class": family, "input_amplitude": amplitude,
                       "quantity": kind, "n_configurations": len(rows),
                       "n_recruited_3sd": _finite_summary(r["n_recruited_3sd_for_10ms"] for r in rows),
                       "mode_participation": _finite_summary(r["response_mode_participation_ratio"] for r in rows),
                       "other_population_energy_fraction": _finite_summary(r["other_population_energy_fraction"] for r in rows),
                       "n_system_recovery_right_censored": sum(r["system_recovery_status"] == "right_censored_at_end" for r in rows)})
    contrast_summary = []
    cgroups = defaultdict(list)
    for row in contrasts:
        cgroups[(row["matching"], row["variant"], row["input_amplitude"], row["quantity"], row["score_clipping_stratum"])].append(row)
    for (matching, variant, amplitude, kind, clipping), rows in sorted(cgroups.items()):
        contrast_summary.append({"matching": matching, "variant": variant, "input_amplitude": amplitude,
                                "quantity": kind, "score_clipping_stratum": clipping,
                                "n_dependent_configuration_contrasts": len(rows),
                                "n_training_validation_supported": sum(r["both_human_training_validation_supported"] is True for r in rows),
                                "other_minus_readout_response_distance": _finite_summary(r["other_minus_readout_response_distance"] for r in rows)})
    return {"status": "completed_existing_full_population_finite_input_analysis", "version": VERSION,
            "n_qualified_control_pulse_pairs": len(proof_rows), "n_population_quantity_responses": len(population_rows),
            "n_system_responses": len(system_rows), "population_names": list(POPULATIONS),
            "independent_units": "enumerated model configurations; reused configurations make contrasts dependent; no human sample or population inference",
            "input_design": "one common all-population native pulse; amplitudes1/10/100 occur across different configurations, not a within-configuration sweep",
            "matching_tolerances": MATCH_TOLERANCES, "matching_disposition": dispositions,
            "family_amplitude_strata": strata, "matched_response_summaries": contrast_summary,
            "bridge_support": compatibility_receipt,
            "fixed_metrics": {"recruitment_baseline_sd": RECRUITMENT_SD, "recruitment_hold_s": RECRUITMENT_HOLD_S,
                              "recovery_fraction_of_finite_peak": RECOVERY_FRACTION, "recovery_hold_s": RECOVERY_HOLD_S},
            "inference": {"new_simulations": False, "clinical_scores_recomputed": False,
                          "human_test_targets_used_for_selection": False,
                          "linear_impulse_response_or_controllability_established": False,
                          "selective_loop_necessity_established": False,
                          "human_physiology_or_causal_mechanism_established": False,
                          "one_spatial_node": True, "uncertainty": "finite-bank descriptive ranges; no pseudo-replicated confidence intervals"}}


def run(inputs, output, seed=20261024, **options):
    """Role-map API; native_cache is a verified aggregate cache directory.

    The root driver owns the target42 and both actual human fine-tuning gates.
    bridge_compatibility must come from the immediately preceding bridge and
    may only describe training/validation support, never held-out outcome wins.
    """
    from native_population import read_cache
    specs = inputs.get("files", inputs)
    require("native_cache" in specs, "Missing native_cache role")
    cache = specs["native_cache"]
    cache = cache.get("path", cache.get("cache_dir")) if isinstance(cache, dict) else cache
    bank = read_cache(Path(cache))
    supported, support_receipt = read_compatibility(specs.get("bridge_compatibility"))
    population, system, proof, contrasts, dispositions = evaluate_bank(bank,
        expected_pairs=int(options.get("expected_pairs", 48)), compatibility=supported)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("population_responses.csv", population), ("system_responses.csv", system),
                       ("pair_qualification.csv", proof), ("matched_response_contrasts.csv", contrasts)):
        _csv(out / name, rows)
    summary = summarize(population, system, proof, contrasts, dispositions, support_receipt)
    summary["seed"] = int(seed)
    summary["native_cache_receipt_sha256"] = _sha(Path(cache) / "receipt.json") if (Path(cache) / "receipt.json").is_file() else None
    _json(out / "summary.json", summary)
    receipt = {"status": summary["status"], "version": VERSION, "seed": int(seed),
               "n_qualified_control_pulse_pairs": len(proof), "output_sha256": {p.name: _sha(p) for p in out.iterdir() if p.is_file()},
               "both_human_data_finetunings_before_comparison": "enforced_by_root_driver",
               "target_participants": 42, "new_simulations": False,
               "human_test_targets_used_for_selection": False}
    _json(out / "receipt.json", receipt)
    return receipt
