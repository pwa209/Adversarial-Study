"""Source-verified balance of the 41 saved postsynaptic voltage pathways.

These are single-node model diagnostics. A saved dendritic voltage is already a
postsynaptic contribution, not a current to be multiplied by coupling again.
The population sum identity must be verified before interpreting cancellation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re

import numpy as np

VERSION = "native_pathway_voltage_balance_v1"
REQUIRED_ROLES = ("native_cache", "bridge_compatibility", "bank_receipt")
DENDRITE_COLUMNS = [f"Dendrite.{i}.V" for i in range(1, 42)]
MAX_RAW_FILES = 281
MAX_RAW_BYTES = 64 * 1024**2
BOOTSTRAP_REPLICATES = 2000
RAW_VOLTAGE_FLOOR = 1e-12
# Actual Dumpf::open uses scientific output with14 fractional decimal digits.
# <=5e-15 relative rounding per field;1e-12 allows summation, subtraction and
# mean-centering arithmetic while remaining far below a biological effect gate.
PRINTED_RELATIVE_ROUNDING_BOUND = 5e-15
SUM_RELATIVE_TOLERANCE = 1e-12


def require(value, message):
    if not value:
        raise ValueError("Pathway balance: " + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned(spec):
    require(isinstance(spec, dict) and re.fullmatch(r"[0-9a-f]{64}", str(spec.get("sha256", ""))),
            "Input requires an exact SHA256 pin")
    path = Path(spec["path"])
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 8 * 1024**2,
            "Missing, symlink or oversized metadata input")
    require(sha(path) == spec["sha256"], "Pinned input changed: " + str(path))
    return path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def parse_pathways(raw, trace):
    """One numeric parse; require saved clocks and cached V/Q/noise to agree."""
    from native_population import COLUMNS, N_ROWS, SEPARATOR, _actual_clock
    require(len(raw) <= MAX_RAW_BYTES, "Oversized native output")
    lines = raw.decode("utf-8", errors="strict").splitlines()
    separators = [i for i, line in enumerate(lines) if line.strip() == SEPARATOR]
    require(len(separators) == 1, "Exact native separator required")
    payload = [x for x in lines[separators[0] + 1:] if x.strip()]
    require(len(payload) == N_ROWS + 2, "Native output must preserve all5000 observations")
    columns = payload[0].split()
    require(columns and columns[0] == "Time" and len(columns) == len(set(columns)),
            "Ambiguous native output columns")
    require(columns == trace["native_columns"], "Raw and cache native column identities differ")
    require(set(DENDRITE_COLUMNS + COLUMNS + ["Pop.10.Q"]).issubset(columns),
            "All41 dendritic voltages and9 population V/Q observations required")
    nodes = payload[1].split()
    require(len(nodes) == len(columns) - 1 and set(nodes) == {"1"}, "One spatial node required")
    data = np.loadtxt(io.StringIO("\n".join(payload[2:])), dtype=float, ndmin=2)
    require(data.shape == (N_ROWS, len(columns)) and np.isfinite(data).all(),
            "Malformed or nonfinite native observations")
    require(_actual_clock(data[:, 0]) and np.array_equal(data[:, 0], trace["times"]),
            "Actual .002..10s native clock differs; no rebasing permitted")
    require(np.array_equal(data[:, [columns.index(x) for x in COLUMNS]], trace["values"]),
            "Raw V/Q values differ from verified cache")
    require(np.array_equal(data[:, columns.index("Pop.10.Q")], trace["noise"]),
            "Raw noise differs from verified cache")
    observed = "Pop.11.Q" in columns
    require(observed == trace["stimulus_observed"], "Stimulus observation identity differs")
    if observed:
        require(np.array_equal(data[:, columns.index("Pop.11.Q")], trace["stimulus"]),
                "Raw pulse differs from verified cache")
    return data[:, [columns.index(x) for x in DENDRITE_COLUMNS]]


def load_pathways(trace, bank_root):
    """Only one source read for each selected configuration, with exact pins."""
    from native_population import _safe_file
    row = trace["row"]
    root = Path(bank_root).resolve()
    path = _safe_file(row["source_output_path"], MAX_RAW_BYTES, root)
    require(path == root / row["simulation_id"] / "native.output", "Native path/identity differs")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == row["source_output_sha256"], "Native output SHA256 changed")
    config = _safe_file(path.parent / "configuration.conf", 1024**2, root)
    require(sha(config) == row["config_sha256"]
            and config.read_text(encoding="utf-8") == trace["config_text"], "Native configuration changed")
    return parse_pathways(raw, trace), len(raw)


def incoming(connections, population, include_external=True):
    selected = [x for x in connections if x["destination_population"] == population
                and (include_external or not x["external_input"])]
    require(selected, "Population lacks configured incoming pathways")
    return [int(x["connection_id"]) - 1 for x in selected]


def verify_voltage_sum(pathways, voltage, connections, *, centered=False, rounding_reference=None):
    """Return the observed identity proof; allow bounded native print rounding.

    Population voltages may have a constant reference term. Centering handles
    that case for zero-input fluctuation analysis; paired deltas remove it.
    The tolerance is fixed at1e-12 times the original summed absolute pathway
    voltage scale. Actual pinned Dumpf source prints14 decimal digits in
    scientific notation. It never depends on baseline standard deviation.
    """
    d, v = np.asarray(pathways, float), np.asarray(voltage, float)
    require(d.ndim == 2 and d.shape[1] == 41 and v.shape == (len(d), 9)
            and np.isfinite(d).all() and np.isfinite(v).all(), "Invalid pathway sum inputs")
    original_reference = [max(float(np.max(np.sum(np.abs(d[:, incoming(connections, population)]), axis=1))),
                              float(np.max(np.abs(v[:, population - 1])))) for population in range(1, 10)]
    if centered:
        d, v = d - d.mean(axis=0), v - v.mean(axis=0)
    proofs = []
    for population in range(1, 10):
        channels = d[:, incoming(connections, population)]
        reconstruction = channels.sum(axis=1)
        error = float(np.max(np.abs(reconstruction - v[:, population - 1])))
        scale = original_reference[population - 1]
        if rounding_reference is not None:
            require(len(rounding_reference) == 9 and np.isfinite(rounding_reference).all(), "Invalid original-voltage rounding reference")
            scale = max(scale, float(rounding_reference[population - 1]))
        tolerance = float(16 * np.finfo(float).tiny + SUM_RELATIVE_TOLERANCE * scale)
        proofs.append({"population": population, "maximum_absolute_sum_error_V": error,
                       "sum_absolute_pathway_scale_V": scale, "fixed_rounding_tolerance_V": tolerance,
                       "voltage_sum_qualified": bool(error <= tolerance)})
    return proofs


def balance_metrics(pathways, time):
    """Physical-unit cancellation with zero-denominator disposition preserved."""
    d, t = np.asarray(pathways, float), np.asarray(time, float)
    require(d.ndim == 2 and len(d) == len(t) >= 2 and np.isfinite(d).all()
            and np.isfinite(t).all() and np.all(np.diff(t) > 0), "Invalid balance observations")
    absolute = np.sum(np.abs(d), axis=1)
    net = np.sum(d, axis=1)
    absolute_mass = float(np.trapezoid(absolute, t))
    net_mass = float(np.trapezoid(np.abs(net), t))
    positive_mass = float(np.trapezoid(np.maximum(d, 0).sum(axis=1), t))
    negative_mass = float(np.trapezoid(-np.minimum(d, 0).sum(axis=1), t))
    total_energy = float(np.trapezoid(np.sum(d*d, axis=1), t))
    net_energy = float(np.trapezoid(net*net, t))
    require(all(np.isfinite(x) for x in (absolute_mass, net_mass, positive_mass, negative_mass,
                                        total_energy, net_energy)), "Physical-unit metric overflow")
    cancellation = 1 - net_mass / absolute_mass if absolute_mass > 0 else None
    require(cancellation is None or -1e-12 <= cancellation <= 1 + 1e-12,
            "Cancellation violates triangle inequality")
    return {"n_incoming_pathways": d.shape[1], "absolute_pathway_mass_V_s": absolute_mass,
            "net_pathway_mass_V_s": net_mass, "positive_response_mass_V_s": positive_mass,
            "negative_response_mass_V_s": negative_mass, "relative_voltage_cancellation": cancellation,
            "individual_pathway_energy_V2_s": total_energy, "net_pathway_energy_V2_s": net_energy,
            "signed_cross_energy_V2_s": net_energy - total_energy,
            "both_response_signs_observed_fraction": float(np.mean((d.max(axis=1) > 0) & (d.min(axis=1) < 0))),
            "denominator_status": "nonzero_raw_voltage_mass" if absolute_mass > 0 else "zero_response"}


def energy_timing(signal, time, onset):
    y, t = np.asarray(signal, float), np.asarray(time, float)
    require(y.shape == t.shape and len(t) >= 2 and np.isfinite(y).all(), "Invalid pathway timing")
    energy_increment = (y[:-1]**2 + y[1:]**2) * np.diff(t) / 2
    cumulative = np.r_[0., np.cumsum(energy_increment)]
    total = float(cumulative[-1])
    require(np.isfinite(total), "Pathway timing energy overflow")
    result = {"response_energy_V2_s": total, "peak_absolute_response_V": float(np.max(np.abs(y))),
              "signed_area_V_s": float(np.trapezoid(y, t)), "timing_status": "nonzero" if total > 0 else "zero_response"}
    for fraction in (.1, .5, .9):
        result[f"energy_t{int(fraction*100)}_s"] = float(t[np.searchsorted(cumulative, fraction * total)] - onset) if total > 0 else None
    result["peak_latency_s"] = float(t[int(np.argmax(np.abs(y)))] - onset) if total > 0 else None
    return result


def cluster_bootstrap(rows, metric, seed, replicates=BOOTSTRAP_REPLICATES):
    """Configuration-cluster percentile interval; never a human P-value."""
    groups = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            require(np.isfinite(value), "Nonfinite bootstrap input")
            groups[row["configuration_cluster"]].append(float(value))
    values = np.asarray([np.mean(groups[k]) for k in sorted(groups)], float)
    result = {"metric": metric, "n_configuration_clusters": len(values), "estimate": float(np.mean(values)) if len(values) else None,
              "ci95_lower": None, "ci95_upper": None, "bootstrap_replicates": int(replicates),
              "inference": "descriptive finite-bank configuration-cluster sensitivity; no population significance"}
    if len(values) >= 2:
        rng = np.random.default_rng(seed)
        means = np.empty(replicates)
        for i in range(replicates):
            means[i] = np.mean(values[rng.integers(0, len(values), len(values))])
        result["ci95_lower"], result["ci95_upper"] = map(float, np.quantile(means, [.025, .975]))
    return result


def _record_meta(trace, supported):
    row = trace["row"]
    return {"simulation_id": row["simulation_id"], "configuration_cluster": row["simulation_id"],
            "variant": row["variant"], "account_class": row["account_class"],
            "human_training_validation_fit_supported": row["simulation_id"] in supported,
            "source_output_sha256": row["source_output_sha256"], "config_sha256": row["config_sha256"],
            "param_gain_factor": row.get("param_gain_factor"),
            "param_coupling_factor": row.get("param_coupling_factor"),
            "param_history_factor": row.get("param_history_factor")}


def pathway_rows(pathways, voltage, time, connections, meta, mode, rounding_reference=None):
    """Raw-tonic, centered fluctuations and paired-response units stay separate."""
    rows = []
    centered = mode == "zero_input_centered_fluctuation"
    d = pathways - pathways.mean(axis=0) if centered else pathways
    v = voltage - voltage.mean(axis=0) if centered else voltage
    if rounding_reference is None:
        rounding_reference = [max(float(np.max(np.sum(np.abs(pathways[:, incoming(connections, population)]), axis=1))),
                                  float(np.max(np.abs(voltage[:, population - 1])))) for population in range(1, 10)]
    proof = verify_voltage_sum(d, v, connections, rounding_reference=rounding_reference)
    for population in range(1, 10):
        for scope, external in (("all_incoming", True), ("endogenous_only", False)):
            positions = incoming(connections, population, external)
            result = balance_metrics(d[:, positions], time)
            valid = proof[population - 1]["voltage_sum_qualified"]
            above_floor = bool(np.max(np.abs(d[:, positions])) > RAW_VOLTAGE_FLOOR)
            rows.append(meta | {"mode": mode, "population": population, "scope": scope,
                                "voltage_sum_qualified": valid, **proof[population - 1], **result,
                                "mechanistic_interpretation_qualified": valid,
                                "raw_voltage_above_fixed_1e-12V_floor": above_floor,
                                "cancellation_for_inference": result["relative_voltage_cancellation"] if valid and above_floor else None})
    return rows


def configuration_summary(rows):
    all_rows = [r for r in rows if r["scope"] == "all_incoming"]
    qualified = all(r["voltage_sum_qualified"] for r in all_rows)
    cancellations = [r["cancellation_for_inference"] for r in all_rows
                     if r["cancellation_for_inference"] is not None]
    return {"all_population_sums_qualified": qualified,
            "n_population_cancellations_above_fixed_floor": len(cancellations),
            "mean_population_cancellation": float(np.mean(cancellations)) if qualified and cancellations else None,
            "total_pathway_absolute_mass_V_s": float(sum(r["absolute_pathway_mass_V_s"] for r in all_rows)),
            "cortical_cancellation": all_rows[0]["cancellation_for_inference"] if qualified else None,
            "cortical_absolute_mass_V_s": all_rows[0]["absolute_pathway_mass_V_s"]}


def compare_records(records, *, kind):
    """Keep all match dispositions and fit-support counts; never loosen matching."""
    from population_perturbations import match_controls, clipping_stratum
    rows, reasons = [], Counter()
    for a, b in itertools.combinations(records, 2):
        for matching in ("score", "score_and_conventional_features"):
            try:
                matched, reason = match_controls(a, b, spectra=matching != "score")
            except (KeyError, ValueError):
                reasons[matching + ":unavailable_score_or_feature"] += 1
                continue
            reasons[matching + ":" + reason] += 1
            if not matched:
                continue
            qualified = a["summary"]["all_population_sums_qualified"] and b["summary"]["all_population_sums_qualified"]
            row = {"analysis": kind, "matching": matching, "variant": a["variant"],
                   "input_amplitude": a["input_amplitude"], "simulation_a": a["simulation_id"],
                   "simulation_b": b["simulation_id"], "family_a": a["account_class"],
                   "family_b": b["account_class"],
                   "both_human_training_validation_fit_supported": a["human_training_validation_fit_supported"] and b["human_training_validation_fit_supported"],
                   "clinical_score_clipping_stratum": clipping_stratum(a["control_row"], b["control_row"]),
                   "all_population_sums_qualified": qualified,
                   "cortical_cancellation_absolute_difference": None,
                   "mean_population_cancellation_absolute_difference": None,
                   "raw_pathway_signature_relative_distance": None,
                   "raw_population_signature_relative_distance": None,
                   "hidden_minus_cortical_raw_response_distance": None,
                   "raw_pathway_signature_rms_difference_V": None,
                   "raw_population_signature_rms_difference_V": None,
                   "common_raw_voltage_scale_V": None,
                   "relative_distance_status": "sum_identity_unqualified",
                   "raw_pathway_mass_V_s_a": a["summary"]["total_pathway_absolute_mass_V_s"],
                   "raw_pathway_mass_V_s_b": b["summary"]["total_pathway_absolute_mass_V_s"]}
            if qualified:
                for metric in ("cortical_cancellation", "mean_population_cancellation"):
                    va, vb = a["summary"][metric], b["summary"][metric]
                    row[metric + "_absolute_difference"] = abs(va - vb) if va is not None and vb is not None else None
                row["raw_pathway_signature_rms_difference_V"] = float(np.sqrt(np.mean((a["pathway_signature"] - b["pathway_signature"])**2)))
                row["raw_population_signature_rms_difference_V"] = float(np.sqrt(np.mean((a["population_signature"] - b["population_signature"])**2)))
                scale = max(float(np.max(np.abs(a["pathway_signature"]))), float(np.max(np.abs(b["pathway_signature"]))),
                            float(np.max(np.abs(a["population_signature"]))), float(np.max(np.abs(b["population_signature"]))))
                row["common_raw_voltage_scale_V"] = scale
                row["relative_distance_status"] = "above_fixed_1e-12V_numerical_floor" if scale > RAW_VOLTAGE_FLOOR else "both_below_fixed_1e-12V_numerical_floor"
                if scale > RAW_VOLTAGE_FLOOR:
                    row["raw_pathway_signature_relative_distance"] = row["raw_pathway_signature_rms_difference_V"] / scale
                    row["raw_population_signature_relative_distance"] = row["raw_population_signature_rms_difference_V"] / scale
                    hidden_rms = float(np.sqrt(np.mean((a["population_signature"][:, 1:] - b["population_signature"][:, 1:])**2)))
                    cortical_rms = float(np.sqrt(np.mean((a["population_signature"][:, :1] - b["population_signature"][:, :1])**2)))
                    row["hidden_minus_cortical_raw_response_distance"] = (hidden_rms - cortical_rms) / scale
            rows.append(row)
    return rows, dict(reasons)


def evaluate_bank(bank, supported, *, raw_loader=load_pathways, expected_pairs=48,
                  expected_supported_zero=185):
    from native_population import qualify_pair
    from population_perturbations import pair_groups
    pairs = pair_groups(bank, expected_pairs=expected_pairs)
    zero_ids = {s for s, trace in bank["traces"].items()
                if s in supported and float(trace["row"]["input_amplitude"]) == 0}
    require(len(zero_ids) == expected_supported_zero,
            f"Expected {expected_supported_zero} fit-supported zero-input configurations; found {len(zero_ids)}")
    require(not (supported - set(bank["traces"])), "Fit support names absent or unqualified configurations")
    selected = zero_ids | {t["row"]["simulation_id"] for pair in pairs for t in pair}
    require(len(selected) <= MAX_RAW_FILES, "Raw-file scope exceeds frozen bound")
    raw, bytes_read = {}, 0
    dispositions = []
    for row in bank["rows"]:
        sid = row["simulation_id"]
        dispositions.append({"simulation_id": sid, "variant": row.get("variant"),
                             "native_status": row.get("status"), "population_status": row.get("population_status"),
                             "human_training_validation_fit_supported": sid in supported,
                             "selected_raw_pathways": sid in selected,
                             "selection_reason": "existing_pair_or_supported_zero_input" if sid in selected else "outside_frozen_existing_data_scope"})
    for count, sid in enumerate(sorted(selected), 1):
        raw[sid], size = raw_loader(bank["traces"][sid], bank["receipt"]["bank_root"])
        require(raw[sid].shape == (len(bank["traces"][sid]["times"]), 41), "Raw pathway matrix shape differs")
        bytes_read += size
        if count % 16 == 0:
            print(json.dumps({"stage": "pathway_voltage_read", "done": count, "total": len(selected)}), flush=True)
    balance, edge, pair_proofs, configs, baseline_records, pulse_records = [], [], [], [], [], []
    for sid in sorted(zero_ids):
        trace = bank["traces"][sid]
        meta = _record_meta(trace, supported)
        time, d, v = trace["times"], raw[sid], trace["V"]
        for mode in ("zero_input_tonic_and_fluctuation", "zero_input_centered_fluctuation"):
            rows = pathway_rows(d, v, time, trace["config"]["connections"], meta, mode)
            balance.extend(rows)
            configs.append(meta | {"mode": mode, "input_amplitude": 0., **configuration_summary(rows)})
        # Static reference terms are removed before comparing fluctuation signatures.
        dc, vc = d - d.mean(axis=0), v - v.mean(axis=0)
        baseline_records.append(meta | {"control_row": trace["row"], "input_amplitude": 0.,
            "input_onset_absolute_s": float(trace["row"]["input_onset_s"]), "input_width_s": float(trace["row"]["input_width_s"]),
            "noise_stream_sha256": hashlib.sha256(np.asarray(trace["noise"], dtype="<f8").tobytes()).hexdigest(),
            "summary": configuration_summary(rows),
            "pathway_signature": np.sqrt(np.mean(dc**2, axis=0, keepdims=True)),
            "population_signature": np.sqrt(np.mean(vc**2, axis=0, keepdims=True))})
    for control, pulse in pairs:
        proof = qualify_pair(control, pulse)
        ca, pb = control["row"]["simulation_id"], pulse["row"]["simulation_id"]
        meta = _record_meta(control, supported) | {"pair_id": proof["pair_id"],
                "configuration_cluster": proof["variant"] + "__" + proof["pair_id"],
                "control_id": ca, "pulse_id": pb, "input_amplitude": proof["amplitude"]}
        require(np.allclose(raw[ca][control["times"] < proof["onset_relative_s"] - .01],
                            raw[pb][control["times"] < proof["onset_relative_s"] - .01], rtol=1e-10, atol=1e-12),
                "Pathways differ before configured pulse")
        time = control["times"]
        post = time >= proof["onset_relative_s"]
        d = raw[pb][post] - raw[ca][post]
        v = pulse["V"][post] - control["V"][post]
        rounding_reference = [max(float(np.max(np.sum(np.abs(raw[s][:, incoming(control["config"]["connections"], pop)]), axis=1))) for s in (ca, pb))
                              for pop in range(1, 10)]
        rows = pathway_rows(d, v, time[post], control["config"]["connections"], meta,
                            "paired_finite_pulse_response", rounding_reference=rounding_reference)
        balance.extend(rows)
        summary = configuration_summary(rows)
        configs.append(meta | {"mode": "paired_finite_pulse_response", **summary})
        pair_proofs.append(meta | proof | {"all_population_delta_voltage_sums_qualified": summary["all_population_sums_qualified"]})
        for connection in control["config"]["connections"]:
            idx = connection["connection_id"] - 1
            edge.append(meta | connection | energy_timing(d[:, idx], time[post], proof["onset_relative_s"]))
        pulse_records.append(meta | {"control_row": control["row"],
            "input_onset_absolute_s": float(control["row"]["input_onset_s"]), "input_width_s": float(control["row"]["input_width_s"]),
            "noise_stream_sha256": hashlib.sha256(np.asarray(control["noise"], dtype="<f8").tobytes()).hexdigest(),
            "summary": summary, "pathway_signature": d, "population_signature": v})
    contrasts, matching = [], {}
    for kind, records in (("supported_zero_input_fluctuations", baseline_records), ("existing_paired_pulse_response", pulse_records)):
        result, reasons = compare_records(records, kind=kind)
        contrasts.extend(result)
        matching[kind] = reasons
    return {"balance": balance, "edge": edge, "pair_proofs": pair_proofs,
            "configurations": configs, "contrasts": contrasts, "matching": matching,
            "dispositions": dispositions, "raw_file_reads": len(selected), "raw_bytes_read": bytes_read,
            "supported_zero_input_configurations": len(zero_ids), "existing_pairs": len(pairs)}


def source_equation_proof(bank_receipt_spec):
    """Verify the actual compiled-source identity and voltage, not current, law."""
    parent_path = pinned(bank_receipt_spec)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    require(parent.get("status") == "native_simulation_bank_completed", "Unqualified native bank receipt")
    result = []
    expected_variants = {"pristine", "scaled_dendrite_v1_explicit_patch"}
    require({v["variant"] for v in parent.get("variants", [])} == expected_variants,
            "Expected both named native implementations")
    needed = ("population.cpp", "firing_response.cpp", "firing_response.h", "dendrite.cpp", "dendrite.h", "dendrite_scaled.cpp", "dumpf.cpp")
    for variant in parent["variants"]:
        root = Path(variant["executable"]).parent / "src"
        texts, files = {}, {}
        for name in needed:
            path = root / name
            require(path.is_file() and not path.is_symlink() and path.resolve() == path.absolute()
                    and path.stat().st_size < 256 * 1024, "Missing or noncanonical source: " + str(path))
            expected = variant["source_hashes"].get(name)
            require(isinstance(expected, str) and sha(path) == expected, "Compiled-source pin differs: " + name)
            texts[name] = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", " ", path.read_text(encoding="utf-8"))
            files[name] = {"path": str(path), "sha256": expected}
        compact = {name: re.sub(r"\s+", "", value) for name, value in texts.items()}
        anchors = {
            "population_delegates_voltage": "returnfiring_response->V();" in compact["population.cpp"],
            "soma_resets_voltage": "v[i]=0;" in compact["firing_response.cpp"],
            "soma_sums_saved_dendrite_voltage": "v[j]+=dendrites[i]->V()[j];" in compact["firing_response.cpp"],
            "soma_voltage_getter": "FiringResponse::V()const{returnv;}" in compact["firing_response.h"],
            "dendrite_voltage_getter": "Dendrite::V()const{returnv;}" in compact["dendrite.h"],
            "dendrite_output_is_voltage": 'output("Dendrite",index+1,"V",v);' in compact["dendrite.cpp"],
            "scaled_output_is_voltage": 'output("Dendrite",index+1,"V",v);' in compact["dendrite_scaled.cpp"],
            "coupling_already_enters_dendrite": "(*de)[2][i]=precouple[i];" in compact["dendrite.cpp"],
            "dendrite_integrator_exports_voltage": "v[i]=(*de)[0][i];" in compact["dendrite.cpp"],
            "scaled_kernel_contains_hfac": "h_fac*y[2]" in compact["dendrite_scaled.cpp"],
            "actual_printed_precision_14": "file.precision(14);" in compact["dumpf.cpp"],
            "actual_output_is_scientific": "file<<scientific;" in compact["dumpf.cpp"]}
        require(all(anchors.values()), "Actual native source lacks the qualified voltage equation: " + str(anchors))
        if variant["variant"] == "scaled_dendrite_v1_explicit_patch":
            require("voidDendriteScaled::step()" in compact["dendrite_scaled.cpp"], "Named patch step absent")
        result.append({"variant": variant["variant"], "source_files": files, "equation_anchors": anchors,
                       "reconstructed_equation": "PopulationV_a(t)=sum_j DendriteV_{j->a}(t)",
                       "printed_fractional_decimal_digits": 14, "scientific_notation": True,
                       "per_field_relative_rounding_bound": PRINTED_RELATIVE_ROUNDING_BOUND,
                       "sum_relative_tolerance_on_original_raw_scale": SUM_RELATIVE_TOLERANCE,
                       "coupling_weight_applied_again": False, "saved_variable_is_electric_current": False})
    return {"status": "actual_compiled_source_voltage_sum_verified", "bank_receipt_sha256": sha(parent_path),
            "variants": result, "printed_sum_identity_also_required": True}


def _cache_location(spec):
    path = Path(spec["path"] if isinstance(spec, dict) else spec)
    receipt = path / "receipt.json"
    if isinstance(spec, dict) and spec.get("receipt_sha256"):
        require(sha(receipt) == spec["receipt_sha256"], "Native cache receipt SHA256 differs")
    return path


def qualification_probe(inputs):
    """Scheduled, bounded real-pair qualification, with no effect-direction gate."""
    from native_population import load_native, qualify_pair, sha256
    require(re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")), "Probe requires Slurm allocation")
    specs = inputs.get("files", inputs)
    source = source_equation_proof(specs["bank_receipt"])
    pinned(specs["bridge_compatibility"])
    from population_perturbations import read_compatibility
    supported, support_receipt = read_compatibility(specs["bridge_compatibility"])
    cache = _cache_location(specs["native_cache"])
    receipt = json.loads((cache / "receipt.json").read_text(encoding="utf-8"))
    require(receipt.get("schema") == "native_nine_population_cache_v2"
            and receipt.get("status") == "native_population_cache_qualified", "Actual v2 native cache required")
    require(receipt.get("bank_receipt_sha256") == specs["bank_receipt"]["sha256"], "Cache compiled-source parent differs")
    metadata_path = cache / "population_metadata.json"
    require(sha256(metadata_path) == receipt["files"][metadata_path.name]["sha256"], "Native metadata changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["traces"]
    groups = defaultdict(list)
    for item in metadata:
        row = item["row"]
        if row.get("pair_id"):
            groups[(row["variant"], row["pair_id"])].append(row)
    require(groups, "No existing paired simulations")
    pair = sorted(next(iter(sorted(groups.items())))[1], key=lambda x: float(x["input_amplitude"]))
    require(len(pair) == 2, "Ambiguous actual probe pair")
    a, b = [load_native(row, receipt["bank_root"]) for row in pair]
    proof = qualify_pair(a, b)
    da, ba = load_pathways(a, receipt["bank_root"])
    db, bb = load_pathways(b, receipt["bank_root"])
    post = a["times"] >= proof["onset_relative_s"]
    rounding_reference = [max(float(np.max(np.sum(np.abs(d[:, incoming(a["config"]["connections"], pop)]), axis=1))) for d in (da, db))
                          for pop in range(1, 10)]
    sums = verify_voltage_sum(db[post] - da[post], b["V"][post] - a["V"][post],
                              a["config"]["connections"], rounding_reference=rounding_reference)
    require(all(x["voltage_sum_qualified"] for x in sums), "Actual paired voltage sum failed")
    return {"status": "actual_pair_pathway_equations_clock_and_voltage_sum_qualified",
            "source": source, "pair": proof, "population_sum_proof": sums,
            "training_validation_fit_support": support_receipt,
            "raw_pathway_files": 2, "raw_pathway_bytes": ba + bb,
            "scientific_effect_direction_tested": False, "raw_output_modified": False}


def run(inputs, output, seed=20261024):
    from native_population import read_cache
    from population_perturbations import read_compatibility
    require(re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")), "Analysis requires Slurm allocation")
    specs = inputs.get("files", inputs)
    require(all(role in specs for role in REQUIRED_ROLES), "Missing required input role")
    source = source_equation_proof(specs["bank_receipt"])
    cache = _cache_location(specs["native_cache"])
    bank = read_cache(cache)
    require(bank["receipt"].get("schema") == "native_nine_population_cache_v2", "Corrected native clock cache required")
    require(bank["receipt"].get("bank_receipt_sha256") == specs["bank_receipt"]["sha256"], "Cache compiled-source parent differs")
    pinned(specs["bridge_compatibility"])
    supported, support_receipt = read_compatibility(specs["bridge_compatibility"])
    result = evaluate_bank(bank, supported)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for name, key in (("population_pathway_balance.csv", "balance"), ("edge_response_timing.csv", "edge"),
                      ("pair_voltage_sum_proof.csv", "pair_proofs"), ("configuration_summary.csv", "configurations"),
                      ("matched_pathway_contrasts.csv", "contrasts"), ("configuration_dispositions.csv", "dispositions")):
        write_csv(out / name, result[key])
    write_json(out / "source_equation_proof.json", source)
    strata = defaultdict(list)
    for row in result["balance"]:
        strata[(row["mode"], row["variant"], row["account_class"], row.get("input_amplitude", 0.), row["scope"], row["population"])].append(row)
    bootstraps = []
    for idx, (key, rows) in enumerate(sorted(strata.items())):
        mode, variant, family, amplitude, scope, population = key
        interval = cluster_bootstrap(rows, "cancellation_for_inference", seed + idx)
        bootstraps.append({"mode": mode, "variant": variant, "account_class": family,
                           "input_amplitude": amplitude, "scope": scope, "population": population, **interval})
    write_json(out / "configuration_cluster_bootstrap.json", bootstraps)
    summary = {"status": "completed_existing_native_pathway_balance", "version": VERSION,
        "seed": int(seed), "raw_file_reads": result["raw_file_reads"], "raw_bytes_read": result["raw_bytes_read"],
        "maximum_raw_file_reads": MAX_RAW_FILES, "existing_control_pulse_pairs": result["existing_pairs"],
        "supported_zero_input_configurations": result["supported_zero_input_configurations"],
        "n_population_balance_rows": len(result["balance"]), "n_edge_timing_rows": len(result["edge"]),
        "pair_sum_identity_passed": sum(r["all_population_delta_voltage_sums_qualified"] for r in result["pair_proofs"]),
        "population_sum_identity_failed_rows": sum(not r["voltage_sum_qualified"] for r in result["balance"]),
        "matched_comparison_dispositions": result["matching"], "bridge_fit_support": support_receipt,
        "matched_comparison_counts": dict(Counter(r["analysis"] + ":" + r["matching"] for r in result["contrasts"])),
        "strict_supported_pulse_contrasts": sum(r["analysis"] == "existing_paired_pulse_response" and r["matching"] == "score_and_conventional_features" and r["both_human_training_validation_fit_supported"] for r in result["contrasts"]),
        "fixed_raw_voltage_floor_V": RAW_VOLTAGE_FLOOR, "per_channel_baseline_SD_scaling": False,
        "inference": {"one_spatial_node": True, "human_mechanism_established": False,
            "independent_human_n": 0, "model_configurations_are_human_participants": False,
            "configuration_bootstrap_is_population_significance": False,
            "unilateral_pathway_or_loop_intervention": False, "new_simulations": False,
            "test_outcomes_used_for_configuration_selection": False,
            "source_variable": "postsynaptic voltage contribution; not reconstructed electric current",
            "both_human_data_finetunings_before_comparison": "enforced_by_root_driver", "target_participants": 42}}
    write_json(out / "summary.json", summary)
    receipt = {"status": summary["status"], "version": VERSION, "seed": int(seed),
               "output_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
               "raw_file_reads": result["raw_file_reads"], "raw_bytes_read": result["raw_bytes_read"],
               "target_participants": 42, "both_human_data_finetunings_before_comparison": "enforced_by_root_driver"}
    write_json(out / "receipt.json", receipt)
    return receipt
