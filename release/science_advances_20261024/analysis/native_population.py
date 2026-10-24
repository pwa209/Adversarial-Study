"""Source-pinned, scheduled loading of all nine native neural populations.

This module qualifies simulator data, not human mechanisms. The two named native
implementations remain separate. Populations 10/11 are observed inputs and never
become additional neural populations. No native output is read during import.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

import numpy as np

SCHEMA = "native_nine_population_cache_v2"
VARIANTS = {"pristine", "scaled_dendrite_v1_explicit_patch"}
COLUMNS = [f"Pop.{i}.{kind}" for i in range(1, 10) for kind in ("V", "Q")]
N_ROWS, DT, MAX_NATIVE_BYTES = 5000, .002, 64 * 1024 * 1024
CLOCK_METADATA = {"output_rows_each": N_ROWS, "output_interval_s": DT,
                  "output_first_time_s": DT, "output_last_time_s": N_ROWS * DT,
                  "output_time_reference": "native_relative_to_configured_output_start",
                  "output_time_labels_preserved": True, "output_clock_rebased": False}
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
SEPARATOR = "=" * 45


class ProvenanceError(ValueError):
    """An identity mismatch is fatal, rather than an analysis exclusion."""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _actual_clock(times):
    """Qualify the native labels without rebasing their physical time reference."""
    return (times.shape == (N_ROWS,) and np.isfinite(times).all()
            and np.allclose(times, np.arange(1, N_ROWS + 1) * DT, rtol=0, atol=1e-9))


def _error_representatives(dispositions):
    """Bound diagnostic size while the CSV retains every full row/reason."""
    grouped = {}
    for row in dispositions:
        if row["population_status"] == "qualified":
            continue
        key = (row["population_status"], row.get("population_error", ""))
        group = grouped.setdefault(key, {"population_status": key[0],
                                         "population_error": key[1][:512], "count": 0,
                                         "simulation_ids": []})
        group["count"] += 1
        if len(group["simulation_ids"]) < 8:
            group["simulation_ids"].append(row["simulation_id"])
    return {"distinct_error_groups": len(grouped), "groups": list(grouped.values())[:20],
            "maximum_groups": 20, "maximum_ids_per_group": 8, "maximum_error_characters": 512}


def _write_dispositions(output_dir, dispositions):
    fields = sorted(set().union(*(r.keys() for r in dispositions)))
    with (output_dir / "population_dispositions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dispositions)


def _safe_file(path, maximum, root=None):
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve() != path:
        raise ProvenanceError("Noncanonical or symlink native path: " + str(path))
    if root is not None and not path.is_relative_to(Path(root).resolve()):
        raise ProvenanceError("Native source leaves the pinned bank root")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
        raise ProvenanceError("Missing, nonregular or oversized native source: " + str(path))
    return path


def _pinned_file(spec, maximum=134217728):
    if not isinstance(spec, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(spec.get("sha256", ""))):
        raise ProvenanceError("Input tables require an exact SHA256 pin")
    path = _safe_file(spec["path"], min(maximum, int(spec.get("max_bytes", maximum))))
    if sha256(path) != spec["sha256"]:
        raise ProvenanceError("Pinned input table changed: " + str(path))
    return path


def _mask_comments(text):
    return re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", " ", text)


def _one_number(text, label):
    values = re.findall(r"\b" + re.escape(label) + r":\s*(" + FLOAT + r")", text)
    if len(values) != 1 or not np.isfinite(float(values[0])):
        raise ValueError("Expected exactly one finite native field: " + label)
    return float(values[0])


def _population_blocks(text):
    starts = list(re.finditer(r"^\s*Population\s+(\d+):\s*([^\n]+)", text, re.M))
    blocks = {}
    for i, match in enumerate(starts):
        number = int(match[1])
        if number in blocks:
            raise ValueError("Duplicate configured population")
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        body = text[match.end():end]
        body = re.split(r"^\s*(?:Propagator\s+\d+:|Coupling\s+\d+:|Output:)", body, maxsplit=1, flags=re.M)[0]
        blocks[number] = {"name": match[2].strip(), "text": body}
    return blocks


def parse_configuration(text):
    """Record actual configured population and connection metadata, including inputs."""
    clean = _mask_comments(text)
    if _one_number(clean, "Nodes") != 1:
        raise ValueError("This release is qualified for one spatial node only")
    blocks = _population_blocks(clean)
    if set(blocks) != set(range(1, 12)):
        raise ValueError("Native configuration must contain nine neural populations and two inputs")
    output_match = re.search(r"^\s*Output:\s*(.*)$", clean, re.M)
    if output_match is None:
        raise ValueError("Missing actual native output configuration")
    output = output_match[1]
    start, interval = _one_number(output, "Start"), _one_number(output, "Interval")
    duration = _one_number(clean[:output_match.start()], "Time")
    integration_step = _one_number(clean[:output_match.start()], "Deltat")
    if not np.allclose([start, interval, duration, integration_step], [10, DT, 20, .0001], rtol=0, atol=1e-12):
        raise ValueError("Native integration/output clock differs from qualified bank")
    matrix_from = re.search(r"^\s*From:\s*([^\n]+)", clean, re.M)
    if matrix_from is None:
        raise ValueError("Missing configured connection matrix")
    sources = [int(x) for x in matrix_from[1].split()]
    if sources != list(range(1, 12)):
        raise ValueError("Connection source population identities changed")
    connections, destinations = [], set()
    for destination, line in re.findall(r"^\s*To\s+(\d+):\s*([^\n]+)", clean, re.M):
        destination = int(destination)
        ids = [int(x) for x in line.split()]
        if len(ids) != 11 or destination in destinations:
            raise ValueError("Malformed or duplicate configured connection destination")
        destinations.add(destination)
        connections.extend({"connection_id": number, "source_population": source,
                            "destination_population": destination,
                            "external_input": source in (10, 11)}
                           for source, number in zip(sources, ids) if number)
    if destinations != set(range(1, 12)) or sorted(x["connection_id"] for x in connections) != list(range(1, 42)):
        raise ValueError("Configured 41-connection bank topology changed")
    for item in connections:
        number = item["connection_id"]
        match = re.search(r"^\s*Coupling\s+" + str(number) + r":\s*(.*?)(?=^\s*(?:Coupling\s+\d+:|Output:)|\Z)", clean, re.M | re.S)
        if match is None:
            raise ValueError("Missing coupling configuration")
        item["nu"] = _one_number(match[1], "nu")
        dendrite = re.search(r"^\s*Dendrite\s+" + str(number) + r":\s*([^\n]+)", clean, re.M)
        propagator = re.search(r"^\s*Propagator\s+" + str(number) + r":\s*([^\n]+)", clean, re.M)
        if dendrite is None or propagator is None:
            raise ValueError("Missing synaptic or propagation configuration")
        item["dendrite"] = dendrite[1].strip()
        item["propagator"] = propagator[1].strip()
    pulse = blocks[11]["text"]
    if not re.search(r"\bStimulus:\s*PulseRect\b", pulse):
        raise ValueError("Population 11 must be the configured rectangular input")
    return {"spatial_nodes": 1, "neural_population_ids": list(range(1, 10)),
            "input_population_ids": [10, 11], "population_names": {str(k): v["name"] for k, v in blocks.items()},
            "connections": sorted(connections, key=lambda x: x["connection_id"]),
            "connection_count": len(connections),
            "endogenous_connection_count": sum(not x["external_input"] for x in connections),
            "external_input_connection_count": sum(x["external_input"] for x in connections),
            "integration_step_s": integration_step, "output_start_s": start, "output_interval_s": interval,
            "input_onset_s": _one_number(pulse, "Onset"), "input_width_s": _one_number(pulse, "Width"),
            "input_amplitude": _one_number(pulse, "Amplitude"), "input_pulses": _one_number(pulse, "Pulses")}


def parse_native_bytes(raw):
    """Parse only the numeric section after an exact standalone NFTsim separator."""
    text = raw.decode("utf-8", errors="strict")
    lines = text.splitlines()
    positions = [i for i, line in enumerate(lines) if line.strip() == SEPARATOR]
    if len(positions) != 1:
        raise ValueError("Expected one exact native configuration/data separator")
    payload = [line for line in lines[positions[0] + 1:] if line.strip()]
    if len(payload) != N_ROWS + 2:
        raise ValueError("Native output does not have exactly 5000 rows")
    columns = payload[0].split()
    if not columns or columns[0] != "Time" or len(set(columns)) != len(columns):
        raise ValueError("Missing Time header or duplicated native outputs")
    nodes = payload[1].split()
    if len(nodes) != len(columns) - 1 or any(x != "1" for x in nodes):
        raise ValueError("Native data must preserve exactly one node per output")
    if not set(COLUMNS + ["Pop.10.Q"]).issubset(columns):
        raise ValueError("All nine voltages/rates and actual noise input must be observed")
    array = np.loadtxt(io.StringIO("\n".join(payload[2:])), dtype=np.float64, ndmin=2)
    if array.shape != (N_ROWS, len(columns)) or not np.isfinite(array).all():
        raise ValueError("Malformed or nonfinite native numeric output")
    times = array[:, 0]
    if not _actual_clock(times):
        raise ValueError("Native output must preserve .002..10 seconds on the fixed 2-ms grid")
    values = array[:, [columns.index(name) for name in COLUMNS]]
    noise = array[:, columns.index("Pop.10.Q")]
    stimulus = array[:, columns.index("Pop.11.Q")] if "Pop.11.Q" in columns else np.zeros(N_ROWS)
    return {"times": times, "values": values, "columns": list(COLUMNS), "noise": noise,
            "stimulus": stimulus, "stimulus_observed": "Pop.11.Q" in columns,
            "native_columns": columns, "native_column_count": len(columns)}


def _aliases(trace):
    trace.update(time=trace["times"], voltage=trace["values"][:, ::2], rate=trace["values"][:, 1::2],
                 V=trace["values"][:, ::2], Q=trace["values"][:, 1::2], noise_Q=trace["noise"],
                 perturb_Q=trace["stimulus"], configuration_text=trace["config_text"])
    return trace


def load_native(row, allowed_root=None):
    row = dict(row)
    simulation_id, variant = row.get("simulation_id", ""), row.get("variant", "")
    if variant not in VARIANTS or not simulation_id.startswith(variant + "__") or not re.fullmatch(r"[A-Za-z0-9_\-]+", simulation_id):
        raise ProvenanceError("Native simulation/implementation identity differs")
    root = Path(allowed_root).resolve() if allowed_root is not None else Path(row["source_output_path"]).absolute().parent.parent
    path = _safe_file(row["source_output_path"], MAX_NATIVE_BYTES, root)
    if path != root / simulation_id / "native.output":
        raise ProvenanceError("Native output path does not match its inventoried simulation")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row.get("source_output_sha256"):
        raise ProvenanceError("Native output SHA256 differs from the pinned inventory")
    configuration = _safe_file(path.parent / "configuration.conf", 1024 * 1024, root)
    config_raw = configuration.read_bytes()
    if hashlib.sha256(config_raw).hexdigest() != row.get("config_sha256"):
        raise ProvenanceError("Actual native configuration SHA256 differs from inventory")
    text = config_raw.decode("utf-8", errors="strict")
    config = parse_configuration(text)
    for key in ("input_onset_s", "input_width_s", "input_amplitude"):
        if not np.isclose(float(row[key]), config[key], rtol=1e-12, atol=1e-12):
            raise ProvenanceError("Inventory/configuration input metadata differs: " + key)
    trace = parse_native_bytes(raw)
    trace.update(row=row, config_text=text, config=config,
                 raw_bytes=len(raw), source_output_sha256=row["source_output_sha256"], config_sha256=row["config_sha256"])
    return _aliases(trace)


load_population = load_native


def _pair_config(text, mask_seed):
    clean = _mask_comments(text)
    blocks = _population_blocks(clean)
    # Replace only the amplitude in Population 11, never another population's field.
    pulse = blocks[11]["text"]
    replaced, count = re.subn(r"\bAmplitude:\s*" + FLOAT, "Amplitude: <PAIRED_INPUT>", pulse)
    if count != 1:
        raise ValueError("Paired input amplitude does not identify one field")
    clean = clean.replace(pulse, replaced, 1)
    if mask_seed:
        # A stream-identity check is required before allowing this metadata change.
        blocks = _population_blocks(clean)
        noise = blocks[10]["text"]
        replaced = re.sub(r"\bWhiteSeed:\s*" + FLOAT, "WhiteSeed: <OBSERVED_SAME_STREAM>", noise)
        clean = clean.replace(noise, replaced, 1)
    return " ".join(clean.split())


def qualify_pair(control, pulse):
    """Reject unmatched backgrounds, hidden parameter changes or unobserved inputs."""
    a, b = control["row"], pulse["row"]
    for key in ("variant", "account_class", "pair_id", "param_gain_factor", "param_coupling_factor", "param_history_factor"):
        if str(a.get(key, "")) != str(b.get(key, "")):
            raise ValueError("Pulse/control metadata differ: " + key)
    if not a.get("pair_id") or float(a["input_amplitude"]) != 0 or float(b["input_amplitude"]) <= 0:
        raise ValueError("Pair must have one actual zero-input control and one positive pulse")
    if not np.array_equal(control["times"], pulse["times"]) or not np.array_equal(control["noise"], pulse["noise"]):
        raise ValueError("Pulse/control clocks or complete observed noise streams differ")
    if _pair_config(control["config_text"], False) != _pair_config(pulse["config_text"], False):
        if _pair_config(control["config_text"], True) != _pair_config(pulse["config_text"], True):
            raise ValueError("Pulse/control configurations differ beyond amplitude/observationally matched WhiteSeed")
        seed_change = True
    else:
        seed_change = False
    cfg = pulse["config"]
    onset, width = cfg["input_onset_s"] - cfg["output_start_s"], cfg["input_width_s"]
    amplitude = cfg["input_amplitude"]
    t = pulse["times"]
    before, active = t < onset - .01, (t >= onset - DT * 1.05) & (t <= onset + width + DT * 1.05)
    if not before.any() or not active.any() or not (t[-1] > onset + width):
        raise ValueError("Observed output lacks a pre-input baseline and post-input response")
    if not control["stimulus_observed"] or not pulse["stimulus_observed"]:
        raise ValueError("Pulse/control population 11 must be physically observed")
    if np.max(np.abs(control["stimulus"])) > 1e-12 or np.max(np.abs(pulse["stimulus"][~active])) > 1e-12:
        raise ValueError("Input occurs outside configured pulse or in control")
    observed = float(np.max(np.abs(pulse["stimulus"][active])))
    if not np.isclose(observed, amplitude, rtol=1e-8, atol=1e-12):
        raise ValueError("Observed population-11 amplitude differs from configuration")
    tolerance = np.maximum(np.max(np.abs(control["values"][before]), axis=0), 1.) * 1e-10
    if not np.allclose(control["values"][before], pulse["values"][before], rtol=0, atol=tolerance):
        raise ValueError("Nine-population trajectories differ before the input")
    return {"status": "qualified_actual_matched_noise_pair", "control_id": a["simulation_id"],
            "pulse_id": b["simulation_id"], "variant": a["variant"], "pair_id": a["pair_id"],
            "onset_relative_s": onset, "width_s": width, "amplitude": amplitude,
            "full_noise_stream_exactly_equal": True, "all_nine_VQ_baseline_matched": True,
            "configuration_diff_only_amplitude_and_observed_same_stream_seed": True,
            "WhiteSeed_metadata_changed": seed_change, "observed_input_peak": observed,
            "baseline_max_absolute_difference": float(np.max(np.abs(control["values"][before] - pulse["values"][before])))}


def _inputs(inputs):
    return inputs.get("files", inputs)


def _csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_bank(inputs, output_dir, seed=20261024):
    """Build one aggregate cache; call this in a scheduled worker, never login nodes."""
    if not re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")):
        raise RuntimeError("Full native bank reads require a scheduled Slurm worker")
    specs = _inputs(inputs)
    inventory_spec = specs.get("bank_inventory", specs.get("native_bank"))
    if inventory_spec is None:
        raise ProvenanceError("Missing bank_inventory pinned input")
    inventory_path = _pinned_file(inventory_spec)
    rows = _csv_rows(inventory_path)
    if not rows or len(rows) > 824 or len({r.get("simulation_id") for r in rows}) != len(rows):
        raise ProvenanceError("Empty, excessive or duplicate native inventory identities")
    parent = None
    if specs.get("bank_receipt"):
        parent = json.loads(_pinned_file(specs["bank_receipt"]).read_text(encoding="utf-8"))
        if parent.get("status") != "native_simulation_bank_completed" or parent.get("inventory_sha256") != inventory_spec["sha256"] or parent.get("candidate_count") != len(rows):
            raise ProvenanceError("Native bank parent receipt does not bind this inventory")
        if {r.get("variant") for r in rows} - {v["variant"] for v in parent.get("variants", [])}:
            raise ProvenanceError("Unqualified implementation in inventory")
    scored_spec = specs.get("scored_bank")
    if scored_spec:
        scored_rows = _csv_rows(_pinned_file(scored_spec))
        scored = {r.get("simulation_id", r.get("job_id")): r for r in scored_rows}
        if len(scored) != len(scored_rows) or set(scored) != {r["simulation_id"] for r in rows}:
            raise ProvenanceError("Scored-bank identities differ from actual native inventory")
        for row in rows:
            extra = scored[row["simulation_id"]]
            for key in ("source_output_sha256", "config_sha256", "variant", "account_class", "signal_sha256"):
                if extra.get(key) and extra[key] != row.get(key):
                    raise ProvenanceError("Scored-bank provenance differs: " + key)
            # A scored status must not replace the native execution disposition.
            row.update({key: value for key, value in extra.items() if key not in row})
            if "status" in extra:
                row["score_status"] = extra["status"]
    root = inputs.get("bank_root", specs.get("bank_root"))
    if root is None:
        roots = {str(Path(r["source_output_path"]).absolute().parent.parent) for r in rows if r.get("source_output_path")}
        if len(roots) != 1:
            raise ProvenanceError("Inventory sources do not resolve to one canonical bank")
        root = next(iter(roots))
    root = Path(root).resolve()
    required = set(inputs.get("required_simulation_ids", []))
    if required - {r["simulation_id"] for r in rows}:
        raise ProvenanceError("Required native simulations are absent from inventory")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        existing = read_cache(output_dir, expected_inventory_sha256=inventory_spec["sha256"])
        if existing["receipt"].get("scored_bank_sha256") != (scored_spec.get("sha256") if scored_spec else None) or existing["receipt"].get("bank_receipt_sha256") != specs.get("bank_receipt", {}).get("sha256") or existing["receipt"].get("bank_root") != str(root):
            raise ProvenanceError("Existing cache does not bind these scored-bank/parent identities")
        if required - set(existing["traces"]):
            raise ProvenanceError("Existing cache lacks a required native simulation")
        return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    traces, dispositions = {}, []
    for index, row in enumerate(sorted(rows, key=lambda r: r["simulation_id"])):
        disposition = dict(row)
        if row.get("status") != "finite":
            disposition.update(population_status="retained_native_failure", population_error=row.get("error", ""))
        else:
            try:
                trace = load_native(row, root)
                traces[row["simulation_id"]] = trace
                disposition.update(population_status="qualified", population_error="", cache_index=len(traces)-1)
            except ProvenanceError:
                raise
            except (ValueError, UnicodeError) as exc:
                disposition.update(population_status="excluded_numeric_or_schema", population_error=str(exc))
        dispositions.append(disposition)
        if (index + 1) % 32 == 0:
            print(json.dumps({"stage": "native_population_qualification", "done": index + 1, "total": len(rows), "qualified": len(traces)}), flush=True)
    missing = required - set(traces)
    # Preserve exclusions even when qualification cannot proceed to an array cache.
    # This file is written once, before the fatal required-member check.
    _write_dispositions(output_dir, dispositions)
    diagnostics = {"schema": SCHEMA, "utc": datetime.now(timezone.utc).isoformat(),
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "seed": int(seed),
                   "inventory_path": str(inventory_path), "inventory_sha256": inventory_spec["sha256"],
                   "bank_root": str(root), "inventoried_rows": len(rows), "qualified_rows": len(traces),
                   "dispositions": dict(Counter(r["population_status"] for r in dispositions)),
                   "rejection_representatives": _error_representatives(dispositions),
                   **CLOCK_METADATA}
    if missing or not traces:
        _write_json(output_dir / "receipt.json", {**diagnostics,
                    "status": "native_population_cache_rejected",
                    "missing_required_simulation_ids": sorted(missing),
                    "files": {"population_dispositions.csv": {
                        "sha256": sha256(output_dir / "population_dispositions.csv"),
                        "bytes": (output_dir / "population_dispositions.csv").stat().st_size}}})
        raise ValueError("Required native outputs did not qualify: " + ",".join(sorted(missing)[:20])
                         + (f" ({len(missing)} missing in total)" if len(missing) > 20 else ""))
    ids = list(traces)
    values = np.stack([traces[s]["values"] for s in ids])
    noise = np.stack([traces[s]["noise"] for s in ids])
    stimulus = np.stack([traces[s]["stimulus"] for s in ids])
    with (output_dir / "population_cache.npz").open("xb") as stream:
        np.savez_compressed(stream, values=values, noise=noise, stimulus=stimulus,
                            times=traces[ids[0]]["times"], simulation_ids=np.asarray(ids, dtype="U160"))
    metadata = [{k: v for k, v in traces[s].items() if k in ("row", "config", "config_text", "stimulus_observed", "native_columns", "native_column_count", "raw_bytes", "source_output_sha256", "config_sha256")} for s in ids]
    _write_json(output_dir / "population_metadata.json", {"schema": SCHEMA, "columns": COLUMNS, "traces": metadata})
    receipt = {**diagnostics, "status": "native_population_cache_qualified", "utc": datetime.now(timezone.utc).isoformat(),
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "seed": int(seed), "inventory_path": str(inventory_path),
               "inventory_sha256": inventory_spec["sha256"], "bank_root": str(root),
               "scored_bank_sha256": scored_spec.get("sha256") if scored_spec else None,
               "bank_receipt_sha256": specs.get("bank_receipt", {}).get("sha256"),
               "inventoried_rows": len(rows), "qualified_rows": len(traces),
               "dispositions": dict(Counter(r["population_status"] for r in dispositions)),
               "variant_counts": dict(Counter(traces[s]["row"]["variant"] for s in ids)),
               "neural_population_ids": list(range(1, 10)),
               "input_population_ids": [10, 11], "spatial_nodes": 1, "columns": COLUMNS,
               "all_qualified_source_and_configuration_hashes_verified": True,
               "original_failures_and_numeric_exclusions_retained": True,
               "source_builds_bound_by_parent_receipt": parent is not None,
               "interpretation": "Observed native population trajectories only; no anatomical EEG forward map or human causal identification",
               "files": {name: {"sha256": sha256(output_dir / name), "bytes": (output_dir / name).stat().st_size}
                         for name in ("population_cache.npz", "population_metadata.json", "population_dispositions.csv")}}
    if sha256(inventory_path) != inventory_spec["sha256"] or (scored_spec and sha256(scored_spec["path"]) != scored_spec["sha256"]):
        raise ProvenanceError("Native inventory or score table changed during qualification")
    _write_json(output_dir / "receipt.json", receipt)
    return {"traces": traces, "rows": dispositions, "receipt": receipt, "cache_dir": str(output_dir)}


def read_cache(cache_dir, expected_inventory_sha256=None):
    if not re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")):
        raise RuntimeError("Full native cache reads require a scheduled Slurm worker")
    cache_dir = Path(cache_dir)
    receipt = json.loads((cache_dir / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "native_population_cache_qualified":
        raise ProvenanceError("Unqualified native population cache")
    if any(receipt.get(key) != value for key, value in CLOCK_METADATA.items()):
        raise ProvenanceError("Native population cache clock metadata differs")
    if expected_inventory_sha256 is not None and receipt.get("inventory_sha256") != expected_inventory_sha256:
        raise ProvenanceError("Existing native cache belongs to a different inventory")
    for name in ("population_cache.npz", "population_metadata.json", "population_dispositions.csv"):
        expected = receipt.get("files", {}).get(name, {})
        path = _safe_file(cache_dir / name, 2 * 1024 ** 3, cache_dir)
        if path.stat().st_size != expected.get("bytes") or sha256(path) != expected.get("sha256"):
            raise ProvenanceError("Native cache aggregate changed: " + name)
    metadata = json.loads((cache_dir / "population_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA or metadata.get("columns") != COLUMNS:
        raise ProvenanceError("Native cache column identity differs")
    traces = {}
    with np.load(cache_dir / "population_cache.npz", allow_pickle=False) as arrays:
        ids, values, times = arrays["simulation_ids"].tolist(), arrays["values"], arrays["times"]
        noise, stimulus = arrays["noise"], arrays["stimulus"]
    if len(ids) != receipt.get("qualified_rows") or len(set(ids)) != len(ids) or len(metadata["traces"]) != len(ids):
        raise ProvenanceError("Native cache identities differ from receipt")
    if values.shape != (len(ids), N_ROWS, 18) or noise.shape != (len(ids), N_ROWS) or stimulus.shape != noise.shape:
        raise ProvenanceError("Native cache shape differs")
    if not all(np.isfinite(x).all() for x in (values, times, noise, stimulus)) or not _actual_clock(times):
        raise ProvenanceError("Native cache contains invalid numeric observations")
    for i, simulation_id in enumerate(ids):
        trace = dict(metadata["traces"][i])
        if trace["row"]["simulation_id"] != simulation_id:
            raise ProvenanceError("Native cache metadata order differs")
        trace.update(times=times, values=values[i], columns=list(COLUMNS), noise=noise[i], stimulus=stimulus[i])
        traces[simulation_id] = _aliases(trace)
    return {"traces": traces, "rows": _csv_rows(cache_dir / "population_dispositions.csv"),
            "receipt": receipt, "cache_dir": str(cache_dir)}


load_cache = read_cache


def run(inputs, output, seed=20261024):
    return load_bank(inputs, output, seed)["receipt"]
