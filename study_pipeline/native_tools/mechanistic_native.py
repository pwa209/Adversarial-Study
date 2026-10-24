"""Scheduled, source-pinned NFTsim banks and explicit mechanistic interventions.

Published source is never modified. A separate named implementation restores the
scaled dendrite integrator and its initialized time constants; it is compared
against pristine execution. Population voltages are model-space proxies, not a
qualified scalp EEG forward map. Clinical inference is a separate batch stage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np

from .figshare_source import BASELINE, DOI, sha256_file
from .nftsim_runtime import (FLOAT, _execute, _mask_comments, compare_outputs,
                             parse_output, require_scheduled_job, verify_release)

PATCH_VARIANT = "scaled_dendrite_v1_explicit_patch"
PATCH_PINS = {
    "dendrite_scaled.h": "43dd9f92ec937e87e05269c31b51a665c08f0597daf17b7505c070bce1a2fde8",
    "dendrite_scaled.cpp": "c9b49d940b95ca8dfea783e9d5b7001e23f779d8980587ba8d77583955fece0b",
}
TEMPORAL_FEATURES = ("feature_acf_100ms", "feature_acf_500ms", "feature_spectral_centroid_hz",
                     "feature_spectral_entropy", "feature_hjorth_mobility", "feature_std_amplitude")
JOBS_FIELDS = ("job_id", "episode_id", "channel_name", "participant_id", "source_id",
               "input_path", "sfreq_hz", "signal_sha256")
INVENTORY_FIELDS = (
    "simulation_id", "account_class", "variant", "published_config_id", "config_sha256",
    "status", "error", "pair_id", "param_gain_factor", "param_coupling_factor",
    "param_history_factor", "input_amplitude", "input_onset_s", "input_width_s",
    "source_output_path", "source_output_sha256", "signal_path", "signal_sha256",
    "seconds", *TEMPORAL_FEATURES, "observed_input_peak", "response_normalized_peak",
    "response_normalized_integral", "response_recovery_s", "response_recovery_status",
)


def _json(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def patch_scaled_sources(header: bytes, implementation: bytes) -> tuple[bytes, bytes, str]:
    """Apply only the pinned, documented derived-integrator repair.

    The base class initializes its own alpha/beta, whereas the derived class
    redeclares zero-valued copies. Inherited step() advances the base integrator,
    leaving the derived hsyn equation unused. The named patch copies initialized
    constants and overrides step(), retaining every population/coupling equation.
    A stable equal-rate limit prevents the optional alpha==beta singularity.
    """
    if hashlib.sha256(header).hexdigest() != PATCH_PINS["dendrite_scaled.h"] or hashlib.sha256(implementation).hexdigest() != PATCH_PINS["dendrite_scaled.cpp"]:
        raise ValueError("Scaled-dendrite repair is defined only for exact pinned v2 source")
    h, cpp = header.decode("utf-8"), implementation.decode("utf-8")
    h_new = h.replace("//void step() override;", "void step() override;")
    if h_new == h:
        raise ValueError("Pinned step declaration anchor absent")
    cpp_new = cpp.replace("Dendrite::init(configf);", "Dendrite::init(configf);\n    // Named v1 repair: use the time constants initialized by the base class.\n    alpha = Dendrite::alpha;\n    beta = Dendrite::beta;", 1)
    original_peak = "tpk = log(beta/alpha)/(beta - alpha);\n    eta = (alpha*beta)*(exp(-alpha*tpk) - exp(-beta*tpk))/(beta - alpha);"
    stable_peak = """// Explicit v1 implementation; equal-rate limit of the same response kernel.
    if (std::abs(beta-alpha) <= 1e-12 * std::max(alpha,beta)) {
        tpk = 1.0/alpha;
        eta = alpha/std::exp(1.0);
    } else {
        tpk = std::log(beta/alpha)/(beta-alpha);
        eta = alpha*beta*std::exp(-alpha*tpk)*(-std::expm1(-(beta-alpha)*tpk))/(beta-alpha);
    }"""
    if original_peak not in cpp_new:
        raise ValueError("Pinned kernel initialization anchor absent")
    cpp_new = cpp_new.replace(original_peak, stable_peak, 1)
    cpp_new = cpp_new.replace('#include <cmath>', '#include <algorithm> // std::max; named v1 repair\n#include <cmath>', 1)
    cpp_new += """

// Named v1 repair: integrate the derived, gain-scaled DE and expose its voltage.
void DendriteScaled::step() {
    for (size_type i=0; i<nodes; ++i) {
        (*de)[2][i] = precouple[i];
    }
    rk4->step();
    for (size_type i=0; i<nodes; ++i) {
        v[i] = (*de)[0][i];
    }
}
"""
    diff = "\n".join(difflib.unified_diff(h.splitlines(), h_new.splitlines(), fromfile="pristine/dendrite_scaled.h", tofile=PATCH_VARIANT + "/dendrite_scaled.h")) + "\n"
    diff += "\n".join(difflib.unified_diff(cpp.splitlines(), cpp_new.splitlines(), fromfile="pristine/dendrite_scaled.cpp", tofile=PATCH_VARIANT + "/dendrite_scaled.cpp")) + "\n"
    return h_new.encode("utf-8"), cpp_new.encode("utf-8"), diff


def set_numeric(text: str, section: str, index: int, field: str, value: float) -> tuple[str, dict]:
    if not math.isfinite(value):
        raise ValueError("Configuration replacement must be finite")
    masked = _mask_comments(text)
    if section == "Stimulus":
        pattern = re.compile(r"^\s*Population " + str(index) + r":[^\n]*\n(?:(?!\n\s*Population).)*?\b" + re.escape(field) + r":\s*(" + FLOAT + r")", re.MULTILINE | re.DOTALL)
    else:
        # Numeric field may lie on a following line (published Coupling nu).
        pattern = re.compile(r"^\s*" + re.escape(section) + " " + str(index) + r":[^\n]*?\b" + re.escape(field) + r":\s*(" + FLOAT + r")", re.MULTILINE)
    matches = list(pattern.finditer(masked))
    if len(matches) != 1:
        raise ValueError(f"Parameter must identify exactly one actual {section} {index}.{field}")
    hit = matches[0]
    before = float(hit[1])
    return text[:hit.start(1)] + format(value, ".17g") + text[hit.end(1):], {"parameter": f"{section} {index}.{field}", "before": before, "after": value}


def scale_numeric(text: str, section: str, index: int, field: str, factor: float) -> tuple[str, dict]:
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Mechanistic factors must be finite positive")
    _, probe = set_numeric(text, section, index, field, 1.)
    return set_numeric(text, section, index, field, probe["before"] * factor)


def candidate_config(baseline: str, *, gain: float = 1., coupling: float = 1., history: float = 1., amplitude: float = 0.) -> tuple[str, list[dict]]:
    """Bounded cortical input gain, recurrent coupling and synaptic memory families.

    History scales alpha/beta together: lower factors lengthen the response;
    delays and integration step remain unchanged. Input is the already published
    Perturb population and its existing all-population couplings.
    """
    changes, text = [], baseline
    for index in (1, 2, 3):
        if gain != 1:
            text, change = scale_numeric(text, "Dendrite", index, "hsyn", gain)
            changes.append(change)
        if history != 1:
            for field in ("alpha", "beta"):
                text, change = scale_numeric(text, "Dendrite", index, field, history)
                changes.append(change)
    if coupling != 1:
        for index in (1, 2, 3, 5, 6, 7, 10, 11, 14, 15):
            text, change = scale_numeric(text, "Coupling", index, "nu", coupling)
            changes.append(change)
    for field, value in (("Onset", 12.), ("Amplitude", amplitude), ("Width", .02)):
        text, change = set_numeric(text, "Stimulus", 11, field, value)
        changes.append(change)
    # Pulse population must actually be output; it is absent from the original
    # output list. Preserve all nine endogenous populations and stimulation.
    text, count = re.subn(r"(^Population:\s*1 2 3 4 5 6 7 8 9 10)\s*$", r"\1 11", text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("Published output populations no longer match pinned model")
    return text, changes


def design_candidates(budget: int, seed: int = 20261004) -> list[dict]:
    if not 0 <= budget <= 128:
        raise ValueError("Search budget must lie in [0,128]")
    rows = []
    for family in ("gain", "coupling", "history"):
        for value in (.5, .8, 1.25, 2.):
            row = {"account_class": family, "gain": 1., "coupling": 1., "history": 1.}
            row[family] = value
            rows.append(row)
    rng = np.random.default_rng(seed)
    count = max(0, budget - len(rows))
    if count:
        # Deterministic Latin hypercube in positive log factors; no score/outcome
        # enters proposal generation. Continuous local refinements happen later.
        coordinates = np.column_stack([(rng.permutation(count) + rng.uniform(size=count)) / count for _ in range(3)])
        for factors in np.exp(np.log(.5) + coordinates * np.log(4.)):
            rows.append(dict(zip(("gain", "coupling", "history"), factors.tolist())) | {"account_class": "combined"})
    return rows[:budget]


def output_signal(parsed: dict, column: str = "Pop.1.V") -> np.ndarray:
    if parsed["columns"].count(column) != 1:
        raise ValueError("Native cortical voltage must resolve exactly one output/node: " + column)
    return np.asarray(parsed["values"][:, parsed["columns"].index(column)], dtype=float)


def pulse_response(control: dict, pulse: dict, *, onset_relative_s: float = 2., width_s: float = .02) -> dict:
    if control["columns"] != pulse["columns"] or control["values"].shape != pulse["values"].shape or not np.array_equal(control["values"][:, 0], pulse["values"][:, 0]):
        raise ValueError("Input response requires identical native time grids and population outputs")
    time_grid = pulse["values"][:, 0]
    a, b = output_signal(control), output_signal(pulse)
    stimulus = output_signal(pulse, "Pop.11.Q")
    before = time_grid < onset_relative_s - .01
    active = (time_grid >= onset_relative_s - .0021) & (time_grid <= onset_relative_s + width_s + .0021)
    after = time_grid >= onset_relative_s
    if not before.any() or not after.any():
        raise ValueError("Input onset has no baseline or post-input samples")
    # Matched default seeds are a runtime claim verified here before attributing
    # any difference to input; uncontrolled noise cannot become a pulse response.
    tolerance = max(np.max(np.abs(a[before])), 1.) * 1e-10
    if not np.allclose(a[before], b[before], rtol=0, atol=tolerance):
        raise ValueError("Paired trajectories already differ before the intervention")
    if np.max(np.abs(stimulus[active])) <= 0:
        raise ValueError("Configured native input was not observed on the output sampling grid")
    delta = np.abs(b-a)
    peak = float(np.max(delta[after]))
    scale = max(float(np.std(a[before], ddof=1)), np.finfo(float).eps)
    result = {"observed_input_peak": float(np.max(np.abs(stimulus[active]))),
              "response_normalized_peak": peak/scale,
              "response_normalized_integral": float(np.trapezoid(delta[after], time_grid[after]))/scale,
              "response_recovery_s": "", "response_recovery_status": "right_censored_at_end"}
    if peak <= np.finfo(float).eps:
        result["response_recovery_status"] = "no_detectable_response"
        return result
    peak_index = int(np.flatnonzero(after)[np.argmax(delta[after])])
    hold_samples = max(1, int(round(.25/pulse["interval"])))
    threshold = peak * .05
    for index in range(peak_index + 1, len(delta) - hold_samples + 1):
        if np.all(delta[index:index + hold_samples] <= threshold):
            result["response_recovery_s"] = float(time_grid[index] - onset_relative_s)
            result["response_recovery_status"] = "returned_below_5pct_peak_for_250ms"
            break
    return result


def _build_variant(release: Path, destination: Path, variant: str, compiler: str, timeout: int) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    source_dir = destination / "src"
    shutil.copytree(release / "src", source_dir)
    patch_record = None
    if variant == PATCH_VARIANT:
        old_h = (source_dir / "dendrite_scaled.h").read_bytes()
        old_cpp = (source_dir / "dendrite_scaled.cpp").read_bytes()
        new_h, new_cpp, diff = patch_scaled_sources(old_h, old_cpp)
        (source_dir / "dendrite_scaled.h").write_bytes(new_h)
        (source_dir / "dendrite_scaled.cpp").write_bytes(new_cpp)
        (destination / "explicit_patch.diff").write_text(diff, encoding="utf-8")
        patch_record = {"variant": variant, "source_pin": PATCH_PINS,
                        "patched_hashes": {"dendrite_scaled.h": hashlib.sha256(new_h).hexdigest(), "dendrite_scaled.cpp": hashlib.sha256(new_cpp).hexdigest()},
                        "diff_sha256": sha256_file(destination / "explicit_patch.diff"),
                        "changes": ["copy base-initialized alpha/beta to derived constants", "advance the derived scaled integrator and expose voltage", "numerically stable equal-rate kernel limit"],
                        "biology": "same published population/coupling equations; repaired numerical implementation is an explicitly separate account"}
        _json(destination / "patch_receipt.json", patch_record)
    executable = destination / "nftsim"
    command = [compiler, "-std=c++11", "-O2", "-Wall", "-Wextra", "-pedantic", "-I", str(source_dir), *sorted(str(p) for p in source_dir.glob("*.cpp")), "-o", str(executable)]
    build = _execute(command, destination / "build.log", cwd=destination, timeout=timeout)
    if build["returncode"] != 0 or not executable.is_file():
        raise ValueError("Native variant did not compile: " + variant)
    return {"variant": variant, "executable": str(executable), "executable_sha256": sha256_file(executable), "build": build,
            "source_hashes": {p.name: sha256_file(p) for p in sorted(source_dir.iterdir()) if p.is_file()}, "patch": patch_record}


def _run_one(spec: dict, executable: Path, directory: Path, timeout: int) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    config = directory / "configuration.conf"
    config.write_text(spec.pop("configuration_text"), encoding="utf-8")
    _json(directory / "configuration_changes.json", {"simulation_id": spec["simulation_id"], "changes": spec.pop("changes")})
    output = directory / "native.output"
    execution = _execute([str(executable), "-i", str(config), "-o", str(output)], directory / "native.log", cwd=directory, timeout=timeout)
    row = spec | {"config_sha256": sha256_file(config), "status": "failed", "error": "", "seconds": execution["seconds"]}
    receipt = row | {"execution": execution, "configuration_path": str(config), "observation_column": "Pop.1.V", "mapping": "model_population_voltage_proxy; scalp_forward_mapping_unqualified"}
    if output.is_file():
        row |= {"source_output_path": str(output), "source_output_sha256": sha256_file(output)}
    try:
        if execution["returncode"] != 0:
            raise ValueError("Native executable did not finish successfully: " + str(execution["returncode"]) + " " + execution["error"])
        if output.stat().st_size > 20_000_000:
            raise ValueError("Native output exceeded the per-candidate 20 MB budget")
        parsed = parse_output(output, expected_rows=5000, expected_interval=.002)
        signal = output_signal(parsed)
        # All population channels remain in the source output, including failed
        # or nonstandardizable trajectories; only eligible exports enter MATLAB.
        with np.errstate(over="ignore", invalid="ignore"):
            signal_std = float(np.std(signal, ddof=1))
        if not math.isfinite(signal_std):
            raise ValueError("Finite native values have numerically unstable variance; export not qualified")
        if signal_std <= np.finfo(float).eps:
            row["status"] = "finite_constant_not_standardizable"
        else:
            from adversarial_study.model_guided import univariate_features
            measurements = univariate_features(signal, 500.)
            if any(not math.isfinite(float(measurements.get(key, float("nan")))) for key in TEMPORAL_FEATURES):
                raise ValueError("Native temporal measurements are nonfinite or missing")
            row.update(measurements)
            signal_csv = directory / "detector_input.csv"
            np.savetxt(signal_csv, signal, delimiter=",", fmt="%.17g")
            row |= {"status": "finite", "signal_path": str(signal_csv), "signal_sha256": sha256_file(signal_csv)}
        receipt["output"] = {key: value for key, value in parsed.items() if key != "values"}
    except Exception as exc:
        row["error"] = type(exc).__name__ + ": " + str(exc)
    receipt.update(row)
    _json(directory / "run_receipt.json", receipt)
    return row


def bank(release_dir: Path, output_dir: Path, *, workers: int = 4, timeout_s: int = 180,
         config_limit: int = 281, search_budget: int = 48, compiler: str = "g++") -> dict:
    job_id = require_scheduled_job()
    if not 1 <= workers <= min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))):
        raise ValueError("Workers exceed the four-worker or actual Slurm allocation budget")
    if not 1 <= timeout_s <= 600 or not 1 <= config_limit <= 281:
        raise ValueError("Candidate timeout/config count outside bounded study design")
    candidates = design_candidates(search_budget)
    release = Path(release_dir).resolve(strict=True)
    manifest = verify_release(release)
    configs = [release / "configuration_files" / BASELINE] + sorted((release / "configuration_files").glob("doc_config*.conf"), key=lambda p: int(re.search(r"\d+", p.stem)[0]))
    if len(configs) != 281 or len({p.name for p in configs}) != 281:
        raise ValueError("Mechanistic bank requires baseline plus all 280 published source-pinned configurations")
    recorded = {r["path"]: r["sha256"] for r in manifest["files"]}
    for path in configs:
        if recorded.get("configuration_files/" + path.name) != sha256_file(path):
            raise ValueError("Unrecorded/modified published configuration: " + path.name)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"status": "native_simulation_bank_not_completed", "slurm_job_id": job_id,
              "release_doi": DOI, "release_dir": str(release), "source_manifest_sha256": sha256_file(release / "SOURCE_MANIFEST.json"),
              "feature_definition": "guided_univariate_v1", "observation_column": "Pop.1.V",
              "scalp_forward_mapping": "unqualified", "human_mechanism": "not_established",
              "parameter_identifiability": "not_established; bank similarity alone cannot identify physiological parameters; hsyn and nu can compensate multiplicatively",
              "workers": workers, "timeout_s": timeout_s, "search_budget": search_budget,
              "design_seed": 20261004, "source_reference_preserved": True,
              "clinical_inference": "separate frozen MATLAB stage; this bank does not assign clinical scores",
              "configuration_families": {"gain": "hsyn for incoming cortical dendrites 1,2,3", "coupling": "signed nu factors for corticothalamic recurrent connections 1,2,3,5,6,7,10,11,14,15", "history": "alpha/beta jointly scaled for cortical incoming dendrites 1,2,3; lower factor extends response kernel", "input": "published Perturb population 11 PulseRect; onset12s, width20ms; amplitude in model units"},
              "resource_limits": {"native_outputs_max_bytes_each": 20_000_000, "candidate_count_max": 824, "expected_source_output_rows_each": 5000, "sample_frequency_hz": 500}}
    try:
        compiler_path = shutil.which(compiler)
        if compiler_path is None:
            raise ValueError("C++ compiler unavailable")
        variants = [_build_variant(release, output / name, name, compiler_path, 600) for name in ("pristine", PATCH_VARIANT)]
        result["variants"] = variants
        native_outputs, rows, specs = [], [], []
        baseline = configs[0].read_text(encoding="utf-8")
        for variant in variants:
            name = variant["variant"]
            for index, path in enumerate(configs[:config_limit]):
                label = f"{name}__published_{index:03d}"
                specs.append({"simulation_id": label, "account_class": "published_reference" if index == 0 else "published_bank", "variant": name,
                    "published_config_id": path.name, "param_gain_factor": 1., "param_coupling_factor": 1., "param_history_factor": 1.,
                    "input_amplitude": 0., "input_onset_s": 9.999, "input_width_s": .001, "pair_id": "", "configuration_text": path.read_text(encoding="utf-8"), "changes": []})
            # Repeat plus the known hsyn/coupling diagnostic in both variants.
            for label, factors in (("repeat", {}), ("gain_diagnostic", {"gain": 2.}), ("coupling_diagnostic", {"coupling": 1.05})):
                text, changes = candidate_config(baseline, **factors)
                specs.append({"simulation_id": name + "__" + label, "account_class": "runtime_diagnostic", "variant": name,
                    "published_config_id": BASELINE, "param_gain_factor": factors.get("gain", 1.), "param_coupling_factor": factors.get("coupling", 1.), "param_history_factor": 1.,
                    "input_amplitude": 0., "input_onset_s": 12., "input_width_s": .02, "pair_id": "", "configuration_text": text, "changes": changes})
        # Derived repair candidates are a named implementation, while the
        # pristine bank provides the source-compatibility comparison.
        for index, proposal in enumerate(candidates):
            factors = {key: proposal[key] for key in ("gain", "coupling", "history")}
            for amplitude in (0., (1., 10., 100.)[index % 3]):
                label = f"{PATCH_VARIANT}__candidate_{index:03d}__{'control' if amplitude == 0 else 'pulse'}"
                text, changes = candidate_config(baseline, amplitude=amplitude, **factors)
                specs.append({"simulation_id": label, "account_class": proposal["account_class"], "variant": PATCH_VARIANT,
                    "published_config_id": BASELINE, "param_gain_factor": factors["gain"], "param_coupling_factor": factors["coupling"], "param_history_factor": factors["history"],
                    "input_amplitude": amplitude, "input_onset_s": 12., "input_width_s": .02, "pair_id": f"candidate_{index:03d}", "configuration_text": text, "changes": changes})
        result["candidate_count"] = len(specs)
        executables = {r["variant"]: Path(r["executable"]) for r in variants}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_one, dict(spec), executables[spec["variant"]], output / "simulations" / spec["simulation_id"], timeout_s) for spec in specs]
            # Resolve in proposal order so the retained table never depends on
            # scheduling speed or numerical success.
            rows = [future.result() for future in futures]
        by_id = {r["simulation_id"]: r for r in rows}
        diagnostics = {}
        for variant in variants:
            name = variant["variant"]
            subset = {}
            for label, sid in (("baseline", name + "__published_000"), ("repeat", name + "__repeat"), ("gain", name + "__gain_diagnostic"), ("coupling", name + "__coupling_diagnostic")):
                row = by_id[sid]
                if row["status"] in {"finite", "finite_constant_not_standardizable"}:
                    subset[label] = parse_output(Path(row["source_output_path"]), expected_rows=5000, expected_interval=.002)
            # Original baseline lacks Pop.11.Q; compare exactly shared outputs.
            if "baseline" in subset:
                columns = subset["baseline"]["columns"]
                for key in subset:
                    parsed = subset[key]
                    indices = [parsed["columns"].index(c) for c in columns]
                    subset[key] = parsed | {"columns": columns, "values": parsed["values"][:, indices]}
            diagnostics[name] = {label: compare_outputs(subset["baseline"], subset[label]) if {"baseline", label} <= set(subset) else {"status": "unavailable_failed_native_run"} for label in ("repeat", "gain", "coupling")}
        for index in range(search_budget):
            prefix = f"{PATCH_VARIANT}__candidate_{index:03d}__"
            control, pulse = by_id[prefix + "control"], by_id[prefix + "pulse"]
            try:
                if control["status"] not in {"finite", "finite_constant_not_standardizable"} or pulse["status"] not in {"finite", "finite_constant_not_standardizable"}:
                    raise ValueError("Paired native run unavailable")
                metrics = pulse_response(parse_output(Path(control["source_output_path"])), parse_output(Path(pulse["source_output_path"])))
                pulse.update(metrics)
            except Exception as exc:
                pulse["response_recovery_status"] = "not_qualified: " + str(exc)
        jobs = []
        for row in rows:
            if row.get("source_output_path"):
                native_outputs.append({"simulation_id": row["simulation_id"], "path": row["source_output_path"], "sha256": row["source_output_sha256"]})
            if row["status"] == "finite":
                jobs.append({"job_id": row["simulation_id"], "episode_id": row["simulation_id"], "channel_name": "Pop.1.V", "participant_id": "nftsim", "source_id": "toker_nftsim_v2",
                             "input_path": row["signal_path"], "sfreq_hz": 500., "signal_sha256": row["signal_sha256"]})
        inventory = output / "simulation_inventory.csv"
        jobs_path = output / "detector_jobs.csv"
        _csv(inventory, rows, INVENTORY_FIELDS)
        _csv(jobs_path, jobs, JOBS_FIELDS)
        result |= {"status": "native_simulation_bank_completed", "inventory_sha256": sha256_file(inventory), "jobs_sha256": sha256_file(jobs_path), "native_outputs": native_outputs,
                   "native_jobs": len(jobs), "retained_outcomes": {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})},
                   "runtime_diagnostics": diagnostics, "interpretation": "Complete means bounded bank execution and retained outcomes; finite output, repaired sensitivity, or detector score does not qualify biological parameters or a human EEG observation map"}
        # Published payload must remain byte-identical after all execution.
        verify_release(release)
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    result["seconds"] = time.monotonic() - started
    _json(output / "bank_receipt.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--config-limit", type=int, default=281)
    parser.add_argument("--search-budget", type=int, default=48)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args(argv)
    result = bank(args.release_dir, args.output_dir, workers=args.workers, timeout_s=args.timeout_s,
                  config_limit=args.config_limit, search_budget=args.search_budget, compiler=args.compiler)
    print(json.dumps({key: result.get(key) for key in ("status", "candidate_count", "native_jobs", "retained_outcomes", "error")}))
    return 0 if result["status"] == "native_simulation_bank_completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
