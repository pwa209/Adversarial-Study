"""Diagnostic tests for same-unit pathway cancellation and its provenance."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if not (HERE / "native_population.py").is_file():
    sys.path.insert(1, str(HERE.parents[1] / "full_population_clock_repair_20261024" / "release"))
import native_population as native
import pathway_balance as p


def topology():
    edges = []
    for edge in range(41):
        edges.append({"connection_id": edge + 1, "source_population": edge % 11 + 1,
                      "destination_population": edge % 9 + 1, "external_input": edge % 11 >= 9})
    return edges


def summed_voltage(d, connections):
    return np.column_stack([d[:, p.incoming(connections, pop)].sum(axis=1) for pop in range(1, 10)])


def raw_fixture():
    columns = ["Time", *native.COLUMNS, "Pop.10.Q", "Pop.11.Q", *p.DENDRITE_COLUMNS]
    t = np.arange(1, 5001) * .002
    data = np.column_stack([t] + [np.sin(t * (j + 1)) * .001 for j in range(len(columns) - 1)])
    text = io.StringIO()
    np.savetxt(text, data, fmt="%.17g")
    raw = ("SYNTHETIC FIXTURE\n" + native.SEPARATOR + "\n" + " ".join(columns)
           + "\n" + " ".join(["1"] * (len(columns) - 1)) + "\n" + text.getvalue()).encode()
    trace = {"native_columns": columns, "times": data[:, 0], "values": data[:, 1:19],
             "noise": data[:, 19], "stimulus": data[:, 20], "stimulus_observed": True}
    return raw, trace, data[:, 21:]


def source_fixture(root):
    """Minimal synthetic compiled-source law; never a published-source substitute."""
    files = {
        "population.cpp": "return firing_response->V();",
        "firing_response.cpp": "v[i] = 0; v[j] += dendrites[i]->V()[j];",
        "firing_response.h": "FiringResponse::V() const {return v;}",
        "dendrite.cpp": 'output("Dendrite",index+1,"V",v); (*de)[2][i] = precouple[i]; v[i] = (*de)[0][i];',
        "dendrite.h": "Dendrite::V() const {return v;}",
        "dendrite_scaled.cpp": 'output("Dendrite",index+1,"V",v); h_fac*y[2]; void DendriteScaled::step() {}',
        "dumpf.cpp": "file.precision(14); file<<scientific;"}
    variants = []
    for name in ("pristine", "scaled_dendrite_v1_explicit_patch"):
        path = root / name / "src"
        path.mkdir(parents=True)
        for key, text in files.items():
            (path / key).write_text(text, encoding="utf-8")
        variants.append({"variant": name, "executable": str(path.parent / "nftsim"),
                         "source_hashes": {key: p.sha(path / key) for key in files}})
    receipt = root / "bank_receipt.json"
    receipt.write_text(json.dumps({"status": "native_simulation_bank_completed", "variants": variants}), encoding="utf-8")
    return {"path": str(receipt), "sha256": p.sha(receipt)}


def comparison_fixture(sid="a", supported=True):
    return {"simulation_id": sid, "variant": "pristine", "account_class": "published",
        "human_training_validation_fit_supported": supported, "input_amplitude": 1.,
        "input_onset_absolute_s": 12., "input_width_s": .02, "noise_stream_sha256": "f" * 64,
        "control_row": {"score_status": "ok", "score": .3, "feature_acf_100ms": .2,
            "feature_acf_500ms": .1, "feature_spectral_centroid_hz": 10., "feature_spectral_entropy": .5,
            "feature_hjorth_mobility": 2., "feature_std_amplitude": .01},
        "summary": {"all_population_sums_qualified": True, "cortical_cancellation": .4,
            "mean_population_cancellation": .5, "total_pathway_absolute_mass_V_s": .1},
        "pathway_signature": np.ones((100, 41)), "population_signature": np.ones((100, 9))}


class PathwayBalanceTests(unittest.TestCase):
    def test_opposing_channels_have_complete_cancellation(self):
        t = np.linspace(2, 3, 101)
        d = np.column_stack([np.ones(len(t)), -np.ones(len(t))])
        result = p.balance_metrics(d, t)
        self.assertAlmostEqual(result["relative_voltage_cancellation"], 1.)
        self.assertAlmostEqual(result["absolute_pathway_mass_V_s"], 2.)
        self.assertAlmostEqual(result["net_pathway_mass_V_s"], 0.)
        self.assertAlmostEqual(result["signed_cross_energy_V2_s"], -2.)

    def test_same_sign_channels_have_no_cancellation(self):
        t = np.linspace(0, 1, 101)
        result = p.balance_metrics(np.ones((len(t), 3)), t)
        self.assertEqual(result["relative_voltage_cancellation"], 0.)
        self.assertAlmostEqual(result["signed_cross_energy_V2_s"], 6.)

    def test_never_divides_by_baseline_standard_deviation(self):
        t = np.arange(100) * .002
        d = np.column_stack([np.full(100, 1e-7), np.full(100, -5e-8)])
        result = p.balance_metrics(d, t)
        self.assertAlmostEqual(result["relative_voltage_cancellation"], 2 / 3)
        self.assertLess(result["absolute_pathway_mass_V_s"], 1e-6)

    def test_common_raw_scale_changes_physical_mass_not_relative_cancellation(self):
        t = np.linspace(0, 1, 100)
        d = np.column_stack([np.sin(t), -.8 * np.sin(t)])
        a, b = [p.balance_metrics(scale * d, t) for scale in (1, 100)]
        self.assertAlmostEqual(a["relative_voltage_cancellation"], b["relative_voltage_cancellation"])
        self.assertAlmostEqual(b["absolute_pathway_mass_V_s"], 100 * a["absolute_pathway_mass_V_s"])

    def test_zero_response_preserves_undefined_relative_metric(self):
        result = p.balance_metrics(np.zeros((10, 2)), np.arange(10) * .002)
        self.assertIsNone(result["relative_voltage_cancellation"])
        self.assertEqual(result["denominator_status"], "zero_response")

    def test_nonfinite_and_overflow_fail(self):
        for d in (np.full((10, 2), np.nan), np.full((10, 2), 1e200)):
            with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(ValueError):
                p.balance_metrics(d, np.arange(10) * .002)

    def test_population_sum_is_reconstructed_without_coupling_reweighting(self):
        rng = np.random.default_rng(42)
        d = rng.normal(size=(100, 41))
        connections = topology()
        v = summed_voltage(d, connections)
        self.assertTrue(all(r["voltage_sum_qualified"] for r in p.verify_voltage_sum(d, v, connections)))

    def test_numpy_computations_return_strict_json_safe_sum_proofs_and_summary(self):
        d = np.ones((100, 41), dtype=np.float64) * .01
        connections = topology()
        v = summed_voltage(d, connections)
        proof = p.verify_voltage_sum(d, v, connections)
        self.assertTrue(all(type(r["voltage_sum_qualified"]) is bool for r in proof))
        self.assertTrue(all(type(r["fixed_rounding_tolerance_V"]) is float for r in proof))
        rows = p.pathway_rows(d, v, np.arange(100) * .002, connections,
                              {"configuration_cluster": "a"}, "paired_finite_pulse_response")
        summary = p.configuration_summary(rows)
        interval = p.cluster_bootstrap(rows, "cancellation_for_inference", 1)
        value = {"population_sum_proof": proof, "rows": rows, "summary": summary, "bootstrap": interval,
                 "passed_population_count": sum(r["voltage_sum_qualified"] for r in proof)}
        encoded = json.dumps(value, allow_nan=False)
        self.assertEqual(json.loads(encoded)["passed_population_count"], 9)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            target = Path(directory) / "strict_result.json"
            p.write_json(target, value)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["summary"], summary)

    def test_falsified_population_sum_is_rejected_even_if_cancellation_is_high(self):
        d = np.ones((100, 41)) * .01
        connections = topology()
        v = summed_voltage(d, connections)
        v[:, 4] += .001
        proof = p.verify_voltage_sum(d, v, connections)
        self.assertFalse(proof[4]["voltage_sum_qualified"])
        self.assertEqual(sum(r["voltage_sum_qualified"] for r in proof), 8)

    def test_source_precision_tolerance_allows_print_rounding_but_rejects_small_real_mismatch(self):
        d = np.ones((100, 41)) * .01
        connections = topology()
        v = summed_voltage(d, connections)
        v[:, 0] += 1e-16
        self.assertTrue(p.verify_voltage_sum(d, v, connections)[0]["voltage_sum_qualified"])
        v[:, 0] += 1e-10
        self.assertFalse(p.verify_voltage_sum(d, v, connections)[0]["voltage_sum_qualified"])

    def test_pulse_subtraction_uses_original_tonic_scale_for_rounding(self):
        d = np.ones((100, 41)) * 1e-15
        connections = topology()
        v = summed_voltage(d, connections)
        v[:, 0] += 1e-17
        self.assertFalse(p.verify_voltage_sum(d, v, connections)[0]["voltage_sum_qualified"])
        self.assertTrue(p.verify_voltage_sum(d, v, connections, rounding_reference=[.01] * 9)[0]["voltage_sum_qualified"])

    def test_centered_fluctuation_handles_constant_reference_offset_only(self):
        t = np.linspace(0, 1, 100)
        d = np.sin(t[:, None] * np.arange(1, 42)[None, :]) * .01
        connections = topology()
        v = summed_voltage(d, connections) + .005
        self.assertFalse(all(r["voltage_sum_qualified"] for r in p.verify_voltage_sum(d, v, connections)))
        self.assertTrue(all(r["voltage_sum_qualified"] for r in p.verify_voltage_sum(d, v, connections, centered=True)))

    def test_full_clock_and41_actual_columns_parse(self):
        raw, trace, expected = raw_fixture()
        result = p.parse_pathways(raw, trace)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(trace["times"][0], .002)
        self.assertEqual(trace["times"][-1], 10.)

    def test_wrong_clock_is_not_rebased(self):
        raw, trace, _ = raw_fixture()
        trace["times"] = trace["times"] - .002
        with self.assertRaisesRegex(ValueError, "clock"):
            p.parse_pathways(raw, trace)

    def test_cache_raw_vq_mismatch_is_fatal(self):
        raw, trace, _ = raw_fixture()
        trace["values"] = trace["values"].copy()
        trace["values"][0, 0] += .01
        with self.assertRaisesRegex(ValueError, "V/Q"):
            p.parse_pathways(raw, trace)

    def test_missing_pathway_column_fails(self):
        raw, trace, _ = raw_fixture()
        raw = raw.replace(b"Dendrite.41.V", b"Unknown.41.V")
        trace["native_columns"][-1] = "Unknown.41.V"
        with self.assertRaisesRegex(ValueError, "All41"):
            p.parse_pathways(raw, trace)

    def test_provenance_pin_rejects_mutation(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "metadata.json"
            path.write_text("{}", encoding="utf-8")
            spec = {"path": str(path), "sha256": p.sha(path)}
            self.assertEqual(p.pinned(spec), path)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                p.pinned(spec)

    def test_actual_source_equation_gate_requires_bank_source_hashes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            spec = source_fixture(Path(directory))
            proof = p.source_equation_proof(spec)
            self.assertEqual(len(proof["variants"]), 2)
            self.assertFalse(proof["variants"][0]["coupling_weight_applied_again"])
            source = Path(directory) / "pristine" / "src" / "dendrite.cpp"
            source.write_text("mutated source", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source pin"):
                p.source_equation_proof(spec)

    def test_source_without_soma_sum_law_is_rejected_even_when_pinned(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            spec = source_fixture(Path(directory))
            source = Path(directory) / "pristine" / "src" / "firing_response.cpp"
            source.write_text("v[i] = 0; v[j] += 2 * dendrites[i]->V()[j];", encoding="utf-8")
            receipt = json.loads(Path(spec["path"]).read_text(encoding="utf-8"))
            receipt["variants"][0]["source_hashes"][source.name] = p.sha(source)
            Path(spec["path"]).write_text(json.dumps(receipt), encoding="utf-8")
            spec["sha256"] = p.sha(spec["path"])
            with self.assertRaisesRegex(ValueError, "voltage equation"):
                p.source_equation_proof(spec)

    def test_source_formatter_precision_is_rechecked_and_not_assumed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            spec = source_fixture(Path(directory))
            source = Path(directory) / "pristine" / "src" / "dumpf.cpp"
            source.write_text("file.precision(6); file<<scientific;", encoding="utf-8")
            receipt = json.loads(Path(spec["path"]).read_text(encoding="utf-8"))
            receipt["variants"][0]["source_hashes"][source.name] = p.sha(source)
            Path(spec["path"]).write_text(json.dumps(receipt), encoding="utf-8")
            spec["sha256"] = p.sha(spec["path"])
            with self.assertRaisesRegex(ValueError, "voltage equation"):
                p.source_equation_proof(spec)

    def test_bootstrap_resamples_configuration_clusters_not_duplicate_rows(self):
        a = [{"configuration_cluster": "a", "x": 0.}, {"configuration_cluster": "b", "x": 1.}]
        b = a + [a[0]] * 100
        one, two = [p.cluster_bootstrap(rows, "x", 3, 500) for rows in (a, b)]
        self.assertEqual(one, two)
        self.assertEqual(one["n_configuration_clusters"], 2)
        self.assertEqual(one["estimate"], .5)
        self.assertNotIn("p_value", one)

    def test_single_configuration_has_no_bootstrap_interval(self):
        r = p.cluster_bootstrap([{"configuration_cluster": "a", "x": .4}] * 100, "x", 1)
        self.assertEqual(r["n_configuration_clusters"], 1)
        self.assertIsNone(r["ci95_lower"])

    def test_bootstrap_seed_reproducible(self):
        rows = [{"configuration_cluster": str(i), "x": i / 7} for i in range(8)]
        self.assertEqual(p.cluster_bootstrap(rows, "x", 2026), p.cluster_bootstrap(rows, "x", 2026))

    def test_energy_latency_uses_actual_output_clock(self):
        t = np.arange(1000, 1501) * .002
        signal = np.exp(-(t - 2) / .05)
        result = p.energy_timing(signal, t, 2.)
        self.assertEqual(result["peak_latency_s"], 0.)
        self.assertTrue(0 < result["energy_t10_s"] < result["energy_t50_s"] < result["energy_t90_s"])

    def test_zero_timing_is_not_false_immediate_recovery(self):
        result = p.energy_timing(np.zeros(5), np.arange(5) * .002 + 2, 2.)
        self.assertIsNone(result["energy_t50_s"])
        self.assertEqual(result["timing_status"], "zero_response")

    def test_failed_sum_excluded_from_configuration_inference(self):
        d = np.ones((100, 41)) * .01
        connections = topology()
        v = summed_voltage(d, connections) + 100
        rows = p.pathway_rows(d, v, np.arange(100) * .002, connections,
                              {"configuration_cluster": "a"}, "paired_finite_pulse_response")
        result = p.configuration_summary(rows)
        self.assertFalse(result["all_population_sums_qualified"])
        self.assertIsNone(result["mean_population_cancellation"])

    def test_below_fixed_physical_floor_does_not_become_large_inference(self):
        d = np.ones((100, 41)) * 1e-16
        connections = topology()
        rows = p.pathway_rows(d, summed_voltage(d, connections), np.arange(100) * .002,
                              connections, {"configuration_cluster": "a"}, "paired_finite_pulse_response")
        self.assertTrue(all(r["cancellation_for_inference"] is None for r in rows))

    def test_strict_matching_does_not_relax_to_create_support(self):
        a, b = comparison_fixture("a"), comparison_fixture("b", False)
        rows, reasons = p.compare_records([a, b], kind="paired")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["both_human_training_validation_fit_supported"] is False for r in rows))
        b["control_row"]["feature_spectral_entropy"] = .9
        rows, reasons = p.compare_records([a, b], kind="paired")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["matching"], "score")
        self.assertEqual(reasons["score_and_conventional_features:feature_spectral_entropy_outside_tolerance"], 1)

    def test_different_background_noise_cannot_be_matched(self):
        a, b = comparison_fixture("a"), comparison_fixture("b")
        b["noise_stream_sha256"] = "a" * 64
        rows, reasons = p.compare_records([a, b], kind="paired")
        self.assertFalse(rows)
        self.assertEqual(sum(reasons.values()), 2)

    def test_hidden_and_cortical_distance_use_one_raw_scale(self):
        a, b = comparison_fixture("a"), comparison_fixture("b")
        a["population_signature"][:] = 0.
        b["population_signature"][:] = 0.
        b["population_signature"][:, :1] = 1e-15
        b["population_signature"][:, 1:] = .1
        rows, _ = p.compare_records([a, b], kind="paired")
        self.assertAlmostEqual(rows[0]["hidden_minus_cortical_raw_response_distance"], .1, places=12)
        self.assertEqual(rows[0]["common_raw_voltage_scale_V"], 1.)


if __name__ == "__main__":
    unittest.main()
