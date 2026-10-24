"""Source-pinned repair, intervention pairing and honest bank outcome checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from native_tools.mechanistic_native import (PATCH_PINS, PATCH_VARIANT, _csv, _run_one,
    candidate_config, design_candidates, output_signal, patch_scaled_sources, pulse_response,
    set_numeric, JOBS_FIELDS, INVENTORY_FIELDS)


ROOT = Path(__file__).resolve().parents[2]


def native_trace(signal, stimulus=None):
    signal = np.asarray(signal, dtype=float)
    stimulus = np.zeros_like(signal) if stimulus is None else np.asarray(stimulus)
    return {"columns": ["Time", "Pop.1.V", "Pop.11.Q"],
            "values": np.column_stack((np.arange(len(signal))*.002, signal, stimulus)), "interval": .002}


class MechanisticNativeTests(unittest.TestCase):
    def baseline(self):
        return (ROOT / "source_records" / "toker_v2" / "conscious_brain_configuration_file.conf").read_text(encoding="utf-8")

    def test_patch_pinned_source_and_exact_scope(self):
        records = json.loads((ROOT / "study_pipeline" / "native_tools" / "mech_source_excerpt.json").read_text(encoding="utf-8"))
        before_h = records["dendrite_scaled.h"]["source"].encode()
        before_cpp = records["dendrite_scaled.cpp"]["source"].encode()
        self.assertEqual(hashlib.sha256(before_h).hexdigest(), PATCH_PINS["dendrite_scaled.h"])
        after_h, after_cpp, diff = patch_scaled_sources(before_h, before_cpp)
        self.assertNotIn(b"//void step() override;", after_h)
        self.assertIn(b"alpha = Dendrite::alpha;", after_cpp)
        self.assertIn(b"void DendriteScaled::step()", after_cpp)
        self.assertIn(b"v[i] = (*de)[0][i];", after_cpp)
        self.assertIn(PATCH_VARIANT, diff)
        self.assertEqual(before_h, records["dendrite_scaled.h"]["source"].encode())

    def test_source_repair_refuses_unpinned_code(self):
        with self.assertRaisesRegex(ValueError, "exact pinned"):
            patch_scaled_sources(b"lookalike source", b"no provenance")

    def test_actual_parameter_addressing_and_supported_input(self):
        before = self.baseline()
        after, changes = candidate_config(before, gain=.5, coupling=2., history=.8, amplitude=100.)
        self.assertEqual(len(changes), 22)
        self.assertEqual(before.count("Connection"), after.count("Connection"))
        self.assertIn("Population: 1 2 3 4 5 6 7 8 9 10 11", after)
        selected = {item["parameter"]: item for item in changes}
        self.assertEqual(selected["Dendrite 1.hsyn"]["after"], selected["Dendrite 1.hsyn"]["before"]*.5)
        self.assertLess(selected["Coupling 2.nu"]["after"], 0)
        self.assertEqual(selected["Stimulus 11.Amplitude"]["after"], 100.)
        # Existing integration step, population transfer functions and noise stay
        # the same when changing the declared mechanistic family.
        self.assertIn("Time: 20 Deltat: 0.0001", after)
        self.assertIn("StdDev: 106.302", after)

    def test_config_comments_cannot_target_parameters(self):
        before = "/* Dendrite 1: Scaled - alpha: 999 */\nDendrite 1: Scaled - alpha: 3 beta: 7 hsyn: 2\n"
        after, change = set_numeric(before, "Dendrite", 1, "alpha", 6)
        self.assertIn("alpha: 999", after)
        self.assertEqual(change["before"], 3)
        with self.assertRaises(ValueError):
            set_numeric(before + before, "Dendrite", 1, "alpha", 6)

    def test_design_is_bounded_deterministic_and_covers_mechanisms(self):
        a, b = design_candidates(48), design_candidates(48)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 48)
        self.assertEqual({r["account_class"] for r in a}, {"gain", "coupling", "history", "combined"})
        self.assertTrue(all(.5 <= row[k] <= 2 for row in a for k in ("gain", "coupling", "history")))
        self.assertEqual(design_candidates(0), [])
        with self.assertRaises(ValueError):
            design_candidates(129)

    def test_response_uses_actual_input_and_returns_recovery(self):
        times = np.arange(5000)*.002
        baseline = .01*np.sin(2*np.pi*8*times)
        response = np.where(times >= 2, np.exp(-(times-2)*5)*.02, 0)
        stimulus = np.where((times >= 2)&(times <= 2.02), 100., 0)
        metrics = pulse_response(native_trace(baseline), native_trace(baseline+response, stimulus))
        self.assertEqual(metrics["observed_input_peak"], 100.)
        self.assertGreater(metrics["response_normalized_peak"], 2)
        self.assertAlmostEqual(metrics["response_recovery_s"], np.log(20)/5, delta=.004)
        self.assertEqual(metrics["response_recovery_status"], "returned_below_5pct_peak_for_250ms")

    def test_response_refuses_pre_input_differences_or_unobserved_stimulus(self):
        baseline = np.sin(np.arange(5000)*.1)
        stimulus = np.zeros(5000); stimulus[1000:1010] = 1
        with self.assertRaisesRegex(ValueError, "before"):
            pulse_response(native_trace(baseline), native_trace(baseline+.01, stimulus))
        with self.assertRaisesRegex(ValueError, "not observed"):
            pulse_response(native_trace(baseline), native_trace(baseline))

    def test_no_response_and_right_censoring_retained(self):
        baseline = np.sin(np.arange(5000)*.1)
        stimulus = np.zeros(5000); stimulus[1000:1010] = 1
        no_response = pulse_response(native_trace(baseline), native_trace(baseline, stimulus))
        self.assertEqual(no_response["response_recovery_status"], "no_detectable_response")
        delta = np.zeros(5000); delta[1000:] = 1
        not_recovered = pulse_response(native_trace(baseline), native_trace(baseline+delta, stimulus))
        self.assertEqual(not_recovered["response_recovery_s"], "")
        self.assertEqual(not_recovered["response_recovery_status"], "right_censored_at_end")

    def test_scalar_node_observation_refuses_ambiguous_nodes(self):
        parsed = native_trace(np.arange(10))
        np.testing.assert_array_equal(output_signal(parsed), np.arange(10))
        parsed["columns"] += ["Pop.1.V"]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            output_signal(parsed)

    def test_failed_native_candidate_retained_with_real_log_receipt(self):
        spec = {"simulation_id": "candidate_01", "variant": PATCH_VARIANT, "account_class": "gain", "configuration_text": self.baseline(), "changes": []}
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "candidate"
            def fail(argv, log_path, **kwargs):
                log_path.write_text("observed native failure\n", encoding="utf-8")
                return {"returncode": 2, "error": "", "seconds": .1}
            with patch("native_tools.mechanistic_native._execute", side_effect=fail):
                result = _run_one(spec, Path("known_simulator"), destination, 30)
            self.assertEqual(result["status"], "failed")
            self.assertIn("did not finish successfully", result["error"])
            receipt = json.loads((destination / "run_receipt.json").read_text())
            self.assertEqual(receipt["simulation_id"], "candidate_01")
            self.assertEqual(receipt["execution"]["returncode"], 2)
            self.assertEqual(receipt["config_sha256"], hashlib.sha256((destination / "configuration.conf").read_bytes()).hexdigest())

    def test_empty_exports_remain_schema_valid_and_inventory_retains_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _csv(root / "detector_jobs.csv", [], JOBS_FIELDS)
            _csv(root / "simulation_inventory.csv", [{"simulation_id": "bad", "status": "failed", "error": "unstable"}], INVENTORY_FIELDS)
            self.assertTrue((root / "detector_jobs.csv").read_text().startswith("job_id,episode_id,"))
            self.assertIn("unstable", (root / "simulation_inventory.csv").read_text())

    def test_constant_and_overflowing_variance_keep_native_output_without_export(self):
        for signal, expected in ((np.ones(5000), "finite_constant_not_standardizable"),
                                  (np.sin(np.arange(5000)) * 1e300, "failed")):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                spec = {"simulation_id": "software_fixture", "variant": PATCH_VARIANT, "account_class": "gain", "configuration_text": self.baseline(), "changes": []}
                def fixture_execution(argv, log_path, **kwargs):
                    # An explicit synthetic file tests the parser/export guard.
                    # It does not provide native simulator qualification.
                    log_path.write_text("synthetic execution double\n", encoding="utf-8")
                    destination = Path(argv[argv.index("-o") + 1])
                    with destination.open("w", encoding="utf-8") as stream:
                        stream.write("SYNTHETIC UNIT FIXTURE\n=============================================\nTime Pop.1.V\n1\n")
                        np.savetxt(stream, np.column_stack((np.arange(5000)*.002, signal)), fmt="%.17g")
                    return {"returncode": 0, "error": "", "seconds": .01}
                with patch("native_tools.mechanistic_native._execute", side_effect=fixture_execution):
                    result = _run_one(spec, Path("simulator_double"), Path(temporary)/"fixture", 30)
                self.assertEqual(result["status"], expected)
                self.assertTrue(Path(result["source_output_path"]).is_file())
                self.assertNotIn("signal_path", result)
                receipt = json.loads((Path(temporary)/"fixture"/"run_receipt.json").read_text())
                self.assertEqual(receipt["source_output_sha256"], result["source_output_sha256"])


if __name__ == "__main__":
    unittest.main()
