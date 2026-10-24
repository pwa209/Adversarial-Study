"""Synthetic checks of configuration leakage, input controls and bank inference."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import native_population
except ModuleNotFoundError:
    # Local source checkout: qualification releases place this pinned v2 module
    # beside the test. Do not fake its strict pair qualification in fixtures.
    fallback = Path(__file__).resolve().parents[2] / "full_population_clock_repair_20261024" / "release"
    if fallback.is_dir():
        sys.path.append(str(fallback))
import mechanism_compensation as m


def pair(index=0, family="gain", factors=(1., 1., 1.), amplitude=1., variant="scaled_dendrite_v1_explicit_patch"):
    t = np.arange(1, 501) * .01
    g, c, h = factors
    common = np.sin(t * (3. + h)) + .2 * np.cos(t * 13)
    values = np.column_stack([(.2 + j * .03) * common * g * c + j + .05 * np.sin(t * (j + 3)) for j in range(18)])
    noise = np.sin(t * 19.)
    response = np.zeros(len(t))
    response[t >= 2.] = amplitude * g * c * np.exp(-(t[t >= 2.] - 2.) * h / .12)
    config = "Time: 20 Deltat: .0001\nNodes: 1\n"
    config += "\n".join(f"Population {j}: Synthetic\nDendrite {j}: Scaled alpha: {2*h} beta: {3*h} hsyn: {g}\n" for j in range(1, 10))
    config += "Population 10: Stimulation\nStimulus: White - Onset: 0 Mean: 1 StdDev: 2 WhiteSeed: 123\n"
    config += "Population 11: Perturb\nStimulus: PulseRect - Onset: 12 Amplitude: {amplitude} Width: .02 Pulses: 1\n"
    config += f"Propagator 1: Map - Tau: .01\nCoupling 1: Map - nu: {c}\nOutput: Node: All Start: 10 Interval: .002\nPopulation: 1 2 3 4 5 6 7 8 9 10 11\n"
    common_row = {"variant": variant, "account_class": family, "pair_id": f"candidate_{index:03d}",
                  "param_gain_factor": g, "param_coupling_factor": c, "param_history_factor": h}
    answer = []
    for tag, amp in (("control", 0.), ("pulse", amplitude)):
        y = values.copy() if amp == 0 else values + response[:, None] * np.linspace(.3, 1.5, 18)
        row = common_row | {"simulation_id": variant + f"__candidate_{index:03d}__{tag}", "input_amplitude": amp}
        answer.append({"row": row, "times": t.copy(), "values": y,
                       "noise": noise.copy(), "stimulus": amp * ((t >= 2.) & (t <= 2.02)),
                       "stimulus_observed": True, "config_text": config.format(amplitude=amp),
                       "config": {"input_onset_s": 12., "output_start_s": 10., "input_width_s": .02, "input_amplitude": amp}})
    return answer


def bank_fixture():
    traces = {}
    for family_number, family in enumerate(("gain", "coupling", "history", "combined")):
        for j, factor in enumerate((.5, .8, 1.25, 2.)):
            factors = [1., 1., 1.]
            if family == "combined":
                factors = [factor, .7 + j * .25, 1.6 - j * .25]
            else:
                factors[family_number] = factor
            for trace in pair(family_number * 4 + j, family, tuple(factors), (1., 10., 100.)[j % 3]):
                traces[trace["row"]["simulation_id"]] = trace
    return {"traces": traces, "receipt": {"schema": "native_nine_population_cache_v2"}}


class ConfigurationCustodyTests(unittest.TestCase):
    def test_pulse_control_seed_and_recorder_share_one_physical_role(self):
        control, pulse = pair()
        pulse["config_text"] = pulse["config_text"].replace("WhiteSeed: 123", "WhiteSeed: 999").replace("Population: 1 2", "Population: 2 1")
        self.assertEqual(m.structural_key(control), m.structural_key(pulse))

    def test_actual_mechanism_change_is_not_masked(self):
        control, _ = pair()
        changed = copy.deepcopy(control)
        changed["config_text"] = changed["config_text"].replace("nu: 1.0", "nu: 1.1")
        self.assertNotEqual(m.structural_key(control), m.structural_key(changed))

    def test_all_pairs_stay_together_and_split_is_order_independent(self):
        bank = bank_fixture()
        roles, rows = m.assign_roles(bank["traces"], 123)
        reversed_roles, _ = m.assign_roles(dict(reversed(list(bank["traces"].items()))), 123)
        self.assertEqual(roles, reversed_roles)
        for key, trace in bank["traces"].items():
            other = key.replace("__control", "__pulse")
            self.assertEqual(roles[key]["role"], roles[other]["role"])
        self.assertEqual(len({row["configuration_group"] for row in rows}), 16)
        self.assertEqual({row["role"] for row in rows}, {"train", "validation", "test"})

    def test_published_nominal_factors_are_never_parameter_ground_truth(self):
        control, _ = pair()
        self.assertTrue(m.parameter_eligible(control))
        control["row"]["account_class"] = "published_bank"
        self.assertFalse(m.parameter_eligible(control))


class DynamicsTests(unittest.TestCase):
    def test_training_reference_cannot_see_held_trace(self):
        a = pair(1, factors=(.8, 1., 1.))[0]
        b = pair(2, factors=(1.2, 1., 1.))[0]
        held = pair(3, factors=(2., 1., 1.))[0]
        before = m.reference_fit([a, b])
        held["values"] *= 1e8
        after = m.reference_fit([a, b])
        np.testing.assert_array_equal(before["axes"], after["axes"])
        self.assertNotIn(held["row"]["simulation_id"], before["training_simulation_ids"])

    def test_pre_input_features_cannot_see_future_control(self):
        a, _ = pair()
        b = pair(1, factors=(.8, 1., 1.))[0]
        reference = m.reference_fit([a, b])
        initial = m.trace_features(a, reference, before=1.99)
        a["values"][a["times"] >= 1.99] += 1e5
        self.assertEqual(initial, m.trace_features(a, reference, before=1.99))

    def test_constant_and_nearly_constant_channels_do_not_create_huge_gain(self):
        a, _ = pair()
        b = pair(1, factors=(.8, 1., 1.))[0]
        for trace in (a, b):
            trace["values"][:, 2:] = 20. + np.sin(trace["times"][:, None]) * 1e-20
        reference = m.reference_fit([a, b])
        features = m.trace_features(a, reference)
        self.assertTrue(all(np.isfinite(list(features.values()))))
        self.assertLessEqual(features["common_scaled_participation"], 2.01)
        self.assertGreaterEqual(reference["scale"][2], 20e-12)

    def test_participation_distinguishes_one_two_and_zero_modes(self):
        self.assertEqual(m._participation([0, 0]), 0.)
        self.assertAlmostEqual(m._participation([0, 1]), 1.)
        self.assertAlmostEqual(m._participation([1, 1]), 2.)
        self.assertAlmostEqual(m._participation([1e-20, 1e-20]), 2.)

    def test_ridge_handles_small_physical_units_and_linear_signal(self):
        x = np.arange(20., dtype=float)[:, None] * 1e-9
        y = 3 * x + 2e-9
        fit = m.ridge_fit(x, y, .01)
        prediction = m.ridge_predict(fit, x)
        self.assertLess(np.mean((prediction - y) ** 2), 1e-21)
        self.assertLess(fit["sd"][0], 1e-7)


class HeldPredictionTests(unittest.TestCase):
    def test_test_targets_cannot_change_hyperparameters_or_predictions(self):
        x = np.arange(12., dtype=float)[:, None]
        y = np.column_stack((x[:, 0] * 3, x[:, 0] ** 2))
        roles = ["train"] * 7 + ["validation"] * 3 + ["test"] * 2
        rows = [{"role": role, "variant": "variant", "account_class": "gain", "configuration_group": str(i)} for i, role in enumerate(roles)]
        first, selection = m.prediction_experiment(rows, {"training_mean": np.empty((12, 0)), "full": x}, y, ("a", "b"))
        y[-2:] += 1e6
        second, selection2 = m.prediction_experiment(rows, {"training_mean": np.empty((12, 0)), "full": x}, y, ("a", "b"))
        self.assertEqual(selection, selection2)
        self.assertEqual([r["predicted_a"] for r in first], [r["predicted_a"] for r in second])

    def test_no_test_configurations_is_explicitly_unsupported(self):
        rows = [{"role": "train"}] * 4
        predictions, receipt = m.prediction_experiment(rows, {"full": np.arange(4.)[:, None]}, np.arange(4.)[:, None], ("a",))
        self.assertFalse(predictions)
        self.assertEqual(receipt["status"], "unsupported_configuration_role_coverage")

    def test_bootstrap_pairs_predictions_and_does_not_count_people(self):
        rows = []
        for i in range(6):
            for model, loss in (("training_mean", i + 2.), ("full_population", i + 1.)):
                rows.append({"variant": "v", "configuration_group": str(i), "account_class": "gain" if i < 3 else "history", "model": model, "standardized_mse": loss})
        result = m.bootstrap_predictions(rows, 123, 200)
        estimate = next(r for r in result["results"] if r["model"] == "full_population")
        np.testing.assert_allclose(estimate["bank_stability_difference_percentiles_2p5_97p5"], [-1., -1.], atol=1e-14)
        self.assertEqual(result["human_participants_resampled"], 0)
        self.assertEqual(estimate["held_configuration_groups"], 6)
        with self.assertRaisesRegex(ValueError, "once"):
            m.bootstrap_predictions(rows + [rows[0]], 123, 200)

    def test_bootstrap_is_reproducible(self):
        rows = [{"variant": "v", "configuration_group": str(i), "account_class": "combined", "model": model, "standardized_mse": i * factor}
                for i in range(5) for model, factor in (("training_mean", 1.), ("full_population", .5))]
        self.assertEqual(m.bootstrap_predictions(rows, 7, 100), m.bootstrap_predictions(rows, 7, 100))


class ExistingPairTests(unittest.TestCase):
    def test_response_metrics_use_actual_observed_input(self):
        control, pulse = pair(amplitude=10.)
        targets, onset, design, proof = m._response_targets(control, pulse)
        self.assertEqual(onset, 2.)
        self.assertAlmostEqual(np.exp(design[0]), 10.)
        self.assertGreater(targets["voltage_peak_rms"], 0.)
        self.assertGreater(targets["voltage_mode_fraction"], .999)
        self.assertTrue(proof["full_noise_stream_exactly_equal"])

    def test_unmatched_background_and_unobserved_pulse_are_rejected(self):
        control, pulse = pair()
        pulse["noise"][0] += .1
        with self.assertRaisesRegex(ValueError, "noise"):
            m._response_targets(control, pulse)
        control, pulse = pair()
        pulse["stimulus"][:] = 0
        with self.assertRaisesRegex(ValueError, "amplitude"):
            m._response_targets(control, pulse)

    def test_end_to_end_synthetic_bank_has_no_pseudo_replicated_units(self):
        result = m.analyze(bank_fixture(), seed=101, repetitions=100, expected_zero=16, expected_pairs=16)
        self.assertEqual(result["n_zero"], 16)
        self.assertEqual(result["n_pairs"], 16)
        self.assertTrue(result["parameter_predictions"])
        self.assertTrue(result["response_predictions"])
        self.assertEqual(len(result["features"]), 16)
        selected = result["response_models"]["scaled_dendrite_v1_explicit_patch"]
        self.assertIn("factors_and_observed_input", selected["models"])
        self.assertIn("observed_input_only", selected["models"])
        self.assertEqual(selected["role_counts"], {"train": 8, "validation": 4, "test": 4})
        self.assertFalse(selected["test_target_used_for_selection"])
        for row in result["response_predictions"]:
            self.assertIn(row["observed_input_amplitude"], (1., 10.000000000000002, 100.00000000000004))
        boot = result["bootstrap"]["finite_response_prediction"]
        self.assertTrue(all(row["held_configuration_groups"] == 4 for row in boot["results"]))
        # Every object written as JSON in run must support the exact native
        # encoder used by Slurm qualification; no permissive default=str hook.
        for key in ("coverage", "parameter_models", "response_models", "mode_summaries", "bootstrap"):
            json.dumps(result[key], allow_nan=False)

    def test_qualification_probe_receipt_is_strictly_json_serializable(self):
        traces = {}
        for index in range(48):
            for trace in pair(index, "combined", (0.6 + index * .01, 1., 1.), (1., 10., 100.)[index % 3]):
                traces[trace["row"]["simulation_id"]] = trace
        template = pair(1000)[0]
        for index in range(568):
            variant = "pristine" if index % 2 else "scaled_dendrite_v1_explicit_patch"
            simulation_id = variant + f"__published_{index:03d}"
            trace = dict(template)
            trace["row"] = dict(template["row"]) | {"simulation_id": simulation_id, "variant": variant,
                    "account_class": "published_bank", "pair_id": ""}
            trace["config_text"] = template["config_text"].replace("Nodes: 1", f"Nodes: 1\nSyntheticConfigurationIdentity: {index}")
            traces[simulation_id] = trace
        bank = {"traces": traces, "receipt": {"schema": "native_nine_population_cache_v2"}}
        with patch.object(m, "_load_input_cache", return_value=(bank, Path("fixture_cache"), {})), patch.object(m, "sha", return_value="f" * 64):
            probe = m.qualification_probe({}, 17)
        self.assertEqual(probe["native_traces"], 664)
        self.assertEqual(probe["zero_input_configurations"], 616)
        self.assertEqual(probe["paired_groups"], 48)
        json.dumps(probe, allow_nan=False)

    def test_wrong_actual_bank_count_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "zero-input"):
            m.analyze(bank_fixture(), expected_zero=616, expected_pairs=48)


if __name__ == "__main__":
    unittest.main()
