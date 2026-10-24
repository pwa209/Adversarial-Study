"""Known-generator and leakage checks, not empirical scientific findings."""
from __future__ import annotations

import csv
from contextlib import redirect_stderr
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adversarial_study.effective_dynamics import (
    FAMILIES, _generate, analyze_recording_windows, compare_effective_dynamics,
    episode_features, main, synthetic_recovery_report, temporal_surrogate,
)


def signal(matrices=None, samples=2400, seed=19):
    matrices = matrices if matrices is not None else [np.array([[.7, .3], [0., .6]])]
    return _generate(matrices, np.eye(2), samples, np.random.default_rng(seed))


FAST = {"temporal_controls": False}


class EffectiveDynamicsTests(unittest.TestCase):
    def test_training_parameters_and_scaling_ignore_later_samples(self):
        x = signal()
        original = compare_effective_dynamics(x, 100., settings=FAST)
        changed = x.copy()
        changed[:, original["split_sample"]:] = changed[:, original["split_sample"]:] * 3 + 15
        updated = compare_effective_dynamics(changed, 100., settings=FAST)
        self.assertEqual(original["training_mean_input_units"], updated["training_mean_input_units"])
        self.assertEqual(original["training_scale_input_units"], updated["training_scale_input_units"])
        for family in FAMILIES:
            a, b = original["models"][family], updated["models"][family]
            for name in ("intercept_standardized", "lag_matrices_standardized",
                         "innovation_covariance_standardized_regularized", "descriptors"):
                self.assertEqual(a[name], b[name])
            self.assertNotEqual(a["one_step_mse_standardized"], b["one_step_mse_standardized"])

    def test_common_targets_recursive_forecasts_and_effective_interpretation(self):
        result = compare_effective_dynamics(signal(), 100., settings=FAST)
        split = result["split_sample"]
        self.assertEqual(result["first_heldout_target_sample"], split + 2)
        self.assertEqual(result["training_transition_count"], split - 2)
        self.assertEqual(result["heldout_transition_count"], 2400 - split - 2)
        self.assertFalse(result["independent_human_validation"])
        self.assertIn("causal coupling", result["interpretation"])
        for report in result["models"].values():
            first = report["forecasts"][0]
            self.assertEqual(first["origins"], result["heldout_transition_count"])
            self.assertAlmostEqual(first["trajectory_mse_standardized"], report["one_step_mse_standardized"])
            self.assertEqual(report["forecasts"][1]["steps"], 5)
            self.assertEqual(report["forecasts"][1]["horizon_s"], .05)
        var1 = result["models"]["coupled_var1"]
        self.assertGreater(var1["forecasts"][1]["terminal_mse_standardized"], var1["one_step_mse_standardized"])
        self.assertEqual(result["models"]["covariance_only"]["lag_matrices_standardized"], [])
        json.dumps(result, allow_nan=False)
        json.dumps(episode_features(result), allow_nan=False)

    def test_cross_channel_and_second_lag_information_are_predictively_diagnostic(self):
        coupled = compare_effective_dynamics(signal([np.array([[.2, .9], [0., .7]])], samples=4000),
                                            100., settings=FAST)["models"]
        self.assertLess(coupled["coupled_var1"]["one_step_mse_standardized"],
                        coupled["diagonal_ar1"]["one_step_mse_standardized"] - .1)
        second_order = signal([np.array([[.8, .2], [0., .5]]), -.6 * np.eye(2)], samples=4000)
        models = compare_effective_dynamics(second_order, 100., settings=FAST)["models"]
        self.assertLess(models["coupled_var2"]["one_step_mse_standardized"],
                        models["coupled_var1"]["one_step_mse_standardized"] - .15)
        self.assertEqual(len(models["coupled_var2"]["descriptors"]["eigenvalues"]), 4)

    def test_channel_selection_uses_recorded_labels_and_never_replaces_flat_channels(self):
        x = np.vstack([signal(), signal(seed=31)])
        result = compare_effective_dynamics(x, 100., channel_names=["A", "B", "C", "D"],
                                           settings={**FAST, "max_channels": 2, "channel_selection": ["D", "B"]})
        self.assertEqual(result["selected_channel_names"], ["D", "B"])
        self.assertEqual(result["selected_channel_indices"], [3, 1])
        default = compare_effective_dynamics(x, 100., settings={**FAST, "max_channels": 2})
        self.assertEqual(default["selected_channel_indices"], [0, 1])
        x[0] = 0
        with self.assertRaisesRegex(ValueError, "training variance"):
            compare_effective_dynamics(x, 100., settings={**FAST, "max_channels": 2})

    def test_input_shape_frequency_timestamp_and_settings_validation(self):
        x = signal(samples=400)
        for bad in (x[0], x[:1], np.full(x.shape, np.nan), x.astype(complex)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                compare_effective_dynamics(bad, 100., settings=FAST)
        for frequency in (0., float("nan"), [100., 100.]):
            with self.subTest(frequency=frequency), self.assertRaises(ValueError):
                compare_effective_dynamics(x, frequency, settings=FAST)
        times = np.arange(400) / 100.
        times[200:] += .01
        with self.assertRaisesRegex(ValueError, "Discontinuous"):
            compare_effective_dynamics(x, 100., sample_times_s=times, settings=FAST)
        with self.assertRaises(ValueError):
            compare_effective_dynamics(x[:, :40], 100., settings=FAST)
        for options in ({"ridge": 0.}, {"train_fraction": 1.}, {"forecast_steps": [1, 1]},
                        {"forecast_steps": [1.5]}, {"channel_selection": [0, 0]}, {"unknown": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                compare_effective_dynamics(x, 100., settings=options)


class TemporalControlTests(unittest.TestCase):
    def test_shuffling_preserves_joint_covariance_phase_preserves_channel_spectra(self):
        x = signal(samples=700)
        split = 400
        shuffled = temporal_surrogate(x, split, "joint_shuffle", 12)
        phase = temporal_surrogate(x, split, "independent_phase", 12)
        for section in (slice(0, split), slice(split, None)):
            np.testing.assert_allclose(np.cov(x[:, section]), np.cov(shuffled[:, section]), atol=1e-12)
            np.testing.assert_allclose(np.abs(np.fft.rfft(x[:, section], axis=1)),
                                       np.abs(np.fft.rfft(phase[:, section], axis=1)), atol=1e-11)
        self.assertFalse(np.allclose(x, phase))
        self.assertFalse(np.allclose(x, shuffled))

    def test_control_training_is_blockwise_and_seeded(self):
        x = signal(samples=1000)
        first = compare_effective_dynamics(x, 100.)
        changed = x.copy()
        changed[:, first["split_sample"]:] += 10
        second = compare_effective_dynamics(changed, 100.)
        self.assertEqual(first, compare_effective_dynamics(x, 100.))
        for kind in ("joint_shuffle", "independent_phase"):
            self.assertTrue(first["controls"][kind]["blockwise_transformation"])
            for family in FAMILIES:
                a = first["controls"][kind]["models"][family]
                b = second["controls"][kind]["models"][family]
                np.testing.assert_allclose(a["lag_matrices_standardized"], b["lag_matrices_standardized"], atol=1e-12)


class RecordingAndRecoveryTests(unittest.TestCase):
    def test_separate_windows_preserve_gaps_exclusions_and_outcome_independence(self):
        recording = {"data": signal(samples=1800), "sfreq_hz": 100., "channel_names": ["A", "B"],
                     "units": "uV", "gap_intervals_s": [[5., 6.]]}
        windows = [{"episode_id": "first", "participant_id": "p1", "start_s": 0., "stop_s": 4., "score": 0.},
                   {"episode_id": "gap", "participant_id": "p1", "start_s": 4., "stop_s": 8.},
                   {"episode_id": "later", "participant_id": "p1", "start_s": 10., "stop_s": 14.}]
        report = analyze_recording_windows(recording, windows, settings=FAST)
        self.assertEqual(report["counts"], {"input_windows": 3, "fitted_windows": 2, "excluded_windows": 1})
        self.assertIn("gap", report["exclusions"][0]["reason"])
        self.assertEqual([row["n_samples"] for row in report["episode_reports"]], [400, 400])
        windows[0]["score"] = 1.
        windows[0]["capacity"] = 1000
        self.assertEqual(report, analyze_recording_windows(recording, windows, settings=FAST))
        self.assertNotIn("score", report["features"][0])

    def test_synthetic_recovery_reports_nested_confusion_without_empirical_claim(self):
        report = synthetic_recovery_report(replicates=2, samples=1800, seed=44)
        self.assertEqual(report["evidence_status"], "ASSUMPTIONS_ONLY_SYNTHETIC")
        self.assertFalse(report["empirical_inference"])
        for counts in report["confusion_counts"].values():
            self.assertEqual(sum(counts.values()), 2)
        errors = [row["correct_family_lag_matrix_rmse_standardized"] for row in report["replicates"]
                  if row["correct_family_lag_matrix_rmse_standardized"] is not None]
        self.assertLess(max(errors), .08)
        self.assertEqual(report, synthetic_recovery_report(replicates=2, samples=1800, seed=44))
        json.dumps(report, allow_nan=False)

    def test_standalone_npz_csv_cli_writes_receipts_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "eeg.npz", data=signal(samples=700), sfreq_hz=100.,
                     channel_names=np.array(["A", "B"]), units="uV")
            with (root / "windows.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["episode_id", "start_s", "stop_s", "participant_id"])
                writer.writeheader()
                writer.writerow({"episode_id": "e1", "start_s": 0, "stop_s": 5, "participant_id": "p1"})
            (root / "settings.json").write_text(json.dumps(FAST))
            args = ["--recording", str(root / "eeg.npz"), "--windows", str(root / "windows.csv"),
                    "--settings-json", str(root / "settings.json"), "--output-dir", str(root / "output")]
            self.assertEqual(main(args), 0)
            receipt = json.loads((root / "output/receipt.json").read_text())
            self.assertEqual(receipt["counts"]["fitted_windows"], 1)
            self.assertEqual(len(receipt["episode_receipts"]), 1)
            self.assertTrue((root / "output/episode_features.csv").is_file())
            self.assertEqual(len(receipt["inputs"]["recording_sha256"]), 64)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)


if __name__ == "__main__":
    unittest.main()
