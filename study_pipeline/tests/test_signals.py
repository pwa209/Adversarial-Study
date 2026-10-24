"""Known-structure implementation checks; these are not empirical results."""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from adversarial_study.data import extract_window, read_recording
from adversarial_study.signals import compute_features, quality_metrics


SFREQ = 250.0


def sine(frequency: float, amplitude: float = 10.0, duration: float = 10.0) -> np.ndarray:
    times = np.arange(int(SFREQ * duration)) / SFREQ
    return amplitude * np.sin(2.0 * np.pi * frequency * times)


def recording(data: np.ndarray) -> dict:
    return {
        "data": data, "sfreq_hz": SFREQ,
        "channel_names": [f"EEG{channel}" for channel in range(len(data))],
        "units": "uV",
    }


class SignalFeatureTests(unittest.TestCase):
    def test_known_sines_identify_all_bands_and_power(self):
        for frequency, band in [(2, "delta"), (6, "theta"), (10, "alpha"), (20, "beta"), (37, "gamma")]:
            with self.subTest(frequency=frequency):
                features = compute_features(sine(frequency)[None, :], SFREQ)
                self.assertGreater(features[f"relative_{band}"], 0.999)
                self.assertAlmostEqual(sum(features[f"relative_{name}"] for name in
                                           ["delta", "theta", "alpha", "beta", "gamma"]), 1.0, places=10)
                self.assertAlmostEqual(features["total_band_power_uv2"], 50.0, delta=0.1)
                self.assertEqual(features["welch_segments"], 9.0)
                self.assertEqual(features["welch_frequency_resolution_hz"], 0.5)

    def test_multichannel_power_is_voltage_weighted(self):
        features = compute_features(np.stack([sine(10, 10), sine(20, 20)]), SFREQ)
        self.assertAlmostEqual(features["relative_alpha"], 0.2, delta=0.001)
        self.assertAlmostEqual(features["relative_beta"], 0.8, delta=0.001)
        self.assertEqual(features["n_channels"], 2)

    def test_covariance_effective_rank_recovers_known_structure(self):
        base = sine(10)
        same = compute_features(np.stack([base, base * 2]), SFREQ)
        independent = compute_features(np.stack([base, sine(20)]), SFREQ)
        self.assertAlmostEqual(same["effective_rank"], 1.0, places=8)
        self.assertAlmostEqual(independent["effective_rank"], 2.0, places=8)
        self.assertAlmostEqual(independent["covariance_participation_rank"], 2.0, places=8)

    def test_autocorrelation_preserves_channel_values_and_time_scale(self):
        features = compute_features(np.stack([sine(10), sine(5)]), SFREQ)
        self.assertEqual(features["acf_lag01_s"], 0.1)
        self.assertAlmostEqual(features["acf_lag01_ch000"], 1.0, delta=0.02)
        self.assertAlmostEqual(features["acf_lag01_ch001"], -1.0, delta=0.02)
        self.assertAlmostEqual(features["acf_100ms"], 0.0, delta=0.02)

    def test_var1_uses_temporal_holdout_and_recovers_stable_ar_process(self):
        rng = np.random.default_rng(29)
        array = np.zeros((1, 2500))
        noise = rng.normal(0.0, 1.0, array.shape[1])
        for index in range(1, array.shape[1]):
            array[0, index] = 0.8 * array[0, index - 1] + noise[index]
        features = compute_features(array, SFREQ)
        self.assertAlmostEqual(features["var1_spectral_radius"], 0.8, delta=0.05)
        self.assertEqual(features["var1_stable"], 1.0)
        self.assertGreater(features["var1_improvement_over_persistence"], 0.03)
        self.assertEqual(features["var1_training_transitions"], 1749)
        self.assertEqual(features["var1_holdout_transitions"], 749)
        self.assertEqual(features["var1_step_s"], 0.004)
        changed_holdout = array.copy()
        changed_holdout[:, 1750:] += 100.0
        changed_features = compute_features(changed_holdout, SFREQ)
        self.assertEqual(features["var1_spectral_radius"], changed_features["var1_spectral_radius"])
        self.assertGreater(changed_features["var1_holdout_mse_standardized"], features["var1_holdout_mse_standardized"])

    def test_nonfinite_signals_fail_without_imputation(self):
        array = sine(10)[None, :]
        for invalid in (np.nan, np.inf):
            with self.subTest(invalid=invalid):
                modified = array.copy()
                modified[0, 5] = invalid
                with self.assertRaisesRegex(ValueError, "finite"):
                    compute_features(modified, SFREQ)
                quality = quality_metrics(modified, SFREQ)
                self.assertFalse(quality["qualified"])
                self.assertFalse(quality["finite"])
                self.assertEqual(quality["reasons"], ["nonfinite_signal"])
                self.assertIsNone(quality["max_abs_uv"])

    def test_flat_channels_and_development_thresholds(self):
        array = np.stack([sine(10), np.zeros(2500)])
        quality = quality_metrics(array, SFREQ)
        self.assertFalse(quality["qualified"])
        self.assertIn("flat_channel_at_development_threshold", quality["reasons"])
        self.assertEqual(quality["min_std_uv"], 0)
        self.assertTrue(quality_metrics(sine(10)[None, :], SFREQ)["qualified"])
        self.assertFalse(quality_metrics(sine(10)[None, :], SFREQ, max_abs_uv=5)["qualified"])
        features = compute_features(np.zeros((2, 2500)), SFREQ)
        self.assertEqual(features["effective_rank"], 0)
        self.assertEqual(features["relative_alpha"], 0)
        self.assertTrue(all(np.isfinite(value) for value in features.values()))

    def test_feature_configuration_rejects_ambiguous_or_unavailable_scales(self):
        array = sine(10)[None, :]
        for settings in [
            {"typo": 1}, {"welch_overlap_fraction": 1}, {"var_ridge": 0},
            {"acf_lags_s": [11]}, {"acf_lags_s": [0.001]},
            {"welch_segment_s": 2.001}, {"bands_hz": {"alpha": (45, 50)}},
        ]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                compute_features(array, SFREQ, settings)
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            compute_features(array, 80.0)
        with self.assertRaisesRegex(ValueError, "shorter"):
            compute_features(array[:, :100], SFREQ)
        with self.assertRaises(ValueError):
            quality_metrics(array, SFREQ, max_abs_uv=-1)
        with self.assertRaises(ValueError):
            quality_metrics(array, 0)


class RecordingAdapterTests(unittest.TestCase):
    def test_npz_preserves_values_units_and_labels(self):
        data = np.stack([sine(10), sine(20)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.npz"
            np.savez(path, data=data, sfreq_hz=SFREQ, channel_names=np.array(["Cz", "Pz"]),
                     units="uV", reference="linked mastoids")
            result = read_recording(path)
            np.testing.assert_array_equal(result["data"], data)
            self.assertEqual(result["channel_names"], ["Cz", "Pz"])
            self.assertEqual(result["sfreq_hz"], SFREQ)
            self.assertEqual(result["units"], "uV")
            self.assertEqual(result["reference"], "linked mastoids")
            window = extract_window(result, 0, 10)
            np.testing.assert_array_equal(window, data)
            window[0, 0] = 99
            self.assertNotEqual(window[0, 0], result["data"][0, 0])

    def test_npz_rejects_missing_keys_nonfinite_values_object_labels_and_wrong_units(self):
        cases = [
            {"data": sine(10)[None, :], "sfreq_hz": SFREQ},
            {"data": [[np.nan]], "sfreq_hz": SFREQ, "channel_names": ["Cz"]},
            {"data": sine(10)[None, :], "sfreq_hz": SFREQ, "channel_names": np.array(["Cz"], dtype=object)},
            {"data": sine(10)[None, :], "sfreq_hz": SFREQ, "channel_names": ["Cz"], "units": "V"},
            {"data": np.stack([sine(10)] * 2), "sfreq_hz": SFREQ, "channel_names": ["Cz", "Cz"]},
            {"data": sine(10)[None, :], "sfreq_hz": SFREQ, "channel_names": np.array([b"Cz"])},
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, case in enumerate(cases):
                with self.subTest(index=index):
                    path = Path(directory) / f"invalid{index}.npz"
                    np.savez(path, **case)
                    with self.assertRaises(ValueError):
                        read_recording(path)

    def test_out_of_bounds_off_grid_and_non_exact_durations_are_rejected(self):
        source = recording(sine(10)[None, :])
        for start, stop, duration in [(-1, 9, 10), (0, 11, 10), (0, 9.996, 10),
                                      (0.001, 10.001, 10), (0, 10.001, 10.001), (5, 5, 10)]:
            with self.subTest(start=start, stop=stop, duration=duration), self.assertRaises(ValueError):
                extract_window(source, start, stop, duration_s=duration)
        self.assertEqual(extract_window(source, 2, 4, duration_s=2).shape, (1, 500))

    def test_gap_annotations_and_zero_duration_boundaries_are_rejected(self):
        source = recording(sine(10)[None, :])
        for interval in ([[4, 5]], [[5, 5]]):
            source["gap_intervals_s"] = interval
            with self.assertRaisesRegex(ValueError, "gap"):
                extract_window(source, 0, 10)
        source["gap_intervals_s"] = [[10, 10]]
        self.assertEqual(extract_window(source, 0, 10).shape, (1, 2500))
        source["gap_intervals_s"] = [[5, 4]]
        with self.assertRaisesRegex(ValueError, "ordered"):
            extract_window(source, 0, 10)

    def test_timestamp_gap_is_not_hidden_by_sample_count(self):
        data = sine(10)[None, :]
        times = 100.0 + np.arange(2500) / SFREQ
        times[1250:] += 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gap.npz"
            np.savez(path, data=data, sfreq_hz=SFREQ, channel_names=["Cz"], sample_times_s=times)
            source = read_recording(path)
            with self.assertRaisesRegex(ValueError, "discontinuous"):
                extract_window(source, 0, 10)
            self.assertEqual(extract_window(source, 0, 2, duration_s=2).shape, (1, 500))

    def test_flagged_bad_channel_requires_explicit_handling(self):
        source = recording(sine(10)[None, :])
        source["bad_channels"] = ["EEG0"]
        with self.assertRaisesRegex(ValueError, "flagged bad"):
            extract_window(source, 0, 10)

    def test_direct_extraction_checks_selected_samples_and_never_repairs(self):
        # Import rejects a globally nonfinite recording. Direct extraction
        # independently verifies the selected slice without rescanning/casting
        # unrelated samples on every window of a long recording.
        data = sine(10, duration=20)[None, :].astype(np.float32)
        data[0, 4000] = np.nan
        source = recording(data)
        window = extract_window(source, 0, 10)
        self.assertEqual(window.dtype, np.float64)
        np.testing.assert_array_equal(window, data[:, :2500])
        with self.assertRaisesRegex(ValueError, "selected window"):
            extract_window(source, 10, 20)
        self.assertTrue(np.isnan(source["data"][0, 4000]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nonfinite-recording.npz"
            np.savez(path, data=data, sfreq_hz=SFREQ, channel_names=["Cz"])
            with self.assertRaisesRegex(ValueError, "non-finite"):
                read_recording(path)

    def test_mne_reader_normalizes_eeg_only_and_preserves_bad_boundaries(self):
        raw = types.SimpleNamespace(
            info={"sfreq": SFREQ, "bads": []}, ch_names=["Cz", "EOG", "Pz"], first_time=2.0,
            annotations=types.SimpleNamespace(onset=[7.0], duration=[0.0], description=["BAD boundary"]),
            get_data=Mock(return_value=np.stack([sine(10), sine(20)]) * 1e-6),
        )
        reader = Mock(return_value=raw)
        fake_mne = types.SimpleNamespace(io=types.SimpleNamespace(read_raw_edf=reader),
                                         pick_types=Mock(return_value=np.array([0, 2])))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.edf"
            path.touch()
            with patch("adversarial_study.data.importlib.import_module", return_value=fake_mne):
                source = read_recording(path, "mne")
            self.assertEqual(source["channel_names"], ["Cz", "Pz"])
            self.assertEqual(source["reference"], "as_stored")
            np.testing.assert_allclose(source["data"], np.stack([sine(10), sine(20)]))
            np.testing.assert_array_equal(source["gap_intervals_s"], [[5.0, 5.0]])
            with self.assertRaisesRegex(ValueError, "gap"):
                extract_window(source, 0, 10)
            raw.get_data.assert_called_once()
            np.testing.assert_array_equal(raw.get_data.call_args.kwargs["picks"], [0, 2])


if __name__ == "__main__":
    unittest.main()
