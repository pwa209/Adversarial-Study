import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("human_spatial", Path(__file__).with_name("human_spatial.py"))
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
AUTHOR_COORDINATES = Path(__file__).parents[3]/"source_records"/"extensions"/"dream_tononi_HydroCelGSN256v10.sfp"


def author_coordinates():
    # In an immutable release this exact author file is shipped beside the code.
    candidates = (Path(__file__).with_name("electrode_coordinates.sfp"), AUTHOR_COORDINATES)
    return next(path for path in candidates if path.is_file())


def cases():
    rows = []
    for role, people in (("train", range(4)), ("validation", range(4, 6)), ("test", range(6, 9))):
        for person in people:
            for episode, label in enumerate(("experience", "no_experience", "experience_without_recall")):
                rows.append({"source_id": "dream_tononi_serial", "participant_id": str(person), "episode_id": f"{person}_{episode}",
                             "sleep_stage": "N2", "report_category": label, "split": role,
                             "score": .4+.01*person+.01*episode, "global_log_delta": 1+.02*person+.01*episode,
                             "posterior_minus_global_delta": .08*(episode-1)+.001*person,
                             "global_log_high_frequency": .3+.01*person,
                             "posterior_minus_global_high_frequency": .02*episode})
    return rows


class HumanSpatialTests(unittest.TestCase):
    def test_actual_author_coordinates_define_disjoint_fixed_rois(self):
        roi = h.sensor_rois(author_coordinates())
        self.assertGreater(len(roi["global"]), 100)
        self.assertGreater(len(roi["posterior"]), 20)
        self.assertGreater(len(roi["frontal"]), 20)
        self.assertFalse(set(roi["posterior"]) & set(roi["frontal"]))
        self.assertTrue(set(roi["posterior"]) <= set(roi["global"]))

    def test_spectral_estimator_recovers_known_physical_power(self):
        fs = 200
        time = np.arange(10*fs)/fs
        signal = 10*np.sin(2*np.pi*2*time)+4*np.sin(2*np.pi*30*time)
        power = h.bandpowers(signal[None, :], fs)
        self.assertAlmostEqual(float(power["delta"][0]), 50, places=2)
        self.assertAlmostEqual(float(power["high_frequency"][0]), 8, places=2)

    def test_common_average_removes_shared_reference_signal(self):
        rois = h.sensor_rois(author_coordinates())
        labels = sorted(set().union(*rois.values()))
        rng = np.random.default_rng(3)
        data = rng.normal(size=(len(labels), 2000))
        common = 30*np.sin(np.arange(2000)*.05)
        a = h.spatial_features(data, 200, labels, labels, set(), rois, "common_average")
        b = h.spatial_features(data+common, 200, labels, labels, set(), rois, "common_average")
        self.assertAlmostEqual(a["posterior_minus_global_delta"], b["posterior_minus_global_delta"], places=10)

    def test_roi_cannot_be_substituted_by_32_pca_montage(self):
        rois = h.sensor_rois(author_coordinates())
        labels = sorted(set().union(*rois.values()))
        with self.assertRaisesRegex(ValueError, "coverage"):
            h.spatial_features(np.ones((len(labels), 1000)), 100, labels, labels[:32], set(), rois, "common_average")

    def test_exact_window_rejects_offgrid_and_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"recording.npz"
            np.savez(path, data=np.ones((2, 1200)), sfreq_hz=100., channel_names=np.array(["a", "b"]), units=np.array("uV"))
            window, fs, names, bad, reference = h.window_from_npz(path, 1, 11)
            self.assertEqual(window.shape, (2, 1000))
            with self.assertRaisesRegex(ValueError, "Off-grid"):
                h.window_from_npz(path, .001, 10.001)
            np.savez(path, data=np.ones((2, 1200)), sfreq_hz=100., channel_names=np.array(["a", "b"]), gap_intervals_s=np.array([[5., 5.]]))
            with self.assertRaisesRegex(ValueError, "gap/boundary"):
                h.window_from_npz(path, 1, 11)

    def test_source_clock_and_units_are_not_inferred(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"recording.npz"
            np.savez(path, data=np.ones((2, 1000)), sfreq_hz=100., channel_names=np.array(["a", "b"]), units=np.array("V"))
            with self.assertRaisesRegex(ValueError, "units"):
                h.window_from_npz(path, 0, 10)
            times = np.arange(1000)/100
            times[300:] += .01
            np.savez(path, data=np.ones((2, 1000)), sfreq_hz=100., channel_names=np.array(["a", "b"]), sample_times_s=times)
            with self.assertRaisesRegex(ValueError, "clock"):
                h.window_from_npz(path, 0, 10)

    def test_test_labels_never_change_selected_training_models(self):
        original = cases()
        _, _, models_a = h.prediction_comparison(original, "experience_presence", h.PANELS[0], 12)
        changed = copy.deepcopy(original)
        for row in changed:
            if row["split"] == "test":
                row["report_category"] = "experience" if row["report_category"] == "no_experience" else "no_experience"
                row["posterior_minus_global_delta"] *= 100
        _, _, models_b = h.prediction_comparison(changed, "experience_presence", h.PANELS[0], 12)
        self.assertEqual(models_a, models_b)
        self.assertTrue(models_a)
        self.assertFalse(set(models_a["spatial"]["training_people"]) & {"6", "7", "8"})

    def test_matching_does_not_select_on_target_spatial_measure(self):
        original = cases()
        _, pairs_a = h.matched_contrast(original, "experience_presence", h.PANELS[0], 3)
        changed = copy.deepcopy(original)
        for index, row in enumerate(changed):
            row["posterior_minus_global_delta"] = 1000-index**2
        _, pairs_b = h.matched_contrast(changed, "experience_presence", h.PANELS[0], 3)
        identities = lambda rows: [(r["positive_episode"], r["negative_episode"]) for r in rows]
        self.assertEqual(identities(pairs_a), identities(pairs_b))
        self.assertTrue(pairs_a)
        used = [(p["participant_id"], e) for p in pairs_a for e in (p["positive_episode"], p["negative_episode"])]
        self.assertEqual(len(used), len(set(used)))

    def test_report_recall_constructs_remain_distinct(self):
        row = {"report_category": "experience_without_recall"}
        self.assertEqual(h.outcome(row, "experience_presence"), 1)
        self.assertEqual(h.outcome(row, "content_recall_given_experience"), 0)
        row["report_category"] = "no_experience"
        self.assertIsNone(h.outcome(row, "content_recall_given_experience"))

    def test_bootstrap_preserves_small_person_count_and_coarse_reference(self):
        result = h.inference([1, 1, 1, 1, 1], 4, 10000)
        self.assertEqual(result["people"], 5)
        self.assertEqual(result["p_signflip"], .0625)
        self.assertEqual(result["ci95"], [1, 1])
        self.assertEqual(result["bootstrap_unit"], "participant")
        # Ten thousand bootstrap samples cannot create p < .05 for five same-sign people.

    def test_holm_counts_entire_declared_estimable_family(self):
        records = [{"p_signflip": .01}, {"p_signflip": .03}, {"p_signflip": .9}, {"p_signflip": None}]
        h.holm(records)
        self.assertEqual(records[0]["p_holm"], .04)
        self.assertEqual(records[1]["p_holm"], .09)
        self.assertIsNone(records[3]["p_holm"])

    def test_role_overlap_and_unpinned_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            h.fixed_roles({"train_participant_ids": ["x"], "validation_participant_ids": [], "test_participant_ids": ["x"]})
        with self.assertRaisesRegex(ValueError, "SHA256"):
            h.checked({"path": "whatever"})

    def test_end_to_end_source_hash_closure_and_original_test_roles(self):
        """Portable fixture checks custody/roles, not an expected scientific effect."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations, trajectories, records, raw_inputs = [], [], [], []
            rng = np.random.default_rng(29)
            labels = [f"Chan {n}" for n in range(1, 258)]
            for row in cases():
                record_id = row["episode_id"]
                path = root/(record_id+".npz")
                np.savez(path, data=rng.normal(size=(257, 1000)).astype("float32"), sfreq_hz=100.,
                         channel_names=np.asarray(labels), units=np.array("uV"), reference=np.array("fixture_reference"))
                record = {"recording_id": record_id, "path": str(path), "sha256": h.sha(path)}
                records.append(record)
                raw_inputs.append({"path": str(path), "sha256": record["sha256"]})
                original = {**{k: row[k] for k in ("source_id", "participant_id", "episode_id", "sleep_stage", "report_category", "score")},
                            "recording_id": record_id, "start_s": "0.0", "stop_s": "10.0", "saturation_status": "no_saturated_channels"}
                observations.append(original)
                trajectories.append({**original, "measurement_status": "actual_multichannel_eeg", "score_origin": "native_frozen_v2",
                                     "synthetic": "False", "trajectory_channel_names": ";".join(labels)})
            paths = {key: root/(key+".json") for key in h.ROLES}
            paths["dream_observations"] = root/"observations.csv"
            paths["dream_trajectories"] = root/"trajectories.csv"
            paths["electrode_coordinates"] = author_coordinates()
            h.write_csv(paths["dream_observations"], observations)
            h.write_csv(paths["dream_trajectories"], trajectories)
            h.write_json(paths["prepared_recordings"], {"recordings": records})
            h.write_json(paths["state_split"], {role+"_participant_ids": sorted({r["participant_id"] for r in cases() if r["split"] == role})
                                              for role in ("train", "validation", "test")})
            h.write_json(paths["state_receipt"], {"status": "neural_state_space_completed",
                                                "inputs": raw_inputs+[{"path": str(paths["prepared_recordings"]), "sha256": h.sha(paths["prepared_recordings"])}]})
            inputs = {key: {"path": str(path), "sha256": h.sha(path)} for key, path in paths.items()}
            probe = h.qualification_probe(inputs)
            # Qualification must survive the standard strict JSON boundary
            # without a NumPy-specific encoder masking an unsupported scalar.
            self.assertEqual(json.loads(json.dumps(probe, allow_nan=False)), probe)
            self.assertFalse(probe["report_effect_fitted_or_tested"])
            self.assertEqual(probe["window_shape"], [257, 1000])
            receipt = h.run(inputs, root/"analysis", 23)
            self.assertEqual(json.loads(json.dumps(receipt, allow_nan=False)), receipt)
            summary = json.loads((root/"analysis"/"summary.json").read_text())
            self.assertEqual(summary["eligible_episodes"], 27)
            self.assertEqual(summary["eligible_people"], 9)
            self.assertEqual(len(summary["comparisons"]), 14)
            predictions = h.read_csv(root/"analysis"/"held_person_predictions.csv")
            self.assertTrue(predictions)
            self.assertEqual({r["participant_id"] for r in predictions}, {"6", "7", "8"})
            self.assertEqual(len(receipt["raw_recordings"]), 27)
            self.assertEqual(receipt["status"], "human_spatial_completed")
            # A changed genuine source cannot be silently reinterpreted as a QC exclusion.
            with Path(records[0]["path"]).open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash/binding mismatch"):
                h.qualification_probe(inputs)


if __name__ == "__main__":
    unittest.main()
