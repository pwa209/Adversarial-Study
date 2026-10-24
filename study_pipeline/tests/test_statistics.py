import json
import unittest

from adversarial_study.statistics import paired_contrast, randomized_contrast, simulate_precision


class ParticipantContrastTests(unittest.TestCase):
    def test_windows_and_episodes_do_not_inflate_participant_count(self):
        rows = []
        # Person A has many windows from one episode. Equal participant
        # weighting gives (.1 + .5) / 2, rather than the window-weighted effect.
        for person, difference, windows in [("A", .1, 100), ("B", .5, 1)]:
            for condition, value in [("target", difference), ("control", 0.)]:
                rows += [{"participant_id": person, "episode_id": condition,
                          "condition": condition, "outcome": value}
                         for _ in range(windows)]
        result = paired_contrast(rows, "outcome", "condition", "target", "control", 200)
        self.assertEqual(result["counts"]["informative_participants"], 2)
        self.assertEqual(result["counts"]["completed_contrast_episodes"], 4)
        self.assertAlmostEqual(result["estimate"], .3)
        self.assertEqual(len(result["participant_contrasts"]), 2)

    def test_equal_episode_weighting_within_participant(self):
        rows = [{"participant_id": "A", "episode_id": "T1", "condition": "T", "y": 2.} for _ in range(99)]
        rows += [{"participant_id": "A", "episode_id": "T2", "condition": "T", "y": 4.},
                 {"participant_id": "A", "episode_id": "C", "condition": "C", "y": 1.}]
        result = paired_contrast(rows, "y", "condition", "T", "C", 100)
        self.assertAlmostEqual(result["estimate"], 2.)
        self.assertIsNone(result["confidence_interval"])
        self.assertEqual(result["counts"]["informative_participants"], 1)

    def test_null_equivalence_uses_separate_90_percent_interval(self):
        rows = []
        for i, value in enumerate([-.02, -.01, 0., .01, .02, 0., 0., 0.]):
            rows.extend([{"participant_id": f"p{i}", "condition": "T", "y": value},
                         {"participant_id": f"p{i}", "condition": "C", "y": 0.}])
        result = paired_contrast(rows, "y", "condition", "T", "C", 2000,
                                 equivalence_margin=.1)
        self.assertAlmostEqual(result["estimate"], 0.)
        self.assertAlmostEqual(result["equivalence"]["confidence"], .9)
        self.assertEqual(result["confidence"], .95)
        self.assertTrue(result["equivalence"]["supported"])
        self.assertIn("not an exact TOST", result["equivalence"]["interpretation"])
        self.assertGreaterEqual(result["equivalence"]["interval"][0], result["confidence_interval"][0])

    def test_wide_interval_does_not_establish_equivalence(self):
        rows = []
        for person, value in [("a", -10.), ("b", 10.)]:
            rows += [{"participant_id": person, "condition": "T", "y": value},
                     {"participant_id": person, "condition": "C", "y": 0.}]
        result = paired_contrast(rows, "y", "condition", "T", "C", 200,
                                 equivalence_margin=.1)
        self.assertFalse(result["equivalence"]["supported"])

    def test_missingness_and_minimum_episode_requirement_are_reported(self):
        rows = [{"participant_id": "A", "condition": "T", "y": 1.},
                {"participant_id": "A", "condition": "C", "y": None},
                {"participant_id": "B", "condition": "T", "y": float("nan")},
                {"participant_id": "B", "condition": "C", "y": 0.},
                {"participant_id": "C", "condition": "T", "y": 4., "eligible": "false"}]
        result = paired_contrast(rows, "y", "condition", "T", "C", 100)
        self.assertIsNone(result["estimate"])
        self.assertIsNone(result["confidence_interval"])
        self.assertEqual(result["counts"]["input_participants"], 3)
        self.assertEqual(result["counts"]["noninformative_participants"], 2)
        self.assertEqual(result["exclusions"]["missing_or_nonfinite_outcome_rows"], 2)
        self.assertEqual(result["exclusions"]["ineligible_rows"], 1)
        json.dumps(result, allow_nan=False)

    def test_global_participant_is_not_counted_twice_across_sources(self):
        rows = [{"participant_id": "A", "source_id": "site1", "condition": "T", "y": 1.},
                {"participant_id": "A", "source_id": "site2", "condition": "C", "y": 0.}]
        result = paired_contrast(rows, "y", "condition", "T", "C", 100)
        self.assertEqual(result["counts"]["informative_participants"], 1)
        self.assertEqual(result["participant_contrasts"][0]["source_ids"], ["site1", "site2"])

    def test_source_singletons_cannot_manufacture_precision(self):
        rows = []
        for p in ("A", "B"):
            rows += [{"participant_id": p, "source_id": p, "condition": "T", "y": .02},
                     {"participant_id": p, "source_id": p, "condition": "C", "y": 0.}]
        result = paired_contrast(rows, "y", "condition", "T", "C", 100, equivalence_margin=.1)
        self.assertIsNone(result["confidence_interval"])
        self.assertIsNone(result["equivalence"]["supported"])


class RandomizedContrastTests(unittest.TestCase):
    def test_missing_randomized_trials_are_counted_and_no_itt_claim(self):
        rows = []
        for person in ("A", "B"):
            for arm, outcome in [("active", 1.), ("control", 0.)]:
                rows.append({"participant_id": person, "trial_id": arm,
                             "treatment": arm, "eligible": "true", "randomized": "1", "y": outcome})
        rows.append({"participant_id": "C", "trial_id": "t1", "treatment": "active",
                     "eligible": True, "randomized": True, "y": None})
        result = randomized_contrast(rows, "y", bootstrap_reps=100)
        self.assertEqual(result["randomization_accounting"]["randomized_eligible_trials"], 5)
        self.assertEqual(result["counts"]["informative_participants"], 2)
        self.assertEqual(result["randomization_accounting"]["by_arm"]["active"]["missing_outcome_trials"], 1)
        self.assertIn("not a full intention-to-treat", result["estimand"])
        self.assertAlmostEqual(result["estimate"], 1.)

    def test_duplicate_windows_do_not_create_randomized_trials(self):
        rows = [{"participant_id": "A", "trial_id": "t1", "treatment": "active",
                 "eligible": True, "randomized": True, "y": 1.} for _ in range(12)]
        result = randomized_contrast(rows, "y", bootstrap_reps=100)
        self.assertEqual(result["randomization_accounting"]["randomized_eligible_trials"], 1)
        self.assertIsNone(result["estimate"])

    def test_requires_assignment_eligibility_and_rejects_post_outcome_selection(self):
        row = {"participant_id": "A", "treatment": "active", "y": 1.}
        with self.assertRaises(ValueError):
            randomized_contrast([row], "y")
        row.update(randomized=True, eligible=True, eligibility_basis="successful_outcome")
        with self.assertRaises(ValueError):
            randomized_contrast([row], "y")

    def test_trial_cannot_have_conflicting_treatments(self):
        rows = [{"participant_id": "A", "trial_id": "t1", "treatment": arm,
                 "eligible": True, "randomized": True, "y": 1.} for arm in ("active", "control")]
        with self.assertRaises(ValueError):
            randomized_contrast(rows, "y")


class PrecisionTests(unittest.TestCase):
    def test_seeded_simulation_is_reproducible_and_synthetic(self):
        settings = {"eligible_n_grid": [60, 120], "replicates": 20, "seed": 7,
                    "informative_yield": .5, "outcome_missingness": .1}
        result = simulate_precision(settings)
        self.assertEqual(result, simulate_precision(settings))
        self.assertEqual(result["evidence_status"], "ASSUMPTIONS_ONLY_SYNTHETIC")
        self.assertEqual(len(result["scenarios"]), 2)
        self.assertGreater(result["scenarios"][1]["mean_informative_participants"],
                           result["scenarios"][0]["mean_informative_participants"])
        json.dumps(result, allow_nan=False)

    def test_zero_yield_produces_missing_inference_not_zero_width(self):
        result = simulate_precision({"eligible_n_grid": [10], "replicates": 5, "informative_yield": 0.})
        scenario = result["scenarios"][0]
        self.assertEqual(scenario["mean_informative_participants"], 0.)
        self.assertIsNone(scenario["median_confidence_interval_width"])
        self.assertIsNone(scenario["synthetic_directional_detection_rate"])


if __name__ == "__main__":
    unittest.main()
