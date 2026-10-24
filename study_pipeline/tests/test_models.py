import json
import copy
import unittest
from unittest.mock import patch

import numpy as np

from adversarial_study.models import effective_recovery, grouped_prediction, fit_prediction_model, evaluate_prediction_model


class GroupedPredictionTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [{"participant_id": f"p{i}", "baseline": float(i % 3),
                 "dynamic": float(i), "capacity": float(2 * i), "episode_id": str(j)}
                for i in range(12) for j in range(2)]

    def test_whole_participant_folds_and_dynamic_increment(self):
        result = grouped_prediction(self.rows(), "capacity", ["baseline"], ["dynamic"], folds=3, ridge=.01)
        self.assertEqual(result["counts"]["complete_case_participants"], 12)
        self.assertGreater(result["incremental_mse_improvement"], 1.)
        predicted_folds = {}
        for row in result["out_of_fold_predictions"]:
            predicted_folds.setdefault(row["participant_id"], set()).add(row["fold"])
        self.assertTrue(all(len(v) == 1 for v in predicted_folds.values()))
        for fold in result["fold_details"]:
            self.assertFalse(set(fold["training_participants"]) & set(fold["test_participants"]))

    def test_scaling_uses_training_only_and_equal_participant_weight(self):
        rows = self.rows()
        rows += [{"participant_id": "p0", "baseline": 0., "dynamic": 0., "capacity": 0.} for _ in range(50)]
        result = grouped_prediction(rows, "capacity", ["baseline"], ["dynamic"], folds=3)
        for fold in result["fold_details"]:
            training = fold["training_participants"]
            expected_dynamic_mean = np.mean([int(p[1:]) for p in training])
            self.assertAlmostEqual(fold["augmented_fit"]["feature_mean"][1], expected_dynamic_mean)
        json.dumps(result, allow_nan=False)

    def test_frozen_training_fit_is_unchanged_by_heldout_outcome(self):
        original = self.rows()
        result = grouped_prediction(original, "capacity", ["baseline"], ["dynamic"], folds=3, seed=5)
        fold = result["fold_details"][0]
        changed = [dict(row) for row in original]
        for row in changed:
            if row["participant_id"] in fold["test_participants"]:
                row["capacity"] = 100000.
                row["dynamic"] += 10000.
        updated = grouped_prediction(changed, "capacity", ["baseline"], ["dynamic"], folds=3, seed=5)
        self.assertEqual(fold["augmented_fit"], updated["fold_details"][0]["augmented_fit"])

    def test_rejects_group_leakage_and_undersampled_folds(self):
        with self.assertRaises(ValueError):
            grouped_prediction(self.rows(), "capacity", ["participant_id"], ["dynamic"], folds=3)
        with self.assertRaises(ValueError):
            grouped_prediction(self.rows()[:6], "capacity", ["baseline"], ["dynamic"], folds=3)
        with self.assertRaises(ValueError):
            grouped_prediction(self.rows(), "capacity", ["baseline"], ["capacity"], folds=3)

    def test_detector_labels_cannot_become_awareness_labels(self):
        rows = self.rows()
        rows[0]["outcome_label_source"] = "detector_selected"
        with self.assertRaises(ValueError):
            grouped_prediction(rows, "capacity", ["baseline"], ["dynamic"], folds=3)

    def test_complete_case_matching_and_missingness(self):
        rows = self.rows()
        rows[0]["dynamic"] = None
        rows[1]["capacity"] = None
        result = grouped_prediction(rows, "capacity", ["baseline"], ["dynamic"], folds=3)
        self.assertEqual(result["counts"]["complete_case_rows"], 22)
        self.assertEqual(result["counts"]["complete_case_participants"], 11)
        self.assertEqual(result["exclusions"]["incomplete_outcome_or_feature_rows"], 2)

    def test_protected_and_development_may_not_mix(self):
        rows = self.rows()
        rows[0]["partition"] = "protected"
        rows[1]["partition"] = "development"
        with self.assertRaises(ValueError):
            grouped_prediction(rows, "capacity", ["baseline"], ["dynamic"], folds=3)


class EffectiveRecoveryTests(unittest.TestCase):
    def test_noiseless_joint_parameter_recovery_and_holdout(self):
        result = effective_recovery({"replicates": 2, "train_steps": 180, "holdout_steps": 80,
                                     "process_noise_sd": 0., "seed": 42})
        self.assertEqual(result["evidence_status"], "ASSUMPTIONS_ONLY_SYNTHETIC")
        for report in result["recovery"].values():
            self.assertLess(report["rmse"], 1e-9)
        joint = next(r for r in result["family_summary"] if r["family"] == "joint_effective")
        self.assertLess(joint["mean_heldout_noiseless_trajectory_mse"], 1e-20)
        self.assertIn("not NFTsim", result["interpretation"])
        json.dumps(result, allow_nan=False)

    def test_unstable_effective_models_are_rejected(self):
        with self.assertRaises(ValueError):
            effective_recovery({"true_parameters": {"timescale": 10., "coupling": .5}, "replicates": 1})

    def test_seed_reproduces_recovery(self):
        settings = {"seed": 17, "replicates": 3, "train_steps": 100, "holdout_steps": 40}
        self.assertEqual(effective_recovery(settings), effective_recovery(settings))


class FrozenPredictionTests(unittest.TestCase):
    def training_rows(self):
        return [{"participant_id": f"train{i}", "source_id": "siteA", "cohort_id": "dev",
                 "role": "development", "baseline": 1., "dynamic": float(i),
                 "capacity": float(2 * i), "episode_id": f"e{i}_{j}"}
                for i in range(10) for j in range(2)]

    def evaluation_rows(self):
        return [{"participant_id": f"eval{i}", "source_id": "siteA", "cohort_id": "new",
                 "role": "evaluation", "baseline": 1., "dynamic": float(i + 1),
                 "capacity": float(2 * (i + 1)), "episode_id": f"v{i}_{j}"}
                for i in range(6) for j in range(2)]

    def artifact(self):
        return fit_prediction_model(self.training_rows(), "capacity", ["baseline"], ["dynamic"], ridge=.01)

    def test_transport_never_calls_fit_and_leaves_artifact_unchanged(self):
        artifact = self.artifact()
        original = copy.deepcopy(artifact)
        with patch("adversarial_study.models._fit_ridge", side_effect=AssertionError("Evaluation attempted refitting")):
            result = evaluate_prediction_model(self.evaluation_rows(), artifact)
        self.assertEqual(artifact, original)
        self.assertLess(result["augmented_mse"], .01)
        self.assertGreater(result["incremental_mse_improvement"], 1.)
        self.assertEqual(result["uncertainty"]["inference_status"], "available")
        self.assertEqual(result["source_ids_shared_with_training"], ["siteA"])
        json.dumps(result, allow_nan=False)
        json.dumps(artifact, allow_nan=False)
        self.assertNotIn("records", artifact)

    def test_any_training_participant_overlap_is_rejected_even_if_outcome_missing(self):
        artifact = self.artifact()
        rows = self.evaluation_rows()
        rows[0]["participant_id"] = "train0"
        rows[0]["capacity"] = None
        rows[0]["eligible"] = False
        with self.assertRaisesRegex(ValueError, "overlaps inspected training"):
            evaluate_prediction_model(rows, artifact)

    def test_excluded_training_people_still_have_exposure_ids(self):
        rows = self.training_rows()
        rows.append({"participant_id": "previously_inspected", "source_id": "siteB", "baseline": 1.,
                     "dynamic": 1., "capacity": None})
        artifact = fit_prediction_model(rows, "capacity", ["baseline"], ["dynamic"])
        self.assertIn("previously_inspected", artifact["training_participant_ids"])
        self.assertNotIn("previously_inspected", artifact["fitted_participant_ids"])
        evaluation = self.evaluation_rows()
        evaluation[0]["participant_id"] = "previously_inspected"
        with self.assertRaises(ValueError):
            evaluate_prediction_model(evaluation, artifact)

    def test_feature_schema_must_exist_and_partial_missingness_is_counted(self):
        artifact = self.artifact()
        rows = self.evaluation_rows()
        for row in rows:
            del row["dynamic"]
        with self.assertRaisesRegex(ValueError, "columns unavailable"):
            evaluate_prediction_model(rows, artifact)
        rows = self.evaluation_rows()
        rows[0]["dynamic"] = None
        result = evaluate_prediction_model(rows, artifact)
        self.assertEqual(result["counts"]["complete_case_rows"], 11)
        self.assertEqual(result["missing_by_column"]["dynamic"], 1)

    def test_constant_training_feature_does_not_use_evaluation_scale(self):
        artifact = self.artifact()
        self.assertEqual(artifact["baseline_fit"]["feature_scale"], [1.])
        rows = self.evaluation_rows()
        result = evaluate_prediction_model(rows, artifact)
        rows[1]["dynamic"] = 1e5
        updated = evaluate_prediction_model(rows, artifact)
        self.assertEqual(result["predictions"][0], updated["predictions"][0])
        self.assertEqual(result["artifact_content_sha256"], updated["artifact_content_sha256"])
        rows[0]["baseline"] = 1e5
        constant_changed = evaluate_prediction_model(rows, artifact)
        self.assertAlmostEqual(constant_changed["predictions"][0]["baseline_prediction"],
                               result["predictions"][0]["baseline_prediction"])

    def test_artifact_rejects_nonfinite_wrong_shape_and_zero_scales(self):
        for field, bad in [("feature_mean", [float("nan")]), ("feature_scale", [0.]),
                           ("coefficient_standardized", [1., 2.])]:
            artifact = self.artifact()
            artifact["baseline_fit"][field] = bad
            with self.assertRaises(ValueError):
                evaluate_prediction_model(self.evaluation_rows(), artifact)

    def test_protected_rows_cannot_be_used_for_training(self):
        with self.assertRaisesRegex(ValueError, "cannot use protected"):
            fit_prediction_model(self.evaluation_rows(), "capacity", ["baseline"], ["dynamic"])

    def test_one_evaluation_participant_retains_estimate_without_false_precision(self):
        result = evaluate_prediction_model(self.evaluation_rows()[:2], self.artifact())
        self.assertIsNotNone(result["baseline_mse"])
        self.assertIsNone(result["uncertainty"]["intervals"]["baseline_mse"])


if __name__ == "__main__":
    unittest.main()
