import copy
import json
import unittest
from unittest.mock import patch

from adversarial_study.mechanisms import fit_candidate_bank, evaluate_candidate_bank


class CandidateBankTests(unittest.TestCase):
    def observations(self, prefix="dev"):
        return [{"participant_id": f"{prefix}{i}", "source_id": "A", "effective_rank": value,
                 "acf_100ms": .5, "score": .3}
                for i, value in enumerate([1.8, 1.9, 2.1, 2.2])]

    def bank(self):
        return [{"candidate_id": name, "mechanism_family": family, "parameter_values": {"gain": gain},
                 "mapping_status": "qualified", "mapping_reference": "validated-forward-map-v1",
                 "effective_rank": rank, "acf_100ms": .5, "score": .3}
                for name, family, gain, rank in [("A", "gain", 1., 2.), ("B", "gain", 2., 4.),
                                                ("C", "coupling", 1., 2.)]]

    def test_nearest_fit_and_observational_ambiguity_are_retained(self):
        artifact = fit_candidate_bank(self.observations(), self.bank(), ["effective_rank", "acf_100ms"],
                                      {"equivalence_tolerance": 0.})
        self.assertEqual(artifact["selected_candidate_id"], "A")
        self.assertEqual(artifact["equivalent_candidate_ids"], ["A", "C"])
        gain = next(f for f in artifact["family_selections"] if f["mechanism_family"] == "gain")
        self.assertEqual(gain["equivalent_candidate_ids"], ["A"])
        self.assertEqual(artifact["training_scaling"]["constant_features"], ["acf_100ms"])
        json.dumps(artifact, allow_nan=False)

    def test_score_only_and_behaviour_only_fit_are_refused(self):
        for columns in [["score"], ["gcs", "score"], ["capacity_value"]]:
            with self.assertRaisesRegex(ValueError, "neural feature"):
                fit_candidate_bank(self.observations(), self.bank(), columns)

    def test_missing_mapping_qualification_is_refused(self):
        for field in ("mapping_status", "mapping_reference"):
            bank = self.bank()
            del bank[0][field]
            with self.assertRaisesRegex(ValueError, "mapping_status"):
                fit_candidate_bank(self.observations(), bank, ["effective_rank"])

    def test_training_scaling_and_participant_weighting(self):
        observations = self.observations()
        observations += [dict(observations[0]) for _ in range(20)]
        artifact = fit_candidate_bank(observations, self.bank(), ["effective_rank"])
        self.assertAlmostEqual(artifact["training_scaling"]["feature_mean"][0], 2.)
        self.assertEqual(artifact["training_counts"]["complete_case_participants"], 4)
        self.assertEqual(artifact["training_counts"]["complete_case_rows"], 24)

    def test_protected_evaluation_does_not_refit_or_select_new_candidates(self):
        artifact = fit_candidate_bank(self.observations(), self.bank(), ["effective_rank"])
        original = copy.deepcopy(artifact)
        observations = self.observations("eval")
        for row in observations:
            row["effective_rank"] = 4.
        with patch("adversarial_study.mechanisms.fit_candidate_bank", side_effect=AssertionError("Refit")):
            result = evaluate_candidate_bank(observations, artifact)
        gain = next(f for f in result["family_evaluation"] if f["mechanism_family"] == "gain")
        self.assertEqual(gain["frozen_selected_candidate_id"], "A")
        self.assertEqual(artifact, original)
        # Development variance=.025, while evaluation variance=0. The
        # expected 4/.025 error proves evaluation did not replace the scale.
        self.assertAlmostEqual(gain["heldout_normalized_mse"], 160.)
        json.dumps(result, allow_nan=False)

    def test_overlap_is_rejected_including_missing_training_rows(self):
        training = self.observations()
        training.append({"participant_id": "exposed", "effective_rank": None})
        artifact = fit_candidate_bank(training, self.bank(), ["effective_rank"])
        self.assertIn("exposed", artifact["training_participant_ids"])
        for person in ("dev0", "exposed"):
            evaluation = self.observations("eval")
            evaluation[0]["participant_id"] = person
            evaluation[0]["effective_rank"] = None
            with self.assertRaisesRegex(ValueError, "overlaps inspected training"):
                evaluate_candidate_bank(evaluation, artifact)

    def test_extrapolation_is_reported_and_not_used_to_exclude_people(self):
        artifact = fit_candidate_bank(self.observations(), self.bank(), ["effective_rank"])
        observations = self.observations("eval")
        observations[0]["effective_rank"] = 10.
        result = evaluate_candidate_bank(observations, artifact)
        self.assertEqual(result["extrapolation"]["participants_beyond_envelope"], 2)
        self.assertEqual(result["counts"]["complete_case_participants"], 4)
        self.assertEqual(result["extrapolation"]["counts_by_feature"]["effective_rank"], 2)

    def test_json_parameter_values_and_invalid_parameters(self):
        bank = self.bank()
        bank[0]["parameter_values"] = '{"gain": 1.0}'
        artifact = fit_candidate_bank(self.observations(), bank, ["effective_rank"])
        self.assertEqual(artifact["candidate_bank"][0]["parameter_values"], {"gain": 1.})
        bank[0]["parameter_values"] = {"gain": float("nan")}
        with self.assertRaises(ValueError):
            fit_candidate_bank(self.observations(), bank, ["effective_rank"])

    def test_protected_rows_cannot_be_used_for_candidate_selection(self):
        rows = self.observations()
        rows[0]["role"] = "evaluation"
        with self.assertRaises(ValueError):
            fit_candidate_bank(rows, self.bank(), ["effective_rank"])

    def test_invalid_scaling_and_unknown_candidate_are_refused(self):
        artifact = fit_candidate_bank(self.observations(), self.bank(), ["effective_rank"])
        artifact["training_scaling"]["feature_scale"] = [0.]
        with self.assertRaises(ValueError):
            evaluate_candidate_bank(self.observations("eval"), artifact)
        artifact = fit_candidate_bank(self.observations(), self.bank(), ["effective_rank"])
        artifact["family_selections"][0]["selected_candidate_id"] = "unknown"
        with self.assertRaises(ValueError):
            evaluate_candidate_bank(self.observations("eval"), artifact)


if __name__ == "__main__":
    unittest.main()
