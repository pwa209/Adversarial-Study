"""Known metadata/native-kernel fixtures validate operations, not human findings."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np

from adversarial_study import core, proposal_mechanisms as pm


class ProposalMechanismTests(unittest.TestCase):
    def cfg(self):
        return pm.DEFAULTS | {"recovery_ridge_grid": [1e-6, .01], "recovery_calibration_units": 8,
                              "recovery_validation_units": 4, "bootstrap_samples": 30}

    def test_candidate_budget_covers_late_sources_accounts_and_inputs(self):
        proposals = []
        for source in ("a", "z"):
            for account in ("combined", "history"):
                for amplitude in (1, 100):
                    for n in range(4):
                        proposals.append({"candidate_id": f"{source}-{account}-{amplitude}-{n}",
                                          "source": source, "account": account, "input": amplitude,
                                          "score": 1000-n, "outcome": "unopened"})
        admitted, census = pm.balanced_budget(proposals, 8, 17, ("source", "account", "input"))
        self.assertEqual(len(admitted), 8)
        self.assertEqual(len({(r["source"], r["account"], r["input"]) for r in admitted}), 8)
        changed = [r | {"score": -100000, "outcome": "changed"} for r in reversed(proposals)]
        again, _ = pm.balanced_budget(changed, 8, 17, ("source", "account", "input"))
        self.assertEqual([r["candidate_id"] for r in admitted], [r["candidate_id"] for r in again])
        self.assertTrue(all(r["admitted"] == 1 and r["excluded_metadata_budget"] == 3 for r in census))

    def search_fixture(self):
        candidates = []; bank = []
        for identity, score, dynamic, conventional in (("a", .2, 0., 1.), ("b", .21, 1., 1.1), ("c", .9, 2., 1.)):
            for pulse in (False, True):
                row = {"simulation_id": identity+("pulse" if pulse else "control"), "score": score+(.2 if pulse else 0),
                       "saturated_fraction": 0., "feature_acf_100ms": dynamic, "feature_acf_500ms": dynamic,
                       "feature_spectral_centroid_hz": conventional, "feature_spectral_entropy": conventional}
                bank.append(row)
            candidates.append({"candidate_id": identity, "source_model_key": "source", "variant": "actual_fixture",
                               "account_class": identity, "control_simulation_id": identity+"control", "pulse_simulation_id": identity+"pulse",
                               "latent_response_path_rms_v1": dynamic, "input_amplitude_model_units": 1., "input_width_s": .02,
                               "control_pulse_geometry_distance": .01, "status": "retained_supported_computational_challenge"})
        scales = {"source": dict.fromkeys(pm.guided.TEMPORAL_FEATURES, 1.)}
        return candidates, bank, scales

    def test_original_families_require_score_feature_dynamics_and_clipping_checks(self):
        candidates, bank, scales = self.search_fixture()
        families, _ = pm.search_families(candidates, bank, scales, self.cfg())
        matched = [r for r in families if r["family"] == "matched_clinical_score_conventional_features_different_dynamics"]
        self.assertEqual([(r["candidate_a"], r["candidate_b"]) for r in matched], [("a", "b")])
        self.assertEqual(len([r for r in families if r["family"] == "changed_score_preserved_specified_property"]), 3)
        bank[2]["saturated_fraction"] = 1
        changed, dispositions = pm.search_families(candidates, bank, scales, self.cfg())
        self.assertFalse(any(r["family"] == "matched_clinical_score_conventional_features_different_dynamics" for r in changed))
        self.assertTrue(any(r["status"] == "retained_unmatched_or_unsupported_account_disagreement" for r in dispositions))
        # Missing clipping is not silently treated as unsaturated.
        bank[2].pop("saturated_fraction")
        missing, _ = pm.search_families(candidates, bank, scales, self.cfg())
        self.assertFalse(any(r["family"] == "matched_clinical_score_conventional_features_different_dynamics" for r in missing))

    def test_no_effect_search_remains_valid_and_retains_dispositions(self):
        candidates, bank, scales = self.search_fixture()
        for row in bank: row["score"] = .2; row["feature_acf_100ms"] = row["feature_acf_500ms"] = 0.
        for row in candidates: row["latent_response_path_rms_v1"] = 0.; row["account_class"] = "same"
        families, dispositions = pm.search_families(candidates, bank, scales, self.cfg())
        self.assertEqual(families, [])
        self.assertEqual(len(dispositions), 3)

    def kernels(self):
        t = np.linspace(0, 10, 160)
        return {"gain\0g0": np.column_stack((np.sin(t), np.cos(t))),
                "gain\0g1": np.column_stack((np.sin(1.7*t), np.cos(1.7*t))),
                "history\0h0": np.column_stack((np.sin(2.7*t), np.cos(2.7*t)))}

    def test_known_native_configuration_recovery_refits_observation_map(self):
        kernels = self.kernels(); W = np.array([[1.3, -.2], [.4, .9]]); bias = np.array([.2, -.1])
        result = pm.recover_generated(kernels, ["gain\0g0", "gain\0g1"], W, bias, self.cfg(), 9)
        self.assertTrue(all(r["correct_configuration"] for r in result))
        self.assertTrue(all(r["later_prediction_standardized_mse"] < 1e-8 for r in result))
        self.assertTrue(all(r["refitted_map_rmse"] < 1e-4 for r in result))
        self.assertTrue(all(r["physiological_recovery_established"] is False for r in result))
        self.assertTrue(all(r["calibration_configuration_count"] == 2 for r in result))

    def test_known_observational_ambiguity_is_retained_as_class(self):
        kernels = self.kernels(); kernels["gain\0g0_clone"] = kernels["gain\0g0"].copy()
        result = pm.recover_generated(kernels, ["gain\0g0", "gain\0g1"], np.eye(2), np.zeros(2), self.cfg(), 4)
        first = next(r for r in result if r["known_generator_id"] == "gain\0g0")
        self.assertEqual(first["equivalence_class_size"], 2)
        self.assertFalse(first["unique_configuration_identifiable_under_generating_map"])
        self.assertTrue(first["recovered_in_equivalence_class"])
        self.assertEqual(set(json.loads(first["generating_equivalence_class"])), {"gain\0g0", "gain\0g0_clone"})

    def test_noise_misspecification_changes_recovery_errors_without_success_gate(self):
        kernels = self.kernels(); args = (kernels, ["gain\0g0", "gain\0g1"], np.eye(2), np.zeros(2), self.cfg(), 5)
        clean = pm.recover_generated(*args)
        noisy = pm.recover_generated(*args, noise_level=.4, noise_model="colored_ar1", misspecification=.5)
        self.assertGreater(np.mean([r["later_prediction_standardized_mse"] for r in noisy]), np.mean([r["later_prediction_standardized_mse"] for r in clean])+.01)
        self.assertTrue(all(r["misspecification_level"] == .5 for r in noisy))
        self.assertTrue(all(r["evidence_status"] == "model_generated_native_inverse_diagnostic" for r in noisy))

    def discrepancy_fixture(self):
        t = np.linspace(0, 10, 30); a = np.column_stack((np.sin(t), np.cos(t))); b = a*np.linspace(1, 2, 30)[:, None]
        states = [{"participant_id": "person", "source_id": "source", "episode_id": e, "state_token": e,
                   "source_model_key": "source_model", "split": "test", "status": "included"} for e in ("a", "b")]
        pred = [{"map_key": "map", "episode_id": e, "source_model_key": "source_model", "account_class": "gain", "participant_id": "person", "conditioning_states": 10} for e in ("a", "b")]
        pair = {"pair_id": "real_corrected_pair", "family": "same_score_different_observed_reports", "participant_id": "person", "source_id": "source", "episode_a": "a", "episode_b": "b"}
        return states, pred, pair, {"a__state": a, "b__state": b}, {"map__a__prediction": a, "map__b__prediction": b}

    def test_discrepancy_prediction_links_exact_corrected_pairs_and_held_people(self):
        states, predictions, pair, actual, estimated = self.discrepancy_fixture()
        output = pm.discrepancy_evaluation([pair], states, predictions, actual, estimated)
        self.assertEqual(output[0]["pair_id"], "real_corrected_pair")
        self.assertEqual(output[0]["split"], "test")
        self.assertAlmostEqual(output[0]["geometry_contrast_mse"], 0.)
        self.assertLess(output[0]["loss_minus_null"], 0.)
        # Fitted early targets are excluded from the explanatory prediction loss.
        estimated["map__a__prediction"] = estimated["map__a__prediction"].copy()
        estimated["map__a__prediction"][:10] += 100
        unchanged = pm.discrepancy_evaluation([pair], states, predictions, actual, estimated)
        self.assertEqual(output, unchanged)
        missing = pair | {"episode_b": "absent"}
        self.assertEqual(pm.discrepancy_evaluation([missing], states, predictions, actual, estimated)[0]["status"], "missing_actual_common_state_domain")
        for row in states: row["split"] = "train"
        self.assertEqual(pm.discrepancy_evaluation([pair], states, predictions, actual, estimated)[0]["status"], "training_case_not_held_person_explanation")

    def baseline_fixture(self):
        state = []; pairs = []
        for p, role in (("p0", "train"), ("p1", "train"), ("p2", "train"), ("p3", "validation"), ("p4", "test"), ("p5", "test")):
            n = int(p[-1])
            for e, offset in (("a", 0), ("b", .02*n)):
                state.append({"participant_id": p, "source_id": "source", "episode_id": p+e, "score": .2+offset,
                              "preclip_gcs": 5.+12*offset, "saturated_fraction": 0.})
            pairs.append({"participant_id": p, "source_id": "source", "source_model_key": "source_model", "split": role,
                          "episode_a": p+"a", "episode_b": p+"b", "activations": {"actual_layer": .01*n},
                          "targets": {"confidence": ("numeric", .1*n, "original_outcome_difference", n, n+.1*n)}})
        return state, pairs

    def test_original_score_baseline_uses_identical_cases_and_train_validation_selection(self):
        state, pairs = self.baseline_fixture()
        predictions, selected, metrics, census = pm.original_score_baselines(pairs, state, ["actual_layer"], self.cfg())
        self.assertEqual({r["predictor"] for r in predictions}, {"frozen_clinical_score", "frozen_score_preclip_saturation", "frozen_score_plus_adapted_activation", "frozen_controls_plus_adapted_activation"})
        self.assertTrue(all(r["test_used_for_selection"] is False for r in selected))
        self.assertEqual(len(predictions), 8)
        pairs[-1]["targets"]["confidence"] = ("numeric", 1e6, "original_outcome_difference", 0, 1e6)
        again = pm.original_score_baselines(pairs, state, ["actual_layer"], self.cfg())
        self.assertEqual(selected, again[1])
        self.assertNotEqual(predictions, again[0])
        self.assertTrue(any(r.get("comparison") == "paired_loss_minus_frozen_clinical_score" for r in metrics))
        self.assertTrue(all(r["status"] == "common_case_included" for r in census))

    def test_missing_preclip_not_imputed_and_common_case_exclusion_explicit(self):
        state, pairs = self.baseline_fixture(); state[-1]["preclip_gcs"] = ""
        predictions, _, metrics, census = pm.original_score_baselines(pairs, state, ["actual_layer"], self.cfg())
        self.assertEqual({r["predictor"] for r in predictions}, {"frozen_clinical_score", "frozen_score_plus_adapted_activation"})
        self.assertEqual(len(predictions), 4)  # Score-only comparisons remain available on the same two test people.
        self.assertEqual(census[-1]["missing_fields"], "preclip_gcs")
        self.assertEqual(census[-1]["status"], "excluded_missing_original_controls")
        self.assertTrue(any(r.get("status") == "insufficient_common_case_people" for r in metrics))

    def test_baseline_cannot_reintroduce_missing_reports_as_observed_categories(self):
        pair = {"participant_id": "p", "source_id": "s", "episode_a": "a", "episode_b": "b",
                "targets": {"report_category": ("nominal", '["ON","missing_original_response"]', "original", "ON", "missing_original_response"),
                            "confidence": ("numeric", .2, "original", 1, 1.2)}}
        observed = {("p", "s", "a"): {"report_eligibility": "observed_original_report"},
                    ("p", "s", "b"): {"report_eligibility": "missing_original_response"}}
        corrected = pm.apply_report_eligibility([pair], observed)[0]
        self.assertIsNone(corrected["targets"]["report_category"][1])
        self.assertEqual(corrected["targets"]["confidence"], pair["targets"]["confidence"])
        self.assertIsNotNone(pair["targets"]["report_category"][1])  # Original parent remains immutable.

    def test_native_file_map_artifacts_feed_balanced_candidates_and_inverse_recovery(self):
        from test_manifold_mechanisms import ManifoldMechanismTests
        helper = ManifoldMechanismTests()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent if os.name == "nt" else None) as folder:
            fixture, directory, _ = helper.run_fixture(Path(folder))
            states, fit, rep, bank, rows, _, _, _ = fixture
            maps, mmcfg = pm.load_maps(directory)
            traces, pairs, rejected = pm.mm._native_bank(bank)
            self.assertEqual(rejected, [])
            cfg = self.cfg() | {"candidate_budget": 4, "recovery_case_budget": 2,
                               "noise_levels": [0., .1], "noise_models": ["white"],
                               "misspecification_levels": [0.], "recovery_replicates": 1}
            with np.load(states/"states.npz", allow_pickle=False) as archive:
                candidates, census, arrays = pm.balanced_candidates(maps, traces, pairs, rows, archive, mmcfg, cfg)
            self.assertEqual(len(candidates), 4)
            self.assertEqual({r["account_class"] for r in candidates}, {"gain", "coupling"})
            self.assertTrue(all(r["control_config_sha256"] and r["native_pulse_output_sha256"] for r in candidates))
            self.assertEqual(len(arrays), 8)
            results, recovery_census = pm.native_recovery(maps, traces, pairs, mmcfg, cfg)
            self.assertEqual(len(results), 8)  # Two sources/accounts x two actual observation types x two noise levels.
            self.assertEqual({r["observation"] for r in results}, {"native_control_trajectory", "native_pulse_minus_control"})
            self.assertTrue(all(r["source_config_sha256"] for r in results))
            self.assertTrue(all(r["physiological_recovery_established"] is False for r in results))
            self.assertEqual(sum(r["admitted"] for r in recovery_census), 2)

    def test_output_hash_verifier_rejects_replaced_artifact(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent if os.name == "nt" else None) as folder:
            root = Path(folder); output = root/"analysis.json"; core.write_json(output, {"value": 1})
            receipt = {"status": "proposal_mechanisms_completed", "source_code_sha256": core.sha256_file(pm.__file__),
                       "inputs": [], "outputs": {"analysis.json": core.sha256_file(output)}}
            core.write_json(root/"receipt.json", receipt); pm.verify_outputs(root)
            core.write_json(output, {"value": 2})
            with self.assertRaisesRegex(core.StudyError, "output changed"): pm.verify_outputs(root)

    def test_end_to_end_additive_output_preserves_parents_and_handles_no_eligible_effect(self):
        from test_manifold_mechanisms import ManifoldMechanismTests
        helper = ManifoldMechanismTests()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent if os.name == "nt" else None) as folder:
            root = Path(folder); fixture, bridge, _ = helper.run_fixture(root)
            states, fit, rep, bank, rows, state_receipt, rep_receipt, _ = fixture
            clinical = root/"clinical"; clinical.mkdir()
            core.write_json(clinical/"training_receipt.json", {"split": {"fixture_only": True}})
            original_config = root/"original.json"; core.write_json(original_config, {"seed": 1})
            corrected = root/"corrected"; corrected.mkdir()
            core.write_json(corrected/"receipt.json", {"status": "explicit_mock_fixture"})
            core.write_csv(corrected/"natural_pairs.csv", [{"pair_id": "actual_source_pair_fixture", "family": "same_score_different_observed_reports",
                           "participant_id": "p4", "source_id": "synthetic_fixture_only", "episode_a": "episode_p4_e0", "episode_b": "episode_p4_e1"}])
            core.write_csv(corrected/"observations.csv", rows)
            config = root/"additional.json"
            core.write_json(config, self.cfg() | {"require_scheduled_execution": False, "candidate_budget": 4,
                            "recovery_case_budget": 2, "recovery_replicates": 1, "noise_levels": [0.],
                            "noise_models": ["white"], "misspecification_levels": [0.]})
            roles = {r["participant_id"]: r["split"] for r in rows}
            rep_receipt = rep_receipt | {"input_paths": {"states": str(states), "clinical_trained": str(clinical),
                                                        "config": str(original_config), "adapted_geometry": "explicit_mock"}}
            before = {p: core.sha256_file(p) for directory in (states, fit, rep, bridge) for p in directory.rglob("*") if p.is_file()}
            with ExitStack() as stack:
                stack.enter_context(patch("adversarial_study.neural_state_space.verify_state_space", return_value=state_receipt))
                stack.enter_context(patch.object(pm.rb, "verify_representation", return_value=rep_receipt))
                stack.enter_context(patch.object(pm.mm, "verify_bridge", return_value={}))
                stack.enter_context(patch("adversarial_study.human_generative_adaptation.verify_fit", return_value=({}, bank, {"fixture_only": True})))
                stack.enter_context(patch("adversarial_study.proposal_discovery.verify_outputs", return_value={}))
                stack.enter_context(patch("native_tools.human_finetune.validate_split", return_value=roles))
                stack.enter_context(patch.object(pm.rb, "_pair_data", return_value=([], [], {}, ["fixture_layer"])))
                receipt = pm.run(states, rep, bridge, fit, [corrected], config, root/"additional")
            self.assertEqual(receipt["status"], "proposal_mechanisms_completed")
            self.assertEqual(before, {p: core.sha256_file(p) for p in before})
            pm.verify_outputs(root/"additional")
            self.assertEqual(core.read_json(root/"additional/strengthened_computational_freeze.json")["candidate_families"], {})
            self.assertFalse(core.read_json(root/"additional/strengthened_computational_freeze.json")["human_intervention_qualified"])
            evaluated = core.read_csv(root/"additional/discrepancy_evaluation.csv")
            self.assertTrue(any(r["status"] == "development_discrepancy_prediction_evaluated" for r in evaluated))


if __name__ == "__main__": unittest.main()
