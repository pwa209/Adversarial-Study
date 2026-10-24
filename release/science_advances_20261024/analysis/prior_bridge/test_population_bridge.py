"""Scientific-contract tests on generated trajectories, never human outcomes."""
import copy
import csv
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import population_bridge as m


def fixture():
    source = "source_model"
    roles = {"train": ["tr1", "tr2"], "validation": ["va1", "va2"], "test": ["te1", "te2"]}
    descriptions = [{"map_key": "map" + str(i), "source_model_key": source,
                     "account_class": "account" + str(i), "variant": "pristine"} for i in range(2)]
    clock = np.linspace(0., m.NOMINAL_STOP, 70)
    grid = np.linspace(.2, m.NOMINAL_STOP, 20)
    cut = 13
    rows, account_rows, states, predictions, maps = [], [], {}, {}, {}
    for role, people in roles.items():
        for person in people:
            token = person + "_0"
            state = np.column_stack([1. + .03 * clock, np.sin(clock) + .2])
            states[token + "__state"], states[token + "__times_s"] = state, clock
            row = dict(source_id="source", source_model_key=source, participant_id=person,
                       episode_id="episode_" + token, state_token=token, split=role, status="included",
                       state_bin_s=str(clock[1] - clock[0]), baseline_support_stop_s="1")
            rows.append(row)
            if role == "train":
                continue
            y = m.interpolate(state, clock, grid)
            scale = float(np.mean(y[:cut] ** 2))
            for description in descriptions:
                key = description["map_key"]
                pred = y + .02
                predictions[key + "__" + token + "__prediction"] = pred
                maps[key + "__times_s"] = grid.copy()
                account_rows.append(description | dict(source_id="source", participant_id=person,
                    episode_id=row["episode_id"], split=role, simulation_id="native",
                    grid_states="20", conditioning_states="13", latent_rank="2",
                    later_state_standardized_mse=m.loss(y[cut:], pred[cut:], scale)))
    selection = {"maps": descriptions, "config": {"lags": [0, 1, 2, 4, 8], "delay_seconds_per_index": .025}}
    training = {"test_outcomes_used_for_selection": False, "human_parameter_fitting_executed": True,
                "maps": 2, "split": roles}
    evaluation = {"test_person_ids": roles["test"], "test_observation_ids": sorted(r["episode_id"] for r in rows if r["split"] == "test")}
    state_selection = {"source_models": [{"source_model_key": source, "training_participant_ids": list(roles["train"]), "validation_participant_ids": list(roles["validation"])}]}
    split = {role + "_participant_ids": people for role, people in roles.items()}
    native_clock = np.linspace(0, 10, 250)
    columns = ["Pop." + str(i) + "." + q for i in range(1, 10) for q in ("V", "Q")]
    traces = {}
    native_rows = []
    for account in range(2):
        for index in range(3):
            identity = "native_" + str(account) + "_" + str(index)
            row = {"simulation_id": identity, "account_class": "account" + str(account),
                   "variant": "pristine", "source_output_sha256": "a" * 64, "config_sha256": "b" * 64,
                   "input_amplitude": "0", "score": ".5", "status": "finite"}
            values = np.column_stack([np.sin(native_clock * (.7 + .03 * j) + index * .2)
                                      + .03 * native_clock * (j + 1) for j in range(18)])
            traces[identity] = {"row": row, "times": native_clock, "values": values, "columns": columns}
            native_rows.append(row)
    bank = {"traces": traces, "rows": [{"simulation_id": r["simulation_id"], "disposition": "qualified"} for r in native_rows], "receipt": {"status": "synthetic_test_only"}}
    return dict(source=source, roles=roles, descriptions=descriptions, clock=clock, grid=grid, cut=cut,
                rows=rows, account_rows=account_rows, states=states, predictions=predictions,
                maps=maps, selection=selection, training=training, evaluation=evaluation,
                state_selection=state_selection, split=split, bank=bank, native_rows=native_rows)


def validate(f):
    return m.validate_design(f["selection"], f["training"], f["state_selection"], f["split"],
                             f["rows"], f["account_rows"], f["maps"], expected_maps=2)


def examples(f, role):
    _, _, index, panels, clocks, cuts, _ = validate(f)
    return m.examples_for(f["source"], role, index, panels, f["descriptions"], f["states"],
                          clocks[f["source"]], cuts[f["source"]])


class ScientificContractTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture()

    def test_frozen_roles_and_original_common_panels(self):
        roles, groups, index, panels, clocks, cuts, census = validate(self.f)
        self.assertEqual(roles["test"], {"te1", "te2"})
        self.assertEqual(len(groups["source_model"]), 2)
        self.assertEqual(cuts["source_model"], 13)
        self.assertEqual([x["common_people"] for x in census], [2, 2])

    def test_person_role_leakage_rejected(self):
        self.f["training"]["split"]["train"].append("te1")
        with self.assertRaisesRegex(ValueError, "roles overlap"):
            validate(self.f)

    def test_neural_basis_training_on_test_people_rejected(self):
        self.f["state_selection"]["source_models"][0]["training_participant_ids"].append("te1")
        with self.assertRaisesRegex(ValueError, "neural basis"):
            validate(self.f)

    def test_silent_common_case_intersection_forbidden(self):
        self.f["account_rows"].pop()
        with self.assertRaisesRegex(ValueError, "case sets differ"):
            validate(self.f)

    def test_different_exact_target_clock_rejected(self):
        self.f["maps"]["map1__times_s"][5] += .001
        with self.assertRaisesRegex(ValueError, "clock differs"):
            validate(self.f)

    def test_different_conditioning_stop_rejected(self):
        self.f["account_rows"][0]["conditioning_states"] = "12"
        with self.assertRaisesRegex(ValueError, "Conditioning stop"):
            validate(self.f)

    def test_parent_nuisance_baseline_cannot_consume_future(self):
        self.f["rows"][0]["baseline_support_stop_s"] = "9"
        with self.assertRaisesRegex(ValueError, "centering reaches future"):
            examples(self.f, "train")

    def test_no_extrapolation_and_episode_joining(self):
        with self.assertRaisesRegex(ValueError, "cannot extrapolate"):
            m.interpolate(np.ones((10, 2)), np.arange(10.), np.array([-1., 1.]))
        with self.assertRaisesRegex(ValueError, "cannot extrapolate"):
            m.interpolate(np.ones((10, 2)), np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4.]), np.array([1., 2.]))

    def test_all_nine_populations_and_matched_eighteen_columns(self):
        trace = next(iter(self.f["bank"]["traces"].values()))
        dims = [m.native_features(trace, self.f["grid"], 13, v).shape[1] for v in m.VIEWS]
        self.assertEqual(dims, [1, 18, 9, 18])
        wrong = copy.deepcopy(trace)
        wrong["columns"][0] = "invented"
        with self.assertRaisesRegex(ValueError, "nine observed"):
            m.native_features(wrong, self.f["grid"], 13, "population_voltage9")

    def test_native_future_never_sets_conditioning_origin_or_scale(self):
        trace = copy.deepcopy(next(iter(self.f["bank"]["traces"].values())))
        first = m.native_features(trace, self.f["grid"], 13, "population_voltage_rate18")
        trace["values"][trace["times"] > self.f["grid"][12] + .1] += 1000
        second = m.native_features(trace, self.f["grid"], 13, "population_voltage_rate18")
        np.testing.assert_allclose(first[:13], second[:13])
        self.assertFalse(np.allclose(first[14:], second[14:]))

    def test_capacity_control_cannot_extrapolate_earlier_native_history(self):
        trace = next(iter(self.f["bank"]["traces"].values()))
        with self.assertRaisesRegex(ValueError, "cannot extrapolate"):
            m.native_features(trace, np.linspace(.1, 9.9, 20), 13, "cortical_delay18")

    def test_actual_first_saved_sample_bounds_delay_without_rebasing_or_padding(self):
        trace = copy.deepcopy(next(iter(self.f["bank"]["traces"].values())))
        actual_times = np.arange(1, 5001) * .002
        trace["values"] = m.interpolate(trace["values"], trace["times"], actual_times)
        trace["times"] = actual_times
        frozen_grid = self.f["grid"].copy()
        support = m.available_history(frozen_grid, [trace], .2)
        self.assertAlmostEqual(support["used_max_delay_s"], .198)
        self.assertEqual(support["requested_max_delay_s"], .2)
        self.assertEqual(support["support_reason"], "limited_by_observed_native_history")
        self.assertFalse(support["native_timestamps_rebased"])
        self.assertFalse(support["human_target_grid_changed"])
        with patch.object(m, "interpolate", wraps=m.interpolate) as observed:
            features = m.native_features(trace, frozen_grid, 13, "cortical_delay18", support["used_max_delay_s"])
        self.assertEqual(features.shape, (20, 18))
        self.assertEqual(len(observed.call_args_list), 18)
        self.assertAlmostEqual(min(call.args[2][0] for call in observed.call_args_list), .002)
        np.testing.assert_array_equal(frozen_grid, self.f["grid"])
        np.testing.assert_array_equal(trace["times"], actual_times)
        with self.assertRaisesRegex(ValueError, "cannot extrapolate"):
            m.native_features(trace, frozen_grid, 13, "cortical_delay18", .2)

    def test_available_history_is_shared_across_all_account_candidates(self):
        candidates = [copy.deepcopy(next(iter(self.f["bank"]["traces"].values()))) for _ in range(2)]
        candidates[0]["times"] = np.linspace(.002, 10, 250)
        candidates[1]["times"] = np.linspace(.004, 10, 250)
        support = m.available_history(self.f["grid"], candidates, .2)
        self.assertAlmostEqual(support["used_max_delay_s"], .196)
        self.assertEqual(support["latest_native_first_sample_s"], .004)
        self.assertEqual(support["candidate_configurations"], 2)
        self.assertTrue(support["shared_across_all_account_candidates"])
        self.assertFalse(support["human_test_targets_used"])

    def test_no_positive_observed_history_is_rejected(self):
        trace = copy.deepcopy(next(iter(self.f["bank"]["traces"].values())))
        for first in (.2, .202):
            trace["times"] = np.linspace(first, 10, 250)
            with self.assertRaisesRegex(ValueError, "Positive shared observed cortical history"):
                m.available_history(self.f["grid"], [trace], .2)
        trace["times"] = np.linspace(.002, 10, 250)
        with self.assertRaisesRegex(ValueError, "Positive requested cortical history"):
            m.available_history(self.f["grid"], [trace], 0.)

    def test_conditioning_assignment_ignores_all_future_targets(self):
        original = examples(self.f, "validation")
        altered = copy.deepcopy(original)
        predictions = np.asarray([original[0]["y"] + .1, original[0]["y"] - .3])
        first, offset1 = m.choose_conditioning(original, predictions, 1.)
        for e in altered:
            e["y"][e["cut"]:] = 1e9
        second, offset2 = m.choose_conditioning(altered, predictions, 1.)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(offset1, offset2)

    def test_equal_person_weights_with_unequal_episode_counts(self):
        train = examples(self.f, "train")
        train.append(copy.deepcopy(train[0]))
        weights = m.weights_for(train)
        self.assertAlmostEqual(weights[0] + weights[2], weights[1])

    def test_feature_basis_uses_only_explicit_training_blocks(self):
        x = [np.column_stack([np.arange(8.), np.sin(np.arange(8.))]), np.column_stack([np.arange(8.) + 1, np.cos(np.arange(8.))])]
        first = m.weighted_basis(x, [.5, .5], 2)
        m.transform(np.full((10, 2), 1e8), first)
        second = m.weighted_basis(x, [.5, .5], 2)
        for key in ("center", "scale", "basis"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_readout_assignments_supported_by_training_only(self):
        train, validation = examples(self.f, "train"), examples(self.f, "validation")
        candidates = [v for v in self.f["bank"]["traces"].values() if v["row"]["account_class"] == "account0"]
        kernels = np.asarray([m.native_features(t, self.f["grid"], 13, "population_voltage_rate18") for t in candidates])
        fitted = m.fit_observation(train, validation, kernels, 4, .1, 1.)
        self.assertEqual(set(fitted["support_indices"]), set(fitted["train_assignments"]))
        self.assertTrue(set(fitted["validation_assignments"]) <= set(fitted["support_indices"]))
        self.assertEqual(fitted["training_people"], ["tr1", "tr2"])
        self.assertTrue(0 < fitted["effective_df_per_output"] <= 5.)
        altered = copy.deepcopy(validation)
        for e in altered:
            e["y"][13:] += 10
        refit = m.fit_observation(train, altered, kernels, 4, .1, 1.)
        for key in ("W", "bias"):
            np.testing.assert_array_equal(fitted[key], refit[key])
        self.assertNotEqual(fitted["validation_loss"], refit["validation_loss"])

    def test_statistical_forecast_preserves_original_sample_clock(self):
        A, bias = np.array([[.5]]), np.array([0.])
        p = m.recurse(np.array([2.]), np.array([.5, 1., 1.5, 2.]), A, bias, 1.)
        np.testing.assert_allclose(p[:, 0], [1.5, 1., .75, .5])

    def test_original_scalar_loss_is_reconstructed_from_actual_arrays(self):
        e = examples(self.f, "test")[0]
        _, _, _, panels, _, _, _ = validate(self.f)
        m.original_prediction(e, self.f["descriptions"][0], self.f["predictions"], panels[("map0", "test")])
        self.f["predictions"]["map0__te1_0__prediction"][17] += 5
        with self.assertRaisesRegex(ValueError, "cannot be reconstructed"):
            m.original_prediction(e, self.f["descriptions"][0], self.f["predictions"], panels[("map0", "test")])

    def test_training_examples_do_not_load_any_validation_or_test_array(self):
        class Protected(dict):
            def __getitem__(self, key):
                if key.startswith(("te", "va")):
                    raise AssertionError("Held-role array read")
                return super().__getitem__(key)
        self.f["states"] = Protected(self.f["states"])
        train = examples(self.f, "train")
        self.assertEqual(len(train), 2)

    def test_exact_reference_is_person_level_and_not_equivalence(self):
        result = m.paired_inference({"p1": -1., "p2": -2., "p3": -1., "p4": -2., "p5": -1.}, 123, 500)
        self.assertEqual(result["n_people"], 5)
        self.assertEqual(result["exact_signflip_two_sided"], .0625)
        self.assertFalse(result["equivalence_established"])

    def test_end_to_end_freeze_and_paired_capacity_provenance(self):
        f = self.f
        original_grid = f["grid"].copy()
        for trace in f["bank"]["traces"].values():
            actual_times = np.arange(1, 5001) * .002
            trace["values"] = m.interpolate(trace["values"], trace["times"], actual_times)
            trace["times"] = actual_times
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            def save_json(name, value):
                path = root / (name + ".json")
                path.write_text(json.dumps(value))
                return path
            def save_csv(name, rows):
                path = root / (name + ".csv")
                m.write_csv(path, rows)
                return path
            def save_npz(name, values):
                path = root / (name + ".npz")
                np.savez(path, **values)
                return path
            paths = {"state_arrays": save_npz("states", f["states"]), "state_predictions": save_npz("predictions", f["predictions"]),
                     "observation_map": save_npz("maps", f["maps"]),
                     "state_models": save_npz("models", {f["source"] + "__transition": np.eye(2) * .9, f["source"] + "__intercept": np.zeros(2)}),
                     "state_rows": save_csv("rows", f["rows"]), "account_predictions": save_csv("account", f["account_rows"]),
                     "native_bank": save_csv("native", f["native_rows"]), "state_split": save_json("split", f["split"]),
                     "mechanism_selection": save_json("selection", f["selection"])}
            f["training"]["observation_map"] = {"sha256": m.sha(paths["observation_map"])}
            paths["mechanism_training"] = save_json("training", f["training"])
            f["evaluation"].update(training_receipt_sha256=m.sha(paths["mechanism_training"]),
                observation_map_sha256=m.sha(paths["observation_map"]), account_predictions_sha256=m.sha(paths["account_predictions"]))
            paths["mechanism_evaluation"] = save_json("evaluation", f["evaluation"])
            f["state_selection"].update(models_sha256=m.sha(paths["state_models"]), split_sha256=m.sha(paths["state_split"]))
            paths["state_selection"] = save_json("state_selection", f["state_selection"])
            specs = {name: {"path": str(path), "sha256": m.sha(path)} for name, path in paths.items()}
            inputs = {"files": specs, "population_cache": "synthetic_cache"}
            loader = types.ModuleType("native_population")
            loader.read_cache = lambda directory: f["bank"]
            out = root / "result"
            with patch.dict(sys.modules, {"native_population": loader}):
                receipt = m.run(inputs, out, bootstrap_replicates=100, expected_maps_per_source=2)
            self.assertEqual(receipt["test_people"], 2)
            self.assertEqual(receipt["test_episodes"], 2)
            self.assertEqual(receipt["models"], 12)
            freeze = m.json_read(out / "training_validation_freeze.json")
            self.assertFalse(freeze["selection_uses_test_targets"])
            support = freeze["cortical_delay_support"][f["source"]]
            self.assertEqual(support["candidate_configurations"], 6)
            self.assertAlmostEqual(support["requested_max_delay_s"], .2)
            self.assertAlmostEqual(support["used_max_delay_s"], .198)
            self.assertEqual(support["delay_columns"], 18)
            for record in m.csv_read(out / "validation_grid.csv"):
                self.assertAlmostEqual(float(record["cortical_delay_used_max_s"]), .198)
                self.assertAlmostEqual(float(record["cortical_delay_requested_max_s"]), .2)
                self.assertEqual(record["cortical_delay_support_reason"], "limited_by_observed_native_history")
            for group in f["descriptions"]:
                a = freeze["models"][group["map_key"] + "__matched_population_voltage_rate18"]
                b = freeze["models"][group["map_key"] + "__matched_cortical_delay18"]
                self.assertEqual([a[k] for k in ("feature_rank", "ridge", "offset_strength")],
                                 [b[k] for k in ("feature_rank", "ridge", "offset_strength")])
                self.assertEqual(a["nominal_readout_coefficients"], b["nominal_readout_coefficients"])
                self.assertEqual(a["training_people"], ["tr1", "tr2"])
            compatibility = m.json_read(out / "compatibility.json")
            self.assertFalse(compatibility["selection_uses_test_targets"])
            self.assertTrue(compatibility["compatible_simulation_ids"])
            report = m.json_read(out / "population_bridge.json")
            self.assertEqual(report["cortical_delay_support"], freeze["cortical_delay_support"])
            np.testing.assert_array_equal(f["grid"], original_grid)
            with np.load(out / "predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved[f["source"] + "__times_s"], original_grid)
            self.assertEqual(len(report["all_account_comparisons"]), 14)
            self.assertEqual(len(report["validation_selected_account_comparisons"]), 7)
            self.assertFalse(report["human_future_data_used_for_conditioning"])
            self.assertEqual(report["behavioral_target"], 42)
            self.assertFalse(report["behavioral_target_is_neural_sample_size"])
            for name, pin in receipt["outputs"].items():
                self.assertEqual(m.sha(out / name), pin)


if __name__ == "__main__":
    unittest.main()
