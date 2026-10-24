"""Synthetic fixtures check source roles, preserved labels and genuine split code.

These tests do not execute GPU fine-tuning or produce empirical study outcomes.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adversarial_study import core, extension_finetune as native, proposal_breadth as breadth, proposal_discovery
from adversarial_study.acquisition import canonical_hash


class ProposalBreadthTests(unittest.TestCase):
    def setUp(self):
        self.cfg = breadth.load_config()

    def text(self, trials=8, confidence="0.2"):
        header = "You see two discs in each trial, varying in brightness. Always focus your eyes on the red cross in the center.\n"
        return header+"\n".join("Left disc has brightness 0.2, right disc has brightness 0.4. You press <<"+
            ("LeftArrow" if trial % 2 else "RightArrow")+">> after 120ms. You rate your confidence with <<"+confidence+">> after 60ms." for trial in range(trials))

    def row(self, person, **extra):
        return {"experiment": breadth.STUDY+"/data_Hellmann_unpublished_LuMagConf.csv", "participant": person,
                "text": self.text(), "is_psych101": False, "is_psych101_test": False, "in_eval": True, **extra}

    def fixture(self, root, rows, *, measured=True):
        source = root/"source"
        source.mkdir()
        path = source/"psych201.jsonl"
        path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
        manifest = root/"manifest.json"
        core.write_json(manifest, {"resources": [{"resource_id": "test_actual_source_file", "source_id": "psych201",
            "relative_path": path.name, "url": "https://example.org/unit-test-fixture.jsonl", "expected_bytes": path.stat().st_size,
            "checksum": {"algorithm": "sha256", "value": core.sha256_file(path)}}]})
        legacy = root/"legacy"
        legacy.mkdir()
        observation = {"source_id": "openneuro_ds001785", "participant_id": "synthetic_discovery_person", "episode_id": "a",
            "recording_id": "r", "score": .8, "report_category": "miss", "start_s": 0, "stop_s": 10,
            "role": "development", "synthetic": False, "score_origin": "native_frozen_v2", "saturated_fraction": 0,
            "detection": 0 if measured else "", "confidence": "", "capacity_value": 0 if measured else "",
            "capacity_construct": "subsequent_tactile_detection", "capacity_units": "binary_felt_yes_no"}
        core.write_csv(legacy/"all_scored_observations.csv", [observation])
        core.write_json(legacy/"receipt.json", {"status": "model_guided_discovery_completed",
            "outputs": {"all_scored_observations.csv": core.sha256_file(legacy/"all_scored_observations.csv")},
            "inputs": [{"path": "synthetic-fixture-original-eeg-reference", "sha256": "a"*64}]})
        discovery = root/"discovery"
        proposal_discovery.run([legacy], proposal_discovery.DEFAULT_CONFIG, discovery)
        config = root/"breadth.json"
        cfg = deepcopy(self.cfg)
        cfg["training"]["people"] = {"train": 2, "validation": 1, "test": 1}
        core.write_json(config, cfg)
        return discovery, source, manifest, config

    def folder(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent if os.name == "nt" else None)

    def test_role_filter_preserves_original_indices_labels_and_confidence_history(self):
        text = self.text()
        original = breadth.ORIGINAL_MARKERS(text)
        choices = breadth.choice_responses(text)
        self.assertEqual([event["index"] for event in choices], list(range(0, 16, 2)))
        self.assertEqual([event["response"] for event in choices], [event["response"] for event in original[::2]])
        self.assertIn("confidence with <<0.2>>", choices[1]["prefix"])
        self.assertNotIn("confidence with <<0.2>>", choices[0]["prefix"])
        self.assertTrue(all(event["prefix"].endswith("You press <<") for event in choices))
        self.assertNotEqual({event["response"] for event in choices}, {"L", "R"})

    def test_missing_confidence_is_retained_and_never_fitted_as_choice(self):
        choices, audit = breadth.qualify_markers(self.text(confidence="nan"))
        self.assertEqual(audit["missing_confidence_events"], 8)
        self.assertEqual(len(choices), 8)
        self.assertIn("<<nan>>", choices[-1]["prefix"])

    def test_confidence_role_cannot_be_relabelled_or_out_of_scale(self):
        for text in (self.text().replace("You rate your confidence with", "You press"), self.text(confidence="2"),
                     self.text()+"<<LeftArrow>>", self.text().replace("brightness 0.2", "brightness nan")):
            with self.assertRaises(ValueError):
                breadth.choice_responses(text)

    def test_exact_source_identity_rejects_substring_collision(self):
        self.assertIsNone(breadth.source_identity(self.row("p", experiment="unrelated_"+breadth.STUDY+"/x")))
        self.assertIsNone(breadth.source_identity(self.row("p", study="another_study")))
        self.assertIsNone(breadth.source_identity(self.row("p", study=None)))
        with self.assertRaises(core.StudyError):
            breadth.source_identity(self.row(1))

    def test_full_source_census_retains_all_tasks_and_exposure_excludes_person(self):
        with self.folder() as folder:
            root = Path(folder)
            rows = [self.row("p1"), self.row("p1", experiment=breadth.STUDY+"/other_session", is_psych101=True),
                    self.row("p2"), self.row("p3", is_psych101=None), self.row("p4", is_psych101_test="false"),
                    self.row("outside", experiment="other_study/raw.csv"), self.row("bad", text="bad text")]
            _, source, manifest, config = self.fixture(root, rows)
            census = breadth.source_census(source, manifest, breadth.load_config(config))
            self.assertEqual(census["all_source_rows"], 7)
            self.assertEqual(census["all_study_session_counts"]["other_study"], 1)
            self.assertEqual(census["eligible_unexposed_people"], 1)
            self.assertEqual(census["eligible_people"], [breadth.STUDY+"\0p2"])
            self.assertTrue(all(not e["eligible"] for e in census["sessions"] if e["participant"] == "p1"))
            self.assertEqual(census["original_choice_alphabet"], ["LeftArrow", "RightArrow"])
            self.assertEqual(sum(e["sessions"] for e in census["all_source_exposure_census"]), 7)

    def test_prepare_genuine_hash_qualified_split_retains_full_raw_transcripts(self):
        with self.folder() as folder:
            root = Path(folder)
            discovery, source, manifest, config = self.fixture(root, [self.row("p"+str(i)) for i in range(6)])
            result = breadth.prepare(discovery, source, manifest, config, breadth.DEFAULT_GENERATOR, root/"prepared")
            self.assertEqual(result["status"], breadth.READY)
            receipt, cfg, split, data = breadth.verify_prepared(root/"prepared", open_test=True)
            self.assertEqual(cfg["response_alphabet"], ["LeftArrow", "RightArrow"])
            self.assertEqual(split["independent_people"], {"train": 2, "validation": 1, "test": 1})
            self.assertEqual(sum(len(rows) for rows in data.values()), 4)
            self.assertEqual(len(set(split["person_roles"])), 4)
            for rows in data.values():
                for row in rows:
                    self.assertEqual(row["text"], self.text())
                    self.assertEqual(row["original_response_count"], 8)
            self.assertFalse(receipt["native_training_executed"])
            self.assertFalse(receipt["neural_to_behavioural_mapping_qualified"])
            with self.assertRaises(core.StudyError):
                breadth.prepare(discovery, source, manifest, config, breadth.DEFAULT_GENERATOR, root/"prepared")

    def test_insufficient_source_reports_disposition_without_fabricated_split(self):
        with self.folder() as folder:
            root = Path(folder)
            discovery, source, manifest, config = self.fixture(root, [self.row("p1"), self.row("p2")])
            result = breadth.prepare(discovery, source, manifest, config, breadth.DEFAULT_GENERATOR, root/"prepared")
            self.assertEqual(result["status"], "BLOCKED_INSUFFICIENT_ELIGIBLE_PEOPLE")
            self.assertFalse((root/"prepared/split").exists())
            self.assertEqual(result["eligible_unexposed_people"], 2)
            with self.assertRaises(core.StudyError):
                breadth.verify_prepared(root/"prepared")

    def test_null_detection_selects_construct_without_low_score_or_effect_gate(self):
        with self.folder() as folder:
            discovery, _, _, _ = self.fixture(Path(folder), [self.row("p")])
            selection = breadth.construct_selection(discovery)
            self.assertEqual(selection["status"], "prespecified_construct_candidate_selected")
            self.assertFalse(selection["result_direction_gate"])
            self.assertFalse(selection["neural_to_behavioural_mapping_qualified"])
            self.assertEqual(selection["human_construct_census"][1]["target"], "detection")
            self.assertEqual(selection["human_construct_census"][1]["measured_rows"], 1)

    def test_unmeasured_construct_cannot_select_unrelated_task(self):
        with self.folder() as folder:
            root = Path(folder)
            discovery, source, manifest, config = self.fixture(root, [self.row("p"+str(i)) for i in range(5)], measured=False)
            result = breadth.prepare(discovery, source, manifest, config, breadth.DEFAULT_GENERATOR, root/"prepared")
            self.assertEqual(result["status"], "BLOCKED_NO_MEASURED_RELEVANT_CONSTRUCT")

    def test_changed_source_or_discovery_fails_before_native_gpu_dispatch(self):
        with self.folder() as folder:
            root = Path(folder)
            discovery, source, manifest, config = self.fixture(root, [self.row("p"+str(i)) for i in range(4)])
            breadth.prepare(discovery, source, manifest, config, breadth.DEFAULT_GENERATOR, root/"prepared")
            with (source/"psych201.jsonl").open("a") as stream:
                stream.write("\n")
            with patch.object(native, "train") as runner:
                with self.assertRaises(core.StudyError):
                    breadth.train(root/"prepared", "centaur", root/"training")
                runner.assert_not_called()

    def test_original_index_budget_accepts_late_choice_index_and_rejects_confidence(self):
        data = {"test": [{"session_id": "s", "person_id": "p", "text": self.text(10)}]}
        cfg = {"max_responses_per_session": 8, "test_events_per_person": 4}
        row = {"session_id": "s", "person_id": "p", "response_index": "14", "response": "LeftArrow"}
        breadth.verify_choice_budget(data, [row], cfg)
        for changed in ({**row, "response_index": "15", "response": "0.2"}, {**row, "response_index": "16", "response": "RightArrow"}, {**row, "response": "L"}):
            with self.assertRaises(core.StudyError):
                breadth.verify_choice_budget(data, [changed], cfg)
        with self.assertRaises(core.StudyError):
            breadth.verify_choice_budget(data, [row, row], cfg)

    def test_context_restores_legacy_functions_even_after_error(self):
        saved = native.marked_responses, native._behaviour_rows, native.write_json, native._json, native._token_events
        with self.assertRaises(RuntimeError):
            with breadth.role_context():
                self.assertEqual([e["index"] for e in native.marked_responses(self.text(2))], [0, 2])
                raise RuntimeError("test interruption")
        self.assertEqual((native.marked_responses, native._behaviour_rows, native.write_json, native._json, native._token_events), saved)

    def test_comparison_keeps_test_sealed_until_both_actual_trainings_qualify(self):
        with patch.object(breadth, "verify_prepared", return_value=({}, {}, {}, {})) as prepared:
            with patch.object(breadth, "_verify_matched_training", side_effect=core.StudyError("actual native training absent")):
                with patch.object(breadth, "qualify_split") as opened:
                    with self.assertRaises(core.StudyError):
                        breadth.compare("prepared", "centaur", "base", "eval_c", "eval_b", "comparison")
                    opened.assert_not_called()
                    prepared.assert_called_once_with("prepared", open_test=False)

    def test_receipt_first_write_corrects_assay_and_binds_actual_source_order(self):
        with self.folder() as folder:
            root = Path(folder)
            cfg = deepcopy(self.cfg["training"])
            cfg["response_alphabet"] = ["LeftArrow", "RightArrow"]
            data = [{"session_id": "s", "person_id": "p", "text": self.text(), "fine_tune_role": "train",
                "text_sha256": hashlib.sha256(self.text().encode()).hexdigest(), "source_id": "psych201",
                "source_resource_id": "fixture", "resource_sha256": "a"*64, "exposure_flags": {"is_psych101": False, "is_psych101_test": False}}]
            def encoding(tokenizer, prefix, alphabet, maximum):
                return [1, 2], {label: [index+3] for index, label in enumerate(alphabet)}
            with patch.object(native, "_encode_candidates", side_effect=encoding):
                with breadth.role_context(training_cfg=cfg, prepared_record={"path": "fixture", "sha256": "b"*64}):
                    train, _ = native._token_events(data, None, cfg)
                    val, _ = native._token_events([{**data[0], "fine_tune_role": "validation"}], None, cfg)
                    val = native._person_event_limit(val, cfg["validation_events_per_person"])
                    order, updates = breadth.optimizer_observations(train, cfg)
                    original = {"status": "human_finetuning_completed", "assay_id": "old hardcoded assay",
                        "train_events_sha256": native.event_fingerprint(train), "validation_events_sha256": native.event_fingerprint(val),
                        "actual_training_order_sha256": canonical_hash(order), "actual_optimizer_steps": updates}
                    native.write_json(root/"training_receipt.json", original)
            stored = core.read_json(root/"training_receipt.json")
            self.assertEqual(stored["assay_id"], breadth.ASSAY)
            self.assertEqual(original["assay_id"], "old hardcoded assay")
            self.assertFalse(stored["human_assay"]["confidence_fitted_as_choice"])
            self.assertEqual([e["response_index"] for e in stored["source_role_training_events"]["train"]["observations"]], list(range(0, 16, 2)))
            self.assertEqual(canonical_hash(stored["actual_optimizer_observation_order"]), canonical_hash(order))

    def test_actual_native_comparator_replays_original_marker_indices_and_null_or_negative_outcome(self):
        # Artifact fixtures exercise native verification, never a GPU/training claim.
        with self.folder() as folder:
            root = Path(folder)
            discovery, source, manifest, config = self.fixture(root, [self.row("p"+str(i)) for i in range(4)])
            prepared = root/"prepared"
            breadth.prepare(discovery, source, manifest, config, breadth.DEFAULT_GENERATOR, prepared)
            _, cfg, split, data = breadth.verify_prepared(prepared, open_test=True)
            trainings = {}
            def encoding(tokenizer, prefix, alphabet, maximum):
                return [1, 2], {label: [index+3] for index, label in enumerate(alphabet)}
            for model in ("centaur", "base"):
                directory = root/(model+"_train")
                (directory/"adapter").mkdir(parents=True)
                weight = directory/"adapter/adapter_model.safetensors"
                weight.write_bytes((model+" SYNTHETIC UNIT ARTIFACT, NOT MODEL WEIGHTS").encode())
                with patch.object(native, "_encode_candidates", side_effect=encoding):
                    with breadth.role_context(training_cfg=cfg, prepared_record=breadth._record(prepared/"prepare_receipt.json")):
                        train, _ = native._token_events(data["train"], None, cfg)
                        val, _ = native._token_events(data["validation"], None, cfg)
                        val = native._person_event_limit(val, cfg["validation_events_per_person"])
                        order, updates = breadth.optimizer_observations(train, cfg)
                        receipt = {"status": "human_finetuning_completed", "model": model, "weights_changed": True,
                            "native_training_executed": True, "actual_optimizer_steps": updates, "optimizer_steps": updates,
                            "epochs": cfg["epochs"], "trainable_parameters": 8, "split_files_sha256": split["files_sha256"],
                            "split_receipt_sha256": core.sha256_file(prepared/"split/split_receipt.json"),
                            "configuration_sha256": core.sha256_file(prepared/"training.config.json"),
                            "initial_trainable_weights_sha256": "before_"+model, "final_trainable_weights_sha256": "after_"+model,
                            "adapter_files_sha256": {"adapter_model.safetensors": core.sha256_file(weight)},
                            "trained_checkpoint": {"path": str(weight.resolve()), "sha256": core.sha256_file(weight)},
                            "train_events_sha256": native.event_fingerprint(train), "validation_events_sha256": native.event_fingerprint(val),
                            "actual_training_order_sha256": canonical_hash(order)}
                        native.write_json(directory/"training_receipt.json", receipt)
                trainings[model] = directory
            hashes = {model: core.sha256_file(directory/"training_receipt.json") for model, directory in trainings.items()}
            evaluations = {}
            human = data["test"][0]
            original = breadth.choice_responses(human["text"])[7]
            with breadth.role_context():
                for model, loglikelihood in (("centaur", -1.0), ("base", -0.5)):
                    directory = root/(model+"_eval")
                    directory.mkdir()
                    table = directory/"heldout_likelihoods.csv"
                    rows = [{"session_id": human["session_id"], "person_id": human["person_id"], "response_index": original["index"],
                        "response": original["response"], "conditional_choice_log_likelihood": loglikelihood}]
                    core.write_csv(table, rows)
                    receipt = {"status": "human_finetuned_heldout_evaluation_completed", "model": model,
                        "both_models_human_finetuned_before_test": True, "test_training_updates": 0,
                        "split_receipt_sha256": core.sha256_file(prepared/"split/split_receipt.json"),
                        "training_receipt_sha256": hashes[model], "both_training_receipts_sha256": hashes,
                        "likelihoods_sha256": core.sha256_file(table), "heldout_rows_sha256": canonical_hash(native._csv_rows(table)),
                        "human_observations_sha256": native.human_observation_fingerprint(data["test"], rows), "test_events_sha256": "synthetic-token-identity-shared",
                        "assay_id": breadth.ASSAY, "test_events": 1, "test_people": 1,
                        "test_observation_ids": [human["session_id"]+":"+str(original["index"])],
                        "test_predictions": {"path": str(table.resolve()), "sha256": core.sha256_file(table)}}
                    core.write_json(directory/"evaluation_receipt.json", receipt)
                    evaluations[model] = directory
            result = breadth.compare(prepared, trainings["centaur"], trainings["base"], evaluations["centaur"], evaluations["base"], root/"comparison")
            self.assertEqual(result["equal_person_mean_loglikelihood_difference"], -.5)
            self.assertFalse(result["neural_to_behavioural_mapping_qualified"])
            verified = breadth.verify_comparison(root/"comparison")
            self.assertEqual(verified["status"], "human_finetuned_comparison_artifacts_verified")
            paired = core.read_csv(root/"comparison/paired_heldout_comparison.csv")
            self.assertEqual(paired[0]["response_index"], "14")
            (trainings["base"]/"adapter/adapter_model.safetensors").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                breadth.verify_comparison(root/"comparison")


if __name__ == "__main__":
    unittest.main()
