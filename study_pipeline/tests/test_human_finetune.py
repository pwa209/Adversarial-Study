"""Synthetic software fixtures only; none establish native MATLAB training."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adversarial_study import core, native
from native_tools import human_finetune as hf


class HumanFinetuneTests(unittest.TestCase):
    def observations(self):
        return [{"participant_id": f"p{i}", "episode_id": f"episode{i}", "source_id": "openneuro_ds001785", "recording_id": f"recording{i}", "detection": i % 2} for i in range(6)]

    def split(self):
        return {"train_participant_ids": ["p0", "p1"], "validation_participant_ids": ["p2", "p3"], "test_participant_ids": ["p4", "p5"]}

    def jobs(self, root):
        jobs = []
        for row in self.observations():
            for channel in ("Fz", "Pz"):
                path = root / (row["episode_id"] + "_" + channel + ".csv")
                path.write_text("1\n2\n3\n", encoding="utf8")
                jobs.append({"job_id": row["episode_id"] + "_" + channel, "participant_id": row["participant_id"], "episode_id": row["episode_id"], "source_id": row["source_id"], "input_path": str(path), "sfreq_hz": "500", "signal_sha256": core.sha256_file(path), "channel_name": channel})
        return jobs

    def fixture(self, root):
        exported = root / "exported"; exported.mkdir()
        examples, _ = hf.build_examples(self.observations(), self.jobs(root), "detection", self.split())
        core.write_csv(exported / "human_examples.csv", examples, list(hf.EXPORT_FIELDS))
        core.write_json(exported / "dataset_split.json", self.split())
        split_sha = core.sha256_file(exported / "dataset_split.json")
        human_assay = hf.ASSAYS["detection"] | {"input_modality": "synthetic_test_fixture_only", "label_vocabulary": ["detected", "not_detected"]}
        receipt = {"status": "human_finetuning_inputs_verified_not_trained", "human_data": True, "human_assay": human_assay, "split": self.split() | {"split_sha256": split_sha}, "data_split_sha256": split_sha, "examples_sha256": core.sha256_file(exported / "human_examples.csv"), "inputs": [], "train_examples_sha256": hashlib.sha256(json.dumps([r for r in core.read_csv(exported / "human_examples.csv") if r["split"] == "train"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        core.write_json(exported / "export_receipt.json", receipt)
        output = root / "trained"; output.mkdir()
        adapted = output / "adapted_human_task_net.mat"; adapted.write_bytes(b"synthetic mock checkpoint; not MATLAB weights")
        changed = output / "backbone_updates.json"
        core.write_json(changed, [{"layer": "conv1", "parameter": "Weights", "before_sha256": "a" * 64, "after_sha256": "b" * 64}])
        predictions = output / "heldout_predictions.csv"
        rows = [{"episode_id": "episode4", "participant_id": "p4", "source_id": "openneuro_ds001785", "label": "not_detected", "prediction": "not_detected", "channel_count": 2, "probability_detected": .2, "probability_not_detected": .8}, {"episode_id": "episode5", "participant_id": "p5", "source_id": "openneuro_ds001785", "label": "detected", "prediction": "detected", "channel_count": 2, "probability_detected": .7, "probability_not_detected": .3}]
        core.write_csv(predictions, rows, list(rows[0]))
        result = {"status": "completed_actual_human_finetune", "native_training_executed": True, "backbone_training": True, "human_data": True, "adaptation_completed": True, "optimizer_steps": 4, "data_split_sha256": split_sha, "split": receipt["split"], "human_assay": human_assay, "original_checkpoint_sha256": native.CHECKPOINT_SHA256, "adapted_checkpoint_sha256": core.sha256_file(adapted), "trained_checkpoint": {"path": str(adapted), "sha256": core.sha256_file(adapted)}, "backbone_updates": {"path": str(changed), "sha256": core.sha256_file(changed)}, "changed_backbone_parameter_count": 1, "explicit_trained_backbone_layers": ["conv1"], "export_dir": str(exported), "export_receipt": {"path": str(exported / "export_receipt.json"), "sha256": core.sha256_file(exported / "export_receipt.json")}, "original_release_dir": str(root / "mock_original"), "test_predictions": {"path": str(predictions), "sha256": core.sha256_file(predictions)}, "test_observation_ids": ["episode4", "episode5"], **self.split()}
        evaluation = {"status": "completed_actual_heldout_human_evaluation", "data_split_sha256": split_sha, "split": receipt["split"], "human_assay": human_assay, "trained_checkpoint": result["trained_checkpoint"], "test_predictions": result["test_predictions"], "test_observation_ids": result["test_observation_ids"]}
        core.write_json(output / "training_receipt.json", result)
        core.write_json(output / "evaluation_receipt.json", evaluation)
        return exported, output, receipt, result, evaluation

    def test_split_is_stable_label_blind_and_person_disjoint(self):
        people = [f"person{i}" for i in range(18)]
        a = hf.participant_split(people)
        b = hf.participant_split(list(reversed(people)))
        self.assertEqual(a, b)
        self.assertEqual(len(a["train_participant_ids"]), 12)
        self.assertEqual(len(a["validation_participant_ids"]), 3)
        self.assertEqual(len(a["test_participant_ids"]), 3)
        self.assertEqual(set(hf.validate_split(a)), set(people))
        a["test_participant_ids"].append(a["train_participant_ids"][0])
        with self.assertRaisesRegex(core.StudyError, "overlap"):
            hf.validate_split(a)

    def test_actual_channel_examples_preserve_labels_and_people(self):
        with tempfile.TemporaryDirectory() as tmp:
            examples, excluded = hf.build_examples(self.observations(), self.jobs(Path(tmp)), "detection", self.split())
            self.assertEqual(len(examples), 12)
            self.assertFalse(excluded)
            self.assertEqual({r["split"] for r in examples if r["participant_id"] == "p4"}, {"test"})
            self.assertEqual({r["label"] for r in examples if r["participant_id"] == "p4"}, {"not_detected"})

    def test_wrong_assay_and_nonbinary_original_response_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = self.jobs(Path(tmp)); observations = self.observations()
            with self.assertRaisesRegex(core.StudyError, "source does not measure"):
                hf.build_examples(observations, jobs, "report_category", self.split())
            observations[0]["detection"] = 3
            with self.assertRaisesRegex(core.StudyError, "original binary"):
                hf.build_examples(observations, jobs, "detection", self.split())

    def test_missing_labels_retained_without_relabeling(self):
        with tempfile.TemporaryDirectory() as tmp:
            observations = self.observations() + [{"participant_id": "missing", "episode_id": "missing_episode", "source_id": "openneuro_ds001785", "recording_id": "missing_recording", "detection": ""}]
            jobs = self.jobs(Path(tmp)); extra = jobs[0] | {"job_id": "missing_example", "participant_id": "missing", "episode_id": "missing_episode"}
            examples, excluded = hf.build_examples(observations, jobs + [extra], "detection", self.split())
            self.assertEqual(len(examples), 12)
            self.assertEqual(excluded[0]["reason"], "missing_original_human_label")

    def test_unseen_held_class_preserves_split_and_refuses_training(self):
        observations = self.observations()
        for row in observations:
            row["source_id"] = "monash_wanderim"; row["report_category"] = "task" if int(row["participant_id"][1:]) % 2 == 0 else "wandering"
        observations[4]["report_category"] = "no_recall"
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [row | {"source_id": "monash_wanderim"} for row in self.jobs(Path(tmp))]
            with self.assertRaisesRegex(core.StudyError, "no training support"):
                hf.build_examples(observations, jobs, "report_category", self.split())

    def test_export_audit_checks_signal_and_split_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            exported, _, _, _, _ = self.fixture(Path(tmp))
            self.assertTrue(hf.verify_export(exported)["human_data"])
            signal = Path(core.read_csv(exported / "human_examples.csv")[0]["input_path"])
            signal.write_text("modified signal", encoding="utf8")
            with self.assertRaisesRegex(core.StudyError, "EEG.*changed"):
                hf.verify_export(exported)

    def test_export_written_numeric_fields_are_roundtrip_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared = root / "prepared"; (prepared / "native").mkdir(parents=True)
            jobs = self.jobs(root)
            for row in jobs: row["sfreq_hz"] = 500.0
            core.write_csv(prepared / "native/native_jobs.csv", jobs, list(jobs[0]))
            split_file = root / "fixed_split.json"; core.write_json(split_file, self.split())
            output = root / "new_export"
            with patch.object(hf, "require_scheduled_job", return_value="test-mock-only"), patch.object(hf.model_guided, "_human", return_value=({"paths": {"release_dir": str(root)}, "_base": str(root)}, prepared, self.observations(), [split_file])), patch.object(native, "verify_release", return_value={"status": "test_mock_only"}):
                receipt = hf.export(root / "mock_pointer", root / "mock_scores", root / "mock_qualification", output, split_path=split_file)
            self.assertFalse(receipt["native_training_executed"])
            self.assertEqual(hf.verify_export(output)["train_examples_sha256"], receipt["train_examples_sha256"])

    def test_native_csv_schema_preserves_every_original_export_field(self):
        # Cross-language contract: MATLAB must request the actual exported
        # participant, signal, original label and split fields. This exercises
        # the writer's real CSV header, rather than guessing inferred names.
        with tempfile.TemporaryDirectory() as tmp:
            exported, _, _, _, _ = self.fixture(Path(tmp))
            with (exported / "human_examples.csv").open(encoding="utf-8") as stream:
                header=stream.readline().strip().split(",")
            source=Path(hf.__file__).with_name("human_finetune_native.m").read_text(encoding="utf-8")
            declaration=re.search(r"required_columns\s*=\s*\{([^}]+)\}",source)
            self.assertIsNotNone(declaration)
            required=re.findall(r"'([^']+)'",declaration.group(1))
            self.assertEqual(set(required),set(header));self.assertEqual(set(header),set(hf.EXPORT_FIELDS))

    def test_actual_completion_contract_and_head_only_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, output, _, result, _ = self.fixture(Path(tmp))
            with patch.object(native, "verify_release", return_value={"status": "test_mock_not_native_execution"}):
                self.assertTrue(hf.verify_adaptation(output)["backbone_training"])
            result["backbone_training"] = False
            core.write_json(output / "training_receipt.json", result)
            with self.assertRaisesRegex(core.StudyError, "actual human backbone"):
                hf.verify_adaptation(output)
        with tempfile.TemporaryDirectory() as tmp:
            _, output, _, result, _ = self.fixture(Path(tmp))
            result["optimizer_steps"] = 0
            core.write_json(output / "training_receipt.json", result)
            with self.assertRaisesRegex(core.StudyError, "actual human backbone"):
                hf.verify_adaptation(output)

    def test_backbone_evidence_and_heldout_identity_are_not_booleans(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, output, _, result, evaluation = self.fixture(Path(tmp))
            core.write_json(output / "backbone_updates.json", [{"layer": "conv1", "parameter": "Weights", "before_sha256": "a", "after_sha256": "a"}])
            result["backbone_updates"]["sha256"] = core.sha256_file(output / "backbone_updates.json")
            core.write_json(output / "training_receipt.json", result)
            with self.assertRaisesRegex(core.StudyError, "backbone update evidence"):
                hf.verify_adaptation(output)
        with tempfile.TemporaryDirectory() as tmp:
            _, output, _, _, evaluation = self.fixture(Path(tmp))
            evaluation["test_observation_ids"] = ["a_different_persons_episode"]
            core.write_json(output / "evaluation_receipt.json", evaluation)
            with patch.object(native, "verify_release", return_value={}):
                with self.assertRaisesRegex(core.StudyError, "prediction identities"):
                    hf.verify_adaptation(output)

    def test_evaluation_cannot_use_original_checkpoint_or_another_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, output, _, _, evaluation = self.fixture(Path(tmp))
            evaluation["trained_checkpoint"] = {"path": "original.mat", "sha256": native.CHECKPOINT_SHA256}
            core.write_json(output / "evaluation_receipt.json", evaluation)
            with patch.object(native, "verify_release", return_value={}):
                with self.assertRaisesRegex(core.StudyError, "adapted checkpoint/fixed split"):
                    hf.verify_adaptation(output)

    def test_export_directory_cannot_be_substituted_by_arbitrary_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, output, _, result, _ = self.fixture(Path(tmp))
            result["trained_checkpoint"]["path"] = str(Path(tmp) / "another_model.mat")
            core.write_json(output / "training_receipt.json", result)
            with self.assertRaisesRegex(core.StudyError, "bound to.*output directory"):
                hf.verify_adaptation(output)


if __name__ == "__main__":
    unittest.main()
