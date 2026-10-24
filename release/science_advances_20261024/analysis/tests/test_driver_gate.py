"""Actual training prerequisite tests, independent of cluster/runtime qualification."""
import copy
from pathlib import Path
import sys
import unittest

for candidate in (Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
    if (candidate / "driver.py").is_file():
        sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError("The real driver.py is required for training-gate tests")

import driver


def valid_records():
    split = {"train_participant_ids": [f"human{i}" for i in range(26)],
             "validation_participant_ids": [f"human{i}" for i in range(26, 34)],
             "test_participant_ids": [f"human{i}" for i in range(34, 42)]}
    template = {"status": "human_finetuning_completed", "native_training_executed": True,
                "human_data": True, "adaptation_completed": True, "weights_changed": True,
                "test_events_consumed": 0, "split": split, "actual_optimizer_steps": 64,
                "optimizer_steps": 64, "initial_trainable_weights_sha256": "1"*64,
                "final_trainable_weights_sha256": "2"*64,
                "train_events_sha256": "3"*64, "validation_events_sha256": "4"*64,
                "actual_training_order_sha256": "5"*64, "epochs": 2,
                "trainable_parameters": 100000, "split_files_sha256": "6"*64,
                "configuration_sha256": "7"*64, "assay_id": "psych201_brightness_feedback_choice"}
    return {name: dict(copy.deepcopy(template), model=name) for name in ("centaur", "base")}


class DriverTrainingGateTests(unittest.TestCase):
    def test_real_actual_training_passes_without_repeating_it(self):
        proof = driver.training_gate(valid_records())
        self.assertEqual(proof["target"], 42)
        self.assertEqual(proof["split"], [26, 8, 8])
        self.assertTrue(proof["both_actual_human_data_finetunings_verified"])
        self.assertFalse(proof["human_finetuning_repeated"])

    def test_both_matching_model_receipts_required(self):
        for removed in ("centaur", "base"):
            with self.subTest(removed=removed):
                records = valid_records(); del records[removed]
                with self.assertRaises(ValueError):
                    driver.training_gate(records)
        records = valid_records(); records["base"]["model"] = "centaur"
        with self.assertRaises(ValueError):
            driver.training_gate(records)

    def test_target42_counts_cannot_be_replaced_by_another_split(self):
        records = valid_records()
        for name in records:
            person = records[name]["split"]["train_participant_ids"].pop()
            records[name]["split"]["validation_participant_ids"].append(person)
        with self.assertRaisesRegex(ValueError, "target42"):
            driver.training_gate(records)

    def test_same_counts_with_cross_partition_person_overlap_rejected(self):
        records = valid_records()
        for name in records:
            records[name]["split"]["test_participant_ids"][0] = records[name]["split"]["train_participant_ids"][0]
        with self.assertRaisesRegex(ValueError, "Disjoint"):
            driver.training_gate(records)

    def test_heldout_training_consumption_rejected_for_either_model(self):
        for name in ("centaur", "base"):
            records = valid_records(); records[name]["test_events_consumed"] = 1
            with self.subTest(model=name), self.assertRaises(ValueError):
                driver.training_gate(records)

    def test_declared_training_without_native_human_updates_rejected(self):
        for field in ("native_training_executed", "human_data", "adaptation_completed", "weights_changed"):
            for name in ("centaur", "base"):
                records = valid_records(); records[name][field] = False
                with self.subTest(field=field, model=name), self.assertRaises(ValueError):
                    driver.training_gate(records)

    def test_optimizer_steps_and_actual_weight_change_required(self):
        for mutation in ("zero_steps", "step_mismatch", "unchanged_weights"):
            records = valid_records()
            if mutation == "zero_steps":
                records["base"]["actual_optimizer_steps"] = records["base"]["optimizer_steps"] = 0
            elif mutation == "step_mismatch":
                records["base"]["actual_optimizer_steps"] = 63
            else:
                records["base"]["final_trainable_weights_sha256"] = records["base"]["initial_trainable_weights_sha256"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                driver.training_gate(records)

    def test_equal_budget_data_order_assay_and_split_are_mandatory(self):
        for field in ("train_events_sha256", "validation_events_sha256", "actual_training_order_sha256", "epochs",
                      "trainable_parameters", "split_files_sha256", "configuration_sha256", "assay_id"):
            records = valid_records(); value = records["base"][field]
            records["base"][field] = value+1 if isinstance(value, int) else value+"different"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "inputs/budgets differ"):
                driver.training_gate(records)
        records = valid_records()
        records["base"]["split"]["test_participant_ids"] = list(reversed(records["base"]["split"]["test_participant_ids"]))
        with self.assertRaisesRegex(ValueError, "inputs/budgets differ"):
            driver.training_gate(records)


if __name__ == "__main__":
    unittest.main()
