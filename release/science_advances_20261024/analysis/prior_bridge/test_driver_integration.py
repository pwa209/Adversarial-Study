"""Integration contracts for adaptation gates and sequential native consumers."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DRIVER_PATH = next((folder / "driver.py" for folder in (HERE, HERE.parent) if (folder / "driver.py").is_file()), None)
if DRIVER_PATH is None:
    raise RuntimeError("Actual full-population driver.py is required")
SPEC = importlib.util.spec_from_file_location("actual_population_system_driver", DRIVER_PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def clinical_records():
    ids = ["openneuro_ds001785_sub-" + str(i).zfill(2) for i in range(1, 18)]
    split = {"train_participant_ids": ids[:11], "validation_participant_ids": ids[11:14], "test_participant_ids": ids[14:]}
    checkpoint = {"path": "/scratch/pwa209/adversarial-study/clinical/adapted_human_task_net.mat", "sha256": "b" * 64}
    clinical = {"status": "completed_actual_human_finetune", "native_training_executed": True,
                "human_data": True, "adaptation_completed": True, "backbone_training": True,
                "actual_optimizer_steps": 5272, "changed_backbone_parameter_count": 6,
                "trained_checkpoint": checkpoint, "split": split}
    adapted = {"status": "actual_human_adapted_activation_geometry_completed", "actual_adapted_native_activations": True,
               "clinical_training_receipt_sha256": "a" * 64, "adapted_checkpoint": copy.deepcopy(checkpoint), "clinical_split": copy.deepcopy(split)}
    return clinical, adapted


class DriverIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=HERE)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        assert self.root.resolve().parent == HERE
        self.temporary.cleanup()

    def test_clinical_dict_checkpoint_and_actual_adaptation_pass(self):
        clinical, adapted = clinical_records()
        result = driver.clinical_gate(clinical, adapted, "a" * 64)
        self.assertTrue(result["actual_human_clinical_backbone_finetuning_verified"])
        self.assertTrue(result["dream_source_held_out"])
        self.assertFalse(result["clinical_training_repeated"])

    def test_clinical_receipt_or_checkpoint_mismatch_rejected(self):
        for alteration in ("training_pin", "checkpoint_sha", "checkpoint_path", "checkpoint_string"):
            clinical, adapted = clinical_records()
            if alteration == "training_pin":
                adapted["clinical_training_receipt_sha256"] = "c" * 64
            elif alteration == "checkpoint_sha":
                adapted["adapted_checkpoint"]["sha256"] = "c" * 64
            elif alteration == "checkpoint_path":
                adapted["adapted_checkpoint"]["path"] += ".different"
            else:
                clinical["trained_checkpoint"] = clinical["trained_checkpoint"]["path"]
            with self.subTest(alteration=alteration), self.assertRaisesRegex(ValueError, "checkpoint"):
                driver.clinical_gate(clinical, adapted, "a" * 64)

    def test_clinical_native_backbone_updates_are_mandatory(self):
        for field in ("native_training_executed", "human_data", "adaptation_completed", "backbone_training",
                      "actual_optimizer_steps", "changed_backbone_parameter_count"):
            clinical, adapted = clinical_records()
            clinical[field] = False if isinstance(clinical[field], bool) else 0
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "backbone adaptation"):
                driver.clinical_gate(clinical, adapted, "a" * 64)

    def test_clinical_exposure_roles_must_match_and_remain_disjoint(self):
        clinical, adapted = clinical_records()
        adapted["clinical_split"]["test_participant_ids"].reverse()
        with self.assertRaisesRegex(ValueError, "partition mismatch"):
            driver.clinical_gate(clinical, adapted, "a" * 64)
        clinical, adapted = clinical_records()
        clinical["split"]["test_participant_ids"][0] = clinical["split"]["train_participant_ids"][0]
        adapted["clinical_split"] = copy.deepcopy(clinical["split"])
        with self.assertRaisesRegex(ValueError, "source-held-out"):
            driver.clinical_gate(clinical, adapted, "a" * 64)

    def test_dream_person_cannot_enter_clinical_exposure_even_with_matching_records(self):
        clinical, adapted = clinical_records()
        clinical["split"]["train_participant_ids"][0] = "dream_tononi_serial_participant01"
        adapted["clinical_split"] = copy.deepcopy(clinical["split"])
        with self.assertRaisesRegex(ValueError, "source-held-out"):
            driver.clinical_gate(clinical, adapted, "a" * 64)

    def value(self):
        return {"files": {name: {"path": str(self.root / (name + ".json")), "sha256": str(i) * 64}
                          for i, name in enumerate(("bank_inventory", "bank_receipt", "scored_bank"), 1)},
                "bank_root": "/scratch/pwa209/adversarial-study/native/simulations"}

    def write_inventory(self, count=664, pair_members=96):
        value = self.value()
        path = Path(value["files"]["bank_inventory"]["path"])
        with path.open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("simulation_id", "status", "pair_id"))
            writer.writeheader()
            writer.writerows({"simulation_id": f"pristine__fixture_{i:03d}", "status": "finite", "pair_id": f"pair{i//2}" if i < pair_members else ""} for i in range(count))
        return value

    def cache_receipt(self):
        cache = self.root / "native_population"
        cache.mkdir()
        (cache / "receipt.json").write_text("{}")
        return cache

    def test_stage_order_is_qualification_then_bridge_then_perturbations(self):
        self.assertEqual(driver.STAGES, ("native_population", "population_bridge", "population_perturbations"))

    def test_loader_gets_exact_frozen_pins_and_every_finite_simulation_required(self):
        value = self.write_inventory()
        inputs = driver.stage_inputs("native_population", value, self.root)
        self.assertEqual(inputs["files"], value["files"])
        self.assertEqual(inputs["bank_root"], value["bank_root"])
        self.assertEqual(len(inputs["required_simulation_ids"]), 664)
        self.assertEqual(len(set(inputs["required_simulation_ids"])), 664)

    def test_frozen_bank_census_change_stops_before_array_loading(self):
        for count, pairs in ((663, 96), (664, 95)):
            value = self.write_inventory(count, pairs)
            with self.subTest(count=count, pairs=pairs), self.assertRaisesRegex(ValueError, "census differs"):
                driver.stage_inputs("native_population", value, self.root)

    def test_neither_consumer_can_start_before_native_receipt(self):
        for stage in ("population_bridge", "population_perturbations"):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "qualification must finish"):
                driver.stage_inputs(stage, self.value(), self.root)

    def test_bridge_gets_consumer_role_pins_and_cache_after_qualification(self):
        cache = self.cache_receipt()
        value = self.value()
        module = types.SimpleNamespace(ROLES=("bank_receipt", "scored_bank"))
        with patch.object(driver.importlib, "import_module", return_value=module):
            inputs = driver.stage_inputs("population_bridge", value, self.root)
        self.assertEqual(inputs["population_cache"], str(cache))
        self.assertEqual(inputs["files"], {key: value["files"][key] for key in module.ROLES})

    def test_perturbations_require_completed_bridge_and_pin_its_compatibility(self):
        cache = self.cache_receipt()
        with self.assertRaisesRegex(ValueError, "Bridge must finish"):
            driver.stage_inputs("population_perturbations", self.value(), self.root)
        bridge = self.root / "population_bridge"
        bridge.mkdir()
        compatibility = bridge / "compatibility.json"
        compatibility.write_text('{"selection_uses_test_targets": false}')
        with self.assertRaisesRegex(ValueError, "Bridge must finish"):
            driver.stage_inputs("population_perturbations", self.value(), self.root)
        (bridge / "receipt.json").write_text("{}")
        inputs = driver.stage_inputs("population_perturbations", self.value(), self.root)
        self.assertEqual(inputs["files"]["native_cache"]["path"], str(cache))
        self.assertEqual(inputs["files"]["bridge_compatibility"], {"path": str(compatibility), "sha256": driver.sha(compatibility)})

    def test_failed_human_gate_prevents_import_and_stage_execution(self):
        with patch.object(driver, "validated_inputs", side_effect=ValueError("human-data fine-tuning gate failed")), \
                patch.object(driver.importlib, "import_module") as importer:
            with self.assertRaisesRegex(ValueError, "fine-tuning gate failed"):
                driver.branch("population_bridge", self.root / "out", 60)
            importer.assert_not_called()
        self.assertFalse((self.root / "out").exists())

    def test_failed_human_gate_prevents_analysis_directory_creation(self):
        with patch.object(driver, "validated_inputs", side_effect=ValueError("human-data fine-tuning gate failed")):
            with self.assertRaisesRegex(ValueError, "fine-tuning gate failed"):
                driver.analysis(self.root / "out", 60)
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
