"""Protect human-data training before cross-system prediction/comparison.

These tests exercise graph admission only. Runtime weight/split receipts remain
required independently; no synthetic fixture establishes human fine-tuning.
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location("finetuning_plan_controller",Path(__file__).parents[1]/"cluster/controller.py")
controller=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(controller)


def stage(name,role,parents=()):
    return {"id":name,"command":["UNIT_FIXTURE_ONLY"],"cpus":1,"memory":"1G","walltime":"00:10:00",
            "evidence_role":role,"depends_on":list(parents)}


class HumanTrainingPreconditions(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/"SOURCE_MANIFEST.json").write_text('{"files":{}}')
        required=["centaur_train","base_train"]
        a=stage("centaur_train","native_human_data_weight_finetuning")
        b=stage("base_train","native_human_data_weight_finetuning")
        bridge=stage("training_receipts","qualification",required)
        evaluate=stage("evaluate","post_finetuning_protected_human_prediction",["training_receipts"])
        evaluate["required_human_training_stages"]=required.copy()
        compare=stage("compare","human_finetuned_cross_system_comparison",["evaluate"])
        compare["required_human_training_stages"]=required.copy()
        self.plan={"source":str(self.root),"source_manifest_sha256":hashlib.sha256((self.root/"SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
                   "max_allocated_cpus":4,"controller_cpus":1,"max_allocated_gpus":0,
                   "cross_system_policy":{"human_finetuning_before_comparison":True},"stages":[a,b,bridge,evaluate,compare]}

    def test_both_training_stages_transitively_precede_prediction_and_comparison(self):
        controller.validate_plan(self.plan)

    def test_one_training_stage_removed_from_ancestry_is_rejected(self):
        self.plan["stages"][2]["depends_on"]=["centaur_train"]
        with self.assertRaisesRegex(ValueError,"Every required human"):controller.validate_plan(self.plan)

    def test_evaluation_stage_cannot_stand_in_for_weight_training(self):
        self.plan["stages"][1]["evidence_role"]="native_pretrained_runtime_qualification"
        with self.assertRaisesRegex(ValueError,"actual native human-data weight"):controller.validate_plan(self.plan)

    def test_comparison_declaring_only_evaluation_as_training_is_rejected(self):
        self.plan["stages"][-1]["required_human_training_stages"]=["evaluate"]
        with self.assertRaisesRegex(ValueError,"actual native human-data weight"):controller.validate_plan(self.plan)

    def test_protected_prediction_cannot_open_test_before_all_human_training(self):
        self.plan["stages"][-2]["depends_on"]=["centaur_train"]
        with self.assertRaisesRegex(ValueError,"Every required human"):controller.validate_plan(self.plan)

    def test_missing_empty_duplicate_or_unknown_declared_train_stages_are_rejected(self):
        target=self.plan["stages"][-1]
        for required in (None,[],["centaur_train","centaur_train"],["absent"],"centaur_train"):
            with self.subTest(required=required):
                target["required_human_training_stages"]=required
                with self.assertRaises(ValueError):controller.validate_plan(self.plan)

    def test_external_input_cannot_masquerade_as_completed_weight_training(self):
        self.plan["stages"][1]["external_input"]="Human recruitment pending"
        with self.assertRaisesRegex(ValueError,"actual native human-data weight"):controller.validate_plan(self.plan)

    def test_legacy_plan_without_policy_retains_original_validation(self):
        self.plan.pop("cross_system_policy")
        self.plan["stages"][-2].pop("required_human_training_stages")
        self.plan["stages"][-1].pop("required_human_training_stages")
        controller.validate_plan(self.plan)

    def test_result_direction_or_performance_is_not_an_admission_condition(self):
        for training in self.plan["stages"][:2]:
            training.update(expected_effect="null_or_adverse",training_loss_status="unknown",minimum_accuracy=None)
        controller.validate_plan(self.plan)


if __name__=="__main__":unittest.main()
