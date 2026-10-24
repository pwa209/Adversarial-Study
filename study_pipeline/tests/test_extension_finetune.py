"""Participant separation, test sealing and actual-adaptation comparison gates."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from adversarial_study.extension_finetune import assign_people,verify_split,explicit_device_map,_verify_training,compare,make_split,verify_comparison,human_observation_fingerprint
from adversarial_study.acquisition import canonical_hash
from adversarial_study.core import sha256_file
from adversarial_study.core import write_csv


def fixture_split(root):
    cfg=root/"config.json";cfg.write_text('{}')
    split=root/"split";split.mkdir();files={};roles={}
    for role in ["train","validation","test"]:
        person=role+"_person";text="Original task <<L>>"
        row={"person_id":person,"session_id":role+"_session","synthetic":False,"text":text,
             "text_sha256":hashlib.sha256(text.encode()).hexdigest()}
        path=split/(role+".jsonl");path.write_text(json.dumps(row)+'\n');files[path.name]=sha256_file(path);roles[person]=role
    receipt={"status":"participant_disjoint_human_split_frozen","configuration_sha256":sha256_file(cfg),"files_sha256":files,"person_roles":roles}
    (split/"split_receipt.json").write_text(json.dumps(receipt));return split,cfg


def fixture_comparison(root):
    split,cfg=fixture_split(root);trains={};evals={}
    for model in ["centaur","base"]:
        train=root/(model+"_train");train.mkdir();(train/"adapter").mkdir();weight=train/"adapter/adapter_model.safetensors"
        weight.write_bytes((model+"_actualtrainedfixture").encode())
        receipt={"status":"human_finetuning_completed","weights_changed":True,"model":model,"native_training_executed":True,
                 "actual_optimizer_steps":1,"optimizer_steps":1,"epochs":1,"trainable_parameters":8,"assay_id":"originalhumanchoice",
                 "split_receipt_sha256":sha256_file(split/"split_receipt.json"),"configuration_sha256":sha256_file(cfg),
                 "adapter_files_sha256":{"adapter_model.safetensors":sha256_file(weight)},
                 "trained_checkpoint":{"path":str(weight),"sha256":sha256_file(weight)},
                 "initial_trainable_weights_sha256":"before_"+model,"final_trainable_weights_sha256":"after_"+model,
                 "train_events_sha256":"sametrain","validation_events_sha256":"samevalidation","actual_training_order_sha256":"sameorder",
                 "split_files_sha256":json.loads((split/"split_receipt.json").read_text())["files_sha256"]}
        (train/"training_receipt.json").write_text(json.dumps(receipt));trains[model]=train
    hashes={name:sha256_file(path/"training_receipt.json") for name,path in trains.items()}
    test=json.loads((split/"test.jsonl").read_text())
    for model,ll in [("centaur",-1.2),("base",-0.2)]:
        evaluate=root/(model+"_eval");evaluate.mkdir();table=evaluate/"heldout_likelihoods.csv"
        rows=[{"session_id":"test_session","person_id":"test_person","response_index":0,"response":"L","conditional_choice_log_likelihood":ll}]
        write_csv(table,rows)
        import csv
        with table.open(newline="") as stream:parsed=list(csv.DictReader(stream))
        receipt={"status":"human_finetuned_heldout_evaluation_completed","model":model,"both_models_human_finetuned_before_test":True,
                 "test_training_updates":0,"split_receipt_sha256":sha256_file(split/"split_receipt.json"),
                 "training_receipt_sha256":hashes[model],"both_training_receipts_sha256":hashes,"test_events_sha256":"sametest",
                 "likelihoods_sha256":sha256_file(table),"heldout_rows_sha256":canonical_hash(parsed),
                 "human_observations_sha256":human_observation_fingerprint([test],rows),"assay_id":"originalhumanchoice",
                 "test_events":1,"test_people":1,"test_observation_ids":["test_session:0"],
                 "test_predictions":{"path":str(table),"sha256":sha256_file(table)}}
        (evaluate/"evaluation_receipt.json").write_text(json.dumps(receipt));evals[model]=evaluate
    output=root/"comparison";result=compare(split,cfg,trains["centaur"],trains["base"],evals["centaur"],evals["base"],output)
    return split,cfg,trains,evals,output,result


class FineTuneTests(unittest.TestCase):
    def test_all_sessions_of_person_stay_together_during_source_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=root/"config.json"
            cfg.write_text(json.dumps({"source_id":"psych201","study":"feedback","people":{"train":1,"validation":1,"test":1},
                                       "seed":42,"require_psych101_unexposed":True,"response_alphabet":["L","R"]}))
            resource={"source_id":"psych201","resource_id":"shard1"};manifest=root/"manifest.json"
            manifest.write_text(json.dumps({"resources":[resource]}))
            rows=[]
            for person in ["1","2","3"]:
                for experiment in ["feedback/exp1","feedback/exp2"]:
                    rows.append((resource,{"sha256":"actualsourcepin"},{"study":"feedback","experiment":experiment,"participant":person,
                                 "text":"Original <<L>>","is_psych101":False,"is_psych101_test":False,"in_eval":False}))
            with mock.patch("adversarial_study.extension_finetune._behaviour_rows",return_value=iter(rows)):
                receipt=make_split(root,manifest,cfg,root/"output")
            self.assertEqual(receipt["sessions"],{"train":2,"validation":2,"test":2})
            self.assertEqual(receipt["independent_people"],{"train":1,"validation":1,"test":1})
            checked,data=verify_split(root/"output",cfg)
            self.assertEqual(sum(len(v) for v in data.values()),6)

    def test_people_assignment_is_disjoint_and_identity_based(self):
        people=["study\0"+str(i) for i in range(12)]
        one=assign_people(people,{"train":6,"validation":3,"test":3},42)
        two=assign_people(list(reversed(people))+people,{"train":6,"validation":3,"test":3},42)
        self.assertEqual(one,two);self.assertEqual(sum(v=="train" for v in one.values()),6)
        self.assertEqual(len(one),12)

    def test_insufficient_real_people_cannot_be_padded_with_sessions(self):
        with self.assertRaisesRegex(ValueError,"insufficient genuine"):
            assign_people(["sameperson"]*100,{"train":2,"validation":1,"test":1},42)

    def test_training_does_not_open_test_outcome_json(self):
        import adversarial_study.extension_finetune as module
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);split,cfg=fixture_split(root);original=module._jsonl
            def reject_test(path):
                if Path(path).name=="test.jsonl":raise AssertionError("test outcomes opened before both systems trained")
                return original(path)
            with mock.patch.object(module,"_jsonl",side_effect=reject_test):
                receipt,data=verify_split(split,cfg,open_test=False)
            self.assertEqual(data["test"],[]);self.assertEqual(len(data["train"]),1)

    def test_participant_leakage_fails_even_with_updated_file_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);split,cfg=fixture_split(root)
            path=split/"test.jsonl";row=json.loads(path.read_text());row["person_id"]="train_person";path.write_text(json.dumps(row)+'\n')
            receipt=json.loads((split/"split_receipt.json").read_text());receipt["files_sha256"][path.name]=sha256_file(path)
            (split/"split_receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,"person crosses"):verify_split(split,cfg)

    def test_explicit_training_map_covers_both_halves_of_original_architecture(self):
        devices=explicit_device_map()
        self.assertEqual(devices["model.layers.0"],0);self.assertEqual(devices["model.layers.39"],0)
        self.assertEqual(devices["model.layers.40"],1);self.assertEqual(devices["model.layers.79"],1)
        self.assertEqual(devices["lm_head"],1)

    def test_released_only_checkpoint_cannot_be_called_human_finetuned(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);split,cfg=fixture_split(root);training=root/"released";training.mkdir()
            (training/"training_receipt.json").write_text(json.dumps({"status":"native_centaur_executed","model":"centaur"}))
            with self.assertRaisesRegex(ValueError,"released-only comparison prohibited"):_verify_training(training,split,cfg,"centaur")

    def test_weight_change_proof_must_match_actual_saved_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);split,cfg=fixture_split(root);training=root/"training";training.mkdir();(training/"adapter").mkdir()
            weights=training/"adapter/adapter_model.safetensors";weights.write_bytes(b"changedtrainedweights")
            receipt={"status":"human_finetuning_completed","weights_changed":True,"model":"centaur",
                     "native_training_executed":True,"actual_optimizer_steps":1,
                     "split_receipt_sha256":sha256_file(split/"split_receipt.json"),"configuration_sha256":sha256_file(cfg),
                     "adapter_files_sha256":{"adapter_model.safetensors":sha256_file(weights)},
                     "trained_checkpoint":{"path":str(weights),"sha256":sha256_file(weights)},
                     "initial_trainable_weights_sha256":"before","final_trainable_weights_sha256":"after"}
            (training/"training_receipt.json").write_text(json.dumps(receipt));weights.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"artifact changed"):_verify_training(training,split,cfg,"centaur")

    def test_no_change_cannot_certify_completed_human_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);split,cfg=fixture_split(root);training=root/"training";training.mkdir()
            receipt={"status":"human_finetuning_completed","weights_changed":True,"model":"centaur",
                     "native_training_executed":True,"actual_optimizer_steps":1,
                     "split_receipt_sha256":sha256_file(split/"split_receipt.json"),"configuration_sha256":sha256_file(cfg),
                     "adapter_files_sha256":{},"initial_trainable_weights_sha256":"same","final_trainable_weights_sha256":"same"}
            (training/"training_receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,"unchanged adapter"):_verify_training(training,split,cfg,"centaur")

    def test_comparison_requires_matched_adaptation_and_keeps_negative_outcome(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);split,cfg,trains,evals,output,result=fixture_comparison(root)
            self.assertEqual(result["equal_person_mean_loglikelihood_difference"],-1.0)
            self.assertTrue(result["all_models_human_finetuned"])
            before={str(p):sha256_file(p) for p in root.rglob("*") if p.is_file()}
            verified=verify_comparison(output)
            self.assertEqual(verified["status"],"human_finetuned_comparison_artifacts_verified")
            self.assertEqual(before,{str(p):sha256_file(p) for p in root.rglob("*") if p.is_file()})
            changed=json.loads((trains["base"]/"training_receipt.json").read_text());changed["optimizer_steps"]=2
            (trains["base"]/"training_receipt.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"identical human training"):
                compare(split,cfg,trains["centaur"],trains["base"],evals["centaur"],evals["base"],root/"invalidcomparison")

    def test_verification_rejects_tampered_native_adapter_after_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            split,cfg,trains,evals,output,result=fixture_comparison(Path(folder))
            (trains["centaur"]/"adapter/adapter_model.safetensors").write_bytes(b"changedaftercomparison")
            with self.assertRaisesRegex(ValueError,"weight artifact changed"):verify_comparison(output)

    def test_verification_rejects_tampered_native_heldout_table(self):
        with tempfile.TemporaryDirectory() as folder:
            split,cfg,trains,evals,output,result=fixture_comparison(Path(folder))
            path=evals["base"]/"heldout_likelihoods.csv";path.write_text(path.read_text().replace("-0.2","-2.0"))
            with self.assertRaisesRegex(ValueError,"heldout result table changed"):verify_comparison(output)

    def test_rehashing_fabricated_paired_result_cannot_bypass_source_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            split,cfg,trains,evals,output,result=fixture_comparison(Path(folder));table=output/"paired_heldout_comparison.csv"
            rows=[{"session_id":"test_session","response_index":0,"person_id":"test_person","centaur_minus_base_loglikelihood":5.0}]
            write_csv(table,rows)
            import csv
            with table.open(newline="") as stream:parsed=list(csv.DictReader(stream))
            receipt=json.loads((output/"comparison_receipt.json").read_text());receipt["paired_table"]["sha256"]=sha256_file(table)
            receipt["paired_table"]["rows_sha256"]=canonical_hash(parsed);(output/"comparison_receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,"differ from actual heldout native evaluations"):verify_comparison(output)

    def test_training_artifact_must_stay_within_own_adapter_role(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);split,cfg,trains,evals,output,result=fixture_comparison(root)
            path=trains["base"]/"training_receipt.json";receipt=json.loads(path.read_text())
            external=root/"other_weights.safetensors";external.write_bytes(b"outsideownrole")
            receipt["adapter_files_sha256"][str(external)]=sha256_file(external);path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,"escapes its adapter role"):_verify_training(trains["base"],split,cfg,"base")


if __name__=="__main__":unittest.main()
