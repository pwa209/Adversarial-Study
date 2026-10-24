"""Synthetic software fixtures; no empirical or native qualification claims."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from adversarial_study import core, model_guided as guided
from adversarial_study import human_generative_adaptation as adaptation


def human(person, number=0, values=None):
    values = values if values is not None else np.asarray([.1,.2,10.,.6,20.])+number*.01
    return {"participant_id": person, "episode_id": person+"-episode-"+str(number), "source_id": "SOFTWARE_TEST",
            "site_id": "TEST_site", "role": "development", "measurement_status": "actual_multichannel_eeg",
            "score_origin": "native_frozen_v2", "synthetic": False,
            **dict(zip(adaptation.FEATURES, values))}


def candidate(name, values=None, account="gain"):
    values = values if values is not None else [.1,.2,10.,.6,20.]
    return {"simulation_id": name, "account_class": account, "variant": "TEST_IMPLEMENTATION",
            "status": "finite", "score": .5, "score_origin": "native_frozen_v2", "config_sha256": "TEST_config_"+name,
            "source_output_path": "TEST_output_"+name, "source_output_sha256": "TEST_hash_"+name,
            "published_config_id": "TEST_published", "param_gain_factor": "1.1",
            **dict(zip(adaptation.FEATURES, values))}


class GenerativeAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.people = ["TEST-person-"+str(i) for i in range(12)]
        self.bank = [candidate("native-A"), candidate("native-B", [.3,.4,14.,.8,24.]),
                     candidate("native-C", [.1,.2,10.,.6,20.], "coupling")]

    def tearDown(self):
        self.temporary.cleanup()

    def write_fit(self, rows, name="fit"):
        directory = guided._new_output(self.root/name)
        core.write_csv(directory/"human_observations.csv", rows, sorted({k for r in rows for k in r}))
        core.write_csv(directory/"scored_bank.csv", self.bank)
        core.write_csv(directory/"retained_simulation_failures.csv", [{"simulation_id": "failed-1", "status": "failed"}])
        bank = self.root/(name+"-native-inventory.csv")
        jobs = self.root/(name+"-jobs.csv")
        scores = self.root/(name+"-scores.csv")
        qualification = self.root/(name+"-qualification.json")
        bank_receipt = self.root/(name+"-bank-receipt.json")
        core.write_csv(bank, self.bank)
        jobs.write_text("job_id\nTEST-only\n", encoding="utf-8")
        scores.write_text("score\n0.5\n", encoding="utf-8")
        core.write_json(qualification, {"status": "TEST_FIXTURE_ONLY"})
        core.write_json(bank_receipt, {"status": "native_simulation_bank_completed", "inventory_sha256": core.sha256_file(bank), "jobs_sha256": core.sha256_file(jobs)})
        core.write_json(Path(str(scores)+".receipt.json"), {"status": "native_batch_completed", "scores_sha256": core.sha256_file(scores),
                        "jobs_sha256": core.sha256_file(jobs), "qualification_sha256": core.sha256_file(qualification)})
        guided._finish(directory, [bank,jobs,scores,Path(str(scores)+".receipt.json"),qualification,bank_receipt], "model_guided_fit_completed")
        return directory

    def native_fixture(self):
        # Only the external NFTsim/MATLAB runtime boundary is mocked. Actual
        # hashes, parent linkage, numerical adaptation and split checks execute.
        return patch.object(guided, "_verified_bank", return_value=(self.bank,[{"simulation_id":"failed-1"}],{}))

    def test_partition_is_outcome_blind_reproducible_and_complete(self):
        split = adaptation.participant_split(self.people)
        self.assertEqual(split, adaptation.participant_split(list(reversed(self.people))))
        assigned = adaptation._assignments(split)
        self.assertEqual(set(assigned), set(self.people))
        self.assertEqual(set(assigned.values()), {"train", "validation", "test"})
        with self.assertRaises(core.StudyError):
            adaptation.participant_split(self.people+[self.people[0]])
        changed = json.loads(json.dumps(split))
        changed["test_participant_ids"][0] = split["train_participant_ids"][0]
        with self.assertRaises(core.StudyError):
            adaptation._assignments(changed)

    def test_full_stage_adapts_actual_parameters_and_retains_all_accounts(self):
        source = self.write_fit([human(person,j, np.asarray([.1,.2,10.,.6,20.])+j*.1)
                                 for person in self.people for j in range(2)])
        with self.native_fixture():
            result = adaptation.adapt(source, self.root/"adapted")
            verified = adaptation.verify_adaptation(self.root/"adapted")
        self.assertTrue(result["human_parameter_fitting_executed"])
        self.assertFalse(result["native_training_executed"])
        self.assertEqual(verified["assay_id"], adaptation.ASSAY_ID)
        fitted = core.read_json(self.root/"adapted/parameter_fit.json")
        self.assertEqual(len(fitted["accounts"]),2)
        self.assertFalse(fitted["physiological_parameter_identification_established"])
        predictions = core.read_csv(self.root/"adapted/heldout_predictions.csv")
        for row in predictions:
            self.assertNotIn(row["held_feature"], row["selection_features"].split(";"))
            self.assertEqual(row["split"],"test")
        evaluation = core.read_json(self.root/"adapted/evaluation_receipt.json")
        self.assertFalse(evaluation["independent_human_validation"])
        self.assertEqual(len(predictions), evaluation["test_episodes"]*2*len(adaptation.FEATURES))
        self.assertEqual(result["retained_native_failures"],1)

    def test_test_perturbation_cannot_change_fitted_parameters_or_selection(self):
        split = adaptation.participant_split(self.people)
        rows = [human(person) for person in self.people]
        altered = [dict(r) for r in rows]
        for row in altered:
            if row["participant_id"] in split["test_participant_ids"]:
                row.update({k:float(row[k])+10000 for k in adaptation.FEATURES})
        a, b = self.write_fit(rows,"a"), self.write_fit(altered,"b")
        with self.native_fixture():
            adaptation.adapt(a,self.root/"a-result")
            adaptation.adapt(b,self.root/"b-result")
        self.assertEqual(core.read_json(self.root/"a-result/parameter_fit.json"),core.read_json(self.root/"b-result/parameter_fit.json"))
        ea = core.read_json(self.root/"a-result/evaluation_receipt.json")
        eb = core.read_json(self.root/"b-result/evaluation_receipt.json")
        self.assertNotEqual(ea["accounts"][0]["metrics"]["equal_participant_standardized_mse"], eb["accounts"][0]["metrics"]["equal_participant_standardized_mse"])

    def test_validation_cannot_change_training_scales_and_support(self):
        train = [human("p1"),human("p2",1)]
        fit = adaptation.train_calibration(train,self.bank[:2],1.)
        x = np.asarray([[float(r[k]) for k in adaptation.FEATURES] for r in train])
        np.testing.assert_allclose(fit["training_mean"], x.mean(axis=0))
        self.assertTrue(set(p["simulation_id"] for p in fit["supported_parameters"]) <= {r["simulation_id"] for r in self.bank})
        _, result = adaptation.predict([human("val",values=np.ones(5)*1e4)],self.bank[:2],fit,"gain","TEST_IMPLEMENTATION","validation")
        self.assertGreater(result["equal_participant_standardized_mse"],1.)
        self.assertGreater(result["out_of_domain_predictions"],0)
        self.assertEqual(fit, adaptation.train_calibration(train,self.bank[:2],1.))

    def test_held_target_cannot_select_native_configuration(self):
        fitted = adaptation.train_calibration([human("p1"),human("p2",values=[.3,.4,14.,.8,24.])],self.bank[:2],0.)
        first, _ = adaptation.predict([human("test")],self.bank[:2],fitted,"gain","TEST_IMPLEMENTATION","test")
        modified = human("test"); modified[adaptation.FEATURES[0]]=9999.
        second, _ = adaptation.predict([modified],self.bank[:2],fitted,"gain","TEST_IMPLEMENTATION","test")
        first_target = next(r for r in first if r["held_feature"]==adaptation.FEATURES[0])
        second_target = next(r for r in second if r["held_feature"]==adaptation.FEATURES[0])
        self.assertEqual(first_target["simulation_id"],second_target["simulation_id"])
        self.assertEqual(first_target["predicted"],second_target["predicted"])

    def test_weak_test_error_does_not_prevent_actual_adaptation_verification(self):
        split = adaptation.participant_split(self.people)
        rows = [human(p,values=np.ones(5)*1e4 if p in split["test_participant_ids"] else None) for p in self.people]
        source = self.write_fit(rows)
        with self.native_fixture():
            result = adaptation.adapt(source,self.root/"result")
            adaptation.verify_adaptation(self.root/"result")
        self.assertTrue(result["human_parameter_fitting_executed"])
        metrics = core.read_json(self.root/"result/evaluation_receipt.json")["accounts"][0]["metrics"]
        self.assertGreater(metrics["equal_participant_standardized_mse"],1e6)

    def test_missing_human_measurement_retains_exclusion_denominator(self):
        rows = [human(p) for p in self.people]
        rows[0][adaptation.FEATURES[0]]=""
        source = self.write_fit(rows)
        with self.native_fixture():
            result = adaptation.adapt(source,self.root/"result")
        self.assertEqual(result["denominator_episodes"],12)
        self.assertEqual(result["missing_measurement_episodes"],1)
        self.assertEqual(len(core.read_csv(self.root/"result/excluded_observations.csv")),1)

    def test_insufficient_participants_are_stage_success_without_completed_adaptation(self):
        source = self.write_fit([human(p) for p in self.people[:4]])
        with self.native_fixture():
            result = adaptation.adapt(source,self.root/"result")
        self.assertEqual(result["status"],"human_generative_adaptation_completed")
        self.assertFalse(result["human_parameter_fitting_executed"])
        self.assertFalse(result["heldout_evaluation_executed"])
        with self.assertRaises(core.StudyError):
            adaptation.verify_adaptation(self.root/"result")

    def test_changed_native_or_adaptation_artifact_cannot_be_verified(self):
        source = self.write_fit([human(p) for p in self.people])
        with self.native_fixture():
            adaptation.adapt(source,self.root/"result")
        with (self.root/"result/parameter_fit.json").open("a") as stream:stream.write(" ")
        with self.assertRaises(core.StudyError):
            adaptation.verify_adaptation(self.root/"result")
        with (source/"scored_bank.csv").open("a") as stream:stream.write(" ")
        with self.assertRaises(core.StudyError):
            adaptation.verify_fit(source)

    def test_failed_bank_status_rejected_even_when_fit_is_hash_pinned(self):
        self.bank[0]["status"]="failed"
        source = self.write_fit([human(p) for p in self.people])
        with self.native_fixture(), self.assertRaises(core.StudyError):
            adaptation.adapt(source,self.root/"result")

    def test_training_receipt_is_written_before_test_evaluation_read(self):
        source = self.write_fit([human(p) for p in self.people])
        original = adaptation._read_partition
        def check(path,assignments,partition,**kwargs):
            if partition=="test":
                self.assertTrue((self.root/"result/training_receipt.json").is_file())
                self.assertTrue((self.root/"result/parameter_fit.json").is_file())
            return original(path,assignments,partition,**kwargs)
        with self.native_fixture(), patch.object(adaptation,"_read_partition",side_effect=check):
            adaptation.adapt(source,self.root/"result")

    def test_rehashed_false_calibration_flag_still_fails_actual_training_reconstruction(self):
        source = self.write_fit([human(p) for p in self.people])
        result = self.root/"result"
        with self.native_fixture():
            adaptation.adapt(source,result)
        parameters = core.read_json(result/"parameter_fit.json")
        parameters["accounts"][0]["training_mean"][0] += 1
        core.write_json(result/"parameter_fit.json", parameters)
        training = core.read_json(result/"training_receipt.json")
        training["trained_checkpoint"]["sha256"] = core.sha256_file(result/"parameter_fit.json")
        core.write_json(result/"training_receipt.json",training)
        evaluation = core.read_json(result/"evaluation_receipt.json")
        evaluation["parameter_fit_sha256"] = training["trained_checkpoint"]["sha256"]
        evaluation["training_receipt_sha256"] = core.sha256_file(result/"training_receipt.json")
        core.write_json(result/"evaluation_receipt.json",evaluation)
        receipt = core.read_json(result/"receipt.json")
        for name in receipt["outputs"]:
            receipt["outputs"][name] = core.sha256_file(result/name)
        core.write_json(result/"receipt.json",receipt)
        with self.native_fixture(), self.assertRaisesRegex(core.StudyError,"not fitted to the declared human training"):
            adaptation.verify_adaptation(result)

    def test_equal_participant_training_mean_ignores_trial_count_imbalance(self):
        rows = [human("one",values=np.ones(5))]*10+[human("two",values=np.ones(5)*3)]
        fitted = adaptation.train_calibration(rows,self.bank[:2],0.)
        np.testing.assert_allclose(fitted["training_mean"],np.ones(5)*2)

    def test_finite_input_numerical_overflow_is_explicit_failure(self):
        with self.assertRaisesRegex(core.StudyError,"finite numerical precision"):
            adaptation.train_calibration([human("one",values=np.ones(5)*1e300),human("two",values=np.ones(5)*-1e300)],self.bank[:2],1.)


if __name__ == "__main__":
    unittest.main()
