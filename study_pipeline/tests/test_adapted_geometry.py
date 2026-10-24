"""Actual file/vector bindings and same-channel geometry contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from adversarial_study import core
from native_tools import adapted_geometry as adapted


def fixture(root):
    jobs=[];outputs=[];predictions=[];values={}
    for person in ["person_one","person_two"]:
        for channel in ["actual_C3","actual_P3"]:
            for episode in ["a","b"]:
                identity=person+channel+episode;path=root/("activation_"+identity+".mat");path.write_bytes(identity.encode())
                jobs.append({"job_id":identity,"episode_id":person+episode,"participant_id":person,"source_id":"actual_source",
                             "channel_name":channel,"clinical_split":"test"})
                outputs.append({"job_id":identity,"path":path.name,"sha256":core.sha256_file(path)})
                predictions.append({"job_id":identity,"status":"ok","error":""})
                values[str(path.resolve())]={"conv_a":np.asarray([1.,0. if episode=="a" else 1.])}
    core.write_csv(root/"adapted_predictions.csv",predictions)
    return jobs,{"layers":["conv_a"]},{"activation_outputs":outputs},values


class AdaptedGeometryTests(unittest.TestCase):
    def test_pairs_never_join_different_people_or_actual_channels(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);jobs,plan,execution,values=fixture(root)
            with mock.patch.object(adapted,"_load_mat",side_effect=lambda path:values[str(path)]):inventory,pairs=adapted._measure(root,plan,execution,jobs)
            self.assertEqual(len(inventory),8);self.assertEqual(len(pairs),4)
            self.assertEqual({r["metric_status"] for r in pairs},{"computed"})
            for row in pairs:self.assertTrue(row["job_id_a"].startswith(row["participant_id"]+row["channel_name"]))

    def test_streaming_sink_emits_every_pair_without_retaining_cohort_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);jobs,plan,execution,values=fixture(root);emitted=[]
            with mock.patch.object(adapted,"_load_mat",side_effect=lambda path:values[str(path)]):
                inventory,pairs=adapted._measure(root,plan,execution,jobs,pair_sink=emitted.append)
            self.assertEqual(len(inventory),8);self.assertEqual(pairs,[]);self.assertEqual(len(emitted),4)

    def test_rehashed_distance_table_must_still_match_actual_native_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);jobs,plan,execution,values=fixture(root)
            with mock.patch.object(adapted,"_load_mat",side_effect=lambda path:values[str(path)]):
                first=adapted._measure(root,plan,execution,jobs)[1];self.assertAlmostEqual(first[0]["cosine_distance"],1-1/np.sqrt(2))
                values[str((root/execution["activation_outputs"][0]["path"]).resolve())]["conv_a"]*=0
                second=adapted._measure(root,plan,execution,jobs)[1]
            self.assertIn("undefined_zero_norm",{r["metric_status"] for r in second})
            self.assertEqual(len(first),len(second))

    def test_changed_native_activation_file_is_rejected_before_vector_use(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);jobs,plan,execution,values=fixture(root);(root/execution["activation_outputs"][0]["path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(core.StudyError,"activation bytes changed"):adapted._measure(root,plan,execution,jobs)

    def test_activation_path_cannot_escape_its_native_output_role(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);jobs,plan,execution,values=fixture(root);execution["activation_outputs"][0]["path"]="../outside.mat"
            with self.assertRaisesRegex(core.StudyError,"escapes its role"):adapted._measure(root,plan,execution,jobs)

    def test_failed_native_job_retains_all_unavailable_pair_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);jobs,plan,execution,values=fixture(root);rows=core.read_csv(root/"adapted_predictions.csv")
            rows[0]["status"]="failed";rows[0]["error"]="actual native input failed";core.write_csv(root/"adapted_predictions.csv",rows)
            execution["activation_outputs"]=execution["activation_outputs"][1:]
            with mock.patch.object(adapted,"_load_mat",side_effect=lambda path:values[str(path)]):inventory,pairs=adapted._measure(root,plan,execution,jobs)
            self.assertEqual(len(pairs),4);self.assertEqual(sum(r["metric_status"]=="activation_unavailable" for r in pairs),1)

    def test_original_or_unqualified_native_forward_cannot_certify_adapted_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaisesRegex(core.StudyError,"not qualified"):
                adapted._verify_execution(root,{}, {"status":"native_batch_completed","forward_correspondence_verified":True,"activation_repeats_verified":True})

    def test_public_verifier_reconstructs_rehashed_pair_table_from_original_bound_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder);root=parent/"geometry";root.mkdir();jobs,plan,execution,values=fixture(root)
            states=parent/"states";states.mkdir();trained=parent/"trained";trained.mkdir()
            core.write_json(states/"receipt.json",{"fixture":"state"});core.write_json(trained/"training_receipt.json",{"fixture":"native human adaptation"})
            adaptation={"trained_checkpoint":{"path":str(trained/"adapted_human_task_net.mat"),"sha256":"actual_fixture_pin"},
                        "split":{"train_participant_ids":["train"],"validation_participant_ids":["validation"],"test_participant_ids":["person_one","person_two"]}}
            core.write_csv(root/"adapted_jobs.csv",jobs)
            plan.update(adapted_checkpoint=adaptation["trained_checkpoint"],states_receipt_sha256=core.sha256_file(states/"receipt.json"),
                        clinical_training_receipt_sha256=core.sha256_file(trained/"training_receipt.json"),jobs_sha256=core.sha256_file(root/"adapted_jobs.csv"),
                        source_code_sha256=core.sha256_file(adapted.__file__),helper_sha256=core.sha256_file(Path(adapted.__file__).with_suffix(".m")))
            core.write_json(root/"export_plan.json",plan)
            execution.update(status="adapted_native_activation_export_completed",forward_correspondence_verified=True,activation_repeats_verified=True,
                             adapted_checkpoint_sha256=plan["adapted_checkpoint"]["sha256"],jobs_sha256=plan["jobs_sha256"],
                             export_plan_sha256=core.sha256_file(root/"export_plan.json"),helper_sha256=plan["helper_sha256"],
                             clinical_training_receipt_sha256=plan["clinical_training_receipt_sha256"],states_receipt_sha256=plan["states_receipt_sha256"],
                             predictions_sha256=core.sha256_file(root/"adapted_predictions.csv"))
            core.write_json(root/"native_receipt.json",execution)
            with mock.patch.object(adapted,"_load_mat",side_effect=lambda path:values[str(path)]):
                inventory,pairs=adapted._measure(root,plan,execution,jobs)
                core.write_json(root/"activation_inventory.json",inventory);core.write_csv(root/"layer_distances.csv",pairs)
                receipt={"status":"actual_human_adapted_activation_geometry_completed","states_path":str(states),"trained_path":str(trained),
                         "adapted_checkpoint":adaptation["trained_checkpoint"],"clinical_split":adaptation["split"],"layers":plan["layers"],
                         "states_receipt_sha256":plan["states_receipt_sha256"],"clinical_training_receipt_sha256":plan["clinical_training_receipt_sha256"],
                         "export_plan_sha256":core.sha256_file(root/"export_plan.json"),"native_receipt_sha256":core.sha256_file(root/"native_receipt.json"),
                         "distance_table_sha256":core.sha256_file(root/"layer_distances.csv"),"activation_inventory_sha256":core.sha256_file(root/"activation_inventory.json")}
                core.write_json(root/"receipt.json",receipt)
                with mock.patch.object(adapted,"source_jobs",return_value=({},adaptation,jobs,[],plan["layers"])):
                    before={p.name:core.sha256_file(p) for p in root.iterdir()};adapted.verify_adapted_geometry(root)
                    self.assertEqual(before,{p.name:core.sha256_file(p) for p in root.iterdir()})
                    pairs[0]["cosine_distance"]=.99;core.write_csv(root/"layer_distances.csv",pairs)
                    receipt["distance_table_sha256"]=core.sha256_file(root/"layer_distances.csv");core.write_json(root/"receipt.json",receipt)
                    with self.assertRaisesRegex(core.StudyError,"Rehashed adapted pair geometry"):
                        adapted.verify_adapted_geometry(root)


if __name__=="__main__":unittest.main()
