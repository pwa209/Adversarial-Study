"""Numerical fixtures test separation/audits; they never certify native training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from adversarial_study import core
from adversarial_study import representation_bridge as bridge


def fixture(root):
    states=root/"states";states.mkdir();adapted=root/"adapted";adapted.mkdir();clinical=root/"clinical";clinical.mkdir()
    cfg=root/"config.json";core.write_json(cfg,{"seed":20261004,"representations":{"ridge_grid":[.01,.1],"bootstrap_samples":20,
             "minimum_training_people":3,"minimum_validation_people":1,"minimum_test_people":2,"maximum_pairs_per_participant":2000,
             "numeric_outcomes":["detection","confidence"],"nominal_outcomes":["report_category"]}})
    people=[("p1","train"),("p2","train"),("p3","train"),("p4","validation"),("p5","test"),("p6","test")]
    roles=dict(people);clinical_roles={"other_train":"train","other_validation":"validation","p5":"test"}
    adaptation={"trained_checkpoint":{"path":str(clinical/"actual_fixture_checkpoint.mat"),"sha256":"fixture_native_weight_pin"},
                "split":{"train_participant_ids":["other_train"],"validation_participant_ids":["other_validation"],"test_participant_ids":["p5"]}}
    core.write_json(clinical/"training_receipt.json",adaptation)
    core.write_json(states/"receipt.json",{"status":"fixture_state_receipt"})
    core.write_json(states/"split.json",{"train_participant_ids":[p for p,r in people if r=="train"],
         "validation_participant_ids":[p for p,r in people if r=="validation"],"test_participant_ids":[p for p,r in people if r=="test"]})
    rows=[];arrays={};jobs=[];head=[];distances=[]
    for number,(person,role) in enumerate(people,1):
        for episode in ["a","b"]:
            token=person+episode;offset=int(episode=="b");time=np.linspace(0,2*np.pi,16)
            arrays[token+"__state"]=np.c_[np.sin(time)*(1+number*.1+offset*.2),np.cos(time)*(1+offset*.1)]
            row={"participant_id":person,"source_id":"actual_source_fixture","episode_id":token,"recording_id":person+"_recording",
                 "split":role,"status":"included","source_model_key":"actual_montage_fixture","state_token":token,
                 "report_category":"focus" if not offset else ["blank","focus","wandering"][number%3],
                 "detection":int(number%2==offset),"confidence":number+offset*.2}
            row.update({field:float(number*.1+offset*(i+1)*.02) for i,field in enumerate(bridge.SPECTRAL+bridge.DYNAMICS)})
            rows.append(row);job={"job_id":token,"episode_id":token,"participant_id":person,"source_id":row["source_id"],"channel_name":"actual_channel"}
            jobs.append(job);prob=.15+number*.05+offset*.04
            head.append({"job_id":token,"status":"ok","human_class_probabilities":json.dumps([prob,1-prob])})
        for layer in ["conv_a","conv_b"]:
            distances.append({"participant_id":person,"source_id":"actual_source_fixture","channel_name":"actual_channel","layer_name":layer,
                              "episode_id_a":person+"a","episode_id_b":person+"b","cosine_distance":number*.05+(0.02 if layer=="conv_b" else 0),"metric_status":"computed"})
    core.write_csv(states/"rows.csv",rows);np.savez(states/"states.npz",**arrays)
    core.write_csv(adapted/"adapted_jobs.csv",jobs);core.write_csv(adapted/"adapted_predictions.csv",head);core.write_csv(adapted/"layer_distances.csv",distances)
    core.write_json(adapted/"receipt.json",{"layers":["conv_a","conv_b"],"states_path":str(states),"trained_path":str(clinical),"adapted_checkpoint":adaptation["trained_checkpoint"]})
    def verified(*args):
        inputs={"states":str(states),"clinical_trained":str(clinical),"adapted_geometry":str(adapted),"config":str(cfg),"geometry":[]}
        pins={"states":core.sha256_file(states/"receipt.json"),"clinical_training":core.sha256_file(clinical/"training_receipt.json"),
              "adapted_geometry":core.sha256_file(adapted/"receipt.json"),"config":core.sha256_file(cfg)}
        return inputs,pins,[],core.read_csv(states/"rows.csv"),roles,clinical_roles,adaptation
    return states,clinical,adapted,cfg,verified,clinical_roles


class RepresentationBridgeTests(unittest.TestCase):
    def test_configuration_uses_root_canonical_keys_and_rejects_conflicting_aliases(self):
        result=bridge.effective_config({"representations":{"ridge_grid":[.001,.01],"bootstrap_samples":37,"minimum_validation_people":1}})
        self.assertEqual(result["ridge_grid"],[.001,.01]);self.assertEqual(result["bootstrap_samples"],37)
        with self.assertRaisesRegex(core.StudyError,"Conflicting"):
            bridge.effective_config({"representations":{"ridge_grid":[1],"ridge_alphas":[2]}})

    def test_equal_participant_fit_does_not_reweight_person_with_duplicate_observations(self):
        x=np.asarray([[0.],[1.],[2.]]);y=np.asarray([0.,1.,3.]);people=["a","b","c"]
        first=bridge.fit_ridge(x,y,people,.1)
        repeated=bridge.fit_ridge(np.r_[x,x[:1],x[:1]],np.r_[y,y[:1],y[:1]],people+["a","a"],.1)
        for key in first:np.testing.assert_allclose(first[key],repeated[key],atol=1e-12)

    def test_test_outcome_changes_neither_training_weights_nor_selected_layer(self):
        with tempfile.TemporaryDirectory() as folder:
            states,clinical,adapted,cfg,verified,clinical_roles=fixture(Path(folder))
            with mock.patch.object(bridge,"_verified_inputs",side_effect=verified):
                first=bridge._analyse(states,clinical,adapted,cfg,[])
                rows=core.read_csv(states/"rows.csv")
                for row in rows:
                    if row["split"]=="test":row["confidence"]="1000" if row["episode_id"].endswith("a") else "-2000";row["report_category"]="unseen_original_report"
                core.write_csv(states/"rows.csv",rows)
                second=bridge._analyse(states,clinical,adapted,cfg,[])
            self.assertEqual(first[3],second[3]);self.assertEqual(set(first[2]),set(second[2]))
            for key in first[2]:np.testing.assert_array_equal(first[2][key],second[2][key])
            self.assertNotEqual(first[4],second[4])
            self.assertTrue(any(m.get("unseen_original_categories") for m in second[-1]["metrics"]))

    def test_clinical_training_or_validation_people_are_excluded_from_comparative_test(self):
        with tempfile.TemporaryDirectory() as folder:
            states,clinical,adapted,cfg,verified,clinical_roles=fixture(Path(folder));clinical_roles["p5"]="validation"
            with mock.patch.object(bridge,"_verified_inputs",side_effect=verified):result=bridge._analyse(states,clinical,adapted,cfg,[])
            census=result[6];self.assertEqual(next(r for r in census if r["participant_id"]=="p5")["reason"],"clinical_adaptation_train_or_validation_person_excluded_from_comparative_test")
            self.assertNotIn("p5",result[-1]["comparative_test_people"])
            self.assertEqual(result[-1]["maps_fitted"],0)  # No padding of the one remaining test person.

    def test_missing_original_outcomes_are_counted_and_not_filled(self):
        with tempfile.TemporaryDirectory() as folder:
            states,clinical,adapted,cfg,verified,clinical_roles=fixture(Path(folder));rows=core.read_csv(states/"rows.csv")
            for row in rows:
                if row["episode_id"]=="p6b":row["confidence"]=""
            core.write_csv(states/"rows.csv",rows)
            with mock.patch.object(bridge,"_verified_inputs",side_effect=verified):result=bridge._analyse(states,clinical,adapted,cfg,[])
            missing=next(r for r in result[-1]["original_outcome_missingness"] if r["target"]=="confidence" and r["split"]=="test")
            self.assertEqual(missing["missing_outcomes"],1)
            self.assertFalse(any(r["target"]=="confidence" for r in result[4]))

    def test_neural_distribution_scale_is_fit_only_to_training_people(self):
        with tempfile.TemporaryDirectory() as folder:
            states,clinical,adapted,cfg,verified,clinical_roles=fixture(Path(folder));rows=core.read_csv(states/"rows.csv")
            configuration=bridge.effective_config(core.read_json(cfg));roles={r["participant_id"]:r["split"] for r in rows}
            first=bridge._pair_data(states,adapted,rows,roles,clinical_roles,configuration)
            with np.load(states/"states.npz") as archive:arrays={k:archive[k].copy() for k in archive.files}
            for key in arrays:
                if key.startswith("p5") or key.startswith("p6"):arrays[key]*=1000
            np.savez(states/"states.npz",**arrays)
            second=bridge._pair_data(states,adapted,rows,roles,clinical_roles,configuration)
            self.assertEqual(first[2],second[2])

    def test_geometry_import_streams_channel_rows_instead_of_materializing_source_table(self):
        with tempfile.TemporaryDirectory() as folder:
            states,clinical,adapted,cfg,verified,clinical_roles=fixture(Path(folder));rows=core.read_csv(states/"rows.csv")
            roles={r["participant_id"]:r["split"] for r in rows};original=core.read_csv
            def refuse_full_table(path,*args,**kwargs):
                if Path(path).name=="layer_distances.csv":raise AssertionError("Whole full-cohort channel/layer table materialized")
                return original(path,*args,**kwargs)
            with mock.patch.object(core,"read_csv",side_effect=refuse_full_table):
                result=bridge._pair_data(states,adapted,rows,roles,clinical_roles,bridge.effective_config(core.read_json(cfg)))
            self.assertEqual(len(result[0]),6);self.assertEqual(len(result[1]),6)

    def test_public_verifier_rejects_rehashed_fitted_map_and_is_read_only_on_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);states,clinical,adapted,cfg,verified,clinical_roles=fixture(root);output=root/"bridge"
            with mock.patch.object(bridge,"_verified_inputs",side_effect=verified):
                report=bridge.run(states,clinical,adapted,cfg,output)
                before={p.name:core.sha256_file(p) for p in output.iterdir()};verified_receipt=bridge.verify_representation(output)
                self.assertTrue(verified_receipt["verified_actual_source_chain"]);self.assertEqual(before,{p.name:core.sha256_file(p) for p in output.iterdir()})
                with np.load(output/"map_weights.npz") as archive:arrays={k:archive[k].copy() for k in archive.files}
                key=next(k for k in arrays if k.endswith("__beta"));arrays[key].flat[0]+=1.;np.savez(output/"map_weights.npz",**arrays)
                receipt=core.read_json(output/"receipt.json");receipt["outputs"]["map_weights.npz"]=core.sha256_file(output/"map_weights.npz");core.write_json(output/"receipt.json",receipt)
                with self.assertRaisesRegex(core.StudyError,"Rehashed representation map"):
                    bridge.verify_representation(output)

    def test_public_verifier_rejects_rehashed_original_outcome_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);states,clinical,adapted,cfg,verified,clinical_roles=fixture(root);output=root/"bridge"
            with mock.patch.object(bridge,"_verified_inputs",side_effect=verified):
                bridge.run(states,clinical,adapted,cfg,output);rows=core.read_csv(output/"pair_predictions.csv");rows[0]["actual"]="fabricated"
                core.write_csv(output/"pair_predictions.csv",rows);receipt=core.read_json(output/"receipt.json")
                receipt["outputs"]["pair_predictions.csv"]=core.sha256_file(output/"pair_predictions.csv");core.write_json(output/"receipt.json",receipt)
                with self.assertRaisesRegex(core.StudyError,"predictions/data differ"):bridge.verify_representation(output)

    def test_native_adaptation_qualification_failure_prevents_any_numerical_map(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);states,clinical,adapted,cfg,verified,clinical_roles=fixture(root)
            with mock.patch.object(bridge,"_verified_inputs",side_effect=core.StudyError("actual native human fine-tuning absent")):
                with self.assertRaisesRegex(core.StudyError,"fine-tuning absent"):bridge.run(states,clinical,adapted,cfg,root/"blocked")
            self.assertFalse((root/"blocked").exists())

    def test_completed_training_receipt_does_not_authorize_another_or_frozen_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            states,clinical,adapted,cfg,verified,clinical_roles=fixture(Path(folder));adaptation=core.read_json(clinical/"training_receipt.json")
            geometry=core.read_json(adapted/"receipt.json");geometry["adapted_checkpoint"]["sha256"]="different_or_released_weights"
            with mock.patch("native_tools.human_finetune.verify_adaptation",return_value=adaptation), \
                 mock.patch("adversarial_study.neural_state_space.verify_state_space",return_value={}), \
                 mock.patch("native_tools.adapted_geometry.verify_adapted_geometry",return_value=geometry):
                with self.assertRaisesRegex(core.StudyError,"actual human-adapted checkpoint"):
                    bridge._verified_inputs(states,clinical,adapted,cfg,[])


if __name__=="__main__":unittest.main()
