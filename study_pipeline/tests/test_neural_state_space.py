"""Actual array operations in synthetic software fixtures; no human qualification."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from adversarial_study import core, neural_state_space as state


class StateSpaceMeasurements(unittest.TestCase):
    def settings(self):
        return {**state.DEFAULTS,"seed":20261004,"minimum_channels":2,"maximum_channels":3,
                "maximum_states":64,"rank_grid":[1,2],"ridge_grid":[.001,.01],"bootstrap_samples":100}

    def test_delay_coordinates_preserve_clock_without_episode_padding(self):
        x=np.arange(30).reshape(10,3)
        delayed,first=state.delay_coordinates(x,[0,1,2])
        self.assertEqual(first,2)
        np.testing.assert_array_equal(delayed[0],np.r_[x[2],x[1],x[0]])
        self.assertEqual(delayed.shape,(8,9))
        # Calling the operator separately on another episode cannot see x.
        next_episode=np.ones((10,3))*999
        second,_=state.delay_coordinates(next_episode,[0,1,2])
        np.testing.assert_array_equal(second,np.ones((8,9))*999)

    def test_baseline_never_uses_late_future_and_states_are_bounded(self):
        settings=self.settings();t=np.arange(2500)/250
        x=np.stack([np.sin(2*np.pi*3*t),np.cos(2*np.pi*5*t)])
        a=state.bin_window(x,250,20,settings)
        changed=x.copy();width=int(round(a[3]*250));changed[:,a[4]*width:]+=10000
        b=state.bin_window(changed,250,20,settings)
        np.testing.assert_array_equal(a[2],b[2])
        self.assertLessEqual(len(a[0]),64)
        self.assertGreaterEqual(a[1][0],20)
        self.assertLess(a[1][-1],30)
        self.assertLess(a[4],len(a[0]))

    def test_channel_set_is_training_only_with_joint_coverage(self):
        settings=self.settings()
        rows=[{"participant_id":"train1","split":"train","trajectory_channel_names":"A;B;C"},
              {"participant_id":"train2","split":"train","trajectory_channel_names":"A;B"},
              {"participant_id":"test","split":"test","trajectory_channel_names":"X;Y"}]
        self.assertEqual(state.fixed_channels(rows,settings),["A","B"])
        rows[-1]["trajectory_channel_names"]="C;D;E"
        self.assertEqual(state.fixed_channels(rows,settings),["A","B"])

    def test_state_descriptors_distinguish_history_with_rotation_invariance(self):
        t=np.arange(100)*.025
        z=np.c_[np.sin(t*3),np.cos(t*5)]
        a,e=state.state_descriptors(z,.025,np.asarray([.5,1.,1.5]))
        rotation=np.asarray([[0.,1.],[-1.,0.]])
        b,_=state.state_descriptors(z@rotation,.025,np.asarray([.5,1.,1.5]))
        for key in a:self.assertAlmostEqual(a[key],b[key],places=10)
        np.testing.assert_allclose(e,np.linalg.eigvalsh(np.cov(z.T)))
        shuffled,_=state.state_descriptors(z[np.random.default_rng(2).permutation(len(z))],.025,np.asarray([.5,1.,1.5]))
        self.assertGreater(shuffled["state_displacement_100ms"],a["state_displacement_100ms"])
        self.assertAlmostEqual(sum(a["state_occupancy_q"+str(i)] for i in range(4)),1.)

    def test_recursive_future_never_inserts_actual_intermediate_targets(self):
        model={"lags":[0],"feature_mean":np.zeros(2),"basis":np.eye(2),"transition":np.eye(2)*.8,"intercept":np.zeros(2)}
        x=np.ones((80,2));row={"state_temporal_split":40,"state_bin_s":.025}
        result=state.evaluate(x,row,model,self.settings())
        # Exact constant targets imply analytical repeated .8 decay errors;
        # resetting predicted history each step would give only .04 MSE.
        self.assertGreater(result["recursive_forecast_mse"],.04)
        self.assertEqual(result["persistence_forecast_mse"],0.)

    def test_participant_uncertainty_does_not_count_windows_as_people(self):
        one=[{"participant_id":"person","error":3.} for _ in range(1000)]
        result=state.uncertainty(one,"error",self.settings())
        self.assertEqual(result["participants"],1)
        self.assertIsNone(result["interval95"])
        two=state.uncertainty(one+[{"participant_id":"other","error":1.}],"error",self.settings())
        self.assertEqual(two["mean"],2.)


class StateSpaceStage(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.config=self.root/"settings.json"
        self.settings=StateSpaceMeasurements().settings()
        core.write_json(self.config,{"schema_version":1,"seed":self.settings["seed"],"state_space":{k:v for k,v in self.settings.items() if k!="seed"}})
        self.source=self.root/"trajectory";self.source.mkdir()
        self.rows=[];self.recordings={};rng=np.random.default_rng(5)
        for i in range(8):
            fs=250.;noise=rng.normal(size=(3,5000))
            x=np.stack([np.sin(np.arange(5000)/13+i*.1)+.1*noise[0],np.cos(np.arange(5000)/23)+.1*noise[1],np.sin(np.arange(5000)/17)+.1*noise[2]])*5
            path=self.root/("recording-"+str(i)+".npz")
            np.savez(path,data=x,sfreq_hz=fs,channel_names=np.asarray(["A","B","C"]),units=np.asarray("uV"))
            rid="recording-"+str(i);self.recordings[rid]={"recording_id":rid,"path":str(path),"sha256":core.sha256_file(path)}
            for episode in range(2):
                self.rows.append({"participant_id":"person-"+str(i),"source_id":"SOFTWARE_TEST","recording_id":rid,"episode_id":rid+"-episode-"+str(episode),
                                  "start_s":str(episode*10),"stop_s":str((episode+1)*10),"score":.2+i*.05,"report_category":"original_A" if i%2 else "original_B",
                                  "capacity_value":float(i),"capacity_construct":"TEST_numeric","capacity_units":"TEST_units","role":"development","synthetic":False,
                                  "trajectory_channel_names":"A;B;C","trajectory_channels":3,"trajectory_effective_rank":2.5,"trajectory_recurrence_fraction":.2,
                                  "signal_path":str(path),"signal_sha256":core.sha256_file(path),"source_trajectory":str(self.source),"source_receipt_sha256":"TEST_BOUNDARY",
                                  "state_token":state._token("SOFTWARE_TEST\0person-"+str(i)+"\0"+rid+"-episode-"+str(episode)),"_context":0})
        self.context={"trajectory_path":str(self.source),"source_receipt_sha256":"TEST_BOUNDARY","pointer_path":"TEST_POINTER","prepared_path":"TEST_PREPARED",
                      "config_path":"TEST_CONFIG","jobs_path":"TEST_JOBS","scores_path":"TEST_SCORES","qualification_path":"TEST_QUALIFICATION",
                      "source_ids":["SOFTWARE_TEST"],"recordings":self.recordings,
                      "cfg":{"channel_quality":{"enabled":True,"minimum_channels":2,"amplitude_reference":"channel_median","max_abs_uv":250.,"flat_std_uv":.01},
                             "quality":{"amplitude_reference":"channel_median","max_abs_uv":250.,"flat_std_uv":.01}}}

    def tearDown(self):self.temp.cleanup()

    def boundary(self):
        return patch.object(state,"_contexts",side_effect=lambda dirs:([copy.deepcopy(self.context)],copy.deepcopy(self.rows),[]))

    def run_stage(self,name="result"):
        with self.boundary():return state.analyze([self.source],self.config,self.root/name)

    def repin(self,directory):
        receipt=core.read_json(directory/"receipt.json")
        for name in receipt["outputs"]:receipt["outputs"][name]=core.sha256_file(directory/name)
        core.write_json(directory/"receipt.json",receipt)

    def replace_npz(self,path,arrays):
        import zipfile
        with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for key,value in arrays.items():state._put(archive,key,value)

    def forged_transform(self,directory):
        """Make internally consistent forged states; source EEG stays unchanged."""
        rows=core.read_csv(directory/"rows.csv");selection=core.read_json(directory/"model_selection.json")
        with np.load(directory/"states.npz") as document:arrays={k:document[k].copy() for k in document.files}
        with np.load(directory/"models.npz") as models,np.load(directory/"sensor_inputs.npz") as cache:
            for row in rows:
                if row["status"]!="included":continue
                key=row["source_model_key"];token=row["state_token"]
                model={name:models[key+"__"+name] for name in ("mean","std","feature_mean","basis","occupancy_edges")}
                model["lags"]=models[key+"__lags"].tolist()
                raw=state._sensor(cache,row,model["mean"],model["std"]);z,first=state.transform(raw,model)
                arrays[token+"__state"]=z;arrays[token+"__sensor"]=raw[first:]
                descriptors,eigen=state.state_descriptors(z,float(row["state_bin_s"]),model["occupancy_edges"])
                arrays[token+"__state_cov_eigenvalues"]=eigen
                row.update(descriptors,state_model_sha256=selection["models_sha256"])
        self.replace_npz(directory/"states.npz",arrays)
        core.write_csv(directory/"rows.csv",rows)
        self.repin(directory)

    def test_actual_arrays_fit_export_transform_and_verify_full_stage(self):
        result=self.run_stage()
        with self.boundary():verified=state.verify_state_space(self.root/"result")
        self.assertEqual(result["status"],"neural_state_space_completed")
        self.assertEqual(result["episode_count"],16)
        self.assertEqual(verified["source_model_count"],1)
        rows=core.read_csv(self.root/"result/rows.csv")
        split=core.read_json(self.root/"result/split.json")
        with np.load(self.root/"result/states.npz",allow_pickle=False) as archive:
            for row in rows:
                token=row["state_token"];z=archive[token+"__state"];sensor=archive[token+"__sensor"]
                self.assertLessEqual(len(z),64)
                self.assertEqual(z.shape[0],sensor.shape[0])
                self.assertEqual(sensor.shape[1],3)
                self.assertIn(row["participant_id"],split[row["split"]+"_participant_ids"])
        self.assertFalse(result["nonlinear_manifold_identification_established"])
        reliability=core.read_json(self.root/"result/reliability.json")
        self.assertTrue(reliability["source_outcome_predictions"])
        self.assertEqual(len(core.read_csv(self.root/"result/montage_sensitivity.csv")),8)

    def test_altered_test_signal_cannot_change_training_basis_or_selected_model(self):
        self.run_stage("first")
        split=core.read_json(self.root/"first/split.json")
        test_people=set(split["test_participant_ids"])
        for row in self.rows:
            if row["participant_id"] not in test_people:continue
            path=Path(row["signal_path"])
            with np.load(path) as document:x=document["data"].copy();fs=document["sfreq_hz"].copy();names=document["channel_names"].copy()
            x=x*1.1
            np.savez(path,data=x,sfreq_hz=fs,channel_names=names)
            self.recordings[row["recording_id"]]["sha256"]=core.sha256_file(path)
            for other in self.rows:
                if other["signal_path"]==str(path):other["signal_sha256"]=core.sha256_file(path)
        self.run_stage("second")
        with np.load(self.root/"first/models.npz") as a,np.load(self.root/"second/models.npz") as b:
            self.assertEqual(a.files,b.files)
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        a=core.read_json(self.root/"first/model_selection.json");b=core.read_json(self.root/"second/model_selection.json")
        self.assertEqual(a["source_models"],b["source_models"])

    def test_missing_test_montage_retains_person_and_exclusion(self):
        split,_=state.fixed_split(self.rows,self.settings["seed"])
        person=split["test_participant_ids"][0]
        for row in self.rows:
            if row["participant_id"]==person:row["trajectory_channel_names"]="A;B"
        result=self.run_stage()
        self.assertEqual(result["excluded_episodes"],2)
        self.assertEqual(len(core.read_csv(self.root/"result/rows.csv")),16)
        excluded=[r for r in core.read_csv(self.root/"result/rows.csv") if r["status"]=="excluded"]
        self.assertTrue(all(r["participant_id"]==person for r in excluded))
        with self.boundary():state.verify_state_space(self.root/"result")

    def test_rehashed_cache_and_consistent_states_cannot_replace_original_eeg(self):
        self.run_stage();directory=self.root/"result"
        row=next(r for r in core.read_csv(directory/"rows.csv") if r["split"]=="test")
        with np.load(directory/"sensor_inputs.npz") as document:arrays={k:document[k].copy() for k in document.files}
        token=row["state_token"];cut=int(row["state_temporal_split"])
        arrays[token+"__bins"][cut:,0]+=3.
        self.replace_npz(directory/"sensor_inputs.npz",arrays)
        self.forged_transform(directory)
        with self.boundary(),self.assertRaisesRegex(core.StudyError,"original raw EEG"):
            state.verify_state_space(directory)

    def test_rehashed_model_and_consistent_states_cannot_replace_training_fit(self):
        self.run_stage();directory=self.root/"result"
        with np.load(directory/"models.npz") as document:arrays={k:document[k].copy() for k in document.files}
        key=next(k for k in arrays if k.endswith("__mean"));arrays[key]+=1.
        self.replace_npz(directory/"models.npz",arrays)
        selection=core.read_json(directory/"model_selection.json");selection["models_sha256"]=core.sha256_file(directory/"models.npz")
        core.write_json(directory/"model_selection.json",selection)
        self.forged_transform(directory)
        with self.boundary(),self.assertRaisesRegex(core.StudyError,"training-only mean"):
            state.verify_state_space(directory)

    def test_rehashed_model_settings_cannot_change_original_configuration(self):
        self.run_stage();directory=self.root/"result"
        selection=core.read_json(directory/"model_selection.json");selection["settings"]["train_fraction"]=.6
        core.write_json(directory/"model_selection.json",selection);self.repin(directory)
        with self.boundary(),self.assertRaisesRegex(core.StudyError,"settings"):
            state.verify_state_space(directory)

    def test_actual_tactile_trace_retains_marker_uncertainty_and_missing_postdata(self):
        for row in self.rows:row["source_trial_type"]="stim-adapt"
        self.run_stage();directory=self.root/"result"
        rows=core.read_csv(directory/"rows.csv")
        with np.load(directory/"states.npz") as archive:
            for row in rows:
                token=row["state_token"]
                if float(row["stop_s"])==10.:
                    self.assertIn("physical_trigger_alignment_unqualified",row["response_marker_status"])
                    self.assertIn(token+"__response_state",archive.files)
                    times=archive[token+"__response_times_s"]
                    self.assertGreaterEqual(times[0],9.);self.assertLess(times[-1],13.)
                else:
                    self.assertTrue(row["response_marker_status"].startswith("missing_actual_response:"))
                    self.assertNotIn(token+"__response_state",archive.files)
        with self.boundary():state.verify_state_space(directory)

    def test_inadequate_source_is_explicit_stage_success(self):
        self.rows=self.rows[:2]
        result=self.run_stage()
        self.assertEqual(result["episode_count"],0)
        self.assertEqual(result["excluded_episodes"],2)
        self.assertEqual(result["scientific_status"],"insufficient_heldout_state_space")
        with self.boundary():state.verify_state_space(self.root/"result")

    def test_changed_artifact_or_original_signal_cannot_verify(self):
        self.run_stage()
        signal=Path(self.rows[0]["signal_path"])
        with signal.open("ab") as stream:stream.write(b"ALTERED")
        with self.boundary(),self.assertRaises(core.StudyError):state.verify_state_space(self.root/"result")
        # The source boundary is deliberately mocked in this fixture, but the
        # final raw-recording input digest remains real and rejects the change.

    def test_clinical_adaptation_roles_are_inherited_for_overlapping_people(self):
        directory=self.root/"clinical";directory.mkdir();core.write_json(directory/"training_receipt.json",{"TEST_ONLY":True})
        result={"split":{"train_participant_ids":["person-0","person-1"],"validation_participant_ids":["person-2"],"test_participant_ids":["person-3"]},"trained_checkpoint":{"sha256":"TEST_actual_updates"}}
        with patch("native_tools.human_finetune.verify_adaptation",return_value=result):
            split,roles=state.fixed_split(self.rows,self.settings["seed"],directory)
        for part in ("train","validation","test"):
            for person in result["split"][part+"_participant_ids"]:self.assertEqual(roles[person],part)
        self.assertEqual(split["clinical_adaptation"]["trained_checkpoint_sha256"],"TEST_actual_updates")

    def test_no_test_transformation_until_model_selection_receipt_exists(self):
        original=state.transform
        # Validation selection calls transform without test rows. Stage row
        # processing supplies a raw test sensor after the model freeze.
        original_sensor=state._sensor
        seen=[]
        def checked(cache,row,mean,std):
            if row["split"]=="test":
                self.assertTrue((self.root/"result/model_selection.json").is_file());seen.append(row["episode_id"])
            return original_sensor(cache,row,mean,std)
        with self.boundary(),patch.object(state,"_sensor",side_effect=checked):state.analyze([self.source],self.config,self.root/"result")
        self.assertTrue(seen)


if __name__=="__main__":unittest.main()
