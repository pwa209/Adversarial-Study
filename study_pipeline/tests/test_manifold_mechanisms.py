"""Known synthetic dynamics test software; no fixtures qualify human/native science."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from adversarial_study import core, manifold_mechanisms as mm


class ManifoldMechanismTests(unittest.TestCase):
    def config(self,root):
        path=root/"config.json"
        core.write_json(path,{"seed":11,"state_space":{"train_fraction":.65},"mechanisms":{"delay_lags":[0,1,2],"ridge_grid":[.001,.1],"observation_rank_grid":[2],"observation_strength_grid":[0.,1.],"maximum_bank_states":32,"maximum_candidates":8,"bootstrap_samples":20,"joint_support_quantile":.95,"minimum_training_people":3,"minimum_validation_people":1,"minimum_test_people":2}})
        return path

    def fixture(self,root,*,with_representation=True,with_response=True):
        states=root/"states";fit=root/"fit";rep=root/"representations"
        for path in (states,fit,rep):path.mkdir()
        native_times=np.arange(5000)/500
        bank=[];control_signals={}
        for account_index,account in enumerate(("gain","coupling")):
            for k in (0,1):
                frequency=1.+.7*k+.1*account_index
                control=np.sin(2*np.pi*frequency*native_times)+.2*np.sin(2*np.pi*3.4*native_times)
                control_signals[account,k]=control
                for amplitude in (0.,1.):
                    identity=f"{account}_{k}_{'control' if amplitude==0 else 'pulse'}"
                    directory=root/identity;directory.mkdir()
                    signal=control.copy();after=native_times>=2
                    signal[after]+=amplitude*.3*np.exp(-(native_times[after]-2)*4)
                    stimulus=amplitude*((native_times>=2)&(native_times<2.02))
                    output=directory/"native.output"
                    with output.open("w",encoding="utf8") as stream:
                        stream.write("Explicit synthetic software fixture\n=============================================\nTime Pop.1.V Pop.11.Q\n1 1\n")
                        np.savetxt(stream,np.column_stack((native_times,signal,stimulus)),fmt="%.17g")
                    config=directory/"configuration.conf";config.write_text("synthetic unit fixture only\n",encoding="utf8")
                    bank.append({"simulation_id":identity,"account_class":account,"variant":"explicit_synthetic_fixture","source_output_path":str(output),"source_output_sha256":core.sha256_file(output),"config_sha256":core.sha256_file(config),"input_amplitude":str(amplitude),"input_width_s":".02","pair_id":f"{account}_{k}","score":".2","param_gain_factor":str(.8 if account=="gain" and k==0 else 1.2 if account=="gain" else 1.),"param_coupling_factor":str(.8 if account=="coupling" and k==0 else 1.2 if account=="coupling" else 1.),"param_history_factor":"1."})
        rows=[];arrays={};residual_rows=[]
        human_times=np.linspace(.01,9.998,400)
        W=np.array([[.6,-.2],[.3,.7],[-.1,.25]])
        for person in range(6):
            role="train" if person<3 else "validation" if person==3 else "test"
            for episode in (0,1):
                token=f"p{person}_e{episode}";identity="episode_"+token
                control=control_signals["gain",episode]
                mean=float(control.mean());std=float(np.std(control,ddof=1))
                lag_seconds=np.array([0.,.3125,.625])
                # Independently specified known two-coordinate linear observation process.
                x=np.column_stack([np.interp(human_times-lag,native_times,(control-mean)/std) for lag in lag_seconds])
                y=x@W
                arrays[token+"__state"]=y;arrays[token+"__times_s"]=100+human_times;arrays[token+"__sensor"]=np.column_stack((y,y))
                row={"episode_id":identity,"participant_id":f"p{person}","source_id":"synthetic_fixture_only","source_model_key":"source_fixture","state_token":token,"split":role,"status":"included","baseline_support_stop_s":106.,"channel_names":"fixture_A;fixture_B;fixture_C;fixture_D","response_event_onset_s":110.}
                if with_response:
                    rt=np.linspace(-1.,3.,180)
                    native_query=2+rt[:,None]-lag_seconds[None,:]
                    response=np.zeros_like(native_query)
                    selected=native_query>=2
                    response[selected]=.3*np.exp(-(native_query[selected]-2)*4)/std
                    arrays[token+"__response_state"]=response@W
                    arrays[token+"__response_times_s"]=110+rt
                    row["response_marker_status"]="synthetic_fixture_not_physical_human_evidence"
                rows.append(row)
                if with_representation:
                    residual_rows.append({"participant_id":f"p{person}","source_id":row["source_id"],"source_model_key":row["source_model_key"],"episode_id":identity,"split":role,"target":"state_distribution_mmd","predictor":"activation","mean_pair_loss":.1+.03*person+.01*episode})
        core.write_csv(states/"rows.csv",rows,list(rows[0]));np.savez(states/"states.npz",**arrays);np.savez(states/"models.npz",explicit_synthetic_fixture=np.asarray([1]))
        core.write_json(states/"split.json",{"fixture_only":True})
        core.write_json(states/"receipt.json",{"status":"mock_parent_fixture_not_scientific_qualification"})
        core.write_json(fit/"receipt.json",{"status":"mock_parent_fixture_only"});core.write_csv(fit/"scored_bank.csv",bank,list(bank[0]))
        core.write_csv(rep/"episode_representation_residuals.csv",residual_rows,["participant_id","source_id","source_model_key","episode_id","split","target","predictor","mean_pair_loss"])
        core.write_json(rep/"receipt.json",{"status":"mock_parent_fixture_only"})
        state_receipt={"split_sha256":core.sha256_file(states/"split.json")}
        rep_receipt={"input_paths":{"states":str(states)},"input_receipts_sha256":{"states":core.sha256_file(states/"receipt.json")},"adapted_checkpoint_sha256":"explicit_synthetic_checkpoint_fixture_only"}
        return states,fit,rep,bank,rows,state_receipt,rep_receipt,self.config(root)

    def mocks(self,fixture):
        states,fit,rep,bank,rows,state_receipt,rep_receipt,config=fixture
        stack=ExitStack()
        stack.enter_context(patch("adversarial_study.neural_state_space.verify_state_space",return_value=state_receipt))
        stack.enter_context(patch("adversarial_study.representation_bridge.verify_representation",return_value=rep_receipt))
        stack.enter_context(patch.object(mm.generative,"verify_fit",return_value=({},bank,{"fixture_only":True})))
        return stack

    def run_fixture(self,root,**kwargs):
        fixture=self.fixture(root,**kwargs);states,fit,rep,*_,config=fixture;out=root/"bridge"
        with self.mocks(fixture):receipt=mm.bridge(states,fit,config,out,rep)
        return fixture,out,receipt

    def rehash(self,out):
        receipt=core.read_json(out/"receipt.json")
        receipt["outputs"]={str(path.relative_to(out)):core.sha256_file(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name!="receipt.json"}
        core.write_json(out/"receipt.json",receipt)

    def test_geometry_rotation_invariant_but_temporal_order_sensitive(self):
        rng=np.random.default_rng(4);y=np.cumsum(rng.normal(size=(80,4)),axis=0)
        q,_=np.linalg.qr(rng.normal(size=(4,4)))
        np.testing.assert_allclose(mm.rotation_geometry(y),mm.rotation_geometry(y@q),rtol=1e-10,atol=1e-10)
        self.assertGreater(np.linalg.norm(mm.rotation_geometry(y)-mm.rotation_geometry(y[rng.permutation(len(y))])),1)

    def test_multivariate_kernel_recovers_known_signal_map(self):
        rng=np.random.default_rng(7);X=[rng.normal(size=(120,3)) for _ in range(4)]
        known=np.array([[.2,.4],[-.7,.1],[.3,-.5]]);bias=np.array([.2,-.3]);Y=[x@known+bias for x in X]
        W,b=mm._fit_weights(X,Y,np.full(4,.25),1e-8,2)
        np.testing.assert_allclose(W,known,atol=1e-7);np.testing.assert_allclose(b,bias,atol=1e-7)
        held=rng.normal(size=(60,3));np.testing.assert_allclose(held@W+b,held@known+bias,atol=1e-7)

    def test_delay_kernel_refuses_unobserved_history_and_episode_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=self.fixture(Path(tmp));traces,_,_=mm._native_bank(fixture[3]);trace=next(iter(traces.values()))
            with self.assertRaisesRegex(core.StudyError,"unobserved history"):
                mm._native_kernel(trace,np.array([.01,.2]),np.array([0.,.3]))
        with self.assertRaisesRegex(core.StudyError,"extrapolate"):
            mm._interpolate(np.ones((20,2)),np.arange(20),np.arange(10,25))

    def test_joint_support_checks_whole_geometry_not_feature_boxes(self):
        t=np.linspace(0,6,80);examples=[]
        for person in range(5):
            for phase in (0.,.03):
                y=np.column_stack((np.sin(t*(1+.01*person)+phase),np.cos(t*(1+.01*person)+phase)))
                examples.append({"y":y,"row":{"participant_id":str(person),"episode_id":str(person)+str(phase)}})
        support=mm.joint_support(examples,.95,3)
        self.assertTrue(mm.check_support(examples[0]["y"],support)["status"].startswith("within"))
        shuffled=examples[0]["y"][np.random.default_rng(3).permutation(len(t))]
        self.assertEqual(mm.check_support(shuffled,support)["status"],"outside_joint_training_manifold_support")
        np.testing.assert_allclose(np.sort(shuffled,axis=0),np.sort(examples[0]["y"],axis=0))

    def test_endpoint_formula_and_no_human_mapping_from_native_pulses(self):
        t=np.linspace(-.9,2.9,96);y=np.column_stack((t,2*t));expected=np.sqrt(2.5)*np.diff(t[t>=0]).sum()
        self.assertAlmostEqual(mm.path_endpoint(y,t),expected)
        with tempfile.TemporaryDirectory() as tmp:
            _,out,_=self.run_fixture(Path(tmp))
            diagnostic=core.read_json(out/"diagnostic.json")
            self.assertIsNone(diagnostic["chosen_candidate"]);self.assertIsNone(diagnostic["primary_neural_endpoint"])
            self.assertFalse(diagnostic["human_intervention_qualified"])
            self.assertEqual(diagnostic["candidate_neural_endpoint"]["name"],"latent_response_path_rms_v1")
            candidates=core.read_csv(out/"candidate_predictions.csv")
            self.assertEqual(len(candidates),4)
            self.assertTrue(all(r["control_config_sha256"] and r["pulse_config_sha256"] for r in candidates))

    def test_full_bridge_has_real_multivariate_predictions_and_semantic_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture,out,receipt=self.run_fixture(Path(tmp))
            self.assertTrue(receipt["human_parameter_fitting_executed"])
            training=core.read_json(out/"training_receipt.json")
            self.assertEqual(training["training_person_ids"],["p0","p1","p2"])
            self.assertFalse(training["test_outcomes_used_for_selection"])
            predictions=core.read_csv(out/"account_predictions.csv")
            self.assertEqual({r["split"] for r in predictions},{"validation","test"})
            self.assertTrue(all(int(r["latent_rank"])==2 for r in predictions))
            self.assertGreater(training["actual_response_fitted_episodes"],0)
            with self.mocks(fixture):self.assertTrue(mm.verify_bridge(out)["human_parameter_fitting_executed"])

    def test_missing_representation_overlap_stays_insufficient_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture,out,_=self.run_fixture(Path(tmp),with_representation=False,with_response=False)
            training=core.read_json(out/"training_receipt.json")
            self.assertTrue(training["human_parameter_fitting_executed"])
            self.assertEqual(training["representation_qualified_source_maps"],0)
            self.assertEqual(training["actual_response_fitted_episodes"],0)
            self.assertTrue(all(r["status"]=="retained_unsupported_challenge" for r in core.read_csv(out/"candidate_predictions.csv")))
            with self.mocks(fixture):mm.verify_bridge(out)

    def test_test_outcome_changes_do_not_change_training_or_candidate_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture=self.fixture(root);states,fit,rep,bank,rows,_,_,config=fixture
            traces,pairs,_=mm._native_bank(bank);cfg=mm._config(config);residuals,_=mm._residuals(rep)
            with np.load(states/"states.npz",allow_pickle=False) as original:
                arrays={k:original[k].copy() for k in original.files}
                a=mm._before_test(rows,original,traces,pairs,cfg,residuals)
                candidates_a=mm._candidate_predictions(a[0],a[6],traces,pairs,cfg)[0]
            for row in rows:
                if row["split"]=="test":arrays[row["state_token"]+"__state"]*=20
            np.savez(states/"changed_test.npz",**arrays)
            with np.load(states/"changed_test.npz",allow_pickle=False) as changed:
                b=mm._before_test(rows,changed,traces,pairs,cfg,residuals)
                candidates_b=mm._candidate_predictions(b[0],b[6],traces,pairs,cfg)[0]
            self.assertEqual(a[1],b[1]);self.assertEqual(candidates_a,candidates_b)

    def test_rehashed_fabricated_map_rejected_by_actual_training_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture,out,_=self.run_fixture(Path(tmp))
            with np.load(out/"observation_map.npz",allow_pickle=False) as original:arrays={k:original[k].copy() for k in original.files}
            key=next(k for k in arrays if k.endswith("__W"));arrays[key]+=.3
            np.savez_compressed(out/"observation_map.npz",**arrays)
            training=core.read_json(out/"training_receipt.json");training["observation_map"]["sha256"]=core.sha256_file(out/"observation_map.npz")
            core.write_json(out/"training_receipt.json",training);self.rehash(out)
            with self.mocks(fixture):
                with self.assertRaisesRegex(core.StudyError,"actual declared training"):
                    mm.verify_bridge(out)

    def test_rehashed_fabricated_heldout_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture,out,_=self.run_fixture(Path(tmp))
            with np.load(out/"state_predictions.npz",allow_pickle=False) as original:arrays={k:original[k].copy() for k in original.files}
            arrays[next(iter(arrays))]+=.25;np.savez_compressed(out/"state_predictions.npz",**arrays);self.rehash(out)
            with self.mocks(fixture):
                with self.assertRaisesRegex(core.StudyError,"actual declared training"):
                    mm.verify_bridge(out)

    def test_native_pair_rejects_differences_before_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=self.fixture(Path(tmp));bank=fixture[3];pulse=next(r for r in bank if r["input_amplitude"]=="1.0")
            path=Path(pulse["source_output_path"]);text=path.read_text();lines=text.splitlines();fields=lines[4].split();fields[1]=str(float(fields[1])+.1);lines[4]=" ".join(fields);path.write_text("\n".join(lines)+"\n")
            _,pairs,rejected=mm._native_bank(bank)
            self.assertEqual(len(pairs),3);self.assertTrue(any("pre-input" in r["reason"] or "before" in r["reason"] for r in rejected))


if __name__=="__main__":unittest.main()
