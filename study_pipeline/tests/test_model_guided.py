"""Meaningful measurement/transport/provenance tests using temporary fixtures.

Fixtures are synthetic software checks. They never qualify the native MATLAB
model, a human EEG observation or an NFTsim-to-scalp biological mapping.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from adversarial_study import core, model_guided as guided


class Measurements(unittest.TestCase):
    def test_univariate_definition_is_scale_and_offset_invariant_except_amplitude(self):
        fs = 250.
        t = np.arange(2500)/fs
        x = np.sin(2*np.pi*10*t)+.2*np.sin(2*np.pi*3*t)
        a = guided.univariate_features(x, fs)
        b = guided.univariate_features(42*x+300, fs)
        for key in guided.TEMPORAL_FEATURES:
            self.assertAlmostEqual(a[key], b[key], places=10)
        self.assertAlmostEqual(a["feature_std_amplitude"]*42, b["feature_std_amplitude"])

    def test_frequency_measurement_and_order_control_are_diagnostic(self):
        fs = 250.
        t = np.arange(2500)/fs
        signal = np.sin(2*np.pi*10*t)
        original = guided.univariate_features(signal, fs)
        shuffled = guided.univariate_features(signal[np.random.default_rng(8).permutation(len(signal))], fs)
        self.assertAlmostEqual(original["feature_spectral_centroid_hz"], 10, places=3)
        self.assertGreater(original["feature_acf_100ms"], .95)
        self.assertLess(abs(shuffled["feature_acf_100ms"]), .1)
        self.assertGreater(shuffled["feature_spectral_entropy"], original["feature_spectral_entropy"])

    def test_bad_signal_never_gets_a_finite_feature_placeholder(self):
        for x, fs in ((np.ones(2500), 250), (np.full(2500, np.nan),250), (np.ones(50),250), (np.arange(2500),80),
                      (np.tile([1e200,-1e200],1250),250)):
            with self.subTest(fs=fs):
                with self.assertRaises(core.StudyError):
                    guided.univariate_features(x, fs)

    def test_actual_multichannel_state_measurements_are_bounded_and_affine_invariant(self):
        t = np.arange(5000)/500.
        x = np.stack([np.sin(2*np.pi*5*t), np.sin(2*np.pi*7*t+.2), np.cos(2*np.pi*3*t)])
        a = guided.multichannel_trajectory(x, 500)
        b = guided.multichannel_trajectory(x*np.array([2,5,20])[:,None]+np.array([100,-20,5])[:,None],500)
        self.assertLessEqual(a["trajectory_states"],400)
        self.assertGreaterEqual(a["trajectory_recurrence_fraction"],0)
        self.assertLessEqual(a["trajectory_recurrence_fraction"],1)
        self.assertLessEqual(a["trajectory_effective_rank"],3+1e-8)
        for key in a:
            self.assertAlmostEqual(a[key],b[key],places=8)

    def test_single_channel_does_not_pretend_to_be_a_multichannel_trajectory(self):
        with self.assertRaises(core.StudyError):
            guided.multichannel_trajectory(np.arange(2500)[None,:],250)

    def test_source_annotated_input_has_real_response_and_missingness_for_probe(self):
        fs=250.;t=np.arange(5000)/fs
        x=np.stack([np.sin(2*np.pi*10*t),np.sin(2*np.pi*12*t)])
        x[:,2500:2625]*=5
        recording={"data":x,"sfreq_hz":fs,"channel_names":["Cz","Pz"]}
        result=guided._event_response(recording,{"source_trial_type":"stim-adapt","stop_s":10,"stim_intensity":.2},[0,1])
        self.assertGreater(result["input_response_rms_peak"],4)
        self.assertIsNotNone(result["input_recovery_time_s"])
        self.assertIn("alignment_unqualified",result["input_response_status"])
        absent=guided._event_response(recording,{"stop_s":10,"source_trial_type":"probe"},[0,1])
        self.assertIsNone(absent["input_response_rms_peak"])

    def test_recovery_is_right_censored_when_not_observed(self):
        fs=250.;t=np.arange(5000)/fs
        x=np.stack([np.sin(2*np.pi*10*t),np.sin(2*np.pi*12*t)])
        x[:,2500:]*=5
        result=guided._event_response({"data":x,"sfreq_hz":fs,"channel_names":["Cz","Pz"]},
                                      {"source_trial_type":"stim-adapt","stop_s":10},[0,1])
        self.assertTrue(result["input_recovery_censored"])
        self.assertIsNone(result["input_recovery_time_s"])


def observation(episode, participant="p", *, start=0, score=.5, capacity=1, report="ON", source="source", construct="detection"):
    return {"episode_id":episode,"participant_id":participant,"source_id":source,"recording_id":participant+"-recording",
            "start_s":start,"stop_s":start+10,"score":score,"capacity_value":capacity,"report_category":report,
            "capacity_construct":construct,"capacity_units":"binary","saturated_fraction":0}


class Discovery(unittest.TestCase):
    def test_only_disjoint_same_recording_same_construct_capacity_pairs(self):
        rows = [observation("a"),observation("overlap",start=5,capacity=0),observation("b",start=20,capacity=0,report="MW"),
                observation("other-construct",start=40,capacity=0,construct="confidence"),observation("other-source",start=60,source="other")]
        pairs,census = guided.natural_pairs(rows)
        capacity_pairs = [p for p in pairs if p["pair_class"].endswith("different_measured_capacity")]
        self.assertEqual([(p["episode_a"],p["episode_b"]) for p in capacity_pairs],[("a","b")])
        self.assertGreater(census["counts"]["overlapping_pairs"],0)
        self.assertFalse(any(p["source_id"]=="other" for p in pairs))

    def test_score_change_same_capacity_and_clipped_ties_are_distinct(self):
        a=observation("a",score=0);b=observation("b",start=20,score=0,report="MW");c=observation("c",start=40,score=.9)
        a["saturated_fraction"]=1;b["saturated_fraction"]=1
        pairs,_=guided.natural_pairs([a,b,c])
        tie=next(p for p in pairs if p["episode_a"]=="a" and p["episode_b"]=="b")
        self.assertEqual(tie["clipping_equivalence"],"not_established")
        self.assertTrue(any(p["pair_class"]=="different_clinical_score_comparable_measured_capacity" for p in pairs))

    def test_truncation_keeps_the_full_candidate_denominator(self):
        rows=[observation(str(i),start=i*20,report="ON" if i%2 else "MW") for i in range(8)]
        pairs,census=guided.natural_pairs(rows,maximum_pairs=2)
        self.assertEqual(len(pairs),2)
        self.assertTrue(census["truncated"])
        self.assertEqual(census["counts"]["all_within_recording_pairs"],28)


class Provenance(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_bound_output_and_inputs_reject_posthoc_change(self):
        source=self.root/"source";source.write_text("observed")
        out=guided._new_output(self.root/"output")
        core.write_json(out/"measurement.json",{"value":2})
        guided._finish(out,[source],"test_fixture")
        guided._verify_output(out)
        source.write_text("different")
        with self.assertRaises(core.StudyError):guided._verify_output(out)

    def test_output_receipt_cannot_escape_its_directory(self):
        out=guided._new_output(self.root/"output")
        external=self.root/"external";external.write_text("x")
        core.write_json(out/"receipt.json",{"feature_definition":guided.FEATURE_DEFINITION,"outputs":{"../external":core.sha256_file(external)},"inputs":[]})
        with self.assertRaises(core.StudyError):guided._verify_output(out)

    def test_scored_bank_requires_actual_native_output_identity_and_hash(self):
        inventory=self.root/"inventory.csv";jobs=self.root/"jobs.csv";output=self.root/"native.out";receipt=self.root/"bank.json"
        output.write_text("NATIVE UNIT FIXTURE ONLY");jobs.write_text("export fixture")
        config=self.root/"configuration.conf";config.write_text("CONFIG UNIT FIXTURE")
        release=self.root/"release";release.mkdir();(release/"SOURCE_MANIFEST.json").write_text("SOURCE UNIT FIXTURE")
        executable=self.root/"nftsim";executable.write_text("EXECUTABLE UNIT FIXTURE")
        row={"simulation_id":"s","account_class":"gain","status":"finite","config_sha256":core.sha256_file(config),
             "source_output_path":str(output),"source_output_sha256":core.sha256_file(output),
             **{k:.5 for k in guided.TEMPORAL_FEATURES}}
        core.write_csv(inventory,[row])
        record={"status":"native_simulation_bank_completed","feature_definition":guided.FEATURE_DEFINITION,
                "inventory_sha256":core.sha256_file(inventory),"jobs_sha256":core.sha256_file(jobs),
                "release_dir":str(release),"source_manifest_sha256":core.sha256_file(release/"SOURCE_MANIFEST.json"),
                "variants":[{"executable":str(executable),"executable_sha256":core.sha256_file(executable),"source_hashes":{}}],
                "native_outputs":[{"simulation_id":"s","path":str(output),"sha256":core.sha256_file(output)}]}
        core.write_json(receipt,record)
        native_double=[{"episode_id":"s","score":.5,"preclip_gcs":9.,"saturated":False}]
        with patch.object(guided.native,"import_scores",return_value=native_double), patch("native_tools.nftsim_runtime.verify_release"):
            complete,failed,_=guided._verified_bank(inventory,jobs,self.root/"score",self.root/"qualification",receipt)
            self.assertEqual(len(complete),1);self.assertEqual(failed,[])
            output.write_text("ALTERED")
            with self.assertRaises(core.StudyError):guided._verified_bank(inventory,jobs,self.root/"score",self.root/"qualification",receipt)


class GroupedPrediction(unittest.TestCase):
    def rows(self):
        rows=[]
        for p in range(4):
            for i in range(8):
                row=observation(f"p{p}-e{i}",f"p{p}",start=i*20,score=i/10,capacity=i%2)
                row.update(site_id=f"site{p%2}", **{k:(i+1)*(j+1)/10 for j,k in enumerate(guided.TEMPORAL_FEATURES)})
                rows.append(row)
        return rows

    def test_participant_folds_keep_people_wholly_apart(self):
        rows=self.rows();x=guided._matrix(rows,guided.TEMPORAL_FEATURES)
        records=guided.grouped_outcome_prediction(rows,x,bank_matrix=x[:8])
        results=[r for r in records if r.get("status")=="development_grouped_prediction"]
        self.assertTrue(results)
        for record in results:
            for fold in record["folds"]:
                self.assertNotIn(fold["held_out_group"],fold.get("training_groups",[]))
                self.assertEqual(fold["held_out_observations"],8)

    def test_matching_scales_refit_inside_grouped_training_folds(self):
        rows=self.rows();x=guided._matrix(rows,guided.TEMPORAL_FEATURES)
        original=guided._nearest
        scales=[]
        def record(human,bank,scale,columns=None):
            scales.append(scale.copy());return original(human,bank,scale,columns)
        # One extreme held-out person exposes train/test scaling separation.
        for row in rows[-8:]:
            for key in guided.TEMPORAL_FEATURES:row[key]*=100
        with patch.object(guided,"_nearest",side_effect=record):
            guided.grouped_outcome_prediction(rows,x,bank_matrix=x[:8])
        self.assertEqual(len(scales),4)
        expected=guided._scales(guided._matrix(rows[:-8],guided.TEMPORAL_FEATURES))
        np.testing.assert_allclose(scales[-1],expected)

    def test_one_person_is_explicit_insufficient_not_window_pseudoreplication(self):
        rows=self.rows()[:8];x=guided._matrix(rows,guided.TEMPORAL_FEATURES)
        records=guided.grouped_outcome_prediction(rows,x)
        self.assertEqual(records[0]["status"],"insufficient_independent_groups")
        self.assertEqual(records[0]["groups"],1)

    def test_original_reports_have_nominal_grouped_predictions_without_forced_order(self):
        rows=self.rows()
        for i,row in enumerate(rows):row["report_category"]=("ON","MW","MB","XX")[i%4]
        x=guided._matrix(rows,guided.TEMPORAL_FEATURES)
        results=guided.grouped_outcome_prediction(rows,x,bank_matrix=x[:8])
        nominal=[r for r in results if r.get("status")=="development_grouped_nominal_report_prediction"]
        self.assertEqual(len(nominal),4)
        for record in nominal:
            self.assertEqual(record["original_categories"],["MB","MW","ON","XX"])
            self.assertEqual(record["independent_participants"],4)
            self.assertGreaterEqual(record["equal_participant_misclassification_rate"],0)
            self.assertLessEqual(record["equal_participant_misclassification_rate"],1)


class Search(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def bank(self):
        values=[]
        for index,account in enumerate(("gain","coupling")):
            values.append({"simulation_id":str(index),"account_class":account,"variant":"explicit_patch","score":.5,
                           "saturated_fraction":0, "published_config_id":"baseline","param_gain_factor":1,"input_amplitude":0,
                           **{k:(index if j<2 else 2) for j,k in enumerate(guided.TEMPORAL_FEATURES)}})
        return values

    def fit_fixture(self, *, adequate=True, support=True):
        directory=guided._new_output(self.root/"fit")
        bank=self.bank();core.write_csv(directory/"scored_bank.csv",bank)
        status="candidate_effective_proxy_predictive_adequacy" if adequate else "weak_proxy_predictive_adequacy"
        core.write_json(directory/"accounts.json",{"training_scale":[1]*5,"accounts":[{"account_class":a,"variant":"explicit_patch","status":status} for a in ("gain","coupling")]})
        fits=[{"account_class":a,"variant":"explicit_patch","simulation_id":str(i),"participant_id":"p"+str(i),
               "fit_status":"within_summary_envelope" if support else "out_of_bank_domain","standardized_summary_distance":.1} for i,a in enumerate(("gain","coupling"))]
        core.write_csv(directory/"empirical_proxy_fits.csv",fits)
        core.write_csv(directory/"retained_simulation_failures.csv",[],["simulation_id","status"])
        guided._finish(directory,[],"model_guided_fit_completed")
        return directory

    def test_score_tie_search_retains_both_accounts_and_declared_difference(self):
        candidates,census=guided.adversarial_pairs(self.bank(),np.ones(5))
        self.assertEqual(len(candidates),1)
        self.assertTrue(candidates[0]["cross_account"])
        self.assertTrue(candidates[0]["conventional_matched"])
        self.assertEqual(census["native_bank_denominator"],2)

    def test_weak_empirical_account_cannot_drive_frozen_human_comparison(self):
        fit=self.fit_fixture(adequate=False)
        search=self.root/"search";guided.search(fit,search)
        self.assertEqual(len(core.read_csv(search/"adversarial_candidates.csv")),1)
        self.assertEqual(core.read_csv(search/"empirically_constrained_candidates.csv"),[])
        guided.diagnostic(search,self.root/"diagnostic")
        artifact=core.read_json(self.root/"diagnostic/frozen_candidate_predictions.json")
        self.assertEqual(artifact["status"],"no_qualified_diagnostic_contrast_available")
        self.assertIsNone(artifact["selected_model_comparison"])

    def test_only_supported_effective_comparison_yields_candidate_not_human_qualification(self):
        fit=self.fit_fixture()
        search=self.root/"search";guided.search(fit,search)
        guided.diagnostic(search,self.root/"diagnostic")
        artifact=core.read_json(self.root/"diagnostic/frozen_candidate_predictions.json")
        self.assertEqual(artifact["status"],"candidate_comparison_requires_mapping_qualification")
        self.assertIsNotNone(artifact["selected_model_comparison"])
        self.assertEqual(artifact["current_human_experiment_status"],"design_candidate_not_recruited_or_executed")
        receipt=core.read_json(self.root/"diagnostic/freeze_receipt.json")
        self.assertEqual(receipt["artifact_sha256"],core.sha256_file(self.root/"diagnostic/frozen_candidate_predictions.json"))

    def test_adequate_account_but_out_of_domain_episode_cannot_drive_diagnostic(self):
        fit=self.fit_fixture(support=False)
        guided.search(fit,self.root/"search")
        self.assertEqual(core.read_csv(self.root/"search/empirically_constrained_candidates.csv"),[])

    def test_input_pair_search_keeps_published_configuration_identity(self):
        a,b=self.bank();b.update(account_class="gain",score=.9,feature_acf_100ms=0,feature_acf_500ms=0,input_amplitude=.1,published_config_id="other")
        candidates,census=guided.adversarial_pairs([a,b],np.ones(5))
        self.assertEqual(census["counts"].get("input_paired_native_conditions",0),0)
        b["published_config_id"]="baseline"
        candidates,census=guided.adversarial_pairs([a,b],np.ones(5))
        self.assertEqual(candidates[0]["candidate_class"],"clinical_score_change_preserved_temporal_property")


class FullFit(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_fit_challenges_held_measurements_and_keeps_native_failures(self):
        # Synthetic integration fixture proves software flow, not human evidence.
        source=guided._new_output(self.root/"observations")
        observations=[]
        for p in range(4):
            for i in range(4):
                row=observation(f"TEST-p{p}-e{i}",f"TEST-p{p}",start=i*20,score=.2+i*.1,capacity=i%2)
                row.update(site_id="TEST-site",synthetic=False,measurement_status="actual_multichannel_eeg",score_origin="native_frozen_v2",
                           **{key:(i+1)*(j+1)/10 for j,key in enumerate(guided.TEMPORAL_FEATURES)})
                observations.append(row)
        core.write_csv(source/"observations.csv",observations)
        guided._finish(source,[],"model_guided_trajectories_completed",denominator_episodes=16,excluded_episodes=0)
        inventory=[]
        for i in range(4):
            inventory.append({"simulation_id":"TEST-s"+str(i),"account_class":"gain","variant":"TEST-implementation",
                              "status":"completed","score":.2+i*.1,"param_gain_factor":1+i*.1,
                              **{key:(i+1)*(j+1)/10 for j,key in enumerate(guided.TEMPORAL_FEATURES)}})
        failed=[{"simulation_id":"TEST-failed","status":"failed","fit_status":"not_scored_or_native_simulation_failed"}]
        paths=[]
        for name in ("bank.csv","jobs.csv","scores.csv","scores.csv.receipt.json","qualification.json","bank_receipt.json"):
            path=self.root/name;path.write_text("UNIT FIXTURE ONLY");paths.append(path)
        with patch.object(guided,"_verified_bank",return_value=(inventory,failed,{})):
            receipt=guided.fit([source],paths[0],paths[1],paths[2],paths[4],paths[5],self.root/"fit")
        self.assertEqual(receipt["independent_participants"],4)
        self.assertFalse(receipt["anatomical_forward_mapping_qualified"])
        summary=core.read_json(self.root/"fit/accounts.json")
        account=summary["accounts"][0]
        self.assertEqual(account["status"],"candidate_effective_proxy_predictive_adequacy")
        self.assertEqual(len(account["held_feature_folds"]),4*len(guided.TEMPORAL_FEATURES))
        for fold in account["held_feature_folds"]:
            self.assertNotIn(fold["held_out_participant"],fold["training_participants"])
            self.assertNotIn(fold["held_out_measurement"],fold["selection_features"])
        self.assertEqual(len(core.read_csv(self.root/"fit/retained_simulation_failures.csv")),1)
        guided._verify_output(self.root/"fit")

    def test_empty_measurements_return_no_account_without_invented_human_fits(self):
        source=guided._new_output(self.root/"observations")
        core.write_csv(source/"observations.csv",[],["episode_id", "participant_id", "source_id", "score",*guided.TEMPORAL_FEATURES])
        guided._finish(source,[],"model_guided_trajectories_completed",denominator_episodes=12,excluded_episodes=12)
        paths=[]
        for name in ("bank","jobs","scores","scores.receipt.json","qualification","bank_receipt"):
            path=self.root/name;path.write_text("UNIT FIXTURE ONLY");paths.append(path)
        with patch.object(guided,"_verified_bank",return_value=([],[],{})):
            result=guided.fit([source],paths[0],paths[1],paths[2],paths[4],paths[5],self.root/"fit")
        self.assertEqual(result["valid_human_episodes"],0)
        self.assertEqual(core.read_json(self.root/"fit/accounts.json")["status"],"no_evaluable_account")
        guided.search(self.root/"fit",self.root/"search")
        guided.diagnostic(self.root/"search",self.root/"diagnostic")
        self.assertEqual(core.read_json(self.root/"diagnostic/frozen_candidate_predictions.json")["status"],"no_qualified_diagnostic_contrast_available")

    def test_full_trajectory_respects_signal_only_mask_and_keeps_raw_native_channels(self):
        # Real array operations in a synthetic unit fixture, no instrument call.
        fs=250.;t=np.arange(5000)/fs
        raw=np.stack([10000+10*np.sin(2*np.pi*10*t),-9000+12*np.sin(2*np.pi*12*t),500*np.sin(2*np.pi*8*t)])
        signal=self.root/"TEST.npz"
        np.savez(signal,data=raw,sfreq_hz=fs,channel_names=np.asarray(["Cz","Pz","BAD"]))
        prepared=self.root/"prepared";prepared.mkdir()
        core.write_json(prepared/"recordings.json",{"recordings":[{"recording_id":"p-recording","path":str(signal),"sha256":core.sha256_file(signal)}]})
        row=observation("TEST-episode")
        row.update(scalp_qualified_channel_names="Cz;Pz",synthetic=False,score_origin="native_frozen_v2")
        cfg={"instrument":{"channel_names":["Cz","Pz"]}, "quality":{"amplitude_reference":"channel_median","max_abs_uv":250.,"flat_std_uv":.01},
             "channel_quality":{"enabled":True,"minimum_channels":2,"amplitude_reference":"channel_median","max_abs_uv":250.,"flat_std_uv":.01}}
        original=raw.copy()
        with patch.object(guided,"_human",return_value=(cfg,prepared,[row],[signal])):
            guided.trajectories(self.root/"pointer",self.root/"scores",self.root/"qualification",self.root/"trajectories")
        observations=core.read_csv(self.root/"trajectories/observations.csv")
        self.assertEqual(observations[0]["trajectory_channel_names"],"Cz;Pz")
        self.assertEqual(observations[0]["trajectory_quality_excluded_channels"],"1")
        self.assertEqual(len(core.read_csv(self.root/"trajectories/trajectory_controls.csv")),3)
        audit=core.read_json(Path(observations[0]["trajectory_channel_quality_audit"]))
        self.assertFalse(audit["channels"][-1]["qualified"])
        with np.load(signal) as saved:np.testing.assert_array_equal(saved["data"],original)
        guided._verify_output(self.root/"trajectories")


if __name__=="__main__":unittest.main()
