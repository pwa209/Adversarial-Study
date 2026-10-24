"""Source truth, no-lookahead likelihood and native cache admission checks."""
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from adversarial_study.extension_sources import marked_responses,binary_prequential,parse_edf_header,read_edf_original
from adversarial_study.extension_centaur import resource_audit,_encode_candidates
from adversarial_study.acquisition import file_checksums,validate_resource


def edf_fixture(path,unit="uV",reserved="",duration=11):
    def field(value,n): return str(value).encode().ljust(n,b" ")
    header=b"".join([field("0",8),field("subject",80),field("recording",80),field("01.01.00",8),field("00.00.00",8),
                      field(512,8),field(reserved,44),field(duration,8),field(1,8),field(1,4)])
    header+=b"".join(field(v,n) for v,n in [("EEG",16),("",80),(unit,8),(-100,8),(100,8),(-32768,8),(32767,8),("",80),(100,8),("",32)])
    values=np.concatenate([np.arange(100,dtype=np.int16)+100*i for i in range(duration)])
    path.write_bytes(header+values.astype("<i2").tobytes());return header


class ExtensionTests(unittest.TestCase):
    def test_real_provider_headers_match_inspected_source_channels(self):
        root=Path(__file__).parents[2]/"source_records/extensions"
        if not root.exists(): self.skipTest("original sourceaudit outside standalonepipelinebundle")
        dense=parse_edf_header((root/"dream_23306054_edf_header.bin").read_bytes())
        self.assertEqual(dense["names"][14],"Chan 15");self.assertEqual(dense["units"][14],"uV")
        self.assertEqual(dense["duration_s"],121)
        sparse=parse_edf_header((root/"dream_22210684_edf_header.bin").read_bytes())
        self.assertEqual(sparse["names"][:3],["Fp1-A2","C3-A2","O1-A2"])
        self.assertEqual(sparse["units"][:3],["","",""])

    def test_authoritative_report_boundary_excludes_extra_export_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"source.edf"; edf_fixture(path)
            result=read_edf_original(path,authoritative_duration_s=10,eeg_names=["EEG"])
            self.assertEqual(result["data"].shape,(1,1000))
            self.assertEqual(result["discarded_export_duration_s"],1)
            self.assertLess(float(result["data"][0,-1]),float(-100+(1100+32768)*200/65535))

    def test_missing_units_stay_missing_instead_of_becoming_microvolts(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"source.edf";edf_fixture(path,unit="")
            result=read_edf_original(path,authoritative_duration_s=10,eeg_names=["EEG"])
            self.assertEqual(result["units"],"UNKNOWN");self.assertEqual(result["missing_unit_channels"],["EEG"])

    def test_discontinuous_edf_fails_before_signal_use(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"source.edf";header=edf_fixture(path,reserved="EDF+D")
            with self.assertRaisesRegex(ValueError,"discontinuous"): parse_edf_header(header)

    def test_data_length_must_match_real_source_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"source.edf";edf_fixture(path);path.write_bytes(path.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError,"byte length"):read_edf_original(path,authoritative_duration_s=10,eeg_names=["EEG"])

    def test_marked_responses_preserve_real_prior_history(self):
        text="Instructions. You press <<L>>. Reward=1. Next: <<R>>."
        events=marked_responses(text)
        self.assertEqual([v["response"] for v in events],["L","R"])
        self.assertIn("<<L>>. Reward=1",events[1]["prefix"])
        self.assertNotIn("R",events[1]["prefix"].split("Next:")[-1])

    def test_malformed_response_events_are_not_repaired(self):
        for text in ["<<L",">>L","<<A<<B>>","<<>>"]:
            with self.assertRaises(ValueError):marked_responses(text)

    def test_prequential_probability_does_not_use_future_choices(self):
        first=binary_prequential(marked_responses("<<L>><<L>><<R>>"))
        second=binary_prequential(marked_responses("<<L>><<L>><<L>>"))
        self.assertEqual(first[:8],second[:8])
        values=[r for r in first if r["model"]=="beta_bernoulli"]
        self.assertAlmostEqual(values[1]["observed_response_probability"],.75)
        self.assertAlmostEqual(values[2]["observed_response_probability"],1/6)
        self.assertEqual(values[2]["training_choices"],2)

    def test_instruction_alphabet_cannot_be_inferred_from_future_responses(self):
        with self.assertRaisesRegex(ValueError,"alphabet"):binary_prequential(marked_responses("<<Z>>"))

    def test_model_cache_admission_rehashes_provider_pins(self):
        with tempfile.TemporaryDirectory() as folder:
            from adversarial_study.extension_centaur import EXPECTED_MODELS,REQUIRED_FILES
            root=Path(folder);resources=[];sources=[]
            for sid,(repository,revision) in EXPECTED_MODELS.items():
                sources.append({"source_id":sid,"repository":repository,"revision":revision,"provider_gated":False})
                (root/sid).mkdir()
                for name in REQUIRED_FILES[sid]:
                    file=root/sid/name;file.write_bytes(b"fixture source")
                    resources.append({"resource_id":sid+":"+name,"source_id":sid,"relative_path":sid+"/"+name,
                                      "url":"https://example.org/model","expected_bytes":file.stat().st_size,
                                      "checksum":{"algorithm":"sha256","value":__import__('hashlib').sha256(file.read_bytes()).hexdigest()}})
            manifest={"sources":sources,"resources":resources}
            path=root/"manifest.json";path.write_text(json.dumps(manifest))
            self.assertEqual(resource_audit(root,path)["status"],"READY_CACHED_MODELS")
            file.write_bytes(b"changedsourcebyte")
            self.assertEqual(resource_audit(root,path)["status"],"SOURCE_TRANSFER_PENDING")

    def test_incomplete_model_inventory_cannot_admit_gpu_measurement(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); manifest={"sources":[],"resources":[{"source_id":"centaur_adapter"}]}
            path=root/"manifest.json";path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"source identity"):resource_audit(root,path)

    def test_native_context_bound_refuses_silent_history_truncation(self):
        class Tokens:
            def encode(self,text,add_special_tokens=True):return [ord(v) for v in text]
        with self.assertRaisesRegex(ValueError,"no silenttruncation"):
            _encode_candidates(Tokens(),"Long original history",["L","R"],4)

    def test_native_response_boundaries_must_be_qualified(self):
        class Tokens:
            def encode(self,text,add_special_tokens=True):return [1] if text.endswith("<<") else [2,3]
        with self.assertRaisesRegex(ValueError,"boundary"):_encode_candidates(Tokens(),"prefix<<",["L","R"],100)

    def test_manifest_resource_pins_and_shared_budget(self):
        path=Path(__file__).parents[1]/"configs/sources.extensions.json"
        manifest=json.loads(path.read_text())
        for resource in manifest["resources"]:validate_resource(resource)
        self.assertLess(sum(r["expected_bytes"] for r in manifest["resources"]),1_000_000_000_000)
        self.assertEqual(len({r["resource_id"] for r in manifest["resources"]}),len(manifest["resources"]))

    def test_relevance_audit_keeps_all_predictions_and_requires_shared_assay(self):
        from adversarial_study.extension_relevance import relevance_audit
        import csv
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); human=root/"human";behaviour=root/"behaviour";human.mkdir();behaviour.mkdir()
            predictions={"results":[{"target":"confidence","source_id":"tactile","construct":"confidence","grouping":"participant_id",
                                      "predictor":"clinical_plus_all_measured_temporal_eeg","status":"development_grouped_prediction",
                                      "independent_participants":3,"equal_participant_mse":0.2},
                                     {"target":"awareness_rating_raw","status":"insufficient_independent_groups"}]}
            (human/"human_outcome_predictions.json").write_text(json.dumps(predictions))
            (behaviour/"source_qualification.json").write_text(json.dumps({"pilot_study":"nasioulas2024feedback","source_overlap_flags":[]}))
            (behaviour/"task_census.csv").write_text("source_id,study,experiment,sessions\npsych201,hellmann_unpublished_brightness,exp1,10\n")
            result=relevance_audit(human,behaviour,root/"output")
            self.assertEqual(result["all_grouped_predictions_retained"],2)
            self.assertEqual(result["candidate_mapping_count"],1)
            self.assertEqual(result["qualified_neural_to_behavioural_mappings"],0)
            self.assertEqual(result["native_discovery_driven_sessions_selected"],0)

    def test_no_neural_capacity_does_not_become_memory_or_attention(self):
        from adversarial_study.extension_relevance import relevance_audit
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);human=root/"human";behaviour=root/"behaviour";human.mkdir();behaviour.mkdir()
            (human/"human_outcome_predictions.json").write_text(json.dumps({"results":[{"target":"awareness_rating_raw","construct":"subjectiverating"}]}))
            (behaviour/"source_qualification.json").write_text('{}')
            (behaviour/"task_census.csv").write_text("source_id,study,experiment,sessions\npsych201,haridi2024memory,exp1,100\n")
            result=relevance_audit(human,behaviour,root/"output")
            self.assertEqual(result["candidate_mapping_count"],0)


if __name__=="__main__":unittest.main()
