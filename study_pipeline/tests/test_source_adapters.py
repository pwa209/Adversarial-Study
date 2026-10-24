"""Scientific adapter semantics using small disposed fixtures, never human EEG."""
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from adversarial_study import source_adapters as adapters
from adversarial_study import core, data


def probes():
    matrix = np.full((3, 38), np.nan)
    matrix[:, 0] = [1, 2, 3]
    matrix[:, 1] = [100, 130, 160]
    matrix[:, 2] = matrix[:, 1]+.02
    matrix[:, 3] = 1
    matrix[:, 4] = 2
    matrix[:, 15] = matrix[:, 1]+3
    matrix[:, 31] = [1, 3, 4]
    matrix[:, 33] = [1, 4, 2]
    return matrix


def markers(sfreq=100):
    result = []
    for onset in (20, 50, 80):
        result.append({"marker_type": "P", "onset_s": onset, "sample_index": round(onset*sfreq)})
        result.append({"marker_type": "C", "onset_s": onset+10, "sample_index": round((onset+10)*sfreq)})
    return result


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def episodes(self, matrix=None, events=None, **kw):
        return adapters.monash_probe_episodes(probes() if matrix is None else matrix,
                                              markers() if events is None else events,
                                              recording_id="fixture", sfreq_hz=100,
                                              duration_s=100, **kw)

    def test_point_report_ten_second_predictor_no_recall_preserved(self):
        rows, disposition, audit = self.episodes()
        self.assertEqual(rows[2]["original_state_label"], "XX")
        self.assertNotEqual(rows[2]["report_category"], "MB")
        for row in rows:
            self.assertEqual(row["stop_s"]-row["start_s"], 10)
            self.assertEqual(row["report_start_s"], row["report_stop_s"])
            self.assertIsNone(row["capacity_value"])
        self.assertEqual(rows[1]["awareness_rating_raw"], 4)
        self.assertLess(audit["max_clock_residual_s"], 1e-9)
        self.assertFalse(audit["photodiode_correction_applied"])

    def test_missing_original_state_and_ratings_preserved(self):
        matrix = probes()
        matrix[1, 31] = np.nan
        matrix[1, 33] = np.nan
        rows, _, _ = self.episodes(matrix)
        self.assertEqual(rows[1]["report_category"], "missing_original_response")
        self.assertIsNone(rows[1]["awareness_rating_raw"])

    def test_count_clock_sequence_and_category_fail_closed(self):
        matrix = probes()
        bads = []
        m = matrix.copy(); m[1, 1] += 1; bads.append(m)
        m = matrix.copy(); m[1, 0] = 7; bads.append(m)
        m = matrix.copy(); m[1, 31] = 5; bads.append(m)
        m = matrix.copy(); m[:, 1] *= 2; bads.append(m)
        for m in bads:
            with self.subTest(matrix=m[:, :2]), self.assertRaises(ValueError):
                self.episodes(m)
        with self.assertRaisesRegex(ValueError, "counts differ"):
            self.episodes(events=markers()[:-1])

    def test_early_windows_and_prefix_dispositions_visible(self):
        early = markers()
        early[0].update(onset_s=5, sample_index=500)
        early[1].update(onset_s=15, sample_index=1500)
        matrix = probes()
        matrix[:, 1] = [85, 130, 160]
        matrix[:, 15] = matrix[:, 1]+3
        rows, disposition, _ = self.episodes(matrix, early)
        self.assertEqual(len(rows), 2)
        self.assertEqual(disposition[0]["status"], "outside_selected_continuous_extent")

    def header(self, *, reference=True):
        path = self.root/"MWI306.vhdr"
        text = ("Brain Vision Data Exchange Header File Version1.0\n[Common Infos]\n"
                "DataFile=MWI306.eeg\nMarkerFile=MWI306.vmrk\nDataFormat=BINARY\n"
                "DataOrientation=MULTIPLEXED\nNumberOfChannels=7\nSamplingInterval=10000\n"
                "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\n"
                "Ch1=Cz,,2,µV\nCh2=Pz,,3,µV\nCh3=Fz,,4,µV\nCh4=Oz,,5,µV\n"
                "Ch5=EOG,,1,mV\nCh6=ECG,,1,mV\nCh7=D1,,.25,µV\n"
                "[Comment]\n"+("Reference Channel Name = AFz\n" if reference else ""))
        path.write_bytes(text.encode("cp1252"))
        return path

    def test_header_units_reference_schema_and_one_based_markers(self):
        path = self.header()
        header = adapters.parse_brainvision_header(path)
        self.assertEqual(header["reference"], "AFz")
        self.assertEqual(header["gain_uv"], [2, 3, 4, 5, 1000, 1000, .25])
        self.assertIn("fallback", header["header_encoding"])
        marker = self.root/"MWI306.vmrk"
        marker.write_text("[Marker Infos]\nMk1=P,P  1,2001,1,0\nMk2=C,C  1,3001,1,0\n")
        parsed = adapters.parse_brainvision_markers(marker, 100, 10000)
        self.assertEqual(parsed[0]["sample_index"], 2000)
        self.assertEqual(parsed[0]["onset_s"], 20)
        with self.assertRaises(ValueError):
            adapters.parse_brainvision_markers(marker, 100, 1000)
        self.header(reference=False)
        with self.assertRaisesRegex(ValueError, "reference absent"):
            adapters.parse_brainvision_header(path)

    def test_monash_binary_conversion_gains_aux_exclusion_receipts(self):
        source = self.root/"raw"
        folder = source/"monash"
        folder.mkdir(parents=True)
        header = self.header()
        header.rename(folder/header.name)
        marker = folder/"MWI306.vmrk"
        marker.write_text("[Marker Infos]\n"+"\n".join(
            f"Mk{i+1}={r['marker_type']},{r['marker_type']}  1,{r['sample_index']+1},1,0"
            for i, r in enumerate(markers())))
        original = np.ones((10000, 7), dtype="<f4")
        original[:, :4] *= np.arange(1, 5)
        original.tofile(folder/"MWI306.eeg")
        behaviour = folder/"wanderIM_behavres_s306_fixture.mat"
        behaviour.write_bytes(b"disposed mock MAT")
        resources = []
        for p in folder.iterdir():
            resources.append({"resource_id": p.name, "source_id": "monash_wanderim",
                              "relative_path": p.relative_to(source).as_posix(),
                              "url": "https://example.org/"+p.name, "expected_bytes": p.stat().st_size,
                              "checksum": {"algorithm": "sha256", "value": hashlib.sha256(p.read_bytes()).hexdigest()},
                              "kind": "monash_behaviour" if p==behaviour else "monash_raw"})
        manifest = source/"manifest.json"
        manifest.write_text(json.dumps({"resources": resources}))
        output = self.root/"converted"
        mock_scipy = types.SimpleNamespace(loadmat=lambda *a, **k: {"probe_res": probes()})
        with patch.object(adapters.importlib, "import_module", return_value=mock_scipy):
            report = adapters.convert_sources("monash", source, manifest, output)
        self.assertFalse(report["synthetic"])
        self.assertEqual(report["converted_participants"], 1)
        cfg = core.load_config(output/"study.config.json")
        self.assertEqual(cfg["mode"], "empirical")
        record = core.read_csv(output/"recordings.csv")[0]
        self.assertEqual(record["participant_id"], "monash_wanderim_306")
        loaded = data.read_recording(output/record["signal_path"])
        self.assertEqual(loaded["channel_names"], ["Cz", "Pz", "Fz", "Oz"])
        np.testing.assert_allclose(loaded["data"][:, 0], [2, 6, 12, 20])
        with np.load(output/record["signal_path"], allow_pickle=False) as archive:
            self.assertEqual(archive["auxiliary_channel_names"].tolist(), ["EOG", "ECG", "D1"])
            np.testing.assert_allclose(archive["auxiliary_data"][:, 0], [1000, 1000, .25])
        self.assertIn("recordings.csv", report["output_tables_sha256"])
        with self.assertRaises(FileExistsError):
            adapters.convert_sources("monash", source, manifest, output)

    def trial_rows(self):
        rows = []
        for onset, typ in [(20, "stim-adapt"), (22, "hit"), (23, "conf"), (25, "conf-resp")]:
            rows.append({"onset": str(onset), "trial_type": typ, "event_sample": str(onset*100),
                         "stimamp": ".6", "stimon": ".3", "confidence": ".98",
                         "response_time": ".7", "response_time2": ".8", "sdt": "h"})
        return rows

    def test_tactile_predictors_precede_stimulus_and_reports_are_points(self):
        rows, dispositions = adapters.tactile_trial_episodes(self.trial_rows(), recording_id="tactile",
                                                            sfreq_hz=100, duration_s=30)
        self.assertEqual(rows[0]["start_s"], 10)
        self.assertEqual(rows[0]["stop_s"], 20)
        self.assertEqual(rows[0]["report_start_s"], 25)
        self.assertEqual(rows[0]["report_stop_s"], 25)
        self.assertEqual(rows[0]["capacity_value"], .98)
        self.assertEqual(rows[0]["first_order_response_s"], 22)
        self.assertEqual(rows[0]["original_sdt"], "h")
        self.assertEqual(rows[0]["source_stimulus_intensity"], .6)

    def test_tactile_reader_excludes_mislabelled_aux_and_keeps_boundary(self):
        source = self.root/"tactileraw"
        folder = source/"ds001785"/"sub-01"/"ses-01"/"eeg"
        folder.mkdir(parents=True)
        prefix = "sub-01_ses-01_task-adapt_run-01"
        setfile = folder/(prefix+"_eeg.set")
        setfile.write_bytes(b"disposed SET fixture")
        setfile.with_suffix(".fdt").write_bytes(b"disposed FDT fixture")
        # Original ds001785 puts the shared task metadata at this inherited
        # filename; it is not a run-specific same-stem sidecar.
        sidecar = folder/"sub-01_ses-01_task-adapt_eeg.json"
        sidecar.write_text(json.dumps({"RecordingType":"continuous", "EEGReference":"Placed on AFz",
                                     "SamplingFrequency":100,"EEGChannelCount":4}))
        labels = ["Cz","Pz","Fz","Oz","EOG","BIP8","BIP3"]
        channels = folder/"sub-01_ses-01_channels.tsv"
        channels.write_text("name\ttype\tunits\tstatus\n"+"\n".join(
            f"{label}\t{'AUDIO' if label=='BIP3' else 'EEG'}\t{'n/a' if label=='BIP3' else 'uV'}\tgood"
            for label in labels))
        events = folder/(prefix+"_events.tsv")
        row = self.trial_rows()
        events.write_text("\t".join(row[0])+"\n"+"\n".join("\t".join(r.values()) for r in row))
        resources = []
        for p in folder.iterdir():
            resources.append({"resource_id":p.name,"source_id":"openneuro_ds001785", "relative_path":p.relative_to(source).as_posix(),
                              "url":"https://example.org/"+p.name,"expected_bytes":p.stat().st_size,
                              "checksum":{"algorithm":"sha256","value":hashlib.sha256(p.read_bytes()).hexdigest()}})
        class Raw:
            ch_names = labels
            info = {"sfreq":100,"bads":[]}
            n_times = 3000
            first_time = 0
            annotations = types.SimpleNamespace(onset=[15],duration=[0],description=["boundary"])
            def get_data(self,picks,start,stop):
                return np.array([np.full(stop-start,1e-6 if i<4 else .5) for i in picks])
            def close(self): pass
        fake_mne = types.SimpleNamespace(io=types.SimpleNamespace(read_raw_eeglab=lambda *a,**k:Raw()))
        output = self.root/"tactileout"
        output.mkdir()
        with patch.object(adapters.importlib,"import_module",return_value=fake_mne):
            record,episodes,audit,_=adapters._tactile_one(setfile,source_root=source,output=output,
                                                         manifest={"resources":resources},maximum=None,capacity="confidence")
        loaded=data.read_recording(output/record["signal_path"])
        self.assertEqual(loaded["channel_names"],labels[:4])
        self.assertEqual(audit["excluded_from_scalp_geometry"],["EOG","BIP8","BIP3"])
        self.assertTrue(audit["source_sidecar_resolution"]["run_entity_omitted"])
        self.assertEqual(audit["source_sidecar_resolution"]["selected_relative_path"],
                         sidecar.relative_to(source).as_posix())
        self.assertIn(sidecar.name, [Path(item["relative_path"]).name for item in audit["source_inputs"]])
        with np.load(audit["unqualified_aux_file"],allow_pickle=False) as archive:
            self.assertEqual(archive["channel_names"].tolist(),["BIP3"])
            self.assertEqual(float(archive["reader_values"][0,0]),.5)
        with self.assertRaisesRegex(ValueError,"gap"):
            data.extract_window(loaded,episodes[0]["start_s"],episodes[0]["stop_s"])

    def tactile_sidecar_fixture(self):
        folder = self.root/"ds001785"/"sub-01"/"ses-01"/"eeg"
        folder.mkdir(parents=True)
        setfile = folder/"sub-01_ses-01_task-adapt_run-01_eeg.set"
        return setfile, setfile.with_suffix(".json"), folder/"sub-01_ses-01_task-adapt_eeg.json"

    def sidecar_resource(self, path):
        path.write_text('{"EEGReference":"AFz"}', encoding="utf-8")
        return {"resource_id":path.name,"source_id":"openneuro_ds001785",
                "relative_path":path.relative_to(self.root).as_posix(),
                "url":"https://example.org/"+path.name,"expected_bytes":path.stat().st_size,
                "checksum":{"algorithm":"sha256","value":hashlib.sha256(path.read_bytes()).hexdigest()}}

    def test_tactile_exact_run_sidecar_retains_checksum_verification(self):
        setfile, exact, _ = self.tactile_sidecar_fixture()
        resource = self.sidecar_resource(exact)
        manifest = {"resources":[resource]}
        selected, audit = adapters._tactile_sidecar(setfile, self.root, manifest)
        self.assertEqual(selected, exact)
        self.assertFalse(audit["run_entity_omitted"])
        adapters._pinned_inputs(self.root, manifest, [selected])
        exact.write_text('{"EEGReference":"wrong"}', encoding="utf-8")
        with self.assertRaises(ValueError):
            adapters._pinned_inputs(self.root, manifest, [selected])

    def test_tactile_inherited_metadata_missing_or_ambiguous_fails_closed(self):
        setfile, exact, inherited = self.tactile_sidecar_fixture()
        inherited_resource = self.sidecar_resource(inherited)
        # An existing file does not become a source pin merely by matching its name.
        with self.assertRaisesRegex(ValueError, "absent from immutable"):
            adapters._tactile_sidecar(setfile, self.root, {"resources":[]})
        exact_resource = self.sidecar_resource(exact)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            adapters._tactile_sidecar(setfile, self.root, {"resources":[inherited_resource,exact_resource]})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            adapters._tactile_sidecar(setfile, self.root, {"resources":[inherited_resource,inherited_resource]})
        selected, _ = adapters._tactile_sidecar(setfile, self.root, {"resources":[inherited_resource]})
        inherited.write_text('{"EEGReference":"forged"}', encoding="utf-8")
        with self.assertRaises(ValueError):
            adapters._pinned_inputs(self.root, {"resources":[inherited_resource]}, [selected])

    def test_tactile_metadata_cannot_cross_task_session_or_participant(self):
        setfile, _, inherited = self.tactile_sidecar_fixture()
        wrong_paths = [inherited.with_name("sub-01_ses-01_task-thrup_eeg.json"),
                       inherited.with_name("sub-02_ses-01_task-adapt_eeg.json"),
                       inherited.with_name("sub-01_ses-02_task-adapt_eeg.json")]
        manifest = {"resources":[self.sidecar_resource(path) for path in wrong_paths]}
        with self.assertRaisesRegex(ValueError, "absent from immutable"):
            adapters._tactile_sidecar(setfile, self.root, manifest)

    def test_original_tactile_cohort_has_checksum_pinned_task_metadata(self):
        manifest_path = Path(__file__).resolve().parent.parent/"configs"/"sources.public.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cohort = [r for r in manifest["resources"] if r["source_id"]=="openneuro_ds001785"
                  and r["relative_path"].endswith(".set") and "task-adapt_" in r["relative_path"]]
        self.assertEqual(len(cohort), 18)
        resources = {r["relative_path"]:r for r in manifest["resources"]}
        for recording in cohort:
            selected, audit = adapters._tactile_sidecar(self.root/recording["relative_path"], self.root, manifest)
            metadata = resources[audit["selected_relative_path"]]
            self.assertTrue(audit["run_entity_omitted"])
            self.assertEqual(metadata["checksum"]["algorithm"], "git_blob_sha1")
            self.assertEqual(metadata["git_commit"], "53be0167e7068aba693d57923eb4c9c037878159")
            self.assertEqual(len(metadata["checksum"]["value"]), 40)

    def test_tactile_missing_response_not_recovered_from_repeated_future_fields(self):
        rows = self.trial_rows()[:1]
        eps, _ = adapters.tactile_trial_episodes(rows, recording_id="tactile", sfreq_hz=100, duration_s=30)
        self.assertIsNone(eps[0]["capacity_value"])
        self.assertIsNone(eps[0]["original_detection"])
        self.assertIsNone(eps[0]["report_start_s"])
        rows = self.trial_rows()
        rows[0]["event_sample"] = "1900"
        with self.assertRaises(ValueError):
            adapters.tactile_trial_episodes(rows, recording_id="tactile", sfreq_hz=100, duration_s=30)

    def test_paris_annotation_quantization_excludes_nonprobe_codes(self):
        raw = types.SimpleNamespace(info={"sfreq": 250}, n_times=25000, first_time=0,
                                    annotations=types.SimpleNamespace(onset=[1.0,20.002,50.002,80.002],
                                                                     description=["Stimulus/S  1","Stimulus/S  3","Stimulus/S  3","Stimulus/S  3"]))
        events = adapters.paris_annotation_markers(raw)
        self.assertEqual(len(events), 3)
        for event in events:
            self.assertLessEqual(abs(event["onset_s"]-event["original_annotation_onset_s"]), .0020001)
        rows, _, audit = adapters.monash_probe_episodes(probes(), events, recording_id="paris",
                                                        sfreq_hz=250, duration_s=100,
                                                        require_end_markers=False, full_recording_duration_s=100)
        self.assertFalse(audit["observed_probe_end_markers_checked"])
        self.assertNotEqual(rows[0]["event_sample_grid_offset_s"], 0)

    def test_paris_zip_adapter_distinct_derivative_and_global_ids(self):
        source = self.root/"rawparis"
        folder = source/"zenodo17649277"
        folder.mkdir(parents=True)
        eeg_archive = folder/"Paris dataset EEG ICA cleaned.zip"
        behaviour_archive = folder/"Paris dataset behaviour.zip"
        with zipfile.ZipFile(eeg_archive, "w") as stream:
            stream.writestr("Paris dataset ICA cleaned/sub_001_ica_raw.fif", b"disposed FIF fixture")
        with zipfile.ZipFile(behaviour_archive, "w") as stream:
            stream.writestr("wanderIM_behavres_s001_fixture.mat", b"disposed MAT fixture")
        items = []
        for p in (eeg_archive, behaviour_archive):
            items.append({"resource_id": p.name, "source_id": "mindblanking_zenodo17649277",
                          "relative_path": p.relative_to(source).as_posix(), "url": "https://example.org/"+p.name.replace(" ","%20"),
                          "expected_bytes": p.stat().st_size,
                          "checksum": {"algorithm": "sha256", "value": hashlib.sha256(p.read_bytes()).hexdigest()}})
        manifest = source/"manifest.json"
        manifest.write_text(json.dumps({"resources": items}))
        class Raw:
            info = {"sfreq":250.0,"bads":[],"custom_ref_applied":True,"projs":[]}
            n_times = 25000
            first_time = 0.0
            ch_names = ["Cz","Pz","Fz","Oz","EOG","D1"]
            annotations = types.SimpleNamespace(onset=[20.,50.,80.], duration=[0.,0.,0.],
                                                description=["Stimulus/S  3"]*3)
            def get_channel_types(self): return ["eeg"]*4+["eog","misc"]
            def get_data(self, picks, start, stop): return np.ones((len(picks), stop-start))*1e-6
            def close(self): pass
        fake_mne = types.SimpleNamespace(io=types.SimpleNamespace(read_raw_fif=lambda *a, **k: Raw()))
        fake_scipy = types.SimpleNamespace(loadmat=lambda *a, **k: {"probe_res": probes()})
        with patch.object(adapters.importlib, "import_module", side_effect=lambda name: fake_mne if name=="mne" else fake_scipy):
            report = adapters.convert_sources("paris", source, manifest, self.root/"outparis")
        self.assertEqual(report["source_id"], "mindblanking_paris")
        self.assertEqual(report["converted_participants"], 1)
        audit = report["recordings"][0]
        self.assertIn("preprocessed", audit["source_signal_kind"])
        self.assertFalse(audit["adapter_reference_changed"])
        self.assertTrue(Path(audit["unqualified_aux_file"]).exists())
        config = core.load_config(self.root/"outparis"/"study.config.json")
        self.assertEqual(config["discovery"]["relationship_columns"], ["awareness_rating_raw"])
        record = core.read_csv(self.root/"outparis"/"recordings.csv")[0]
        self.assertEqual(record["participant_id"], "mindblanking_paris_001")
        loaded = data.read_recording(self.root/"outparis"/record["signal_path"])
        self.assertEqual(loaded["channel_names"], ["Cz","Pz","Fz","Oz"])

    def test_extraction_duplicate_or_budget_rejected(self):
        archive = self.root/"archive.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("../source-name-does-not-control-output", b"12345")
        destination = self.root/"chosen"/"input.bin"
        with self.assertRaisesRegex(ValueError, "bound"):
            adapters._extract_member(archive, "../source-name-does-not-control-output", destination, max_member_bytes=4)
        adapters._extract_member(archive, "../source-name-does-not-control-output", destination)
        self.assertEqual(destination.read_bytes(), b"12345")
        with self.assertRaises(ValueError):
            adapters._extract_member(archive, "../source-name-does-not-control-output", destination)


if __name__ == "__main__":
    unittest.main()
