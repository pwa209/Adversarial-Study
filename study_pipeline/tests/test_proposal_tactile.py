import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from adversarial_study import acquisition, core, source_adapters
from adversarial_study.proposal_tactile import threshold_episodes, staircase_analysis, qualified_converter, convert


PIPELINE = Path(__file__).resolve().parents[1]


def runtime_module():
    spec = importlib.util.spec_from_file_location("tactile_test_runtime", PIPELINE / "cluster/proposal_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qualified_helpers():
    bundled = PIPELINE / "qualified_tools"
    if (bundled / "serialization_repair.py").exists():
        return bundled / "tactile_adapter_repair.py", bundled / "serialization_repair.py"
    workspace = PIPELINE.parent
    return (workspace / "deployment/tactile_integrity_repair_20261006/tactile_adapter_repair.py",
            workspace / "deployment/repair_20261005/serialization_repair.py")


class ThresholdAnalysisTests(unittest.TestCase):
    def event(self, time, amp):
        return dict(onset=time, event_sample=time * 100, trial_type="stim-thr",
                    stimamp=amp, stimon=.5, sdt="n/a", confidence="n/a")

    def test_original_threshold_markers_do_not_create_experience(self):
        rows, census = threshold_episodes([self.event(5, 1), self.event(20, 2)],
            recording_id="threshold", sfreq_hz=100, duration_s=30)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["capacity_value"], "")
        self.assertEqual(rows[0]["report_category"], "unqualified_threshold_report_export")
        self.assertEqual(rows[0]["source_report_export_status"], "missing_original_report_fields")
        self.assertEqual(rows[0]["source_original_sdt"], "n/a")
        self.assertEqual(rows[0]["source_trial_type"], "stim-thr")
        self.assertEqual(census[0]["status"], "outside_full_continuous_extent")

    def test_mismatched_clock_rejected(self):
        event = self.event(20, 2); event["event_sample"] += 10
        with self.assertRaises(ValueError):
            threshold_episodes([event], recording_id="threshold", sfreq_hz=100, duration_s=30)

    def test_observed_export_is_preserved_without_inventing_report_or_capacity(self):
        missing, observed = self.event(20, 2), self.event(40, 3)
        observed.update(sdt="hit", confidence="75")
        rows, census = threshold_episodes([missing, observed], recording_id="r", sfreq_hz=100, duration_s=50)
        self.assertEqual(rows[0]["source_report_export_status"], "missing_original_report_fields")
        self.assertEqual(rows[1]["source_report_export_status"], "observed_fields_without_qualified_report_mapping")
        self.assertEqual(rows[1]["source_original_sdt"], "hit")
        self.assertEqual(rows[1]["source_original_confidence"], "75")
        self.assertEqual(rows[1]["source_sdt_export_status"], "observed_unqualified")
        self.assertEqual(rows[0]["report_category"], rows[1]["report_category"])
        self.assertTrue(all(row[k] == "" for row in rows for k in ("detection", "confidence", "capacity_value", "report_point_s")))
        # A source absence/availability code must never become a report dissociation.
        from adversarial_study.model_guided import natural_pairs
        pairs, _ = natural_pairs([row | {"participant_id": "p", "source_id": "s", "score": .2} for row in rows])
        self.assertEqual(pairs, [])
        self.assertEqual(census[1]["source_report_export_status"], rows[1]["source_report_export_status"])

    def test_explicit_acquisition_selects_all_thresholds_and_original_channels(self):
        runtime = runtime_module()
        manifest = json.loads((PIPELINE / "configs/sources.public.json").read_text())
        selected = runtime.threshold_resources(manifest)
        sets = [row for row in selected if row["relative_path"].endswith(".set")]
        channels = [row for row in selected if row["relative_path"].endswith("_channels.tsv")]
        self.assertEqual(len(sets), 36)
        self.assertEqual(len(channels), 18)
        self.assertTrue(all("\\" not in row["relative_path"] for row in selected))
        self.assertTrue(all(not row.get("default_acquire") for row in selected))
        with patch.object(acquisition, "acquire_manifest", return_value={"selected_files": len(selected)}) as acquire:
            runtime.acquire_threshold_sources(manifest, Path("unused_cache"))
        args, kwargs = acquire.call_args
        self.assertEqual(set(kwargs["resource_ids"]), {r["resource_id"] for r in selected})
        self.assertEqual(args[0]["resources"], selected)
        self.assertEqual(kwargs["max_files"], 1000)
        self.assertEqual(kwargs["max_bytes"], sum(r["expected_bytes"] for r in selected) + 1024**2)
        # Unrelated adapt recordings must not leak through the original manifest.
        self.assertFalse(any("task-adapt_" in r["relative_path"] for r in args[0]["resources"]))

    def test_missing_original_channel_table_rejected_before_acquisition(self):
        runtime = runtime_module()
        manifest = json.loads((PIPELINE / "configs/sources.public.json").read_text())
        channels = [r["resource_id"] for r in runtime.threshold_resources(manifest) if r["relative_path"].endswith("_channels.tsv")]
        manifest["resources"] = [r for r in manifest["resources"] if r["resource_id"] != channels[0]]
        with patch.object(acquisition, "acquire_manifest") as acquire:
            with self.assertRaisesRegex(ValueError, "pinned channel tables"):
                runtime.acquire_threshold_sources(manifest, Path("unused_cache"))
        acquire.assert_not_called()

    def test_serialization_repair_is_installed_in_fresh_conversion_process(self):
        integrity, serialization = qualified_helpers()
        original = source_adapters.write_json
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent if os.name == "nt" else None) as directory:
            output = Path(directory) / "child.json"
            script = (
                "import sys,numpy as np; from pathlib import Path; "
                "from adversarial_study.proposal_tactile import qualified_converter; "
                "from adversarial_study import source_adapters as a; original=a.write_json; "
                "context=qualified_converter(sys.argv[1],sys.argv[2],sys.argv[3]); context.__enter__(); "
                "a.write_json(Path(sys.argv[4]),{'integer':np.int64(7),'float':np.float64(.5),'flag':np.bool_(False)}); "
                "context.__exit__(None,None,None); assert a.write_json is original"
            )
            result = subprocess.run([sys.executable, "-c", script, str(integrity), str(serialization),
                                     core.sha256_file(serialization), str(output)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(core.read_json(output), {"integer": 7, "float": .5, "flag": False})
        self.assertIs(source_adapters.write_json, original)

    def test_serialization_helper_pin_and_exact_integer_report_are_preserved(self):
        import numpy as np
        integrity, serialization = qualified_helpers()
        original = source_adapters.write_json
        with self.assertRaisesRegex(ValueError, "manifest pin"):
            with qualified_converter(integrity, serialization, "0"*64):
                self.fail("Wrong helper pin must never enter conversion")
        self.assertIs(source_adapters.write_json, original)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent if os.name == "nt" else None) as directory:
            out = Path(directory) / "conversion"
            def source_conversion(*args, **kwargs):
                out.mkdir()
                config = {"study_id": "test", "analysis": {}, "discovery": {}, "source_conversion": {}}
                source_adapters.write_json(out / "study.config.json", config)
                answer = {"output_tables_sha256": {"study.config.json": core.sha256_file(out / "study.config.json")},
                          "actual_sample_count": np.int64(2**54+3)}
                source_adapters.write_json(out / "source_qualification.json", answer)
                return answer
            with patch.object(source_adapters, "convert_sources", side_effect=source_conversion):
                answer = convert("source", "manifest", "thrdown", out, "native", "qa", integrity,
                                 serialization, core.sha256_file(serialization))
            saved = core.read_json(out / "source_qualification.json")
            self.assertEqual(saved["actual_sample_count"], 2**54+3)
            self.assertEqual(answer["actual_sample_count"], saved["actual_sample_count"])
            self.assertEqual(saved["serialization_helper_sha256"], core.sha256_file(serialization))
        self.assertIs(source_adapters.write_json, original)

    def test_staircase_confounding_is_retained_not_interpreted_causally(self):
        rows = [dict(source_id="s", participant_id="p", recording_id="r", score=.1 * i,
            stim_intensity=4 - i, stop_s=10 * i) for i in range(1, 4)]
        result = staircase_analysis(rows)
        person = result["participant_recordings"][0]
        self.assertEqual(person["status"], "descriptive_intensity_time_confounding")
        self.assertIsNone(person["partial_slope_given_time"])
        self.assertFalse(result["threshold_experience_or_capacity_inferred"])

    def test_multiple_recordings_do_not_multiply_people(self):
        rows = [dict(source_id="s", participant_id="p", recording_id=r, score=.5,
            stim_intensity=i, stop_s=10 * i) for r in ("r1", "r2") for i in range(1, 4)]
        result = staircase_analysis(rows)
        self.assertEqual(result["source_summaries"][0]["informative_people"], 1)
        self.assertEqual(result["source_summaries"][0]["mean_within_person_slope"], 0.)
        self.assertIsNone(result["source_summaries"][0]["participant_bootstrap_ci"])


if __name__ == "__main__":
    unittest.main()
