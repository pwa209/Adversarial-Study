"""Workflow guards and a disposable, explicitly synthetic integration run."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from adversarial_study import cli, core, demo, native
from adversarial_study.core import StudyError


def metadata(recording_id: str = "dev-recording", role: str = "development", synthetic: str = "false") -> dict:
    return {
        "recording_id": recording_id, "participant_id": f"person-{recording_id}",
        "source_id": f"source-{role}", "cohort_id": f"cohort-{role}", "site_id": "site-one",
        "session_id": "session-one", "role": role, "signal_path": f"{recording_id}.npz",
        "signal_format": "npz", "modality": "EEG", "units": "uV", "duration_s": "10",
        "sfreq_hz": "250", "channel_count": "2", "detector_exposure": "no",
        "analyst_exposure": "no" if role in {"evaluation", "diagnostic"} else "yes", "synthetic": synthetic,
    }


def specification(mode: str = "empirical") -> dict:
    return {
        "schema_version": 1, "study_id": "DISPOSABLE_UNIT_TEST", "mode": mode,
        "path_base": ".", "seed": 7,
        "paths": {"recordings": "data/recordings.csv", "data_root": "data", "output_dir": "runs",
                  "episodes_development": "data/episodes_development.csv",
                  "episodes_evaluation": "data/episodes_evaluation.csv",
                  "episodes_diagnostic": "data/episodes_diagnostic.csv",
                  "freeze": "freeze.json", "release_dir": "release", "qualification": "qualification.json"},
        "instrument": {"channel_names": ["Cz"], "channel_aggregation": "median"},
        "analysis": {"claim_id": "TEST_ONLY", "primary_contrast_name": "A-B", "value_column": "effective_rank",
                     "condition_column": "report_category", "target_value": "A", "comparator_value": "B",
                     "construct": "test fixture", "timing_definition": "ten-second fixture", "independent_unit": "participant",
                     "minimum_episodes_per_condition": 1, "bootstrap_reps": 100},
    }


def episode(recording_id: str = "dev-recording", report_start: float = 5, report_stop: float = 10) -> dict:
    return {"episode_id": "fixture-episode", "recording_id": recording_id, "start_s": 0, "stop_s": 10,
            "report_start_s": report_start, "report_stop_s": report_stop, "report_category": "A", "capacity_value": 1.0}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.signal_dir = self.directory / "data"
        self.signal_dir.mkdir()
        self.config_path = self.directory / "config.json"
        self.manifest = self.signal_dir / "recordings.csv"
        self.config_record = specification()
        self.rows = [metadata()]
        self.save_config()
        self.save_manifest()
        core.write_csv(self.signal_dir / "episodes_development.csv", [episode()])
        times = np.arange(2500) / 250.0
        signal = np.stack([10 * np.sin(2 * np.pi * 10 * times), 8 * np.sin(2 * np.pi * 20 * times)])
        np.savez(self.signal_dir / "dev-recording.npz", data=signal, sfreq_hz=250.0,
                 channel_names=["Cz", "Pz"], units="uV")

    def save_config(self):
        core.write_json(self.config_path, self.config_record)
        self.cfg = core.load_config(self.config_path)

    def save_manifest(self):
        core.write_csv(self.manifest, self.rows, list(core.RECORDING_COLUMNS))

    def test_configuration_tracks_file_hash_and_resolves_relative_paths(self):
        self.assertEqual(self.cfg["_config_sha256"], core.sha256_file(self.config_path))
        self.assertEqual(core.resolve_path(self.cfg, "data_root", True), self.signal_dir)
        self.assertEqual(self.cfg["_config_path"], str(self.config_path.resolve()))
        for key, invalid in [("schema_version", 2), ("mode", "production"), ("study_id", ""),
                             ("seed", True), ("seed", -1), ("seed", 2.5)]:
            with self.subTest(key=key, invalid=invalid):
                changed = copy.deepcopy(self.config_record)
                changed[key] = invalid
                core.write_json(self.config_path, changed)
                with self.assertRaises(StudyError):
                    core.load_config(self.config_path)
        with self.assertRaises(StudyError):
            core.resolve_path(self.cfg, "missing")

    def test_global_person_and_cohort_role_overlap_is_rejected(self):
        for field in ["participant_id", "cohort_id"]:
            with self.subTest(field=field):
                protected = metadata("eval-recording", "evaluation")
                protected[field] = self.rows[0][field]
                with self.assertRaisesRegex(StudyError, "crosses assigned roles"):
                    core.validate_recordings([self.rows[0], protected])
        protected = metadata("eval-recording", "evaluation")
        protected["detector_exposure"] = "unknown"
        with self.assertRaisesRegex(StudyError, "confirmed no"):
            core.validate_recordings([self.rows[0], protected])

    def test_metadata_requires_finite_positive_numbers_and_integral_channel_count(self):
        for key in ["duration_s", "sfreq_hz", "channel_count"]:
            for invalid in ["NaN", "Infinity", "0", "-1"]:
                with self.subTest(key=key, invalid=invalid):
                    row = metadata()
                    row[key] = invalid
                    with self.assertRaises(StudyError):
                        core.validate_recordings([row])
        row = metadata()
        row["channel_count"] = "1.5"
        with self.assertRaises(StudyError):
            core.validate_recordings([row])

    def test_synthetic_rows_cannot_enter_empirical_selection_or_preparation(self):
        self.rows[0]["synthetic"] = "true"
        self.save_manifest()
        self.config_record["mode"] = "empirical"
        self.save_config()
        with self.assertRaisesRegex(StudyError, "Synthetic"), \
             patch("adversarial_study.data.read_recording") as signal_reader:
            cli.prepare(self.cfg)
        signal_reader.assert_not_called()

    def test_development_protected_gate_runs_before_any_outcomes_or_signals_are_opened(self):
        self.config_record["mode"] = "development"
        self.save_config()
        for role in ["evaluation", "diagnostic"]:
            with self.subTest(role=role):
                with patch("adversarial_study.core.read_csv") as outcome_reader, \
                     patch("adversarial_study.data.read_recording") as signal_reader:
                    with self.assertRaisesRegex(StudyError, "empirical"):
                        cli.prepare(self.cfg, role)
                    outcome_reader.assert_not_called()
                    signal_reader.assert_not_called()

    def test_nonsynthetic_rows_cannot_enter_synthetic_development_mode(self):
        self.config_record["mode"] = "development"
        self.save_config()
        with patch("adversarial_study.data.read_recording") as signal_reader:
            with self.assertRaises(StudyError):
                cli.prepare(self.cfg)
            signal_reader.assert_not_called()

    def test_empirical_protected_gate_rejects_bad_freeze_before_opening_outcomes(self):
        self.config_record["mode"] = "empirical"
        self.save_config()
        core.write_json(self.directory / "freeze.json", {"fixture": "not a valid freeze"})
        for role in ["evaluation", "diagnostic"]:
            with self.subTest(role=role):
                with patch("adversarial_study.core.verify_freeze", side_effect=StudyError("changed freeze")) as verifier, \
                     patch("adversarial_study.core.read_csv") as outcome_reader, \
                     patch("adversarial_study.data.read_recording") as signal_reader:
                    with self.assertRaisesRegex(StudyError, "changed freeze"):
                        cli.prepare(self.cfg, role)
                    with self.assertRaisesRegex(StudyError, "changed freeze"):
                        cli.analyze(self.cfg, self.directory / "protected-outcomes.csv", role)
                    self.assertEqual(verifier.call_count, 2)
                    outcome_reader.assert_not_called()
                    signal_reader.assert_not_called()

    def test_wrong_role_episode_file_rejects_foreign_recordings_before_eeg_read(self):
        core.write_csv(self.signal_dir / "episodes_development.csv", [episode("eval-recording")])
        with patch("adversarial_study.data.read_recording") as signal_reader:
            with self.assertRaisesRegex(StudyError, "outside that role"):
                cli.prepare(self.cfg)
            signal_reader.assert_not_called()

    def test_signal_path_cannot_escape_declared_data_root(self):
        self.rows[0]["signal_path"] = "../outside.npz"
        self.save_manifest()
        with patch("adversarial_study.data.read_recording") as signal_reader:
            with self.assertRaisesRegex(StudyError, "escapes"):
                cli.prepare(self.cfg)
            signal_reader.assert_not_called()

    def test_genuine_ten_second_window_does_not_invent_five_second_report_coverage(self):
        run = cli.prepare(self.cfg)
        feature = core.read_csv(run / "features.csv")[0]
        self.assertEqual(float(feature["duration_s"]), 10)
        self.assertEqual(float(feature["report_window_coverage"]), 0.5)
        self.assertEqual(float(feature["report_interval_overlap_s"]), 5)
        self.assertFalse(core.flag(feature["report_covers_entire_window"]))
        self.assertEqual(feature["report_category"], "A")
        self.assertEqual(float(feature["report_start_s"]), 5)
        self.assertEqual(feature["score_origin"], "not_scored")
        jobs = core.read_csv(run / "native" / "native_jobs.csv")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["channel_name"], "Cz")

    def test_prepared_parent_rejects_changed_manifest_episode_and_signal_inputs(self):
        run = cli.prepare(self.cfg)
        table = run / "features.csv"
        for path in [self.manifest, self.signal_dir / "episodes_development.csv", self.signal_dir / "dev-recording.npz"]:
            with self.subTest(path=path):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"\n")
                    with self.assertRaises(StudyError):
                        cli._validate_parent(self.cfg, table, "development")
                finally:
                    path.write_bytes(original)

    def test_analysis_table_hash_rejects_edit_before_rows_are_opened(self):
        run = core.new_run(self.cfg, "synthetic_mock_scores")
        table = run / "analysis.csv"
        rows = [{"episode_id": "fixture-episode", "participant_id": "fixture-person", "source_id": "fixture-source",
                 "role": "development", "synthetic": True, "score_origin": "synthetic_mock_not_toker"}]
        core.write_csv(table, rows)
        core.write_json(run / "analysis_table.json", {"analysis_sha256": core.sha256_file(table), "native_receipts_verified": False})
        core.finish_run(run)
        table.write_bytes(table.read_bytes() + b"\n")
        with patch("adversarial_study.core.read_csv") as row_reader:
            with self.assertRaisesRegex(StudyError, "changed"):
                cli.analyze(self.cfg, table)
            row_reader.assert_not_called()

    def test_freeze_detects_config_code_qualification_and_release_changes(self):
        self.config_record["mode"] = "empirical"
        self.rows.append(metadata("eval-recording", "evaluation"))
        self.save_manifest()
        self.save_config()
        (self.directory / "release").mkdir()
        qualification = self.directory / "qualification.json"
        core.write_json(qualification, {"status": "native_forward_pass_reproducible",
            "checkpoint_sha256": native.CHECKPOINT_SHA256, "reference_id": "DISPOSABLE_UNIT_TEST_NOT_EMPIRICAL"})
        release_record = {"checkpoint_sha256": native.CHECKPOINT_SHA256, "files": copy.deepcopy(native.PINNED_HASHES)}
        freeze_path = self.directory / "freeze.json"
        with patch("adversarial_study.native.verify_release", return_value=release_record), \
             patch("adversarial_study.core.code_hashes", return_value={"python/fixture.py": "fixed-test-code-hash"}):
            receipt = core.freeze_analysis(self.cfg, self.manifest, freeze_path)
            self.assertEqual(receipt["status"], "frozen_specification_not_preregistration_or_results")
            core.verify_freeze(self.cfg, self.manifest, freeze_path)
            with self.assertRaisesRegex(StudyError, "already exists"):
                core.freeze_analysis(self.cfg, self.manifest, freeze_path)
            original = qualification.read_bytes()
            qualification.write_bytes(original + b"\n")
            with self.assertRaisesRegex(StudyError, "artifact changed"):
                core.verify_freeze(self.cfg, self.manifest, freeze_path)
            qualification.write_bytes(original)
            with patch("adversarial_study.core.code_hashes", return_value={"python/fixture.py": "changed"}), \
                 self.assertRaisesRegex(StudyError, "code_sha256"):
                core.verify_freeze(self.cfg, self.manifest, freeze_path)
            changed_release = copy.deepcopy(release_record)
            changed_release["files"]["predict_consciousness_level.m"] = "changed"
            with patch("adversarial_study.native.verify_release", return_value=changed_release), \
                 self.assertRaisesRegex(StudyError, "release changed"):
                core.verify_freeze(self.cfg, self.manifest, freeze_path)
            self.config_record["seed"] += 1
            self.save_config()
            with self.assertRaisesRegex(StudyError, "config_sha256"):
                core.verify_freeze(self.cfg, self.manifest, freeze_path)


class SyntheticIntegrationTests(unittest.TestCase):
    def test_complete_demo_stays_synthetic_and_never_runs_native_matlab(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("adversarial_study.native.import_scores") as score_importer, \
             patch("adversarial_study.native.verify_release") as release_loader:
            output = Path(directory) / "demo"
            summary = demo.run_demo(output)
            self.assertEqual(summary["status"], "synthetic_pipeline_completed")
            self.assertIs(summary["human_evidence"], False)
            self.assertIs(summary["native_matlab_executed"], False)
            score_importer.assert_not_called()
            release_loader.assert_not_called()
            prepared = Path(summary["prepared"])
            features = core.read_csv(prepared / "features.csv")
            self.assertEqual(len({row["participant_id"] for row in features}), 16)
            self.assertEqual(len(features), 96)
            self.assertTrue(all(core.flag(row["synthetic"]) for row in features))
            self.assertEqual(len(core.read_csv(prepared / "native" / "native_jobs.csv")), 96)
            for stage in ["contrast", "grouped_prediction"]:
                report = core.read_json(Path(summary[stage]) / "result.json")
                self.assertIs(report["synthetic"], True)
                self.assertEqual(report["evidence_status"], "synthetic_software_check_only")
            for run_record in output.glob("runs/*/run.json"):
                receipt = core.read_json(run_record)
                self.assertEqual(receipt["status"], "completed")
                self.assertIs(receipt["synthetic"], True)
            for path in output.rglob("*.json"):
                def reject_constant(value):
                    raise AssertionError(f"nonfinite JSON token {value} in {path}")
                json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


if __name__ == "__main__":
    unittest.main()
