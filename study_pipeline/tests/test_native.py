"""Bridge/config integrity checks with temporary fixtures, never model execution.

Runtime receipts below are test doubles in disposable directories. They do not
qualify a detector or create any empirical result/qualification in this project.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from adversarial_study import native
from adversarial_study.core import StudyError, read_csv, sha256_file, write_csv, write_json
from adversarial_study.simulation import _replace_parameter, prepare_nftsim_bank


WRAPPER_SHA = "a9b6073428f86bc55b93666867d5d63e5af4de8310fae41e395b05ad73e6fa23"
MAPPING_SHA = "9b1a8bfdfd1e0cd36425ab73f5d97232215ea9c482d9fd57fccc16c704f90c68"
BOUND_SHA = "96aeb138b1ddf6dc5c874c38501ad5015db6d3636b4ba1aaede98996c547033f"
FAKE_CHECKPOINT_SHA = "a" * 64
SCORE_COLUMNS = ["job_id", "gcs", "score", "preclip_gcs", "status", "signal_sha256", "error"]


def real_shape_window() -> dict:
    times = np.arange(2500) / 250.0
    return {"data": np.stack([10 * np.sin(2 * np.pi * 10 * times), 5 * np.sin(2 * np.pi * 20 * times)]),
            "sfreq_hz": 250.0, "channel_names": ["Cz", "Pz"],
            "episode_id": "test-episode", "participant_id": "fixture-participant", "source_id": "fixture-source"}


class NativeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.pinned_checkpoint = patch.object(native, "CHECKPOINT_SHA256", FAKE_CHECKPOINT_SHA)
        self.pinned_checkpoint.start()
        self.addCleanup(self.pinned_checkpoint.stop)
        self.jobs = native.export_jobs([real_shape_window()], self.directory / "jobs")
        self.job_rows = read_csv(self.jobs)
        self.qualification = self.directory / "test_qualification.json"
        self.qualification_record = {
            "status": "native_forward_pass_reproducible", "checkpoint_sha256": FAKE_CHECKPOINT_SHA,
            "wrapper_sha256": WRAPPER_SHA, "mapping_sha256": MAPPING_SHA,
            "bound_layer_sha256": BOUND_SHA, "helper_sha256": "b" * 64,
            "reference_id": "UNIT_TEST_FIXTURE_NOT_EMPIRICAL", "reference_kind": "analyst_declared_empirical_reference",
            "reference_sha256": self.job_rows[0]["signal_sha256"], "sfreq_hz": 250.0, "samples": 2500,
            "repeat_difference": 0.0, "qualified_layers": [], "extraction_status": "not_available",
        }
        write_json(self.qualification, self.qualification_record)
        self.scores = self.directory / "scores.csv"

    def write_scores(self, rows: list[dict] | None = None, receipt_changes: dict | None = None):
        if rows is None:
            rows = [{"job_id": row["job_id"], "gcs": 9.0, "score": 0.5,
                     "preclip_gcs": 9.0, "status": "ok", "signal_sha256": row["signal_sha256"], "error": ""}
                    for row in self.job_rows]
        write_csv(self.scores, rows, SCORE_COLUMNS)
        receipt = {
            "status": "native_batch_completed", "jobs": len(self.job_rows), "successful_jobs": len(rows),
            "checkpoint_sha256": FAKE_CHECKPOINT_SHA, "wrapper_sha256": WRAPPER_SHA,
            "mapping_sha256": MAPPING_SHA, "bound_layer_sha256": BOUND_SHA, "helper_sha256": "b" * 64,
            "scores_sha256": sha256_file(self.scores), "jobs_sha256": sha256_file(self.jobs),
            "qualification_sha256": sha256_file(self.qualification),
        }
        receipt.update(receipt_changes or {})
        write_json(Path(str(self.scores) + ".receipt.json"), receipt)

    def valid_rows(self) -> list[dict]:
        return [{"job_id": row["job_id"], "gcs": 9.0, "score": 0.5, "preclip_gcs": 9.0,
                 "status": "ok", "signal_sha256": row["signal_sha256"], "error": ""}
                for row in self.job_rows]

    def test_export_preserves_individual_channel_samples_and_hashes(self):
        source = real_shape_window()
        self.assertEqual(len(self.job_rows), 2)
        for index, row in enumerate(self.job_rows):
            np.testing.assert_array_equal(np.loadtxt(row["input_path"], delimiter=","), source["data"][index])
            self.assertEqual(row["signal_sha256"], sha256_file(row["input_path"]))
            self.assertEqual(row["participant_id"], source["participant_id"])
        with self.assertRaises(StudyError):
            native.export_jobs([source], self.jobs.parent)

    def test_export_rejects_nonfinite_sampling_and_signal_values(self):
        for index, frequency in enumerate([np.nan, np.inf, 90.0, 0.0]):
            with self.subTest(frequency=frequency):
                window = real_shape_window()
                window["sfreq_hz"] = frequency
                with self.assertRaises(StudyError):
                    native.export_jobs([window], self.directory / f"invalid-frequency-{index}")
        window = real_shape_window()
        window["data"][0, 10] = np.nan
        with self.assertRaises(StudyError):
            native.export_jobs([window], self.directory / "nonfinite-data")

    def test_export_rejects_nonconstant_requirement_and_duplicate_jobs(self):
        window = real_shape_window()
        window["data"][0] = 0.0
        with self.assertRaises(StudyError):
            native.export_jobs([window], self.directory / "flat-data")
        with self.assertRaises(StudyError):
            native.export_jobs([real_shape_window(), real_shape_window()], self.directory / "duplicates")
        window = real_shape_window()
        window["data"] = window["data"][:, :2000]
        with self.assertRaises(StudyError):
            native.export_jobs([window], self.directory / "short-data")

    def test_import_retains_saturated_raw_values_and_missing_extraction(self):
        rows = self.valid_rows()
        rows[0].update(gcs=3.0, score=0.0, preclip_gcs=-2.0)
        rows[1].update(gcs=15.0, score=1.0, preclip_gcs="")
        self.write_scores(rows)
        result = native.import_scores(self.scores, self.jobs, self.qualification)
        self.assertEqual(result[0]["preclip_gcs"], -2.0)
        self.assertTrue(result[0]["saturated"])
        self.assertIsNone(result[1]["preclip_gcs"])
        self.assertTrue(result[1]["saturated"])
        self.assertEqual(result[0]["score_origin"], "native_frozen_v2")

    def test_import_requires_reproducible_matching_runtime_qualification(self):
        self.write_scores()
        with self.assertRaises(StudyError):
            native.import_scores(self.scores, self.jobs)
        self.qualification_record["status"] = "static_inspection_only"
        write_json(self.qualification, self.qualification_record)
        self.write_scores()
        with self.assertRaises(StudyError):
            native.import_scores(self.scores, self.jobs, self.qualification)
        self.qualification_record["status"] = "native_forward_pass_reproducible"
        self.qualification_record["checkpoint_sha256"] = "c" * 64
        write_json(self.qualification, self.qualification_record)
        self.write_scores()
        with self.assertRaises(StudyError):
            native.import_scores(self.scores, self.jobs, self.qualification)

    def test_import_rejects_tampering_of_each_hashed_input(self):
        for input_kind in ["scores", "jobs", "qualification", "signal"]:
            with self.subTest(input_kind=input_kind):
                self.write_scores()
                paths = {"scores": self.scores, "jobs": self.jobs, "qualification": self.qualification,
                         "signal": Path(self.job_rows[0]["input_path"])}
                path = paths[input_kind]
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"\n")
                    with self.assertRaises(StudyError):
                        native.import_scores(self.scores, self.jobs, self.qualification)
                finally:
                    path.write_bytes(original)

    def test_import_rejects_missing_duplicate_unknown_and_failed_scores(self):
        valid = self.valid_rows()
        unknown = self.valid_rows()
        unknown[0]["job_id"] = "unknown-job"
        failed = self.valid_rows()
        failed[0].update(status="failed", error="native fixture error")
        cases = [valid[:1], valid + valid[:1], unknown, failed]
        for index, rows in enumerate(cases):
            with self.subTest(index=index):
                # Even a receipt claiming the full set must not hide missing
                # or duplicate score rows.
                self.write_scores(rows, {"successful_jobs": len(self.job_rows)})
                with self.assertRaises(StudyError):
                    native.import_scores(self.scores, self.jobs, self.qualification)

    def test_import_rejects_invalid_bounds_remapping_and_preclip_values(self):
        cases = [
            {"gcs": 2.0, "score": 0.0}, {"gcs": 16.0, "score": 1.0},
            {"gcs": 9.0, "score": -0.01}, {"gcs": 9.0, "score": 1.01},
            {"gcs": "NaN"}, {"score": "Infinity"}, {"score": 0.4},
            {"preclip_gcs": "NaN"}, {"preclip_gcs": 20.0},
        ]
        for index, invalid in enumerate(cases):
            with self.subTest(index=index):
                rows = self.valid_rows()
                rows[0].update(invalid)
                self.write_scores(rows)
                with self.assertRaises(StudyError):
                    native.import_scores(self.scores, self.jobs, self.qualification)

    def test_import_rejects_signal_hash_mismatch_even_when_scores_receipt_is_updated(self):
        rows = self.valid_rows()
        rows[0]["signal_sha256"] = "d" * 64
        self.write_scores(rows)
        with self.assertRaises(StudyError):
            native.import_scores(self.scores, self.jobs, self.qualification)

    def test_import_rejects_missing_wrapper_hashes_and_incomplete_batch_status(self):
        del self.qualification_record["wrapper_sha256"]
        write_json(self.qualification, self.qualification_record)
        self.write_scores(receipt_changes={"wrapper_sha256": None})
        with self.assertRaises(StudyError):
            native.import_scores(self.scores, self.jobs, self.qualification)
        self.qualification_record["wrapper_sha256"] = WRAPPER_SHA
        write_json(self.qualification, self.qualification_record)
        for status in [None, "started"]:
            with self.subTest(status=status):
                self.write_scores(receipt_changes={"status": status})
                with self.assertRaises(StudyError):
                    native.import_scores(self.scores, self.jobs, self.qualification)


class NativeReleaseIntegrityTests(unittest.TestCase):
    def test_all_release_files_are_pinned_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in native.RELEASE_FILES:
                (path / name).write_bytes(("DISPOSABLE UNIT TEST FIXTURE " + name).encode("utf-8"))
            hashes = {name: sha256_file(path / name) for name in native.RELEASE_FILES}
            checkpoint = path / native.RELEASE_FILES[0]
            md5 = hashlib.md5(checkpoint.read_bytes()).hexdigest()
            with patch.object(native, "CHECKPOINT_SHA256", hashes[checkpoint.name]), \
                 patch.object(native, "CHECKPOINT_MD5", md5), patch.object(native, "PINNED_HASHES", hashes):
                verified = native.verify_release(path)
                self.assertEqual(verified["status"], "static_release_hash_verified_not_native_execution")
                self.assertEqual(verified["files"], hashes)
                for name in native.RELEASE_FILES:
                    with self.subTest(name=name):
                        target = path / name
                        original = target.read_bytes()
                        try:
                            target.write_bytes(original + b" changed")
                            with self.assertRaises(StudyError):
                                native.verify_release(path)
                        finally:
                            target.write_bytes(original)
                (path / "remap_gcs_scale.m").unlink()
                with self.assertRaises(StudyError):
                    native.verify_release(path)


class NFTsimBankTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.template = self.directory / "template.conf"
        self.template.write_text("Nodes: 1\nCoupling 1: Map - nu:\n0.1\nCoupling 2: Map - nu: -0.2\n", encoding="utf-8")

    def test_scalar_bank_preserves_other_parameters_and_records_hash(self):
        manifest = prepare_nftsim_bank(self.template, [{"Coupling 1.nu": 0.15}, {"Coupling 1.nu": 0.2}],
                                       self.directory / "bank", {"Coupling 1.nu": [0.1, 0.2]})
        rows = read_csv(manifest)
        self.assertEqual(len(rows), 2)
        for row in rows:
            candidate = Path(row["config_path"])
            self.assertEqual(row["config_sha256"], sha256_file(candidate))
            self.assertEqual(row["nodes"], "1")
            self.assertIn("Coupling 2: Map - nu: -0.2", candidate.read_text(encoding="utf-8"))
        self.assertEqual(self.template.read_text(encoding="utf-8"),
                         "Nodes: 1\nCoupling 1: Map - nu:\n0.1\nCoupling 2: Map - nu: -0.2\n")

    def test_bank_rejects_missing_reversed_or_violated_bounds_and_nonfinite_values(self):
        cases = [([], {"Coupling 1.nu": [0, 1]}), ([{}], {"Coupling 1.nu": [0, 1]}),
                 ([{"Coupling 1.nu": 0.5}], None), ([{"Coupling 1.nu": 0.5}], {"Coupling 2.nu": [0, 1]}),
                 ([{"Coupling 1.nu": 0.5}], {"Coupling 1.nu": [1, 0]}),
                 ([{"Coupling 1.nu": 2.0}], {"Coupling 1.nu": [0, 1]}),
                 ([{"Coupling 1.nu": np.nan}], {"Coupling 1.nu": [0, 1]})]
        for index, (parameters, bounds) in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(StudyError):
                prepare_nftsim_bank(self.template, parameters, self.directory / f"invalid-bank-{index}", bounds)

    def test_parser_rejects_absent_duplicate_and_unsupported_fields(self):
        for text, key in [
            ("Nodes: 1\nCoupling 2: Map - nu: 0.1\n", "Coupling 1.nu"),
            ("Coupling 1: Map - nu: 0.1\nCoupling 1: Map - nu: 0.2\n", "Coupling 1.nu"),
            ("Coupling 1: Map - nu: 0.1\n", "Propagator 1.Tau"),
        ]:
            with self.subTest(text=text, key=key), self.assertRaises(StudyError):
                _replace_parameter(text, key, 0.5)

    def test_parser_rejects_comments_and_multiple_value_fields(self):
        for text in [
            "/* Coupling 1: Map - nu: 0.1 */\n", "// Coupling 1: Map - nu: 0.1\n",
            "Coupling 1: Map - nu: 0.1 0.2\n", "Coupling 1: Map - nu: 0.1,0.2\n",
        ]:
            with self.subTest(text=text), self.assertRaises(StudyError):
                _replace_parameter(text, "Coupling 1.nu", 0.5)

    def test_bank_rejects_zero_or_duplicate_nodes(self):
        for index, node_header in enumerate(["Nodes: 0", "Nodes: 1\nNodes: 2"]):
            self.template.write_text(node_header + "\nCoupling 1: Map - nu: 0.1\n", encoding="utf-8")
            with self.subTest(node_header=node_header), self.assertRaises(StudyError):
                prepare_nftsim_bank(self.template, [{"Coupling 1.nu": 0.5}], self.directory / f"invalid-nodes-{index}",
                                    {"Coupling 1.nu": [0, 1]})


if __name__ == "__main__":
    unittest.main()
