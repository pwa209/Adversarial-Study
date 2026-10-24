"""Numerical/provenance harness tests; fixtures are synthetic, never qualified data."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from native_tools.detector_worker import matlab_command, matlab_string, validate_reference
from native_tools.figshare_source import BASELINE, DOI, DETECTOR_FILES, REQUIRED_SOURCE, acquire_release, selected_files, sha256_file
from native_tools.nftsim_runtime import compare_outputs, parse_output, perturb_config, require_scheduled_job, verify_release


def fake_metadata():
    names = sorted(REQUIRED_SOURCE | {BASELINE, "LICENSE", "license.txt"})
    files, payloads = [], {}
    for file_id, name in enumerate(names, 100):
        payload = ("Synthetic unit-test payload for " + name).encode()
        url = f"https://ndownloader.figshare.com/files/{file_id}"
        payloads[url] = payload
        files.append({"name": name, "id": file_id, "size": len(payload), "computed_md5": hashlib.md5(payload).hexdigest(),
                      "download_url": url, "is_link_only": False})
    return {"id": 30906557, "doi": DOI, "files": files}, payloads


class AcquisitionTests(unittest.TestCase):
    def test_published_local_metadata_has_complete_source_and_configs(self):
        path = Path(__file__).resolve().parents[2] / "source_records" / "toker_v2" / "figshare_v2_metadata.json"
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        chosen = selected_files(data, True, True)
        self.assertEqual(sum(path.startswith("src/") for _, path in chosen), 63)
        self.assertEqual(sum(Path(path).name.startswith("doc_config") for _, path in chosen), 280)
        self.assertTrue(DETECTOR_FILES <= {Path(path).name for _, path in chosen})

    def test_bytes_checked_reconstructed_and_manifest_frozen(self):
        metadata, payloads = fake_metadata()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "release"
            result = acquire_release(root, metadata, reader=lambda url, _: payloads[url])
            self.assertEqual(result["status"], "pinned_payload_verified")
            self.assertTrue((root / "src" / "dendrite_scaled.cpp").is_file())
            verify_release(root)
            with self.assertRaisesRegex(ValueError, "existing source manifest"):
                acquire_release(root, metadata, reader=lambda url, _: payloads[url])
            (root / "src" / "dendrite_scaled.cpp").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed or is missing"):
                verify_release(root)

    def test_download_mismatch_never_becomes_verified(self):
        metadata, payloads = fake_metadata()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "size/MD5"):
                acquire_release(Path(temp), metadata, reader=lambda url, _: b"corrupt")
            self.assertFalse((Path(temp) / "SOURCE_MANIFEST.json").exists())

    def test_wrong_release_arbitrary_urls_and_traversal_rejected(self):
        metadata, _ = fake_metadata()
        with self.assertRaisesRegex(ValueError, "version 2"):
            selected_files(metadata | {"doi": "different"})
        metadata["files"][0]["download_url"] = "https://example.org/private"
        with self.assertRaisesRegex(ValueError, "actual Figshare"):
            selected_files(metadata)
        metadata["files"][0]["name"] = "../main.cpp"
        with self.assertRaisesRegex(ValueError, "basenames"):
            selected_files(metadata)

    def test_added_unmanifested_source_rejected(self):
        metadata, payloads = fake_metadata()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            acquire_release(root, metadata, reader=lambda url, _: payloads[url])
            (root / "src" / "extra.cpp").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unrecorded"):
                verify_release(root)


class NumericalOutputTests(unittest.TestCase):
    def output(self, text):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "output"
        path.write_text("Time: 20\n=============================================\n\nTime Pop.1.Q Dendrite.1.V\n 1 1\n" + text, encoding="utf-8")
        return path

    def test_finite_sampling_and_shape_checked(self):
        parsed = parse_output(self.output(".002 20 .01\n.004 21 .02\n"), expected_rows=2, expected_interval=.002)
        self.assertEqual(parsed["rows"], 2)
        self.assertEqual(parsed["columns"], ["Time", "Pop.1.Q", "Dendrite.1.V"])
        with self.assertRaisesRegex(ValueError, "expected 3"):
            parse_output(self.output(".002 20 .01\n.004 21 .02\n"), expected_rows=3)
        with self.assertRaisesRegex(ValueError, "sampling interval"):
            parse_output(self.output(".002 20 .01\n.007 21 .02\n"), expected_interval=.002)

    def test_nonfinite_malformed_nonmonotonic_rejected(self):
        for text in (".002 nan .01\n.004 21 .02\n", ".002 20\n.004 21 .02\n", ".004 20 .01\n.002 21 .02\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_output(self.output(text))

    def test_repeats_and_control_distinguish_time_from_state(self):
        baseline = parse_output(self.output(".002 20 .01\n.004 21 .02\n"))
        same = parse_output(self.output(".002 20 .01\n.004 21 .02\n"))
        changed = parse_output(self.output(".002 22 .01\n.004 21 .02\n"))
        self.assertTrue(compare_outputs(baseline, same)["exactly_equal"])
        self.assertEqual(compare_outputs(baseline, changed)["max_absolute_difference"], 2)
        changed["values"][0, 0] += .001
        with self.assertRaisesRegex(ValueError, "times changed"):
            compare_outputs(baseline, changed)

    def test_single_parameter_change_with_comments_masked(self):
        text = "/* Dendrite 1: Scaled - alpha: 2 beta: 6 hsyn: 800 */\n Dendrite 1: Scaled - alpha: 2 beta: 6 hsyn: 19.555\nCoupling 1: Map - nu:\n0.0001\n"
        altered, record = perturb_config(text, "hsyn")
        self.assertIn("hsyn: 800", altered)
        self.assertAlmostEqual(record["after"], 39.11)
        self.assertIn("alpha: 2 beta: 6", altered)
        self.assertEqual(record["before"], 19.555)
        altered, record = perturb_config(text, "coupling", 1.05)
        self.assertAlmostEqual(record["after"], .000105)
        self.assertIn("hsyn: 19.555", altered)
        with self.assertRaises(ValueError):
            perturb_config(text + "\nCoupling 1: Map - nu: 1", "coupling")

    def test_login_execution_guard(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "scheduled Slurm"):
            require_scheduled_job()
        with patch.dict(os.environ, {"SLURM_JOB_ID": "42"}):
            self.assertEqual(require_scheduled_job(), "42")


class EmpiricalProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.raw = root / "unit_test_synthetic_raw.bin"
        self.raw.write_bytes(b"Synthetic test fixture; this is not a qualified empirical recording")
        self.signal = root / "unit_test_synthetic_signal.csv"
        np.savetxt(self.signal, np.sin(np.arange(1000) / 20), fmt="%.17g")
        # This tests schema mechanics only; no MATLAB worker or qualification is run.
        self.reference = {"reference_kind": "empirical_eeg", "synthetic": False, "continuity_verified": True,
                          "reference_id": "unit-test-schema", "participant_id": "unit-test-person", "source_id": "unit-test-source", "channel": "Cz",
                          "source_recording_path": str(self.raw), "source_recording_sha256": sha256_file(self.raw),
                          "signal_sha256": sha256_file(self.signal), "start_sample": 100, "stop_sample": 1100, "sfreq_hz": 100}

    def test_hash_span_and_sampling_checked(self):
        checked = validate_reference(self.signal, self.reference, 100)
        self.assertEqual(checked["samples"], 1000)
        with self.assertRaisesRegex(ValueError, "sampling frequency"):
            validate_reference(self.signal, self.reference, 101)
        with self.assertRaisesRegex(ValueError, "sample span"):
            validate_reference(self.signal, self.reference | {"stop_sample": 1099}, 100)
        self.raw.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            validate_reference(self.signal, self.reference, 100)

    def test_synthetic_unverified_or_missing_reference_rejected(self):
        for change in ({"synthetic": True}, {"synthetic": "false"}, {"continuity_verified": False}, {"reference_kind": "analyst_declared_empirical_reference"}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "must declare empirical"):
                validate_reference(self.signal, self.reference | change, 100)
        missing = self.reference.copy()
        missing.pop("source_recording_path")
        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_reference(self.signal, missing, 100)

    def test_matlab_arguments_quote_apostrophes_without_shell(self):
        self.assertEqual(matlab_string("O'Neil.csv"), "'O''Neil.csv'")
        with self.assertRaises(ValueError):
            matlab_string("evil\ncommand")
        command = matlab_command("qualify", release_dir=Path("release"), signal_csv=Path("O'Neil.csv"),
                                 sfreq=100, output_json=Path("out.json"), reference_json=Path("ref.json"), layers=["fc"])
        self.assertIn("'O''Neil.csv'", command)
        self.assertIn("qualify_empirical_native", command)
        self.assertIn("{'fc'}", command)


if __name__ == "__main__":
    unittest.main()
