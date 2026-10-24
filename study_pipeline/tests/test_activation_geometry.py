"""Disposable receipt/MAT-struct fixtures; no detector execution or human results."""
from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from adversarial_study import activation_geometry as geometry, native
from adversarial_study.core import StudyError, read_csv, sha256_file, write_csv, write_json


class ActivationGeometryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        times = np.arange(2500) / 250.
        windows = []
        for episode, participant, source in (("e1", "p1", "s1"), ("e2", "p1", "s1"), ("e3", "p1", "s1"),
                                               ("other_source", "p1", "s2"), ("other_person", "p2", "s1")):
            windows.append({"episode_id": episode, "participant_id": participant, "source_id": source,
                            "data": np.stack([np.sin(2 * np.pi * 10 * times), np.sin(2 * np.pi * 20 * times)]),
                            "channel_names": ["Cz", "Pz"], "sfreq_hz": 250.})
        self.jobs_path = native.export_jobs(windows, self.root / "jobs")
        self.jobs = read_csv(self.jobs_path)
        self.qualification_path = self.root / "FIXTURE_qualification.json"
        self.q = {"status": "native_forward_pass_reproducible", "checkpoint_sha256": native.CHECKPOINT_SHA256,
                  "wrapper_sha256": native.PINNED_HASHES["predict_consciousness_level.m"],
                  "mapping_sha256": native.PINNED_HASHES["remap_gcs_scale.m"],
                  "bound_layer_sha256": native.PINNED_HASHES["BoundedOutputLayer.m"],
                  "qualified_layers": ["relu", "dense.layer"], "helper_sha256": "b" * 64,
                  "extraction_status": "forward_correspondence_verified", "activation_repeats_verified": True,
                  "reference_id": "UNIT_TEST_ONLY_NOT_EMPIRICAL"}
        self.scores_path = self.root / "scores.csv"
        self.rows, self.outputs, self.values = [], [], {}
        for job in self.jobs:
            episode = job["episode_id"]
            gcs = 3. if episode in {"e1", "e2"} else 9.
            raw = -1. if episode == "e1" else (-2. if episode == "e2" else 9.)
            self.rows.append({"job_id": job["job_id"], "gcs": gcs, "score": (gcs - 3) / 12,
                              "preclip_gcs": raw, "status": "ok", "signal_sha256": job["signal_sha256"],
                              "extraction_status": "forward_correspondence_verified"})
            path = Path(str(self.scores_path) + "." + job["job_id"] + ".activations.mat")
            # Deliberately not a real .mat: the parser is mocked below. These
            # temporary files/receipts never qualify a detector or a dataset.
            path.write_bytes(b"UNIT_TEST_ACTIVATION_FIXTURE_NOT_EMPIRICAL")
            self.outputs.append({"job_id": job["job_id"], "path": str(path), "sha256": sha256_file(path)})
            vector = {"e1": [1., 0.], "e2": [0., 1.], "e3": [-1., 0.]}.get(episode, [1., 1.])
            self.values[str(path)] = {"relu": np.array([vector]), "dense_layer": np.array([[1., 2., 3.]])}
        self.write_receipts()
        self.loader = patch.object(geometry, "_load_mat", side_effect=lambda path: self.values[str(path)])
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def write_receipts(self, outputs=None):
        write_json(self.qualification_path, self.q)
        write_csv(self.scores_path, self.rows)
        receipt = {"status": "native_batch_completed", "jobs": len(self.jobs), "successful_jobs": len(self.jobs),
                   "checkpoint_sha256": native.CHECKPOINT_SHA256, "wrapper_sha256": self.q["wrapper_sha256"],
                   "scores_sha256": sha256_file(self.scores_path), "jobs_sha256": sha256_file(self.jobs_path),
                   "qualification_sha256": sha256_file(self.qualification_path),
                   "activation_outputs": self.outputs if outputs is None else outputs}
        write_json(Path(str(self.scores_path) + ".receipt.json"), receipt)

    def compute(self):
        return geometry.compute_activation_geometry(self.jobs_path, self.scores_path, self.qualification_path)

    def test_full_registry_pairs_never_cross_person_source_or_channel(self):
        result = self.compute()
        self.assertEqual(result["counts"]["within_person_source_channel_job_pairs"], 6)
        self.assertEqual(result["counts"]["layer_pair_rows"], 12)
        self.assertEqual(result["counts"]["computed_distances"], 12)
        self.assertEqual({row["participant_id"] for row in result["pairs"]}, {"p1"})
        self.assertEqual({row["source_id"] for row in result["pairs"]}, {"s1"})
        jobs = {row["job_id"]: row for row in self.jobs}
        for row in result["pairs"]:
            self.assertEqual(jobs[row["job_id_a"]]["channel_name"], jobs[row["job_id_b"]]["channel_name"])
        known = next(row for row in result["pairs"] if row["layer_name"] == "relu" and row["episode_id_a"] == "e1" and row["episode_id_b"] == "e2")
        self.assertEqual(known["cosine_distance"], 1.)
        self.assertEqual(known["absolute_score_difference"], 0.)
        self.assertEqual(known["absolute_preclip_gcs_difference"], 1.)
        self.assertTrue(known["saturated_a"])
        self.assertEqual(known["boundary_a"], "floor")
        json.dumps(result, allow_nan=False)

    def test_missing_shape_and_zero_norm_are_retained_without_padding(self):
        selected = [row for row in self.jobs if row["episode_id"] == "e1"]
        path = str(self.scores_path) + "." + selected[0]["job_id"] + ".activations.mat"
        self.values[path]["relu"] = np.zeros((1, 2))
        path2 = str(self.scores_path) + "." + selected[1]["job_id"] + ".activations.mat"
        self.values[path2]["relu"] = np.ones((1, 4))
        missing = next(row for row in self.outputs if row["job_id"] == next(job["job_id"] for job in self.jobs if job["episode_id"] == "e2" and job["channel_name"] == "Cz"))
        Path(missing["path"]).unlink()
        result = self.compute()
        self.assertEqual(len(result["pairs"]), 12)
        statuses = {row["metric_status"] for row in result["pairs"]}
        self.assertTrue({"activation_unavailable", "shape_mismatch", "undefined_zero_norm"} <= statuses)
        for row in result["pairs"]:
            if row["metric_status"] != "computed":
                self.assertIsNone(row["cosine_distance"])

    def test_activation_hash_and_path_binding_are_required(self):
        path = Path(self.outputs[0]["path"])
        path.write_bytes(b"CHANGED UNIT TEST FIXTURE")
        result = self.compute()
        job = next(row for row in result["activation_inventory"] if row["job_id"] == self.outputs[0]["job_id"])
        self.assertEqual(job["status"], "native_activation_hash_mismatch")
        self.outputs[0]["sha256"] = sha256_file(path)
        self.outputs[0]["path"] = str(self.root / "wrong.mat")
        self.write_receipts()
        job = next(row for row in self.compute()["activation_inventory"] if row["job_id"] == self.outputs[0]["job_id"])
        self.assertEqual(job["status"], "native_activation_path_mismatch")

    def test_exact_layer_set_and_repeated_extraction_qualification(self):
        self.values[self.outputs[0]["path"]]["unexpected_layer"] = np.ones((1, 2))
        result = self.compute()
        job = next(row for row in result["activation_inventory"] if row["job_id"] == self.outputs[0]["job_id"])
        self.assertEqual(job["status"], "qualified_layer_set_mismatch")
        self.q["activation_repeats_verified"] = False
        self.write_receipts()
        result = self.compute()
        self.assertFalse(result["qualified_extraction"])
        self.assertEqual(result["counts"]["computed_distances"], 0)

    def test_legacy_unbound_singleton_and_invalid_output_registries(self):
        self.write_receipts(outputs=[])
        result = self.compute()
        self.assertEqual(result["counts"]["computed_distances"], 0)
        self.assertEqual(result["activation_inventory"][0]["status"], "missing_native_activation_binding")
        self.write_receipts(outputs=self.outputs[0])
        result = self.compute()
        self.assertEqual(sum(item["status"] == "qualified_layer_set_verified" for item in result["activation_inventory"]), 1)
        self.write_receipts(outputs=self.outputs + self.outputs[:1])
        with self.assertRaisesRegex(StudyError, "Duplicate"):
            self.compute()

    def test_score_provenance_is_checked_before_activation_loading(self):
        with self.scores_path.open("a") as stream:
            stream.write("\n")
        with patch.object(geometry, "_load_mat", side_effect=AssertionError("Loader called before score verification")):
            with self.assertRaisesRegex(StudyError, "scores_sha256"):
                self.compute()

    def test_layer_name_mapping_collision_and_unsafe_transformations(self):
        self.assertEqual(geometry.matlab_layer_field_name("dense.layer"), "dense_layer")
        self.q["qualified_layers"] = ["dense.layer", "dense_layer"]
        self.write_receipts()
        with self.assertRaisesRegex(StudyError, "collide"):
            self.compute()
        with self.assertRaises(StudyError):
            geometry.matlab_layer_field_name("ambiguous whitespace")

    def test_cli_pointer_exports_integrity_and_preserves_existing_results(self):
        pointer = self.root / "prepared.json"
        write_json(pointer, {"jobs": str(self.jobs_path)})
        out = self.root / "output"
        args = ["--pointer", str(pointer), "--scores", str(self.scores_path), "--qualification", str(self.qualification_path),
                "--output-dir", str(out), "--single-table"]
        self.assertEqual(geometry.main(args), 0)
        result = json.loads((out / "receipt.json").read_text())
        self.assertEqual(result["counts"]["computed_distances"], 12)
        self.assertEqual(len(read_csv(out / "layer_distances.csv")), 12)
        self.assertEqual(result["inputs"]["pointer_sha256"], sha256_file(pointer))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            geometry.main(args)

    def test_participant_streaming_matches_whole_pairs_with_one_complete_import(self):
        expected = self.compute()
        out = self.root / "streaming_output"
        with patch.object(native, "import_scores", wraps=native.import_scores) as importer:
            result = geometry.export_participant_geometry(self.jobs_path, self.scores_path, self.qualification_path, out)
        self.assertEqual(importer.call_count, 1)
        self.assertEqual(result["counts"], expected["counts"])
        self.assertEqual(result["counts"]["participants"], 2)
        self.assertEqual(result["metric_status_counts_by_layer"], expected["metric_status_counts_by_layer"])
        combined = []
        for item in result["participant_partitions"]:
            directory = Path(item["directory"])
            self.assertEqual(len(directory.name), 64)
            rows = read_csv(directory / "layer_distances.csv")
            self.assertTrue(all(row["participant_id"] == item["participant_id"] for row in rows))
            child = json.loads((directory / "receipt.json").read_text())
            self.assertEqual(child["complete_native_batch_job_count"], len(self.jobs))
            combined.extend(rows)
        key = lambda row: (row["job_id_a"], row["job_id_b"], row["layer_name"])
        self.assertEqual({key(row) for row in combined}, {key(row) for row in expected["pairs"]})
        expected_metrics = {key(row): (str(row["cosine_distance"]), row["metric_status"]) for row in expected["pairs"]}
        self.assertEqual({key(row): (row["cosine_distance"], row["metric_status"]) for row in combined}, expected_metrics)
        self.assertFalse((out / "layer_distances.csv").exists())


class MatlabStructAndNumericalTests(unittest.TestCase):
    def test_lazy_scipy_parser_preserves_actual_mat_struct_shapes(self):
        struct = types.SimpleNamespace(_fieldnames=["relu"], relu=np.array([[1., 2.], [3., 4.]]))
        array = np.empty((1, 1), dtype=object)
        array[0, 0] = struct
        fake_scipy = types.SimpleNamespace(loadmat=lambda *args, **kwargs: {"activation_values": array})
        with patch.object(geometry.importlib, "import_module", return_value=fake_scipy) as loader:
            values = geometry._load_mat(Path("UNIT_TEST_ONLY.mat"))
            self.assertEqual(loader.call_args.args, ("scipy.io",))
        vector, descriptor = geometry._vector(values["relu"])
        np.testing.assert_array_equal(vector, [1., 3., 2., 4.])
        self.assertEqual(descriptor["shape"], [2, 2])

    def test_nonfinite_values_and_numerically_large_vectors(self):
        vector, descriptor = geometry._vector(np.array([[float("nan"), 1.]]))
        self.assertIsNone(vector)
        self.assertEqual(descriptor["status"], "invalid_activation_values")
        self.assertAlmostEqual(geometry._cosine(np.array([1e300, 1e300]), np.array([1e300, 1e300])), 0., places=12)
        self.assertIsNone(geometry._cosine(np.zeros(2), np.ones(2)))


if __name__ == "__main__":
    unittest.main()
