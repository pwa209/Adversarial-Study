"""Cache tests use synthetic environment metadata, never scientific artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cluster import runtime_cache as cache


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class RuntimeCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "adversarial-study"
        self.source_run = self.root / "runs" / "old-release"
        self.run = self.root / "runs" / "new-release"
        self.source = self.root / "environments" / "old-release"
        self.destination = self.root / "environments" / "new-release"
        for path in (self.source_run / "receipts", self.run, self.source / "bin"):
            path.mkdir(parents=True)
        (self.source / "bin/python").write_text("synthetic Python metadata fixture")
        (self.source / "bin/activate").write_text("synthetic activation fixture")
        (self.source / "pyvenv.cfg").write_text("synthetic virtual environment fixture")
        self.runtime = dict(zip(cache.CPU_RUNTIME, ("3.11", "2.4", "1.17", "1.9", "3.16", "19", "arrow/19")))
        self.resolved = self.source_run / "environment-resolved.txt"
        self.resolved.write_text("numpy==2.4\nmne==1.9\n")
        self.runtime_path = self.source_run / "environment-runtime.json"
        write(self.runtime_path, self.runtime)
        self.receipt = self.source_run / "receipts/environment.json"
        write(self.receipt, {"stage": "environment", "status": "completed", "exit_code": 0,
                             "source_manifest_sha256": "a" * 64, "plan_sha256": "b" * 64})
        self.config = self.root / "cache.json"
        self.entry = {"source_release": "old-release", "environment": str(self.source), "stage": "environment",
                      "receipt": self.pin(self.receipt), "resolved": self.pin(self.resolved), "runtime": self.pin(self.runtime_path)}
        self.specification = {"schema_version": 1, "project_roots": [str(self.root)], "caches": {"cpu": self.entry}}
        self.save_config()

    def pin(self, path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def save_config(self):
        write(self.config, self.specification)

    def link(self, kind="cpu"):
        try:
            return cache.reuse(kind, self.destination, self.run, self.config)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                self.skipTest("Host does not permit directory symlinks")
            raise

    def recheck(self, kind="cpu"):
        with patch.object(cache.sys, "prefix", str(self.source)):
            return cache.reuse(kind, self.destination, self.run, self.config, phase="recheck")

    def fresh(self, kind="cpu"):
        _, resolved_name, runtime_name = cache.FILES[kind]
        (self.run / resolved_name).write_bytes(self.resolved.read_bytes())
        (self.run / runtime_name).write_bytes(self.runtime_path.read_bytes())

    def test_link_and_active_recheck_bind_evidence(self):
        linked = self.link()
        self.assertTrue(self.destination.is_symlink())
        self.assertEqual(self.destination.resolve(), self.source)
        self.assertEqual(linked["previous_source_manifest_sha256"], "a" * 64)
        self.assertEqual(linked["pinned_inputs"]["receipt"]["sha256"], self.entry["receipt"]["sha256"])
        self.fresh()
        verified = self.recheck()
        self.assertEqual(verified["status"], "active_cached_environment_verified_current_release_tests_required")
        self.assertTrue(verified["current_release_tests_required"])
        self.assertFalse(verified["trained_weights_reused"])
        self.assertFalse(verified["scientific_results_reused"])
        self.assertEqual(verified["active_resolved"]["sha256"], self.entry["resolved"]["sha256"])

    def test_missing_or_changed_source_receipt_refuses_link(self):
        self.receipt.write_text("{}")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.link()
        self.assertFalse(self.destination.exists())

    def test_rehashed_failed_source_stage_refuses_link(self):
        previous = json.loads(self.receipt.read_text())
        previous["status"] = "failed"
        write(self.receipt, previous)
        self.entry["receipt"] = self.pin(self.receipt)
        self.save_config()
        with self.assertRaisesRegex(ValueError, "did not complete"):
            self.link()

    def test_source_freeze_and_runtime_pins_are_mandatory(self):
        for path in (self.resolved, self.runtime_path):
            initial = path.read_bytes()
            path.write_bytes(initial + b" ")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                self.link()
            path.write_bytes(initial)

    def test_absent_kind_has_no_install_fallback(self):
        with self.assertRaisesRegex(ValueError, "refusing duplicate"):
            self.link("centaur")
        self.assertFalse(self.destination.exists())

    def test_destination_and_run_must_be_scoped_matching_new_release(self):
        outside = Path(self.temp.name) / "outside"
        with self.assertRaisesRegex(ValueError, "outside configured"):
            cache.reuse("cpu", outside, self.run, self.config)
        with self.assertRaisesRegex(ValueError, "same new release"):
            cache.reuse("cpu", self.destination, self.source_run, self.config)

    def test_existing_directory_or_link_is_never_overwritten(self):
        self.destination.mkdir()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.link()
        self.assertTrue(self.destination.is_dir())

    def test_changed_environment_metadata_fails_recheck(self):
        self.link()
        self.fresh()
        (self.source / "bin/activate").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed after link"):
            self.recheck()

    def test_runtime_recheck_requires_actual_cached_active_prefix(self):
        self.link()
        self.fresh()
        with patch.object(cache.sys, "prefix", str(self.run)):
            with self.assertRaisesRegex(ValueError, "activated cached"):
                cache.reuse("cpu", self.destination, self.run, self.config, phase="recheck")

    def test_changed_fresh_freeze_or_versions_fails_recheck(self):
        self.link()
        self.fresh()
        (self.run / "environment-resolved.txt").write_text("numpy==99\n")
        with self.assertRaisesRegex(ValueError, "pip freeze differs"):
            self.recheck()
        self.fresh()
        actual = dict(self.runtime, numpy="99")
        write(self.run / "environment-runtime.json", actual)
        with self.assertRaisesRegex(ValueError, "runtime versions differ"):
            self.recheck()

    def test_changed_config_after_link_fails_recheck(self):
        self.link()
        self.fresh()
        self.specification["extra"] = "modified after link"
        self.save_config()
        with self.assertRaisesRegex(ValueError, "changed after link"):
            self.recheck()

    def test_fresh_runtime_alias_to_old_evidence_is_rejected(self):
        self.link()
        self.fresh()
        actual = self.run / "environment-runtime.json"
        actual.unlink()
        actual.symlink_to(self.runtime_path)
        with self.assertRaisesRegex(ValueError, "newly generated"):
            self.recheck()

    def test_centaur_rechecks_new_allocation_not_old_gpu_metadata(self):
        self.source.rename(self.source.with_name("old-release-centaur"))
        self.source = self.source.with_name("old-release-centaur")
        self.destination = self.destination.with_name("new-release-centaur")
        self.resolved = self.source_run / cache.FILES["centaur"][1]
        self.resolved.write_text("torch==2.14\n")
        self.runtime_path = self.source_run / cache.FILES["centaur"][2]
        gpu = {"versions": {key: "1.0" for key in cache.GPU_PACKAGES}, "cuda": "12.6", "utc": "old",
               "status": "native_dependencies_available",
               "devices": [{"name": "A100", "total_memory_bytes": 40 * 1024**3}] * 2}
        write(self.runtime_path, gpu)
        self.receipt = self.source_run / "receipts/centaur_dependencies.json"
        write(self.receipt, {"stage": "centaur_dependencies", "status": "completed", "exit_code": 0,
                             "source_manifest_sha256": "c" * 64, "plan_sha256": "d" * 64})
        self.entry = {"source_release": "old-release", "environment": str(self.source), "stage": "centaur_dependencies",
                      "receipt": self.pin(self.receipt), "resolved": self.pin(self.resolved), "runtime": self.pin(self.runtime_path)}
        self.specification["caches"]["centaur"] = self.entry
        self.save_config()
        self.link("centaur")
        self.fresh("centaur")
        fresh = dict(gpu, utc="new", devices=[{"name": "NEW ACTUAL A100", "total_memory_bytes": 40 * 1024**3}] * 2)
        write(self.run / cache.FILES["centaur"][2], fresh)
        self.assertEqual(self.recheck("centaur")["runtime_identity"], {"versions": gpu["versions"], "cuda": "12.6"})
        # A separate new receipt cannot be overwritten; inspect the fresh-device
        # validator directly rather than pretending another verification occurred.
        fresh["devices"] = fresh["devices"][:1]
        with self.assertRaisesRegex(ValueError, "fresh allocated GPU"):
            cache._runtime_identity(fresh, "centaur")


if __name__ == "__main__":
    unittest.main()
