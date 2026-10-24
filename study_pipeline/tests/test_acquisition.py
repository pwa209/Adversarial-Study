"""Network-free acquisition integrity tests; all bytes are disposable fixtures."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adversarial_study import acquisition as acq


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data)), "ETag": '"fixture"'}


def resource(data=b"real_fixture_bytes", relative="source/raw.bin"):
    return {"resource_id": "object1", "source_id": "source", "relative_path": relative,
            "url": "https://example.org/version1/raw.bin", "expected_bytes": len(data),
            "checksum": {"algorithm": "sha256", "value": hashlib.sha256(data).hexdigest()},
            "default_acquire": True}


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_acquire(self, item, data):
        with patch.object(acq.urllib.request, "urlopen", return_value=Response(data)):
            return acq.acquire_manifest({"resources": [item]}, self.root,
                                        max_bytes_per_second=1e12, request_interval_s=0, retries=0)

    def test_atomic_published_checksum_and_reuse_rehash(self):
        item = resource()
        result = self.run_acquire(item, b"real_fixture_bytes")
        target = self.root/item["relative_path"]
        self.assertEqual(result["resources"][0]["status"], "published_verified")
        self.assertFalse(target.with_name(target.name+".part").exists())
        with patch.object(acq.urllib.request, "urlopen", side_effect=AssertionError("no network")):
            reused = acq.acquire_manifest({"resources": [item]}, self.root)
        self.assertEqual(reused["transferred_bytes"], 0)
        target.write_bytes(b"tamper_fixture_byts")
        with self.assertRaises(ValueError):
            acq.verify_acquired(self.root, item)

    def test_checksum_failure_never_publishes(self):
        item = resource(b"expected")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_acquire(item, b"modified")
        self.assertFalse((self.root/item["relative_path"]).exists())
        self.assertTrue((self.root/(item["relative_path"]+".part")).exists())

    def make_partial(self, item, prefix, etag='"fixture"'):
        target = self.root/item["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.with_name(target.name+".part").write_bytes(prefix)
        target.with_name(target.name+".part.json").write_text(json.dumps({
            "resource_manifest_sha256": acq.canonical_hash(item), "etag": etag}))

    def test_resume_exact_range(self):
        data = b"abcdefghij"
        item = resource(data)
        self.make_partial(item, data[:4])
        response = Response(data[4:], 206, {"Content-Range": "bytes 4-9/10", "Content-Length": "6", "ETag": '"fixture"'})
        with patch.object(acq.urllib.request, "urlopen", return_value=response) as opened:
            result = acq.acquire_manifest({"resources": [item]}, self.root, max_bytes_per_second=1e12, request_interval_s=0)
        self.assertEqual(opened.call_args.args[0].headers["Range"], "bytes=4-")
        self.assertEqual(result["transferred_bytes"], 6)
        self.assertEqual((self.root/item["relative_path"]).read_bytes(), data)

    def test_range_ignored_restarts_instead_of_appending(self):
        data = b"abcdefghij"
        item = resource(data)
        self.make_partial(item, data[:4])
        self.run_acquire(item, data)
        self.assertEqual((self.root/item["relative_path"]).read_bytes(), data)

    def test_wrong_range_and_changed_etag_fail(self):
        item = resource(b"abcdefghij")
        self.make_partial(item, b"abcd")
        for headers in ({"Content-Range": "bytes 0-5/10", "ETag": '"fixture"'},
                        {"Content-Range": "bytes 4-9/10", "ETag": '"changed"'}):
            with self.subTest(headers=headers), patch.object(acq.urllib.request, "urlopen", return_value=Response(b"efghij", 206, headers)):
                with self.assertRaises(ValueError):
                    acq.acquire_manifest({"resources": [item]}, self.root, request_interval_s=0)

    def test_orphan_partial_or_changed_identity_rejected(self):
        item = resource(b"abcdefghij")
        self.make_partial(item, b"abcd")
        item["url"] += "?different"
        with self.assertRaisesRegex(ValueError, "identity changed"):
            acq.acquire_manifest({"resources": [item]}, self.root)
        (self.root/(item["relative_path"]+".part.json")).unlink()
        with self.assertRaisesRegex(ValueError, "no pinned identity"):
            acq.acquire_manifest({"resources": [item]}, self.root)

    def test_budget_and_selection_preflight_open_no_http(self):
        item = resource()
        for kwargs in ({"max_bytes": 1}, {"resource_ids": ["missing"]}, {"source_ids": ["missing"]}):
            with self.subTest(kwargs=kwargs), patch.object(acq.urllib.request, "urlopen", side_effect=AssertionError("network forbidden")):
                with self.assertRaises(ValueError):
                    acq.acquire_manifest({"resources": [item]}, self.root, **kwargs)

    def test_containment_https_and_duplicates(self):
        for path in ("../escape", "/absolute", "x\\escape", "C:/unsafe"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                acq.acquire_manifest({"resources": [resource(relative=path)]}, self.root)
        item = resource()
        item["url"] = "http://example.org/raw"
        with self.assertRaises(ValueError):
            acq.acquire_manifest({"resources": [item]}, self.root)
        item = resource()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            acq.acquire_manifest({"resources": [item, item]}, self.root)

    def test_git_blob_hash_not_file_sha1_and_md5(self):
        data = b"source metadata\n"
        for algorithm, value in (("git_blob_sha1", hashlib.sha1(b"blob 16\0"+data).hexdigest()),
                                  ("md5", hashlib.md5(data).hexdigest())):
            with self.subTest(algorithm=algorithm):
                item = resource(data, relative=algorithm+"/meta.txt")
                item["checksum"] = {"algorithm": algorithm, "value": value}
                self.run_acquire(item, data)
                self.assertEqual(acq.verify_acquired(self.root, item)["sha256"], hashlib.sha256(data).hexdigest())

    def test_kernel_lock_prevents_simultaneous_writer_and_survives_old_lockfile(self):
        lock = self.root/"same.lock"
        lock.write_bytes(b"\0")  # previous worker's inode remains; no stale-PID gate
        with acq._resource_lock(lock):
            with self.assertRaises(OSError):
                with acq._resource_lock(lock):
                    self.fail("simultaneous writer acquired same resource")
        with acq._resource_lock(lock):
            self.assertTrue(lock.exists())

    def test_checked_in_inventory_has_no_duplicates_and_truthful_counts(self):
        path = Path(__file__).resolve().parents[1]/"configs"/"sources.public.json"
        manifest = json.loads(path.read_text())
        for item in manifest["resources"]:
            acq.validate_resource(item)
        self.assertEqual(len({r["resource_id"] for r in manifest["resources"]}), len(manifest["resources"]))
        monash = next(s for s in manifest["sources"] if s["source_id"] == "monash_wanderim")
        self.assertEqual(monash["raw_recordings_available"], 27)
        self.assertEqual(monash["matched_behaviour_available"], 26)
        self.assertIsNone(monash["license"])
        self.assertEqual(sum(r["default_acquire"] for r in manifest["resources"]), 4)


if __name__ == "__main__":
    unittest.main()
