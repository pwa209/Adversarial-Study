"""Release-level lineage checks, separate from scientific software fixtures."""
import hashlib
import importlib
import json
from pathlib import Path
import unittest

from adversarial_study import core


class AdditiveReleaseTests(unittest.TestCase):
    def test_legacy_inventory_and_bytes_are_unchanged(self):
        source = Path(core.__file__).resolve().parents[2]
        parent_path = source / "PARENT_SOURCE_MANIFEST.json"
        if not parent_path.is_file():
            self.skipTest("Requires staged immutable additive release")
        expected = json.loads(parent_path.read_text())["files"]
        for name, digest in expected.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), digest)
        package = "study_pipeline/adversarial_study/"
        matlab = "study_pipeline/matlab/"
        original = {}
        for name, digest in expected.items():
            if name.startswith(package) and name[len(package):].endswith(".py") and "/" not in name[len(package):]:
                original["python/" + name[len(package):]] = digest
            if name.startswith(matlab) and name.endswith(".m") and "/" not in name[len(matlab):]:
                original["matlab/" + name[len(matlab):]] = digest
        self.assertEqual(core.code_hashes(), original)

    def test_additions_are_explicitly_separate_and_pinned(self):
        source = Path(core.__file__).resolve().parents[2]
        if not (source / "PARENT_SOURCE_MANIFEST.json").is_file():
            self.skipTest("Requires staged immutable additive release")
        manifest = json.loads((source / "SOURCE_MANIFEST.json").read_text())
        for name in ("discovery", "mechanisms", "validation", "breadth", "tactile"):
            module = importlib.import_module("adversarial_study.proposal_" + name)
            path = Path(module.__file__).resolve()
            self.assertEqual(path.parent, source / "study_pipeline/proposal_additions")
            relative = path.relative_to(source).as_posix()
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest["files"][relative])


if __name__ == "__main__":
    unittest.main()
