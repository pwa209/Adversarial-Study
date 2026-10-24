"""Cross-assay inference guards; fixture receipts are not executed training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adversarial_study import cross_system


class CrossSystemGuards(unittest.TestCase):
    def fixtures(self, root, biological=True):
        for name, filename, record in (
            ("clinical", "training_receipt.json", {"native_training_executed": True, "assay_id": "tactile_detection"}),
            ("behaviour", "comparison_receipt.json", {"all_models_human_finetuned": True, "assay_id": "lottery_choice"}),
            ("biological", "training_receipt.json", {"human_parameter_fitting_executed": biological, "assay_id": "temporal_proxy"}),
        ):
            (root / name).mkdir()
            (root / name / filename).write_text(json.dumps(record))

    def test_adaptation_does_not_equate_unrelated_human_assays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            with patch("native_tools.human_finetune.verify_adaptation") as clinical, patch("adversarial_study.extension_finetune.verify_comparison") as behavioural, patch("adversarial_study.human_generative_adaptation.verify_adaptation") as generative:
                report = cross_system.audit(root / "clinical", root / "biological", root / "behaviour", root / "output")
            for verifier in (clinical, behavioural, generative):
                verifier.assert_called_once()
            self.assertTrue(report["all_systems_human_adapted"])
            self.assertFalse(report["cross_modal_comparison_executed"])
            self.assertFalse(report["shared_human_assay_qualified"])
            self.assertNotEqual(report["clinical_assay"], report["behavioural_assay"])

    def test_no_evaluable_parameter_fit_is_recorded_without_permitting_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root, biological=False)
            with patch("native_tools.human_finetune.verify_adaptation"), patch("adversarial_study.extension_finetune.verify_comparison"), patch("adversarial_study.human_generative_adaptation.verify_adaptation"):
                report = cross_system.audit(root / "clinical", root / "biological", root / "behaviour", root / "output")
            self.assertEqual(report["status"], "cross_system_human_training_prerequisites_missing")
            self.assertFalse(report["all_systems_human_adapted"])

    def test_boolean_receipt_cannot_bypass_changed_checkpoint_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            with patch("native_tools.human_finetune.verify_adaptation", side_effect=ValueError("adapted checkpoint changed")):
                with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                    cross_system.audit(root / "clinical", root / "biological", root / "behaviour", root / "output")
            self.assertFalse((root / "output").exists())

    def test_partial_manifold_evidence_cannot_be_claimed_as_connected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            with patch("native_tools.human_finetune.verify_adaptation"), patch("adversarial_study.extension_finetune.verify_comparison"), patch("adversarial_study.human_generative_adaptation.verify_adaptation"):
                with self.assertRaisesRegex(ValueError, "supplied together"):
                    cross_system.audit(root / "clinical", root / "biological", root / "behaviour", root / "output", manifold=root / "unmatched")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
