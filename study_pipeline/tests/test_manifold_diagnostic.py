"""Design-based randomized inference and real future-data admission requirements."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

from adversarial_study import core, manifold_diagnostic as diagnostic


def trials(outcome_active=.8, outcome_control=.2):
    return [{"participant_id": p, "trial_id": t, "eligible": "1", "assigned_arm": arm,
             "assignment_probability_active": ".5", "observed_outcome": str(value), "missing_reason": ""}
            for p in ("new1", "new2", "new3")
            for t, arm, value in (("a", "active", outcome_active), ("c", "control", outcome_control))]


class RandomizedDiagnosticTests(unittest.TestCase):
    def estimate(self, rows):
        return diagnostic.randomized_estimate(rows, outcome_min=0, outcome_max=1, bootstrap_samples=100)

    def test_design_contrast_negative_and_null_retained(self):
        self.assertAlmostEqual(self.estimate(trials())["point_estimate"], .6)
        self.assertAlmostEqual(self.estimate(trials(.2, .8))["point_estimate"], -.6)
        self.assertAlmostEqual(self.estimate(trials(.3, .3))["point_estimate"], 0)

    def test_people_are_equal_weight_despite_unequal_trial_counts(self):
        rows = trials()
        for index in range(20):
            rows.append({**rows[0], "trial_id": "extra" + str(index)})
        answer = self.estimate(rows)
        expected = (((.8 * 2 - .2 * 2 + 20 * .8 * 2) / 22) + .6 + .6) / 3
        self.assertAlmostEqual(answer["point_estimate"], expected)
        self.assertEqual(answer["independent_participants"], 3)

    def test_all_missing_trials_remain_in_bounds_and_denominator(self):
        rows = trials()
        rows[0].update(observed_outcome="", missing_reason="equipment_failure")
        answer = self.estimate(rows)
        self.assertIsNone(answer["point_estimate"])
        self.assertEqual(answer["denominators"]["eligible_rows"], 6)
        self.assertEqual(answer["missing_outcomes"], 1)
        lo, hi = answer["identification_interval"]
        self.assertLessEqual(lo, .6)
        self.assertGreaterEqual(hi, .6)
        self.assertEqual(answer["participant_estimates"][0]["eligible_randomized_trials"], 2)

    def test_missing_control_sign_gives_correct_identification_bounds(self):
        rows = trials()
        rows[1].update(observed_outcome="", missing_reason="no_recording")
        answer = self.estimate(rows)
        self.assertAlmostEqual(answer["identification_interval"][0], (.8 - 1 + .6 + .6) / 3)
        self.assertAlmostEqual(answer["identification_interval"][1], (.8 + .6 + .6) / 3)

    def test_invalid_randomization_or_duplicate_outcomes_fail(self):
        for change in ({"assignment_probability_active": "0"}, {"assignment_probability_active": "nan"},
                       {"assigned_arm": "unknown"}, {"observed_outcome": "1.1"},
                       {"observed_outcome": "", "missing_reason": ""}):
            rows = trials(); rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(core.StudyError):
                self.estimate(rows)
        rows = trials(); rows.append(rows[0])
        with self.assertRaises(core.StudyError): self.estimate(rows)

    def test_computational_candidate_does_not_authorize_human_analysis(self):
        with patch.object(diagnostic, "verify_freeze", return_value={"human_intervention_qualified": False}):
            with self.assertRaisesRegex(core.StudyError, "computational candidate"):
                diagnostic.evaluate("unused", "unused", "unused", "unused", "unused")

    def test_timezone_and_changed_raw_provenance_fail(self):
        with self.assertRaises(core.StudyError): diagnostic._utc("2026-10-04T12:00:00")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.bin"; source.write_bytes(b"original")
            pins = [diagnostic._pin(source)]; diagnostic._verify_pins(pins)
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(core.StudyError, "changed"): diagnostic._verify_pins(pins)

    def future_fixture(self, root):
        frozen = root / "frozen"; frozen.mkdir()
        core.write_json(frozen / "frozen_protocol.json", {"fixture": "prospective_guard_test"})
        core.write_json(frozen / "receipt.json", {"fixture": "prospective_guard_test"})
        protocol = {"human_intervention_qualified": True, "primary_neural_endpoint": {"name": "fixture_measured_endpoint", "unit": "normalized"},
                    "created_utc": "2026-10-04T10:00:00Z", "development_participant_ids": ["development1"]}
        registry = root / "assignments.csv"; rows = trials()
        core.write_csv(registry, [{k:r[k] for k in ("participant_id", "trial_id", "eligible", "assigned_arm", "assignment_probability_active")} for r in rows])
        outcomes = root / "outcomes.csv"; core.write_csv(outcomes, rows)
        source = root / "raw.bin"; source.write_bytes(b"synthetic fixture only; not human evidence")
        registration = root / "registration.json"
        core.write_json(registration, {"frozen_protocol_sha256": core.sha256_file(frozen / "frozen_protocol.json"),
            "primary_neural_endpoint": protocol["primary_neural_endpoint"], "registered_utc": "2026-10-04T11:00:00Z", "outcome_min": 0, "outcome_max": 1, "bootstrap_samples": 100})
        data_path = root / "data_receipt.json"
        data = {"frozen_protocol_sha256": core.sha256_file(frozen / "frozen_protocol.json"), "observations_sha256": core.sha256_file(outcomes),
            "registration_sha256": core.sha256_file(registration), "human_data": True, "synthetic": False,
            "raw_source_artifacts": [diagnostic._pin(source)], "assignment_registry": diagnostic._pin(registry),
            "first_outcome_acquired_utc": "2026-10-05T11:00:00Z", "participant_ids": sorted({r["participant_id"] for r in rows}),
            "complete_eligible_randomization_denominator": True}
        core.write_json(data_path, data)
        return protocol, frozen, outcomes, registration, data_path, rows, data

    def test_original_assignment_manifest_blocks_omitted_trials_even_with_rehashed_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol, frozen, outcomes, registration, dp, rows, data = self.future_fixture(root)
            core.write_csv(outcomes, rows[:-1]); data["observations_sha256"] = core.sha256_file(outcomes); core.write_json(dp, data)
            with patch.object(diagnostic, "verify_freeze", return_value=protocol):
                with self.assertRaisesRegex(core.StudyError, "every original assignment"):
                    diagnostic.evaluate(frozen, outcomes, registration, dp, root / "output")

    def test_original_assignment_arm_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol, frozen, outcomes, registration, dp, rows, data = self.future_fixture(root)
            rows[0]["assigned_arm"] = "control"; core.write_csv(outcomes, rows)
            data["observations_sha256"] = core.sha256_file(outcomes); core.write_json(dp, data)
            with patch.object(diagnostic, "verify_freeze", return_value=protocol):
                with self.assertRaisesRegex(core.StudyError, "differs from original assignment"):
                    diagnostic.evaluate(frozen, outcomes, registration, dp, root / "output")

    def test_new_people_and_prospective_timing_are_required(self):
        for fault in ("person_overlap", "late_registration", "synthetic_data"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); protocol, frozen, outcomes, registration, dp, rows, data = self.future_fixture(root)
                if fault == "person_overlap": protocol["development_participant_ids"].append("new1")
                elif fault == "late_registration": data["first_outcome_acquired_utc"] = "2026-10-04T09:00:00Z"
                else: data["synthetic"] = True
                core.write_json(dp, data)
                with patch.object(diagnostic, "verify_freeze", return_value=protocol), self.assertRaises(core.StudyError):
                    diagnostic.evaluate(frozen, outcomes, registration, dp, root / "output")

    def test_complete_source_bound_randomization_runs_without_a_favourable_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol, frozen, outcomes, registration, dp, rows, data = self.future_fixture(root)
            with patch.object(diagnostic, "verify_freeze", return_value=protocol):
                answer = diagnostic.evaluate(frozen, outcomes, registration, dp, root / "output")
            self.assertAlmostEqual(answer["point_estimate"], .6)
            self.assertTrue((root / "output/participant_estimates.csv").is_file())


if __name__ == "__main__": unittest.main()
