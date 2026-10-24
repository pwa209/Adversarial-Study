"""Scientific edge cases for the additive source-bound discovery audit."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest

from adversarial_study import core, proposal_discovery as discovery


class ProposalDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = discovery.load_config()

    def row(self, episode, *, person="p", source="monash_wanderim", recording="r", score=.2,
            report="ON", start=0, stage="", capacity=None):
        return {"source_id": source, "participant_id": person, "episode_id": episode,
            "recording_id": recording, "score": score, "report_category": report,
            "start_s": start, "stop_s": start+10, "sleep_stage": stage,
            "role": "development", "synthetic": False, "score_origin": "native_frozen_v2",
            "saturated_fraction": 0, "capacity_value": capacity,
            "capacity_construct": "subsequent_tactile_detection" if capacity is not None else "",
            "capacity_units": "binary_felt_yes_no" if capacity is not None else ""}

    def sleep(self, episode, **kw):
        return self.row(episode, source="dream_tononi_serial", recording=episode, stage="N2",
                        report=kw.pop("report", "experience"), **kw)

    def test_missing_sart_and_tactile_reports_are_never_observed_discordances(self):
        rows = [self.row("a"), self.row("b", report="missing_original_response", start=30),
                self.row("c", source="openneuro_ds001785", report="hit", start=60),
                self.row("d", source="openneuro_ds001785", report="missing_first_order_response", start=90)]
        result = discovery.analyse(rows, self.cfg)
        self.assertEqual(result["natural_pairs"], [])
        self.assertEqual(len(result["observations"]), 4)
        self.assertEqual(sum(r["report_eligibility"] == "missing_original_report" for r in result["observations"]), 2)
        self.assertEqual(result["audit"]["all_scored_participants"], 1)

    def test_sleep_pairs_cross_disconnected_recordings_only_within_person_source_stage(self):
        rows = [self.sleep("a"), self.sleep("b", report="no_experience"),
                self.sleep("c", person="other", report="no_experience"),
                {**self.sleep("d", report="no_experience"), "sleep_stage": "REM"},
                {**self.sleep("e", report="no_experience"), "source_id": "dream_oudiette_n1"}]
        result = discovery.analyse(rows, self.cfg)
        pairs = result["natural_pairs"]
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0]["episode_a"], pairs[0]["episode_b"]}, {"a", "b"})
        self.assertEqual(pairs[0]["pairing_scope"], "within_person_source_stage_disconnected_awakening")
        self.assertEqual(len(result["sleep_participant_contrasts"]), 1)
        self.assertEqual(pairs[0]["sleep_stage"], "N2")

    def test_sleep_contrast_weights_people_once_and_keeps_stages_separate(self):
        rows = [self.sleep("a", person="p1", score=.2), self.sleep("b", person="p1", report="no_experience", score=.8)]
        rows += [self.sleep(f"c{i}", person="p2", score=.6) for i in range(20)]
        rows += [self.sleep("d", person="p2", report="no_experience", score=.4)]
        rows += [{**self.sleep("e", person="p1", score=.1), "sleep_stage": "REM"}]
        result = discovery.analyse(rows, self.cfg)
        group = result["audit"]["sleep_comparisons"][0]
        self.assertEqual(group["independent_participants"], 2)
        self.assertEqual(group["sleep_stage"], "N2")
        self.assertAlmostEqual(group["mean_within_person_score_difference"], (-.6+.2)/2)
        self.assertIsNotNone(group["participant_bootstrap_95_interval"])

    def test_low_score_sensitivity_retains_nulls_missing_people_and_comparators(self):
        rows = [self.row("a", score=.1), self.row("b", score=.2, start=20),
                self.row("c", score=.8, start=40), self.row("d", score=.9, start=60),
                self.row("e", person="missing", report="missing_original_response", score=.1),
                self.row("f", person="no_low", score=.9)]
        result = discovery.analyse(rows, self.cfg)
        main = [r for r in result["low_score_participant_yield"] if r["main_threshold"]]
        self.assertEqual(len(main), 3)
        self.assertTrue(next(r for r in main if r["participant_id"] == "p")["informative_under_declared_episode_budget"])
        missing = next(r for r in main if r["participant_id"] == "missing")
        self.assertEqual(missing["missing_or_unsupported_report_rows"], 1)
        self.assertEqual(missing["low_supported_report_rows"], 0)
        summary = next(r for r in result["audit"]["low_score_yield"] if r["main_threshold"])
        self.assertEqual(summary["all_scored_participants"], 3)
        self.assertAlmostEqual(summary["informative_yield_among_all_scored_people"], 1/3)
        self.assertEqual(next(r for r in main if r["participant_id"] == "no_low")["missing_comparison_reason"], "no_low_case")
        self.assertEqual({r["threshold"] for r in result["low_score_cases"]}, {.1, .25, .5})

    def test_repeated_or_overlapping_windows_cannot_satisfy_reliability_budget(self):
        rows = [self.row(f"a{i}", start=i, score=.1) for i in range(5)]
        rows += [self.row(f"b{i}", start=40+i, score=.9) for i in range(5)]
        result = discovery.analyse(rows, self.cfg)
        main = next(r for r in result["low_score_participant_yield"] if r["main_threshold"])
        self.assertEqual(main["low_supported_report_rows"], 5)
        self.assertEqual(main["low_disjoint_windows_within_recordings"], 1)
        self.assertEqual(main["same_report_disjoint_comparator_windows"], 1)
        self.assertFalse(main["informative_under_declared_episode_budget"])
        reliability = next(r for r in result["low_score_reliability"] if r["main_threshold"])
        self.assertFalse(reliability["sampled_recurrence_available"])
        self.assertIn("not_qualified", reliability["independent_repeat_session_reliability_status"])

    def test_tactile_low_support_stays_subsequent_and_saturation_is_explicit(self):
        rows = [self.row("a", source="openneuro_ds001785", report="hit", capacity=1, score=0),
                self.row("b", source="openneuro_ds001785", report="miss", capacity=0, start=30, score=0)]
        rows[0]["saturated_fraction"] = 1
        result = discovery.analyse(rows, self.cfg)
        self.assertEqual(result["natural_pairs"], [])
        case = next(r for r in result["low_score_cases"] if r["main_threshold"])
        self.assertEqual(case["case_class"], "low_score_with_subsequent_detection")
        self.assertFalse(case["experience_during_score_window_certified"])
        self.assertEqual(case["saturation_status"], "at_least_one_saturated_channel")

    def test_bounded_pair_output_preserves_each_person_and_full_denominators(self):
        cfg = deepcopy(self.cfg)
        cfg["maximum_pairs_per_participant_family"] = 1
        rows = [self.row(f"{p}{i}", person=p, report="ON" if i%2 else "MW", start=20*i) for p in ("p1", "p2") for i in range(6)]
        result = discovery.analyse(rows, cfg)
        self.assertEqual(len(result["natural_pairs"]), 2)
        self.assertEqual({r["participant_id"] for r in result["natural_pairs"]}, {"p1", "p2"})
        self.assertTrue(result["audit"]["pair_census"]["truncated"])
        self.assertAlmostEqual(sum(r["participant_weight"] for r in result["natural_pairs"]), 1)
        for row in result["natural_pairs"]:
            self.assertEqual(row["all_eligible_pairs_for_person_family"], 9)

    def test_invalid_score_and_unqualified_source_remain_in_census(self):
        rows = [self.row("a", score=""), self.row("b", source="new_source", report="some_label")]
        result = discovery.analyse(rows, self.cfg)
        self.assertEqual(len(result["observations"]), 2)
        self.assertEqual(result["natural_pairs"], [])
        self.assertEqual(result["low_score_cases"], [])
        self.assertEqual(result["observations"][1]["report_eligibility"], "source_report_policy_unqualified")

    def test_no_result_gate_empty_branch_completes_and_protected_data_reject(self):
        result = discovery.analyse([], self.cfg)
        self.assertEqual(result["audit"]["status"], discovery.STATUS)
        self.assertFalse(result["audit"]["result_direction_gate"])
        with self.assertRaises(core.StudyError):
            discovery.analyse([{**self.row("a"), "role": "evaluation"}], self.cfg)
        with self.assertRaises(core.StudyError):
            discovery.analyse([self.row("a"), self.row("a")], self.cfg)

    def test_source_bound_io_preserves_legacy_and_detects_changed_artifacts(self):
        test_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=test_root if os.name == "nt" else None) as folder:
            root = Path(folder)
            if os.name == "nt":
                self.assertTrue(root.resolve().is_relative_to(test_root))
            legacy = root/"legacy"
            legacy.mkdir()
            rows = [self.row("a"), self.row("b", report="MW", start=20)]
            core.write_csv(legacy/"all_scored_observations.csv", rows)
            core.write_json(legacy/"receipt.json", {"status": "model_guided_discovery_completed",
                "outputs": {"all_scored_observations.csv": core.sha256_file(legacy/"all_scored_observations.csv")},
                "inputs": [{"path": "original-source-bound-eeg", "sha256": "a"*64}]})
            original_hash = core.sha256_file(legacy/"receipt.json")
            result = discovery.run([legacy], discovery.DEFAULT_CONFIG, root/"audit")
            self.assertEqual(result["status"], discovery.STATUS)
            self.assertEqual(core.sha256_file(legacy/"receipt.json"), original_hash)
            discovery.verify_outputs(root/"audit")
            with self.assertRaises(core.StudyError):
                discovery.run([legacy], discovery.DEFAULT_CONFIG, root/"audit")
            with (root/"audit"/"natural_pairs.csv").open("a") as stream:
                stream.write("tampered\n")
            with self.assertRaises(core.StudyError):
                discovery.verify_outputs(root/"audit")
            core.write_csv(legacy/"all_scored_observations.csv", [self.row("changed")])
            with self.assertRaises(core.StudyError):
                discovery.read_legacy([legacy], self.cfg)


if __name__ == "__main__":
    unittest.main()
