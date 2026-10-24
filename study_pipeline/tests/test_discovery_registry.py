"""Complementary scientific discovery views retain their measurement limits."""
import copy
import unittest

from adversarial_study import discovery
from adversarial_study.core import StudyError


def episode(index, score=.5, capacity=None, report=None, person="p", recording="r", **extra):
    return {"episode_id": f"{person}-{index}", "recording_id": recording,
            "participant_id": person, "source_id": "source", "start_s": index * 20,
            "stop_s": index * 20 + 10, "score": score, "capacity_value": capacity,
            "report_category": report, "saturated_fraction": 0, **extra}


class DiscoveryRegistryTests(unittest.TestCase):
    def test_complementary_registry_finds_reverse_contrast_without_assuming_a_finding(self):
        rows = [episode(0, .1, .8, "experienced"), episode(1, .8, .81, "experienced")]
        result = discovery.discovery_registry(rows, {})
        self.assertEqual(result["same_score_different_capacity"]["pair_count"], 0)
        self.assertEqual(result["same_capacity_different_score"]["pair_count"], 1)
        self.assertEqual(result["same_score_different_reports"]["pair_count"], 0)
        self.assertEqual(result["score_distribution"]["input_rows"], 2)
        self.assertEqual(result["temporal_comparisons"]["comparison_count"], 1)

    def test_distinct_report_categories_do_not_require_numeric_capacity_or_merge_recall(self):
        rows = [episode(0, .5, report="mind blanking"), episode(1, .51, report="don't remember")]
        result = discovery.match_report_pairs(rows, {})
        self.assertEqual(result["pair_count"], 1)
        self.assertEqual(result["pairs"][0]["report_a"], "mind blanking")
        self.assertEqual(result["pairs"][0]["report_b"], "don't remember")
        self.assertIsNone(result["pairs"][0]["capacity_difference_abs"])
        self.assertIsNone(result["pairs"][0]["report_support_a"])
        self.assertEqual(len(discovery.score_distribution(rows)["groups"]), 2)

    def test_unknown_saturation_remains_unknown_and_bounded_ties_are_not_preclip_ties(self):
        rows = [episode(0, 1, .2, "A", preclip_gcs=16), episode(1, 1, .8, "B", preclip_gcs=21)]
        rows[0].pop("saturated_fraction")
        rows[1]["saturated_fraction"] = 1
        summary = discovery.score_distribution(rows)
        unknown = next(g for g in summary["groups"] if g["original_report_category"] == "A")
        self.assertEqual(unknown["rows_with_unknown_saturation"], 1)
        self.assertIsNone(unknown["mean_saturated_channel_fraction"])
        self.assertEqual(discovery.match_report_pairs(rows, {})["pair_count"], 0)
        pair = discovery.match_report_pairs(rows, {"exclude_saturated": False})["pairs"][0]
        self.assertEqual(pair["score_difference_abs"], 0)
        self.assertEqual(pair["preclip_gcs_b"] - pair["preclip_gcs_a"], 5)
        self.assertEqual(pair["saturation_status_a"], "unknown")
        temporal = discovery.temporal_registry(rows)["comparisons"][0]
        self.assertEqual(temporal["bounded_score_change"], 0)
        self.assertEqual(temporal["preclip_gcs_change"], 5)

    def test_unselected_relationships_give_people_equal_weight_and_do_not_select_score_tails(self):
        rows = [episode(i, score, 1 - score, person="many") for i, score in enumerate([.1, .2, .3, .4, .5, .6])]
        rows += [episode(i, score, score, person="few") for i, score in enumerate([.1, .3, .8])]
        rows += [episode(0, 1, None, person="missing", saturated_fraction=1)]
        group = discovery.unselected_relationships(rows)["groups"][0]
        self.assertAlmostEqual(group["mean_within_person_correlation_equal_participant_weight"], 0)
        self.assertEqual(group["participants_with_estimable_relationship"], 2)
        self.assertEqual(group["input_rows"], 10)
        self.assertEqual(group["exclusions"]["missing_measured_value"], 1)
        distinct_source = copy.deepcopy(rows[:3])
        for row in distinct_source:
            row["source_id"] = "different_capacity_units"
            row["episode_id"] = "other-source-" + row["episode_id"]
        self.assertEqual(len(discovery.unselected_relationships(rows + distinct_source)["groups"]), 2)

    def test_missing_reports_do_not_hide_available_scores_and_constant_capacity_has_no_correlation(self):
        rows = [episode(i, .1 + .08 * i, .8, report="A" if i % 2 else None) for i in range(9)]
        self.assertEqual(len(discovery.score_distribution(rows)["groups"]), 2)
        group = discovery.unselected_relationships(rows)["groups"][0]
        self.assertEqual(group["participants_with_paired_values"], 1)
        self.assertEqual(group["participants_with_estimable_relationship"], 0)
        self.assertIsNone(group["mean_within_person_correlation_equal_participant_weight"])

    def test_temporal_change_uses_actual_report_and_capacity_support(self):
        rows = [episode(0, .2, .4, "white dream", report_start_s=5, report_stop_s=10,
                        report_timing_basis="analyst_assigned_interval", capacity_start_s=0, capacity_stop_s=10),
                episode(1, .7, .9, "reported no experience", report_start_s=25, report_stop_s=30,
                        report_timing_basis="explicit_interval_report", capacity_start_s=20, capacity_stop_s=30)]
        pair = discovery.temporal_registry(rows)["comparisons"][0]
        self.assertTrue(pair["sampled_report_category_changed"])
        self.assertEqual(pair["unobserved_gap_between_score_windows_s"], 10)
        self.assertEqual(pair["unobserved_gap_between_report_intervals_s"], 15)
        self.assertEqual(pair["report_support_a"]["timing_basis"], "analyst_assigned_interval")
        self.assertAlmostEqual(pair["measured_capacity_change"], .5)
        rows[1].pop("capacity_stop_s")
        rows[1]["report_start_s"], rows[1]["report_stop_s"] = 35, 40
        pair = discovery.temporal_registry(rows)["comparisons"][0]
        self.assertIsNone(pair["sampled_report_category_changed"])
        self.assertIsNone(pair["measured_capacity_change"])

    def test_comparisons_require_known_nonoverlapping_recording_clock_and_do_not_bridge_invalid_scores(self):
        rows = [episode(0, .1, .8, "A"), episode(1, None, .8, "B"), episode(2, .8, .8, "B")]
        temporal = discovery.temporal_registry(rows)
        self.assertEqual(temporal["comparison_count"], 0)
        self.assertEqual(temporal["excluded_by_reason"]["adjacent_pair_missing_valid_score"], 2)
        for row in rows:
            row.pop("recording_id")
        self.assertEqual(discovery.temporal_registry(rows)["comparison_count"], 0)
        self.assertEqual(discovery.match_capacity_pairs(rows, {})["pair_count"], 0)
        self.assertEqual(discovery.match_score_pairs(rows, {})["pair_count"], 0)
        rows = [episode(0, .1, .8, "A"), episode(1, .8, .8, "B")]
        rows[1]["start_s"], rows[1]["stop_s"] = 5, 15
        self.assertEqual(discovery.temporal_registry(rows)["comparison_count"], 0)
        self.assertEqual(discovery.match_capacity_pairs(rows, {"min_time_separation_s": 0})["pair_count"], 0)
        rows[1]["start_s"], rows[1]["stop_s"] = 20, 30
        rows[1]["recording_id"] = "different_clock"
        self.assertEqual(discovery.temporal_registry(rows)["comparison_count"], 0)
        self.assertEqual(discovery.match_capacity_pairs(rows, {})["pair_count"], 0)

    def test_numeric_matching_does_not_reuse_episodes_within_a_registry(self):
        rows = [episode(i, score, .8, "A") for i, score in enumerate([.1, .8, .2, .9])]
        result = discovery.match_capacity_pairs(rows, {})
        self.assertEqual(result["pair_count"], 2)
        ids = [pair[k] for pair in result["pairs"] for k in ("episode_a", "episode_b")]
        self.assertEqual(len(ids), len(set(ids)))

    def test_invalid_settings_and_duplicate_observations_are_rejected(self):
        rows = [episode(0, .1, .8, "A")]
        with self.assertRaises(StudyError):
            discovery.unselected_relationships(rows, {"minimum_relationship_episodes": 2})
        with self.assertRaises(StudyError):
            discovery.match_capacity_pairs(rows, {"same_capacity_tolerance": -1})
        with self.assertRaises(StudyError):
            discovery.temporal_registry(rows, {"maximum_temporal_gap_s": -1})
        with self.assertRaises(StudyError):
            discovery.match_report_pairs(rows + rows, {})
        with self.assertRaises(StudyError):
            discovery.temporal_registry(rows + rows)


if __name__ == "__main__":
    unittest.main()
