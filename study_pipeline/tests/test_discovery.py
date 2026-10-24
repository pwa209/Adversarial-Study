import unittest

import numpy as np

from adversarial_study import discovery


class DiscoveryTests(unittest.TestCase):
    def rows(self):
        return [{"episode_id": f"e{i}", "recording_id": "r", "participant_id": "p", "source_id": "s",
                 "start_s": i * 20, "stop_s": i * 20 + 10, "score": .5 + i * .005,
                 "capacity_value": i % 2, "saturated_fraction": 0, "report_category": f"report_{i % 2}"}
                for i in range(4)]

    def test_matching_is_within_person_recording_and_without_episode_reuse(self):
        rows = self.rows()
        result = discovery.match_score_pairs(rows, {"same_score_tolerance": .05, "capacity_difference_min": .2})
        self.assertEqual(result["pair_count"], 2)
        ids = [p[key] for p in result["pairs"] for key in ("episode_a", "episode_b")]
        self.assertEqual(len(set(ids)), 4)
        for row in rows[1::2]:
            row["recording_id"] = "other"
        self.assertEqual(discovery.match_score_pairs(rows, {})["pair_count"], 0)

    def test_saturation_and_overlap_do_not_create_cases(self):
        rows = self.rows()
        for row in rows:
            row["saturated_fraction"] = 1
        self.assertEqual(discovery.match_score_pairs(rows, {})["pair_count"], 0)
        rows = self.rows()
        for row in rows:
            row["start_s"], row["stop_s"] = 0, 10
        self.assertEqual(discovery.match_score_pairs(rows, {"min_time_separation_s": 0})["pair_count"], 0)

    def test_distribution_preserves_reports_and_participant_weight(self):
        rows = self.rows()
        rows.append({**rows[0], "participant_id": "other", "episode_id": "other", "score": 1.0})
        result = discovery.score_distribution(rows)
        group = next(g for g in result["groups"] if g["original_report_category"] == "report_0")
        self.assertEqual(group["participants"], 2)
        self.assertAlmostEqual(group["mean_score_participant_weighted"], (.505 + 1) / 2)

    def test_phase_surrogate_preserves_spectrum_and_shared_cross_spectrum(self):
        array = np.random.default_rng(4).normal(size=(3, 500))
        transformed = discovery.phase_surrogate(array, seed=5, shared_phases=True)
        original_spectrum = np.fft.rfft(array, axis=1)
        surrogate_spectrum = np.fft.rfft(transformed, axis=1)
        np.testing.assert_allclose(abs(original_spectrum), abs(surrogate_spectrum), atol=1e-12)
        np.testing.assert_allclose(original_spectrum[0] * original_spectrum[1].conj(),
                                   surrogate_spectrum[0] * surrogate_spectrum[1].conj(), atol=1e-10)
        self.assertFalse(np.allclose(array, transformed))


if __name__ == "__main__":
    unittest.main()
