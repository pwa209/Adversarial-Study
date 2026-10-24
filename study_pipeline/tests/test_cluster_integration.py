import importlib.util
import tempfile
import unittest
from pathlib import Path
from adversarial_study.core import read_csv

spec = importlib.util.spec_from_file_location("cluster_pilot", Path(__file__).parents[1] / "cluster/pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class DynamicsExportTests(unittest.TestCase):
    def test_metadata_rich_exclusion_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            row = {"episode_id": "episode", "participant_id": "person", "source_id": "source",
                   "cohort_id": "cohort", "recording_id": "record", "role": "development", "reason": "annotated gap"}
            pilot.write_dynamics_tables(directory, [{"episode_id": "other", "effective_coupled_var1_one_step_mse": 1.2}], [row])
            self.assertEqual(read_csv(Path(directory) / "exclusions.csv"), [row])

    def test_all_excluded_has_empty_features_and_preserves_missingness(self):
        with tempfile.TemporaryDirectory() as directory:
            row = {"episode_id": "episode", "participant_id": "person", "reason": "missing qualified channel"}
            pilot.write_dynamics_tables(directory, [], [row])
            self.assertEqual(read_csv(Path(directory) / "episode_features.csv", ("episode_id", "participant_id")), [])
            self.assertEqual(read_csv(Path(directory) / "exclusions.csv"), [row])


if __name__ == "__main__": unittest.main()
