"""Original-scope graph and source-QC regression checks; no empirical findings."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from adversarial_study import signals

PIPELINE = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("original_plan", PIPELINE / "cluster/make_plan.py")
plan_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan_module)


class OriginalProgrammeTests(unittest.TestCase):
    def test_original_loop_and_public_branches_are_executable(self):
        manifest = json.loads((PIPELINE / "configs/sources.public.json").read_text())
        extensions = json.loads((PIPELINE / "configs/sources.extensions.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "SOURCE_MANIFEST.json").write_text('{"files":{}}')
            plan = plan_module.make_plan(source, "fixture", manifest, extensions)
        stages = {s["id"]: s for s in plan["stages"]}
        for name in ["biological_bank", "simulation_score", "joint_mechanistic_fit", "joint_adversarial_search", "joint_diagnostic_freeze", "convert_tactile_cohort", "convert_dream_cohort", "behavioural_breadth", "behavioural_relevance", "centaur_human_finetune", "base_human_finetune", "clinical_human_finetune", "generative_human_adaptation", "human_finetuned_behavioural_comparison"]:
            self.assertIn("command", stages[name])
            self.assertNotIn("external_input", stages[name])
        self.assertIn("simulation_score", stages["joint_mechanistic_fit"]["depends_on"])
        self.assertEqual(stages["joint_adversarial_search"]["depends_on"], ["joint_mechanistic_fit"])
        self.assertEqual(stages["joint_diagnostic_freeze"]["depends_on"], ["joint_adversarial_search"])
        self.assertNotIn("centaur_native", stages)
        self.assertEqual(stages["centaur_human_finetune"]["gpus"], 2)
        self.assertEqual(stages["base_human_finetune"]["gpus"], 2)
        self.assertTrue(plan["cross_system_policy"]["human_finetuning_before_comparison"])
        for name in ("human_neural_state_space", "adapted_clinical_geometry", "human_representation_bridge",
                     "manifold_mechanistic_bridge", "manifold_diagnostic_freeze"):
            self.assertIn("command", stages[name])
        self.assertIn("clinical_human_finetune", stages["human_neural_state_space"]["depends_on"])
        self.assertEqual(stages["adapted_clinical_geometry"]["required_human_training_stages"], ["clinical_human_finetune"])
        self.assertIn("human_representation_bridge", stages["manifold_mechanistic_bridge"]["depends_on"])
        mechanism_command = stages["manifold_mechanistic_bridge"]["command"]
        self.assertIn("--states", mechanism_command)
        self.assertIn("--representations", mechanism_command)
        self.assertEqual(stages["independent_neural_replication"]["depends_on"], ["manifold_diagnostic_freeze"])
        self.assertIn("manifold_diagnostic_freeze", stages["new_randomized_human_experiment"]["depends_on"])
        self.assertIn("manifold_mechanistic_bridge", stages["cross_system_training_audit"]["depends_on"])
        for name in ["centaur_human_evaluation", "base_human_evaluation"]:
            self.assertEqual(set(stages[name]["depends_on"]), {"centaur_human_finetune", "base_human_finetune"})
        self.assertEqual({s["id"] for s in plan["stages"] if s.get("external_input")}, {"independent_neural_replication", "new_randomized_human_experiment"})
        self.assertFalse(plan["result_direction_gates"])
        self.assertTrue(plan["transfer"]["continue_independent_groups"])
        # Every executable prerequisite resolves without a cyclic invented gate.
        resolved = set()
        while len(resolved) < len(stages):
            ready = {name for name, s in stages.items() if name not in resolved and set(s.get("depends_on", [])) <= resolved}
            self.assertTrue(ready)
            resolved.update(ready)

    def test_dc_offset_screen_preserves_native_signal_and_rejects_real_artifact(self):
        t = np.arange(5000) / 500
        clean = 10 * np.sin(2 * np.pi * 10 * t) + 20000
        flat = np.full(5000, 20000.)
        artifact = clean.copy(); artifact[200] += 1000
        window = np.stack([clean, artifact, flat])
        original = window.copy()
        self.assertFalse(signals.quality_metrics(clean[None], 500)["qualified"])
        self.assertTrue(signals.quality_metrics(clean[None], 500, amplitude_reference="channel_median")["qualified"])
        indices, audit = signals.channel_quality_selection(window, 500, ["clean", "artifact", "flat"], {"enabled": True, "amplitude_reference": "channel_median"})
        self.assertEqual(indices, [0])
        self.assertEqual(len(audit), 3)
        np.testing.assert_array_equal(window, original)


if __name__ == "__main__": unittest.main()
