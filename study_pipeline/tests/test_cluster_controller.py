"""Scheduler fault tests: accepted jobs are never blindly submitted again."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("cluster_controller", Path(__file__).parents[1] / "cluster/controller.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "receipts").mkdir()
        self.plan = {"source": str(self.root), "run_root": str(self.root), "account": "test", "max_allocated_cpus": 8,
                     "controller_cpus": 1, "stages": []}
        (self.root / "SOURCE_MANIFEST.json").write_text('{"files":{}}')
        self.plan["source_manifest_sha256"] = hashlib.sha256((self.root / "SOURCE_MANIFEST.json").read_bytes()).hexdigest()

    def tearDown(self): self.temp.cleanup()

    def stage(self, name, cpus=4, parents=()):
        return {"id": name, "cpus": cpus, "depends_on": list(parents), "command": ["python", "x.py"], "memory": "4G", "walltime": "01:00:00"}

    def state(self, statuses):
        self.path = self.root / "plan.json"
        self.path.write_text(json.dumps(self.plan))
        return {"plan_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(), "stages": statuses}

    def test_running_receipt_does_not_release_resources_or_dependency(self):
        self.plan["stages"] = [self.stage("one"), self.stage("two", parents=["one"])]
        state = self.state({"one": {"state": "ACCEPTED", "job_id": "42"}, "two": {"state": "WAITING"}})
        (self.root / "receipts/one.json").write_text(json.dumps({"job_id": "42", "plan_sha256": state["plan_sha256"], "exit_code": 0}))
        with patch.object(controller, "inspect_job", return_value="RUNNING"), patch.object(controller.subprocess, "run") as call:
            controller.cycle(self.path, self.root / "state.json", state)
        call.assert_not_called()
        self.assertEqual(state["stages"]["one"]["state"], "ACCEPTED")
        self.assertEqual(state["stages"]["two"]["state"], "WAITING")

    def test_pending_allocations_consume_budget(self):
        self.plan["stages"] = [self.stage("one"), self.stage("two")]
        state = self.state({"one": {"state": "ACCEPTED", "job_id": "42"}, "two": {"state": "WAITING"}})
        with patch.object(controller, "inspect_job", return_value="PENDING"), patch.object(controller.subprocess, "run") as call:
            controller.cycle(self.path, self.root / "state.json", state)
        call.assert_not_called()

    def test_submission_intent_persisted_and_unknown_outcome_not_replayed(self):
        self.plan["stages"] = [self.stage("one")]
        state = self.state({"one": {"state": "WAITING"}})
        saved = self.root / "state.json"
        def unknown(*args, **kwargs):
            self.assertEqual(json.loads(saved.read_text())["stages"]["one"]["state"], "SUBMISSION_INTENT")
            return SimpleNamespace(returncode=0, stdout="ambiguous", stderr="")
        with patch.object(controller.subprocess, "run", side_effect=unknown):
            with self.assertRaises(RuntimeError): controller.cycle(self.path, saved, state)
        with patch.object(controller.subprocess, "run") as call:
            status = controller.cycle(self.path, saved, state)
        call.assert_not_called()
        self.assertEqual(status, "REQUIRES_RECONCILIATION")

    def test_failed_allocation_does_not_advance_dependents(self):
        self.plan["stages"] = [self.stage("one"), self.stage("two", parents=["one"])]
        state = self.state({"one": {"state": "ACCEPTED", "job_id": "42"}, "two": {"state": "WAITING"}})
        with patch.object(controller, "inspect_job", return_value="OUT_OF_MEMORY"), patch.object(controller.subprocess, "run") as call:
            status = controller.cycle(self.path, self.root / "state.json", state)
        call.assert_not_called()
        self.assertEqual(status, "FAILED_STAGES_RETAINED")
        self.assertEqual(state["stages"]["two"]["state"], "BLOCKED_UPSTREAM")

    def test_receipt_must_match_actual_allocation(self):
        self.plan["stages"] = [self.stage("one")]
        state = self.state({"one": {"state": "ACCEPTED", "job_id": "42"}})
        (self.root / "receipts/one.json").write_text(json.dumps({"job_id": "99", "plan_sha256": state["plan_sha256"], "exit_code": 0}))
        with patch.object(controller, "inspect_job", return_value="COMPLETED"):
            with self.assertRaises(RuntimeError): controller.cycle(self.path, self.root / "state.json", state)

    def test_plan_cycles_duplicate_ids_resources_and_changed_manifest(self):
        self.plan["stages"] = [self.stage("one"), self.stage("two", parents=["one"])]
        controller.validate_plan(self.plan)
        self.plan["stages"][0]["depends_on"] = ["two"]
        with self.assertRaises(ValueError): controller.validate_plan(self.plan)
        self.plan["stages"] = [self.stage("one"), self.stage("one")]
        with self.assertRaises(ValueError): controller.validate_plan(self.plan)
        self.plan["stages"] = [self.stage("one", cpus=8)]
        with self.assertRaises(ValueError): controller.validate_plan(self.plan)
        self.plan["stages"] = [self.stage("one")]
        (self.root / "SOURCE_MANIFEST.json").write_text('{"changed":true}')
        with self.assertRaises(ValueError): controller.validate_plan(self.plan)

    def test_expired_queue_record_uses_accounting(self):
        responses = [SimpleNamespace(returncode=1, stdout="", stderr="Invalid job id"),
                     SimpleNamespace(returncode=0, stdout="COMPLETED|0:0\n", stderr="")]
        with patch.object(controller.subprocess, "run", side_effect=responses):
            self.assertEqual(controller.inspect_job("42"), "COMPLETED")

    def test_transfers_wait_without_allocating_cpu_until_pinned_marker(self):
        self.plan["stages"] = [self.stage("one")]
        self.plan["stages"][0]["transfer_group"] = "one"
        state = self.state({"one": {"state": "WAITING"}})
        with patch.object(controller.subprocess, "run") as call:
            controller.cycle(self.path, self.root / "state.json", state)
        call.assert_not_called()
        self.assertEqual(state["stages"]["one"]["transfer_state"], "WAITING_BOUNDED_HTTPS_TRANSFER")
        (self.root / "transfer").mkdir()
        marker = {"status": "complete_verified", "group": "one", "plan_sha256": state["plan_sha256"],
                  "source_manifest_sha256": self.plan["source_manifest_sha256"]}
        (self.root / "transfer/one.complete.json").write_text(json.dumps(marker))
        with patch.object(controller.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="43", stderr="")) as call:
            controller.cycle(self.path, self.root / "state.json", state)
        call.assert_called_once()
        self.assertEqual(state["stages"]["one"]["state"], "ACCEPTED")

    def test_transfer_failure_is_preserved_without_submission(self):
        self.plan["stages"] = [self.stage("one")]
        self.plan["stages"][0]["transfer_group"] = "one"
        state = self.state({"one": {"state": "WAITING"}})
        (self.root / "transfer").mkdir()
        failure = {"status": "failed", "group": "one", "plan_sha256": state["plan_sha256"],
                   "source_manifest_sha256": self.plan["source_manifest_sha256"], "error": "provider unavailable"}
        (self.root / "transfer/one.failed.1.json").write_text(json.dumps(failure))
        with patch.object(controller.subprocess, "run") as call:
            self.assertEqual(controller.cycle(self.path, self.root / "state.json", state), "FAILED_STAGES_RETAINED")
        call.assert_not_called()
        self.assertEqual(state["stages"]["one"]["state"], "BLOCKED_UPSTREAM")

    def test_unrelated_transfer_failure_does_not_block_independent_group(self):
        self.plan["stages"] = [self.stage("one")]
        self.plan["stages"][0]["transfer_group"] = "one"
        state = self.state({"one": {"state": "WAITING"}})
        (self.root / "transfer").mkdir()
        (self.root / "transfer/other.failed.1.json").write_text(json.dumps({"group": "other", "plan_sha256": state["plan_sha256"], "source_manifest_sha256": self.plan["source_manifest_sha256"]}))
        with patch.object(controller.subprocess, "run") as call:
            controller.cycle(self.path, self.root / "state.json", state)
        call.assert_not_called()
        self.assertEqual(state["stages"]["one"]["state"], "WAITING")

    def test_gpu_account_gres_and_pending_gpu_budget(self):
        self.plan.update(max_allocated_gpus=2, gpu_account="test_gpu")
        first, second = self.stage("first", cpus=2), self.stage("second", cpus=2)
        for stage in (first, second): stage.update(gpus=2, gpu_type="a100", account="test_gpu")
        self.plan["stages"] = [first, second]
        controller.validate_plan(self.plan)
        state = self.state({"first": {"state": "WAITING"}, "second": {"state": "WAITING"}})
        with patch.object(controller.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="43", stderr="")) as call:
            controller.cycle(self.path, self.root / "state.json", state)
        self.assertEqual(call.call_count, 1)
        self.assertIn("--account=test_gpu", call.call_args.args[0])
        self.assertIn("--gres=gpu:a100:2", call.call_args.args[0])
        self.assertEqual(state["stages"]["second"]["state"], "WAITING")
        first["gpus"] = 3
        with self.assertRaises(ValueError): controller.validate_plan(self.plan)


if __name__ == "__main__": unittest.main()
