"""Transaction tests; fake scheduler observations never authorize real actions."""
from contextlib import nullcontext
import copy
import hashlib
from pathlib import Path
import re
import unittest

from live_parent_submit import submit_live_parent


class Ops:
    CONTROL = Path("/scratch/pwa209/adversarial-study/runs/existing-data-strengthening-20261024")
    POLICY = {"qualify": {"memory": "8G"}, "analysis": {"memory": "64G"}}
    def __init__(self):
        root = "/scratch/pwa209/adversarial-study/maintenance/existing-data-strengthening-20261024"
        self.held = {"JobId": "6944368", "JobName": "acs-existing-data-qualify", "UserId": "pwa209(3164768)",
                     "Account": "def-ptewarie_cpu", "NumCPUs": "4", "NumTasks": "1", "CPUs/Task": "4", "Requeue": "0",
                     "TimeLimit": "02:00:00", "Command": root + "/worker.sh", "WorkDir": root,
                     "Comment": "acs:existing-data-strengthening-20261024:qualify", "StdOut": str(self.CONTROL / "qualify.out"),
                     "NumNodes": "1", "MinMemoryNode": "8G", "ReqTRES": "cpu=4,mem=8G,node=1,billing=4",
                     "Dependency": "(null)", "JobState": "PENDING", "Reason": "JobHeldUser", "AllocTRES": "(null)", "NodeList": "(null)", "Restarts": "0"}
        self.live = dict(self.held, Reason="Priority")
        self.slots, self.mutations, self.verified = {}, [], []
        self.actual = None
        self.qualified = False
        self.acceptance = {"returncode": 0, "stdout": "6945000\n", "stderr": ""}
        self.release = {"returncode": 0, "stdout": "", "stderr": ""}
    @staticmethod
    def require(ok, message):
        if not ok: raise ValueError(message)
    @staticmethod
    def text(row): return " ".join(k + "=" + v for k, v in row.items())
    @staticmethod
    def fields(raw):
        return {m.group(1): m.group(2).strip() for m in re.finditer(r"([^\s=]+)=(.*?)(?=\s+[^\s=]+=|$)", raw)}
    @staticmethod
    def digest(raw): return hashlib.sha256(raw).hexdigest()
    def sha(self, path): return self.digest(repr(self.slots[path.name]).encode())
    def verify_packet(self, pin, full_upstream=False): self.verified.append(pin)
    def locked(self): return nullcontext()
    def state(self, create=False): self.require(create is False, "No new qualifier state")
    def maybe(self, path): return self.slots.get(path.name)
    def custody(self, phase, pin):
        if phase == "analysis": return None
        return {"job_id": "6944368", "values": {"held": {"raw": self.text(self.held)}}}
    def accounting(self, job, phase): return self.actual
    def observe(self, argv): return self.text(self.live) if argv[3] == "6944368" else "held-analysis"
    def qualification_proof(self, pin, job):
        self.require(self.qualified, "Original completed qualification not available")
        return {"job_id": job, "original_qualification_verified": True}
    def queue_guard(self, allowed): return {"allowed": allowed, "requested_study_cpus": 20, "active_study_cpus": 0}
    def submit_argv(self, phase, pin, dep): return ["sbatch", "--hold", "--dependency=" + dep, phase, pin]
    def write_slot(self, path, value):
        self.require(path.name not in self.slots, "Never rewrite slot")
        self.slots[path.name] = value
    def mutate_once(self, argv):
        self.mutations.append(argv)
        return copy.deepcopy(self.acceptance if argv[0] == "sbatch" else self.release)
    def validate_allocation(self, raw, job, phase, dependency=None, held=False, fulfilled=False):
        if phase == "qualify":
            self.require(self.live["JobState"] == "RUNNING" and self.live.get("AllocTRES") == "cpu=4,mem=8G,node=1,billing=4", "Original running allocation verifier rejected")
        else:
            self.require(held and dependency == "afterok:6944368" and job == "6945000", "Unchanged analysis held allocation required")
        return {"validated": True}
    def utc(self): return "2026-10-24T00:00:00+00:00"


class LiveParentTests(unittest.TestCase):
    def test_real_current_pending_can_supply_missing_accounting_without_invention(self):
        ops = Ops()
        self.assertEqual(submit_live_parent(ops, "pin", "6944368"), "6945000")
        proof = ops.slots["analysis.intent.json"]["live_parent_evidence"]
        self.assertIsNone(proof["accounting_observation"])
        self.assertFalse(proof["accounting_row_synthesized"])
        self.assertEqual(proof["state"], "PENDING")
        self.assertEqual(len(ops.mutations), 2)
        self.assertIn("--dependency=afterok:6944368", ops.mutations[0])

    def test_running_uses_original_allocation_validator(self):
        ops = Ops(); ops.live.update(JobState="RUNNING", Reason="None", AllocTRES="cpu=4,mem=8G,node=1,billing=4", NodeList="nc1001")
        self.assertEqual(submit_live_parent(ops, "pin", "6944368"), "6945000")

    def test_literal_none_is_valid_pending_reason_before_backfill(self):
        ops = Ops(); ops.live["Reason"] = "None"
        self.assertEqual(submit_live_parent(ops, "pin", "6944368"), "6945000")
        self.assertEqual(ops.slots["analysis.intent.json"]["live_parent_evidence"]["state"], "PENDING")

    def test_identity_and_resources_are_verified_before_intent(self):
        for field, changed in (("Command", "/wrong"), ("WorkDir", "/wrong"), ("Account", "wrong"),
                               ("JobId", "77"), ("NumCPUs", "8"), ("ReqTRES", "cpu=4,mem=8G,node=1,gres/gpu=1"),
                               ("MinMemoryNode", "16G"), ("Comment", "wrong"), ("StdOut", "/wrong")):
            ops = Ops(); ops.live[field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError): submit_live_parent(ops, "pin", "6944368")
            self.assertFalse(ops.slots); self.assertFalse(ops.mutations)

    def test_failed_cancelled_and_unknown_state_rejected(self):
        for state in ("FAILED", "CANCELLED", "TIMEOUT", "UNKNOWN", "COMPLETING", "CONFIGURING"):
            ops = Ops(); ops.live["JobState"] = state
            with self.subTest(state=state), self.assertRaises(ValueError): submit_live_parent(ops, "pin", "6944368")
            self.assertFalse(ops.slots); self.assertFalse(ops.mutations)

    def test_pending_held_or_unknown_allocation_rejected(self):
        for field, changed in (("Reason", "JobHeldUser"), ("Reason", "Unknown"), ("Reason", ""),
                               ("AllocTRES", "cpu=4"), ("NodeList", "nc1001"), ("Restarts", "1")):
            ops = Ops(); ops.live[field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError): submit_live_parent(ops, "pin", "6944368")
            self.assertFalse(ops.slots)

    def test_missing_pending_reason_rejected_before_intent(self):
        ops = Ops(); del ops.live["Reason"]
        with self.assertRaisesRegex(ValueError, "reason is unknown"): submit_live_parent(ops, "pin", "6944368")
        self.assertFalse(ops.slots); self.assertFalse(ops.mutations)

    def test_completed_without_original_qualification_proof_cannot_fallback(self):
        ops = Ops(); ops.live["JobState"] = "COMPLETED"
        with self.assertRaisesRegex(ValueError, "Original completed"): submit_live_parent(ops, "pin", "6944368")
        self.assertFalse(ops.slots); self.assertFalse(ops.mutations)

    def test_accounted_completion_requires_original_proof_and_marks_fulfilled(self):
        ops = Ops(); ops.actual = {"row": ["", "", "", "", "", "COMPLETED"]}; ops.qualified = True
        self.assertEqual(submit_live_parent(ops, "pin", "6944368"), "6945000")
        self.assertTrue(ops.slots["analysis.held.json"]["dependency_already_qualified"])

    def test_failed_accounting_cannot_be_overridden_by_pending_live(self):
        ops = Ops(); ops.actual = {"row": ["", "", "", "", "", "FAILED"]}
        with self.assertRaises(ValueError): submit_live_parent(ops, "pin", "6944368")
        self.assertFalse(ops.slots)

    def test_ambiguous_submission_preserved_and_never_retried(self):
        ops = Ops(); ops.acceptance = {"returncode": None, "ambiguous": True}
        with self.assertRaisesRegex(ValueError, "Ambiguous submission"): submit_live_parent(ops, "pin", "6944368")
        self.assertEqual(len(ops.mutations), 1)
        with self.assertRaisesRegex(ValueError, "never replay"): submit_live_parent(ops, "pin", "6944368")
        self.assertEqual(len(ops.mutations), 1)

    def test_ambiguous_release_preserved_and_never_retried(self):
        ops = Ops(); ops.release = {"returncode": None, "ambiguous": True}
        with self.assertRaisesRegex(ValueError, "Ambiguous release"): submit_live_parent(ops, "pin", "6944368")
        self.assertIn("analysis.released.json", ops.slots)
        with self.assertRaisesRegex(ValueError, "never replay"): submit_live_parent(ops, "pin", "6944368")
        self.assertEqual(len(ops.mutations), 2)

    def test_wrong_qualifier_job_does_not_create_intent(self):
        ops = Ops()
        with self.assertRaises(ValueError): submit_live_parent(ops, "pin", "77")
        self.assertFalse(ops.slots); self.assertFalse(ops.mutations)


if __name__ == "__main__": unittest.main()
