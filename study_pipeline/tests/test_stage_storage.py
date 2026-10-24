"""Bounded quota admission tests; no Lustre RPC or remote operation is used."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location("quota_stage_worker",Path(__file__).parents[1]/"cluster/stage_worker.py")
worker=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(worker)

HEADER="Disk quotas for usr pwa209 (uid 3164768):\n Filesystem kbytes quota limit grace files quota limit grace\n"
ACTUAL=HEADER+" /scratch 126144812 20000000000 20000000000 - 77864 1000000 1000000 -\n"


class QuotaParsing(unittest.TestCase):
    def test_actual_narval_summary_uses_kib_and_inodes_not_exact_study_tree(self):
        parsed=worker.parse_personal_quota(ACTUAL,"pwa209","/scratch/pwa209")
        self.assertEqual(parsed["personal_allocated_bytes"],126144812*1024)
        self.assertEqual(parsed["personal_inode_count"],77864)

    def test_wrapped_long_personal_path_and_over_quota_marks(self):
        text=HEADER+"/scratch/pwa209\n 100* 200 300 6days 44* 100 200 -\n"
        parsed=worker.parse_personal_quota(text,"pwa209","/scratch/pwa209")
        self.assertEqual(parsed["personal_allocated_bytes"],102400)
        self.assertEqual(parsed["personal_inode_count"],44)
        self.assertTrue(parsed["quota_usage_marked_over_limit"])

    def test_no_space_uid_is_parsed_as_actual_probe_reports(self):
        worker.parse_personal_quota(ACTUAL.replace("uid 3164768","uid3164768"),"pwa209","/scratch/pwa209")

    def test_wrong_user_unknown_counts_human_units_wrong_header_and_duplicate_rows_fail(self):
        cases=[ACTUAL.replace("pwa209","other"),ACTUAL.replace("126144812","-"),ACTUAL.replace("77864","78K"),
               ACTUAL.replace("kbytes","bytes"),ACTUAL.replace("126144812","-100"),ACTUAL.replace("77864","nan"),
               ACTUAL+" /scratch/pwa209 100 200 300 - 1 100 200 -\n",ACTUAL.replace("/scratch ","/project "),
               ACTUAL.replace("126144812",str(2**63)),HEADER+"/scratch\n"]
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):worker.parse_personal_quota(text,"pwa209","/scratch/pwa209")


class FakeQuotaProcess:
    pid=12345
    def __init__(self,command,*,stdin,stdout,stderr,start_new_session,body=ACTUAL,errors="",exit_code=0,timeout=False):
        self.command=command;self.timeout=timeout;self.exit_code=exit_code;self.killed=False;self.wait_calls=[]
        stdout.write(body.encode());stdout.flush();stderr.write(errors.encode());stderr.flush()
    def wait(self,timeout):
        self.wait_calls.append(timeout)
        if self.timeout:raise subprocess.TimeoutExpired(self.command,timeout)
        return self.exit_code
    def kill(self):self.killed=True


class BoundedQuotaAdmission(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{"SLURM_JOB_ID":"44"});self.env.start();self.addCleanup(self.env.stop)
        self.user=patch.object(worker.getpass,"getuser",return_value="pwa209");self.user.start();self.addCleanup(self.user.stop)

    def fake(self,**options):
        self.processes=[]
        def make(*args,**kwargs):
            process=FakeQuotaProcess(*args,**kwargs,**options);self.processes.append(process);return process
        return patch.object(worker.subprocess,"Popen",side_effect=make)

    def test_success_uses_plain_lfs_command_and_explicit_personal_scope(self):
        with self.fake():snapshot=worker.storage_snapshot("/scratch/pwa209/adversarial-study")
        self.assertEqual(snapshot["status"],"VERIFIED_PERSONAL_QUOTA")
        self.assertEqual(snapshot["scope"],"personal_scratch_all_projects_not_exact_own_study")
        self.assertEqual(snapshot["command"],["lfs","quota","-u","pwa209","/scratch"])
        self.assertEqual(self.processes[0].wait_calls,[30])
        self.assertNotIn("entries",snapshot)
        self.assertNotIn("file_bytes",snapshot)
        worker.guard_storage(snapshot,{"entry_allowance":150000,"storage_allowance_bytes":10**12})

    def test_timeout_is_unknown_kills_only_quota_helper_and_never_waits_twice(self):
        with self.fake(timeout=True):snapshot=worker.storage_snapshot("/scratch/pwa209/adversarial-study")
        self.assertEqual(snapshot["status"],"UNKNOWN")
        self.assertIsNone(snapshot["personal_allocated_bytes"])
        self.assertIsNone(snapshot["personal_inode_count"])
        self.assertTrue(self.processes[0].killed)
        self.assertEqual(self.processes[0].wait_calls,[30])
        with self.assertRaises(ValueError):worker.guard_storage(snapshot,{"entry_allowance":150000,"storage_allowance_bytes":10**12})

    def test_command_failure_warning_and_malformed_output_never_become_zero(self):
        for options in ({"exit_code":4},{"errors":"quota lookup incomplete"},{"body":"unknown quota"},{"body":ACTUAL.replace("77864","-")}):
            with self.subTest(options=options),self.fake(**options):
                snapshot=worker.storage_snapshot("/scratch/pwa209/adversarial-study")
                self.assertEqual(snapshot["status"],"UNKNOWN")
                self.assertIsNone(snapshot["personal_inode_count"])

    def test_non_slurm_wrong_user_root_and_traversal_never_launch_quota(self):
        with patch.object(worker.subprocess,"Popen") as process:
            for root in ("/scratch/other/study","/scratch/pwa209/../other/study","relative"):
                self.assertEqual(worker.storage_snapshot(root)["status"],"UNKNOWN")
            with patch.dict(os.environ,{"SLURM_JOB_ID":""}):
                self.assertEqual(worker.storage_snapshot("/scratch/pwa209/study")["status"],"UNKNOWN")
            process.assert_not_called()

    def test_guard_rejects_exact_cap_and_near_cap_on_either_metric(self):
        plan={"entry_allowance":150000,"storage_allowance_bytes":10**12}
        snapshot={"status":"VERIFIED_PERSONAL_QUOTA","personal_inode_count":149999,"personal_allocated_bytes":10**12-1}
        worker.guard_storage(snapshot,plan)
        for changed in ({"personal_inode_count":150000},{"personal_allocated_bytes":10**12}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):worker.guard_storage({**snapshot,**changed},plan)


class ActualExitPreservation(unittest.TestCase):
    def test_unknown_after_execution_preserves_actual_exit_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"source";source.mkdir();(source/"study_pipeline").mkdir()
            (source/"SOURCE_MANIFEST.json").write_text(json.dumps({"files":{}}))
            run=root/"run";run.mkdir();plan_path=root/"plan.json"
            plan={"source":str(source),"source_manifest_sha256":worker.digest(source/"SOURCE_MANIFEST.json"),"project":str(root),
                  "scratch":"/scratch/pwa209/adversarial-study","run_root":str(run),"entry_allowance":150000,"storage_allowance_bytes":10**12,
                  "stages":[{"id":"fixture","command":["UNIT_FIXTURE_ONLY"]}]}
            plan_path.write_text(json.dumps(plan))
            (run/"execution_state.json").write_text(json.dumps({"plan_sha256":worker.digest(plan_path),"stages":{"fixture":{"job_id":"44","state":"ACCEPTED"}}}))
            before={"status":"VERIFIED_PERSONAL_QUOTA","personal_inode_count":77864,"personal_allocated_bytes":126144812*1024}
            after={"status":"UNKNOWN","personal_inode_count":None,"personal_allocated_bytes":None,"error":"bounded timeout"}
            resource=SimpleNamespace(RUSAGE_CHILDREN=-1,getrusage=lambda which:SimpleNamespace(ru_maxrss=100))
            with patch.dict(os.environ,{"SLURM_JOB_ID":"44"}),patch.dict(sys.modules,{"resource":resource}),\
                 patch.object(worker,"storage_snapshot",side_effect=[before,after]),\
                 patch.object(worker.platform,"node",return_value="UNIT_FIXTURE_NODE"),\
                 patch.object(worker.subprocess,"run",return_value=SimpleNamespace(returncode=0)):
                self.assertEqual(worker.run(plan_path,"fixture"),0)
            receipt=json.loads((run/"receipts/fixture.json").read_text())
            self.assertEqual(receipt["status"],"completed")
            self.assertEqual(receipt["exit_code"],0)
            self.assertEqual(receipt["storage_after"]["status"],"UNKNOWN")
            self.assertIn("actual execution",receipt["storage_audit_limitation"])

    def test_unknown_before_execution_fails_closed_without_scientific_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"source";source.mkdir();(source/"study_pipeline").mkdir()
            (source/"SOURCE_MANIFEST.json").write_text(json.dumps({"files":{}}))
            run=root/"run";run.mkdir();plan_path=root/"plan.json"
            plan={"source":str(source),"source_manifest_sha256":worker.digest(source/"SOURCE_MANIFEST.json"),"project":str(root),
                  "scratch":"/scratch/pwa209/adversarial-study","run_root":str(run),"entry_allowance":150000,"storage_allowance_bytes":10**12,
                  "stages":[{"id":"fixture","command":["UNIT_FIXTURE_ONLY"]}]}
            plan_path.write_text(json.dumps(plan))
            unknown={"status":"UNKNOWN","personal_inode_count":None,"personal_allocated_bytes":None}
            resource=SimpleNamespace(RUSAGE_CHILDREN=-1,getrusage=lambda which:SimpleNamespace(ru_maxrss=100))
            with patch.dict(os.environ,{"SLURM_JOB_ID":"44"}),patch.dict(sys.modules,{"resource":resource}),\
                 patch.object(worker.platform,"node",return_value="UNIT_FIXTURE_NODE"),\
                 patch.object(worker,"storage_snapshot",return_value=unknown),patch.object(worker.subprocess,"run") as child:
                self.assertEqual(worker.run(plan_path,"fixture"),2)
                child.assert_not_called()
            receipt=json.loads((run/"receipts/fixture.json").read_text())
            self.assertEqual(receipt["status"],"failed")
            self.assertIn("UNKNOWN",receipt["error"])


if __name__=="__main__":unittest.main()
