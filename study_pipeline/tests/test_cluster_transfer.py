"""Transfer-only fixtures: no network, scheduler, human EEG, or scientific fit."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/"cluster"/"transfer.py"
SPEC=importlib.util.spec_from_file_location("bounded_study_transfer",SCRIPT)
transfer=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


class Response(io.BytesIO):
    status=200
    def __init__(self,payload):
        super().__init__(payload)
        self.headers={"Content-Length":str(len(payload)),"ETag":'"fixture"'}


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base=Path(self.temporary.name).resolve()
        self.project=self.base/"project"
        self.source=self.project/"source"
        self.run=self.project/"runs"/"v2"
        self.scratch=self.base/"scratch"
        for path in (self.source/"study_pipeline/configs",self.source/"source_records/toker_v2",self.run/"receipts",self.scratch):
            path.mkdir(parents=True)
        self.payloads={}
        native_names=sorted(transfer.figshare.REQUIRED_SOURCE|transfer.figshare.DETECTOR_FILES|
                            {transfer.figshare.BASELINE,"LICENSE","license.txt","doc_config1.conf"})
        items=[]
        for i,name in enumerate(native_names,start=1000):
            payload=("native disposed fixture:"+name).encode()
            url=f"https://ndownloader.figshare.com/files/{i}"
            self.payloads[url]=payload
            items.append({"id":i,"name":name,"size":len(payload),"computed_md5":hashlib.md5(payload).hexdigest(),
                          "download_url":url,"is_link_only":False})
        self.native_path=self.source/"source_records/toker_v2/figshare_v2_metadata.json"
        self.native_path.write_text(json.dumps({"id":30906557,"doi":transfer.figshare.DOI,"files":items}))
        public=[]
        for i in range(2):
            payload=("public disposed fixture:"+str(i)).encode()
            url=f"https://example.org/public/{i}"
            self.payloads[url]=payload
            public.append({"resource_id":f"object{i}","source_id":"monash_wanderim","relative_path":f"monash/object{i}.bin",
                           "url":url,"expected_bytes":len(payload),"checksum":{"algorithm":"sha256","value":hashlib.sha256(payload).hexdigest()}})
        self.public_path=self.source/"study_pipeline/configs/sources.public.json"
        self.public_path.write_text(json.dumps({"resources":public}))
        source_manifest={"files":{p.relative_to(self.source).as_posix():transfer.digest(p) for p in (self.native_path,self.public_path)}}
        (self.source/"SOURCE_MANIFEST.json").write_text(json.dumps(source_manifest))
        self.plan={"schema_version":1,"source":str(self.source),"project":str(self.project),"scratch":str(self.scratch),
                   "run_root":str(self.run),"source_manifest_sha256":transfer.digest(self.source/"SOURCE_MANIFEST.json"),
                   "storage_allowance_bytes":1_000_000,"entry_allowance":100,
                   "transfer":{"max_rate_bytes_per_second":8*1024*1024,"groups":[
                       {"id":"native_sources","kind":"figshare_v2","metadata":"{source}/source_records/toker_v2/figshare_v2_metadata.json",
                        "destination":str(self.scratch/"native"),"include_detector":True,"include_config_bank":True},
                       {"id":"acquire_monash_pilot","kind":"public_manifest","manifest":"{pipeline}/configs/sources.public.json",
                        "destination":str(self.scratch/"data"),"resource_ids":["object0"]}]}}
        self.plan_path=self.run/"execution_plan.json"
        self.save_plan()

    def save_plan(self): self.plan_path.write_text(json.dumps(self.plan))

    def fake_open(self,request,**kwargs): return Response(self.payloads[request.full_url])

    def run_fixture(self,**kwargs):
        with patch.object(transfer,"login_guard"),patch.object(transfer.acq.urllib.request,"urlopen",side_effect=self.fake_open),\
             patch.object(transfer.acq.time,"sleep",return_value=None):
            return transfer.run_transfer(self.plan_path,**kwargs)

    def test_finite_groups_native_layout_and_scheduled_verify(self):
        result=self.run_fixture()
        self.assertEqual(result["completed_groups"],["native_sources","acquire_monash_pilot"])
        native=transfer.load(self.scratch/"native/SOURCE_MANIFEST.json")
        self.assertEqual(native["status"],"pinned_payload_verified")
        self.assertEqual(native["version"],2)
        self.assertIn("src/main.cpp",[f["path"] for f in native["files"]])
        plan,pins,groups=transfer.load_context(self.plan_path)
        marker=transfer.validate_marker(plan,pins,groups[1])
        self.assertEqual(marker["status"],"complete_verified")
        with patch.dict(os.environ,{"SLURM_JOB_ID":"123"}),patch.object(transfer.acq.urllib.request,"urlopen",side_effect=AssertionError("network forbidden")):
            verified=transfer.verify_group(self.plan_path,"acquire_monash_pilot")
        self.assertFalse(verified["network_access"])
        self.assertEqual(verified["resource_count"],1)

    def test_complete_marker_reuse_does_not_download_again(self):
        self.run_fixture()
        before=(self.run/"transfer/native_sources.complete.json").read_bytes()
        with patch.object(transfer,"login_guard"),patch.object(transfer.acq.urllib.request,"urlopen",side_effect=AssertionError("no second download")):
            result=transfer.run_transfer(self.plan_path)
        self.assertEqual(result["status"],"complete_verified")
        self.assertEqual(before,(self.run/"transfer/native_sources.complete.json").read_bytes())

    def test_modified_cache_fails_scheduled_hash_check(self):
        self.run_fixture(group_ids=["acquire_monash_pilot"])
        target=self.scratch/"data/monash/object0.bin"
        value=bytearray(target.read_bytes());value[0]^=1;target.write_bytes(value)
        with patch.dict(os.environ,{"SLURM_JOB_ID":"124"}),self.assertRaisesRegex(ValueError,"checksum"):
            transfer.verify_group(self.plan_path,"acquire_monash_pilot")

    def test_receipt_marker_and_plan_pins_cannot_be_changed(self):
        self.run_fixture(group_ids=["acquire_monash_pilot"])
        receipt=self.run/"transfer/acquire_monash_pilot.receipt.json"
        receipt.write_text(receipt.read_text()+" ")
        with patch.dict(os.environ,{"SLURM_JOB_ID":"125"}),self.assertRaisesRegex(ValueError,"receipt changed"):
            transfer.verify_group(self.plan_path,"acquire_monash_pilot")
        self.plan["extra_annotation"]="changed plan"
        self.save_plan()
        plan,pins,groups=transfer.load_context(self.plan_path)
        with self.assertRaisesRegex(ValueError,"marker differs"):
            transfer.validate_marker(plan,pins,groups[1])

    def test_marker_resource_list_tampering_is_rejected(self):
        self.run_fixture(group_ids=["acquire_monash_pilot"])
        path=self.run/"transfer/acquire_monash_pilot.complete.json"
        marker=transfer.load(path)
        marker["resources"]=[]
        path.write_text(json.dumps(marker))
        plan,pins,groups=transfer.load_context(self.plan_path)
        with self.assertRaisesRegex(ValueError,"resource list"):
            transfer.validate_marker(plan,pins,groups[1])

    def test_concurrent_status_publication_has_unique_temporaries(self):
        path=self.run/"transfer/status.json"
        errors=[]
        barrier=threading.Barrier(4)
        def write(index):
            try:
                barrier.wait()
                for iteration in range(5): transfer.save(path,{"thread":index,"iteration":iteration})
            except BaseException as error: errors.append(error)
        threads=[threading.Thread(target=write,args=(i,)) for i in range(4)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertFalse(errors)
        self.assertEqual(transfer.load(path)["iteration"],4)
        self.assertFalse(list(path.parent.glob("status.json.*.tmp")))

    def test_source_metadata_tampering_blocks_before_any_network(self):
        self.public_path.write_text(self.public_path.read_text()+" ")
        with patch.object(transfer.acq.urllib.request,"urlopen",side_effect=AssertionError("network forbidden")),self.assertRaisesRegex(ValueError,"source pin"):
            transfer.load_context(self.plan_path)

    def add_cohort(self):
        self.plan["transfer"]["groups"].append({"id":"acquire_monash_cohort","kind":"public_manifest",
            "manifest":"{pipeline}/configs/sources.public.json","destination":str(self.scratch/"data"),
            "source_ids":["monash_wanderim"],"requires_stage":"qualify_detector"})
        self.save_plan()

    def qualification(self,status="completed",exit_code=0,pins_override=None):
        plan,pins,_=transfer.load_context(self.plan_path)
        pins.update(pins_override or {})
        (self.run/"receipts/qualify_detector.json").write_text(json.dumps({"stage":"qualify_detector",**pins,
            "status":status,"exit_code":exit_code,"job_id":"777"}))
        (self.run/"execution_state.json").write_text(json.dumps({"plan_sha256":pins["plan_sha256"],
            "stages":{"qualify_detector":{"job_id":"777","state":"COMPLETED"}}}))

    def test_full_cohort_wait_uses_only_scheduled_integrity_not_result_direction(self):
        self.add_cohort()
        self.qualification()
        result=self.run_fixture(group_ids=["acquire_monash_cohort"])
        self.assertEqual(result["status"],"complete_verified")
        self.assertTrue((self.scratch/"data/monash/object1.bin").exists())

    def test_failed_or_mismatched_qualification_blocks_cohort(self):
        self.add_cohort()
        self.qualification(status="failed",exit_code=2)
        with self.assertRaisesRegex(ValueError,"qualification failed"):
            self.run_fixture(group_ids=["acquire_monash_cohort"])
        failures=list((self.run/"transfer").glob("acquire_monash_cohort.failed.*.json"))
        self.assertEqual(len(failures),1)
        self.assertEqual(transfer.load(failures[0])["status"],"failed")
        self.qualification(pins_override={"source_manifest_sha256":"0"*64})
        with self.assertRaisesRegex(ValueError,"source pins"):
            self.run_fixture(group_ids=["acquire_monash_cohort"])
        self.assertEqual(len(list((self.run/"transfer").glob("acquire_monash_cohort.failed.*.json"))),2)

    def test_budget_path_duplicate_group_and_result_gate_validation(self):
        original=json.loads(json.dumps(self.plan))
        mutations=[lambda p:p["transfer"].update(max_bytes=1),
                   lambda p:p["transfer"].update(max_rate_bytes_per_second=9*1024*1024),
                   lambda p:p["transfer"]["groups"][0].update(destination=str(self.base/"outside")),
                   lambda p:p["transfer"]["groups"].append(p["transfer"]["groups"][0]),
                   lambda p:p["transfer"]["groups"][1].update(requires_stage="positive_result")]
        for mutate in mutations:
            self.plan=json.loads(json.dumps(original));mutate(self.plan);self.save_plan()
            with self.subTest(plan=self.plan),self.assertRaises(ValueError): transfer.load_context(self.plan_path)

    def test_verification_requires_allocation_and_wait_is_bounded(self):
        with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(RuntimeError,"Slurm"):
            transfer.verify_group(self.plan_path,"native_sources")
        with self.assertRaises(TimeoutError):
            transfer.wait_group(self.plan_path,"native_sources",timeout_s=.001)

    def test_shared_rate_limits_checksum_and_network_without_idle_credit(self):
        limiter=transfer.SharedReadLimit(100,max_network_bytes=10)
        with patch.object(transfer.time,"monotonic",return_value=100),patch.object(transfer.time,"sleep") as slept:
            limiter.next_time=50
            limiter.consume(5,"network")
            limiter.consume(10,"checksum")
        self.assertEqual(limiter.network_bytes,5)
        self.assertEqual(limiter.checksum_bytes,10)
        self.assertGreaterEqual(slept.call_args_list[0].args[0],.049)
        self.assertGreaterEqual(slept.call_args_list[1].args[0],.149)
        with self.assertRaisesRegex(ValueError,"budget"):
            limiter.consume(6,"network")

    def test_login_guard_rejects_wrong_account_host_and_compute(self):
        with patch.object(transfer.getpass,"getuser",return_value="other"),self.assertRaises(RuntimeError): transfer.login_guard(self.plan)
        with patch.object(transfer.getpass,"getuser",return_value="pwa209"),patch.object(transfer.socket,"gethostname",return_value="ng0001"),self.assertRaises(RuntimeError): transfer.login_guard(self.plan)
        with patch.object(transfer.getpass,"getuser",return_value="pwa209"),patch.object(transfer.socket,"gethostname",return_value="narval1"),patch.dict(os.environ,{"SLURM_JOB_ID":"1"}),self.assertRaises(RuntimeError): transfer.login_guard(self.plan)


if __name__=="__main__":unittest.main()
