"""Scheduled integrity gates for the full-population extension."""
import json
import inspect
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import unittest

import audit_ops as ops

AFTER_GUARD_RESERVE_SECONDS=3800


def remaining():
    value=ops._DEADLINE-time.monotonic()
    ops.require(value>0,'Finite scheduled phase deadline exhausted')
    return value


def storage(phase,label):
    ops.require(remaining()>3660,'Insufficient actual allocation time for full original A4 census')
    guard=ops.guard_module()
    capture=ops.RUN/(phase+'.storage.'+label+'.json')
    ops.require(not capture.exists() and not capture.is_symlink(),'Existing storage capture; never replay')
    value=guard.guard(guard.SOURCE,reserve=ops.RESERVES[phase],capture=capture)
    ops.require(value.get('guard_admission_passed') is True and value.get('guard_admission_error') is None,
        'Full original A4 storage admission failed')
    return {'snapshot':value,'capture':str(capture),'sha256':ops.sha(capture)}


def tests(minimum):
    suite=unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.discover(str(ops.ROOT),pattern='test_*.py'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    value={'run':result.testsRun,'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped)
        -len(result.expectedFailures),'failed':len(result.failures),'errors':len(result.errors),
        'skipped':len(result.skipped),'expected_failures':len(result.expectedFailures),
        'unexpected_successes':len(result.unexpectedSuccesses)}
    ops.require(result.wasSuccessful() and value['passed']>=minimum and not any(value[key] for key in
        ('failed','errors','skipped','expected_failures','unexpected_successes')),'Actual scheduled qualification tests failed or skipped')
    return value


def runtime_probe(manifest):
    argv=[str(ops.PYTHON),'-B',str(ops.analysis_driver(manifest)),'--runtime-check']
    timeout=min(630,remaining()-3660)
    ops.require(timeout>0,'Insufficient phase time for actual analysis ABI probe plus full A4 guard')
    def text(value):return value.decode('utf8',errors='replace') if isinstance(value,bytes) else (value or '')
    try:
        child=subprocess.run(argv,text=True,capture_output=True,stdin=subprocess.DEVNULL,timeout=timeout,check=False)
        code=child.returncode;stdout=child.stdout;stderr=child.stderr;error=None
    except Exception as failure:
        code=None;stdout=text(getattr(failure,'stdout',None));stderr=text(getattr(failure,'stderr',None))
        error=type(failure).__name__+': '+str(failure)
    actual={'argv':argv,'returncode':code,'timeout_seconds':timeout,'command_retried':False,
        'stdout':stdout[:32768],'stderr':stderr[:8192],'stdout_sha256':ops.digest(stdout.encode()),
        'stderr_sha256':ops.digest(stderr.encode()),'stdout_bytes':len(stdout.encode()),'stderr_bytes':len(stderr.encode()),
        'truncated':len(stdout.encode())>32768 or len(stderr.encode())>8192,'error':error}
    capture=ops.RUN/'qualify.runtime_probe.json';ops.write_slot(capture,actual)
    ops.require(type(code) is int and code==0 and error is None and actual['truncated'] is False,
        'Actual analysis ABI probe failed; bounded stdout/stderr preserved on scratch')
    proof=json.loads(stdout);ops.validate_runtime_proof(proof,manifest)
    return {**actual,'proof':proof,'capture_path':str(capture),'capture_sha256':ops.sha(capture)}


def guard_context_probe():
    """Exercise real frozen metadata qualification/imports; never call a census."""
    capture=ops.RUN/'qualify.guard_context.json'
    try:
        guard=ops.guard_module();guard.verify()
        common=sys.modules['bridge_common'];qualified=common.qualified();policy=common.policy();policy_path,spec=policy.read_policy()
        ops.require(common.__file__==str(ops.BRIDGE/'bridge_common.py') and policy.__file__==str(ops.BRIDGE/'project_policy.py')
            and list(inspect.signature(guard.guard).parameters)==['source','reserve','capture']
            and list(inspect.signature(guard.validate_snapshot).parameters)==['snapshot','plan','reserve']
            and list(inspect.signature(policy.snapshot).parameters)==['root','original_personal_snapshot','timeout_s'],
            'Actual original A4/common/policy ABI differs')
        proof={'status':'original_A4_runtime_context_qualified','job_id':os.environ['SLURM_JOB_ID'],
            'bridge_manifest_sha256':ops.BRIDGE_PIN,'canonical_bridge_source_sha256':ops.BRIDGE_COMMON_PIN,
            'original_A4_manifest_sha256':ops.A4_PIN,'canonical_dependency_name':'bridge_common',
            'bridge_qualification_sha256':ops.sha(ops.BRIDGE/'qualification.json'),
            'bridge_qualification_status':qualified['status'],'policy_path':str(policy_path),
            'policy_entry_limit':spec['project_entry_limit'],'original_guard_source':guard.__file__,
            'scientific_arrays_opened':False,'census_executed':False,'historical_guard_sources_modified':False}
        ops.write_slot(capture,proof)
        return {'proof':proof,'capture_path':str(capture),'capture_sha256':ops.sha(capture)}
    except Exception as error:
        failure={'status':'original_A4_runtime_context_failed','error':str(error),'traceback':traceback.format_exc(),
            'scientific_arrays_opened':False,'census_executed':False}
        try:ops.write_slot(capture,failure)
        except Exception:pass
        raise




def child_budget():
    deadline=remaining()-AFTER_GUARD_RESERVE_SECONDS
    ops.require(deadline>0,'Insufficient phase time for science plus mandatory full after-storage guard')
    return deadline


def execute(argv,deadline):
    ops.require(all(isinstance(arg,str) for arg in argv),'Fixed actual scientific argv required')
    code=None;error=None;trace=None
    with ops.log_slot(ops.CONTROL/'analysis.child.stdout.log') as stdout,\
        ops.log_slot(ops.CONTROL/'analysis.child.stderr.log') as stderr:
        try:
            completed=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                timeout=deadline,check=False)
            code=completed.returncode
        except Exception as failure:
            error=type(failure).__name__+': '+str(failure);trace=traceback.format_exc()[-32768:]
    record={'argv':argv,'returncode':code,'timeout_seconds':deadline,'error':error,'traceback':trace,
        'command_retried':False,'scheduled_child_jobs_submitted':False,
        'stdout':ops.log_capture(ops.CONTROL/'analysis.child.stdout.log'),
        'stderr':ops.log_capture(ops.CONTROL/'analysis.child.stderr.log')}
    capture=ops.CONTROL/'analysis.child.execution.json';ops.write_slot(capture,record)
    ops.require(type(code) is int and code==0 and error is None,
        'Scientific process did not complete successfully; actual streams and execution record are preserved on scratch')
    return {**record,'capture_path':str(capture),'capture_sha256':ops.sha(capture)}


def native_schema_probe(manifest):
    capture=ops.RUN/'qualify.native_schema.json'
    module=ops.load_pinned('_full_population_native_probe',ops.ROOT/'native_probe.py',manifest['files']['native_probe.py'])
    try:
        proof=module.run()
        ops.require(proof.get('status')=='actual_native_clock_and_pair_schema_qualified'
            and proof.get('job_id')==os.environ['SLURM_JOB_ID'], 'Actual native clock/pair schema did not qualify')
        ops.write_slot(capture,proof)
        return {'proof':proof,'capture_path':str(capture),'capture_sha256':ops.sha(capture)}
    except Exception as error:
        ops.write_slot(capture,{'status':'actual_native_clock_and_pair_schema_failed','error':str(error),
            'error_type':type(error).__name__,'traceback':traceback.format_exc()[-32768:],
            'job_id':os.environ['SLURM_JOB_ID'],'full_bank_executed':False})
        raise


def parent_completion_probe():
    capture=ops.RUN/'qualify.parent_completion.json'
    proof=ops.completed_dream_proof();ops.write_slot(capture,proof)
    return {'proof':proof,'capture_path':str(capture),'capture_sha256':ops.sha(capture)}


def run(phase,pin):
    job=os.environ.get('SLURM_JOB_ID','')
    try:
        ops.require(phase in ops.POLICY and job.isdigit() and socket.getfqdn().endswith('.narval.calcul.quebec'),
            'Actual Narval scheduled allocation required')
        ops.scheduled_deadline(phase);manifest=ops.verify_packet(pin)
        context=ops.worker_context(phase,pin,job)
        ops.write_slot(ops.RUN/(phase+'.started.json'),{'utc':ops.utc(),'phase':phase,'job_id':job,
            'manifest_sha256':pin,'context':context,'prior_attempts_replayed':False})
        import numpy
        runtime={'python':sys.version,'python_executable':sys.executable,'numpy':numpy.__version__}
        test_source=ops.qualification_test_environment(manifest)
        if phase=='qualify':
            parent=parent_completion_probe();remaining()
            probe=runtime_probe(manifest);checked=tests(manifest['minimum_qualification_tests']);remaining()
            native_probe=native_schema_probe(manifest);remaining()
            guard_context=guard_context_probe();remaining()
            actual=storage(phase,'qualification');ops.verify_packet(pin)
            ops.write_slot(ops.CONTROL/'qualification.json',{'utc':ops.utc(),'status':'full_population_system_qualified',
                'job_id':job,'manifest_sha256':pin,'tests':checked,'runtime':runtime,'runtime_probe':probe,
                'guard_runtime_context':guard_context,'native_schema_probe':native_probe,'allocation':context['allocation'],'parent_completion':parent,
                'qualification_source_context':test_source,
                'storage':actual['snapshot'],'storage_capture_path':actual['capture'],
                'storage_capture_sha256':actual['sha256'],'results_or_hypotheses_used_as_qualification_gate':False})
            return
        qualifier=context['custody']['values']['intent']['qualifier_job_id']
        qualified=ops.qualification_proof(pin,qualifier)
        parent=ops.completed_dream_proof();remaining()
        before=storage(phase,'before');remaining()
        output=ops.RESULT/'analysis'
        ops.require(not output.exists() and not output.is_symlink(),'Exclusive fresh scientific output required')
        budget=child_budget();ops.require(phase=='analysis' and budget>31,'Only finite follow-up science execution is permitted')
        argv=[str(ops.PYTHON),'-B',str(ops.analysis_driver(manifest)),'--output',str(output),
            '--deadline-seconds',str(max(1,int(budget)-30))]
        command=execute(argv,budget);ops.require(output.is_dir() and output.resolve()==output,'Actual output directory is absent')
        ops.verify_packet(pin);after=storage(phase,'after');remaining()
        ops.write_slot(ops.CONTROL/'analysis_receipt.json',{'utc':ops.utc(),'status':'full_population_system_completed',
            'job_id':job,'manifest_sha256':pin,'output_root':str(output),'runtime':runtime,'qualifier':qualified,
            'completed_dream_parent':parent,'command':command,'before_storage':before,'after_storage':after,
            'results_or_hypotheses_used_as_execution_gate':False,'previous_outputs_modified':False})
    except Exception as error:
        failure={'utc':ops.utc(),'phase':phase,'job_id':job,'manifest_sha256':pin,'error_type':type(error).__name__,
            'error':str(error),'traceback':traceback.format_exc(),'retry_attempted':False}
        try:ops.write_slot(ops.RUN/(phase+'.failure.json'),failure)
        except Exception:traceback.print_exc()
        raise
