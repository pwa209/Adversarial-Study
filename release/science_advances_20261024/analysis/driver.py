"""Finite scheduled full-population bridge and existing-pair perturbation analysis."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STAGES = ('mechanism_compensation', 'human_spatial', 'pathway_balance')
STATUS = 'cross_mechanistic_system_runtime_qualified'


def require(value, message):
    if not value:
        raise ValueError('Cross-mechanistic extension: '+message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def training_gate(records):
    require(set(records) == {'centaur', 'base'}, 'Both human-data fine-tunings are mandatory')
    for name, record in records.items():
        require(record.get('model') == name and record.get('status') == 'human_finetuning_completed'
            and all(record.get(k) is True for k in ('native_training_executed', 'human_data', 'adaptation_completed', 'weights_changed'))
            and record.get('test_events_consumed') == 0,
            'Actual changed-weight human training with no test updates is required: '+name)
        split = record['split']
        groups = [split[k] for k in ('train_participant_ids', 'validation_participant_ids', 'test_participant_ids')]
        require(list(map(len, groups)) == [26, 8, 8] and len(set().union(*map(set, groups))) == 42,
                'Disjoint target42 26/8/8 split required')
        require(record['actual_optimizer_steps'] == record['optimizer_steps'] > 0
            and record['initial_trainable_weights_sha256'] != record['final_trainable_weights_sha256'],
            'Actual optimizer updates and changed weights are required')
    keys = ('configuration_sha256', 'train_events_sha256', 'validation_events_sha256', 'actual_training_order_sha256', 'optimizer_steps',
            'actual_optimizer_steps', 'epochs', 'trainable_parameters', 'split_files_sha256', 'assay_id', 'split')
    require(all(records['centaur'][k] == records['base'][k] for k in keys), 'Human-data training inputs/budgets differ')
    return {'target': 42, 'split': [26, 8, 8], 'both_actual_human_data_finetunings_verified': True,
            'test_events_consumed': 0, 'human_finetuning_repeated': False}


def clinical_gate(clinical, adapted, training_pin):
    require(clinical.get('status') == 'completed_actual_human_finetune'
            and all(clinical.get(k) is True for k in ('native_training_executed', 'human_data', 'adaptation_completed', 'backbone_training'))
            and clinical.get('actual_optimizer_steps', 0) > 0 and clinical.get('changed_backbone_parameter_count', 0) > 0,
            'Actual human clinical backbone adaptation is mandatory')
    require(adapted.get('status') == 'actual_human_adapted_activation_geometry_completed'
            and adapted.get('actual_adapted_native_activations') is True
            and adapted.get('clinical_training_receipt_sha256') == training_pin
            and isinstance(clinical.get('trained_checkpoint'), dict)
            and adapted.get('adapted_checkpoint') == clinical['trained_checkpoint'],
            'Actual adapted checkpoint and human training receipt must match')
    roles = []
    for role in ('train', 'validation', 'test'):
        people = clinical.get('split', {}).get(role+'_participant_ids', [])
        require(people and people == adapted.get('clinical_split', {}).get(role+'_participant_ids'),
                'Clinical exposure partition mismatch')
        roles.extend(people)
    require(len(roles) == len(set(roles)) and all(p.startswith('openneuro_ds001785_') for p in roles),
            'DREAM must remain source-held-out from tactile-only clinical adaptation')
    return {'actual_human_clinical_backbone_finetuning_verified': True,
            'adapted_geometry_training_receipt_bound': True, 'dream_source_held_out': True,
            'clinical_training_repeated': False}


def configuration():
    value = json.loads((HERE/'inputs.json').read_text())
    require(value['current_behavioral_target'] == 42 and value['behavioral_split'] == [26, 8, 8]
            and value['both_human_data_finetunings_before_comparison'] is True
            and value['retrospective_development'] is True and value['new_downloads'] is False
            and value['new_simulations'] is False and value['stages'] == list(STAGES)
            and value['reuse_completed_native_cache'] is True and value['result_direction_gate'] is False,
            'Target, adaptation order or existing-data scope differs')
    require(value['maximum_simultaneous_science_processes'] == 3 and value['maximum_threads_per_process'] == 4,
            'Bounded CPU policy differs')
    return value


def runtime_check():
    for name in STAGES:
        module = importlib.import_module(name)
        require(callable(module.run) and {'inputs', 'output'}.issubset(inspect.signature(module.run).parameters)
                and callable(module.qualification_probe), 'Stage ABI differs: '+name)
    configuration()
    return {'status': STATUS, 'driver_sha256': sha(__file__), 'input_manifest_sha256': sha(HERE/'inputs.json'),
            'scientific_arrays_opened': False, 'output_directories_created': False,
            'scientific_analysis_executed': False, 'stages': list(STAGES)}


def parent_cache_gate(value, parent, cache):
    require(parent.get('status') == 'full_population_system_completed' and parent.get('job_id') == '6963704'
            and set(parent.get('stages', {})) == {'native_population','population_bridge','population_perturbations'}
            and all(record['returncode'] == 0 for record in parent['stages'].values()),
            'Completed full-population parent required; never regenerate it')
    for role, member in (('native_cache_receipt','native_population/receipt.json'),
                         ('bridge_compatibility','population_bridge/compatibility.json')):
        require(parent['outputs'].get(member) == value['files'][role]['sha256'], 'Completed parent output binding differs: '+role)
    require(cache.get('schema') == 'native_nine_population_cache_v2'
            and cache.get('status') == 'native_population_cache_qualified'
            and cache.get('inventoried_rows') == cache.get('qualified_rows') == 664
            and cache.get('bank_receipt_sha256') == value['files']['bank_receipt']['sha256']
            and cache.get('inventory_sha256') == value['files']['bank_inventory']['sha256']
            and cache.get('scored_bank_sha256') == value['files']['scored_bank']['sha256']
            and cache.get('output_time_labels_preserved') is True and cache.get('output_clock_rebased') is False,
            'Original native cache provenance or clock differs')
    require(Path(value['native_cache_dir']) == Path(value['files']['native_cache_receipt']['path']).parent,
            'Cache path differs from completed receipt')
    return {'completed_population_job_adopted':'6963704', 'native_cache_reused': True,
            'native_cache_receipt_sha256':value['files']['native_cache_receipt']['sha256'],
            'native_cache_regenerated': False}


def validated_inputs():
    value = configuration()
    total, records = 0, {}
    for role, entry in value['files'].items():
        path = Path(entry['path']); before = path.lstat()
        require(not path.is_symlink() and path.resolve() == path and stat.S_ISREG(before.st_mode)
                and any(path.is_relative_to(root) for root in ('/scratch/pwa209/adversarial-study','/home/pwa209/adversarial-study')),
                'Foreign or linked input: '+role)
        require(before.st_uid == 3164768 and before.st_size <= entry.get('max_bytes', 134217728),
                'Input ownership/size differs: '+role)
        total += before.st_size
        require(total <= 536870912 and sha(path) == entry['sha256'], 'Input content differs: '+role)
        after = path.lstat()
        identity = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        require(identity(before) == identity(after), 'Input changed during verification')
        if role in ('centaur_training_receipt','base_training_receipt'):
            records[role.split('_')[0]] = json.loads(path.read_text())
    proof = training_gate(records)
    paths = {name:Path(spec['path']) for name,spec in value['files'].items()}
    proof.update(clinical_gate(json.loads(paths['clinical_training_receipt'].read_text()),
                              json.loads(paths['adapted_receipt'].read_text()), value['files']['clinical_training_receipt']['sha256']))
    parent = json.loads(paths['followup_scientific_receipt'].read_text())
    require(parent['status'] == 'dream_existing_data_followup_completed' and parent['analytic_people'] == 31
            and parent['analytic_episodes'] == 190 and len(parent['stages']) == 10
            and all(r['returncode'] == 0 for r in parent['stages']), 'Existing DREAM prerequisite incomplete')
    proof.update(parent_cache_gate(value, json.loads(paths['population_completion_receipt'].read_text()),
                                  json.loads(paths['native_cache_receipt'].read_text())))
    proof.update(input_files_verified=len(value['files']), input_bytes_verified=total,
                 input_manifest_sha256=sha(HERE/'inputs.json'))
    return value, proof


def stage_inputs(name, value):
    require(name in STAGES, 'Unknown stage')
    cache = {'path':value['native_cache_dir'], 'receipt_sha256':value['files']['native_cache_receipt']['sha256']}
    if name == 'mechanism_compensation':
        return {'files':{'population_cache':cache}, 'bootstrap_repetitions':2000}
    if name == 'human_spatial':
        return {'files':{role:value['files'][role] for role in importlib.import_module(name).ROLES}}
    return {'files':{'native_cache':cache, **{role:value['files'][role] for role in ('bridge_compatibility','bank_receipt')}}}


def branch(name, output, seconds):
    require(name in STAGES and seconds > 0, 'Unknown stage or expired deadline')
    value, proof = validated_inputs()
    module = importlib.import_module(name)
    result = module.run(stage_inputs(name,value), Path(output), seed=value['seed'])
    require(isinstance(result,dict) and isinstance(result.get('status'),str), 'Stage receipt missing')
    print(json.dumps({'status':'cross_mechanistic_stage_completed','stage':name,'training_gate':proof,
                      'source_sha256':sha(module.__file__), 'stage_status':result['status']}, separators=(',',':')), flush=True)


def execute_branches(output, deadline):
    """Independent bounded children; a failure never discards a successful sibling."""
    records, children, handles = {}, {}, []
    env = dict(os.environ, OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               NUMEXPR_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    try:
        for name in STAGES:
            remaining = int(deadline-time.monotonic())-30
            require(remaining > 0, 'Finite science deadline exhausted before launch')
            argv = [sys.executable,'-B',str(HERE/'driver.py'),'--branch',name,
                    '--output',str(output/name),'--deadline-seconds',str(remaining)]
            record = {'argv':argv,'command_retried':False,'started_utc':datetime.now(timezone.utc).isoformat()}
            records[name] = record
            stdout = (output/(name+'.stdout.log')).open('xb'); handles.append(stdout)
            stderr = (output/(name+'.stderr.log')).open('xb'); handles.append(stderr)
            children[name] = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env)
        for name, child in children.items():
            try:
                records[name]['returncode'] = child.wait(timeout=max(.01,deadline-time.monotonic()-10))
            except subprocess.TimeoutExpired:
                child.kill(); records[name]['returncode'] = child.wait(); records[name]['timed_out'] = True
            records[name]['finished_utc'] = datetime.now(timezone.utc).isoformat()
    finally:
        for name, child in children.items():
            if child.poll() is None:
                child.kill(); child.wait(); records[name]['interrupted'] = True
            records[name].setdefault('returncode', child.returncode)
        for handle in handles: handle.close()
        write(output/'child_execution.json',records)
    require(set(records) == set(STAGES) and all(r['returncode'] == 0 and not r.get('timed_out') for r in records.values()),
            'At least one branch failed; each branch and completed sibling output are preserved')
    return records


def analysis(output, seconds):
    started = time.monotonic(); deadline = started+seconds
    value, proof = validated_inputs()
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), 'Fresh immutable result directory required')
    require(os.environ.get('SLURM_JOB_ID','').isdigit() and os.environ.get('SLURM_CPUS_PER_TASK') == '16',
            'Sixteen-CPU scheduled allocation required')
    output.mkdir(); write(output/'training_and_input_gate.json',proof); write(output/'configuration.json',value)
    records = execute_branches(output,deadline)
    _, final_proof = validated_inputs()
    require(proof == final_proof, 'Human training/input provenance changed during analysis')
    products = {str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    receipt = {'status':'cross_mechanistic_system_completed','utc':datetime.now(timezone.utc).isoformat(),
               'job_id':os.environ['SLURM_JOB_ID'],'stages':records,'outputs':products,
               'training_gate':proof,'elapsed_seconds':time.monotonic()-started,
               'current_behavioral_target':42,'behavioral_split':[26,8,8],
               'both_human_data_finetunings_before_comparison':True,'result_direction_gate':False,
               'retrospective_development':True,'independent_human_replication_completed':False,
               'new_simulations':False,'new_downloads':False,'completed_work_repeated':False,
               'bootstrap_adds_independent_people':False,'maximum_science_threads':12}
    write(output/'receipt.json',receipt); print(json.dumps(receipt,separators=(',',':')),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-check',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--deadline-seconds',type=int,default=72000)
    parser.add_argument('--branch',choices=STAGES)
    args = parser.parse_args()
    if args.runtime_check: print(json.dumps(runtime_check()))
    elif args.branch: branch(args.branch,args.output,args.deadline_seconds)
    else: analysis(args.output,args.deadline_seconds)
