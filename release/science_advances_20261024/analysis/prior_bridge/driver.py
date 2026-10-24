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
STAGES = ('native_population', 'population_bridge', 'population_perturbations')
STATUS = 'full_population_system_runtime_qualified'


def require(value, message):
    if not value:
        raise ValueError('Full-population extension: '+message)


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
            and value['new_simulations'] is False and value['stages'] == list(STAGES),
            'Target, adaptation order or scope differs')
    return value


def runtime_check():
    for name in STAGES:
        module = importlib.import_module(name)
        require(callable(module.run) and {'inputs', 'output'}.issubset(inspect.signature(module.run).parameters),
                'Stage ABI differs: '+name)
    configuration()
    return {'status': STATUS, 'driver_sha256': sha(__file__), 'input_manifest_sha256': sha(HERE/'inputs.json'),
            'scientific_arrays_opened': False, 'output_directories_created': False,
            'scientific_analysis_executed': False, 'stages': list(STAGES)}


def validated_inputs():
    value = configuration()
    total, records = 0, {}
    for role, entry in value['files'].items():
        path = Path(entry['path']); before = path.lstat()
        require(not path.is_symlink() and path.resolve() == path and stat.S_ISREG(before.st_mode)
                and any(path.is_relative_to(root) for root in ('/scratch/pwa209/adversarial-study', '/home/pwa209/adversarial-study')),
                'Foreign or linked input: '+role)
        require(before.st_uid == 3164768 and before.st_size <= entry.get('max_bytes', 134217728),
                'Input ownership/size differs: '+role)
        total += before.st_size
        require(total <= 536870912 and sha(path) == entry['sha256'], 'Input content differs: '+role)
        after = path.lstat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(identity(before) == identity(after), 'Input changed during verification')
        if role in ('centaur_training_receipt', 'base_training_receipt'):
            records[role.split('_')[0]] = json.loads(path.read_text())
    proof = training_gate(records)
    paths = {name: Path(spec['path']) for name, spec in value['files'].items()}
    proof.update(clinical_gate(json.loads(paths['clinical_training_receipt'].read_text()),
                              json.loads(paths['adapted_receipt'].read_text()), value['files']['clinical_training_receipt']['sha256']))
    parent = json.loads(paths['followup_scientific_receipt'].read_text())
    require(parent['status'] == 'dream_existing_data_followup_completed' and parent['analytic_people'] == 31
            and parent['analytic_episodes'] == 190 and len(parent['stages']) == 10
            and all(r['returncode'] == 0 for r in parent['stages']), 'Existing DREAM prerequisite incomplete')
    proof.update(input_files_verified=len(value['files']), input_bytes_verified=total,
                 input_manifest_sha256=sha(HERE/'inputs.json'))
    return value, proof


def stage_inputs(name, value, result_root):
    require(name in STAGES, 'Unknown stage')
    result_root = Path(result_root)
    if name == 'native_population':
        with Path(value['files']['bank_inventory']['path']).open(newline='', encoding='utf8') as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 664 and sum(bool(r.get('pair_id')) for r in rows) == 96, 'Frozen bank census differs')
        return {'files': {r: value['files'][r] for r in ('bank_inventory', 'bank_receipt', 'scored_bank')},
                'bank_root': value['bank_root'],
                'required_simulation_ids': [r['simulation_id'] for r in rows if r['status'] == 'finite']}
    cache = result_root/'native_population'
    require((cache/'receipt.json').is_file(), 'Population qualification must finish before either analysis')
    if name == 'population_bridge':
        return {'files': {r: value['files'][r] for r in importlib.import_module(name).ROLES}, 'population_cache': str(cache)}
    compatibility = result_root/'population_bridge/compatibility.json'
    require(compatibility.is_file() and (result_root/'population_bridge/receipt.json').is_file(),
            'Bridge must finish before perturbation support comparison')
    return {'files': {'native_cache': {'path': str(cache)},
                      'bridge_compatibility': {'path': str(compatibility), 'sha256': sha(compatibility)}}}


def branch(name, output, seconds):
    require(name in STAGES and seconds > 0, 'Unknown stage or expired deadline')
    value, proof = validated_inputs()
    module = importlib.import_module(name)
    inputs = stage_inputs(name, value, Path(output).parent)
    result = module.run(inputs, Path(output), seed=value['seed'], **value['settings'][name])
    require(isinstance(result, dict) and isinstance(result.get('status'), str), 'Stage receipt missing')
    print(json.dumps({'status': 'full_population_stage_completed', 'stage': name, 'training_gate': proof,
                      'source_sha256': sha(module.__file__), 'stage_status': result['status']}, separators=(',', ':')), flush=True)


def analysis(output, seconds):
    started = time.monotonic(); deadline = started+seconds
    value, proof = validated_inputs()
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), 'Fresh immutable result directory required')
    require(os.environ.get('SLURM_JOB_ID', '').isdigit() and os.environ.get('SLURM_CPUS_PER_TASK') == '16',
            'Sixteen-CPU scheduled allocation required')
    output.mkdir(); write(output/'training_and_input_gate.json', proof)
    write(output/'configuration.json', value)
    records = {}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               NUMEXPR_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    try:
        for name in STAGES:
            remaining = int(deadline-time.monotonic())-30
            require(remaining > 0, 'Finite science deadline exhausted')
            argv = [sys.executable, '-B', str(HERE/'driver.py'), '--branch', name,
                    '--output', str(output/name), '--deadline-seconds', str(remaining)]
            record = {'argv': argv, 'command_retried': False, 'started_utc': datetime.now(timezone.utc).isoformat()}
            records[name] = record
            with (output/(name+'.stdout.log')).open('xb') as stdout, (output/(name+'.stderr.log')).open('xb') as stderr:
                try:
                    child = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                           timeout=remaining, check=False, env=env)
                    record['returncode'] = child.returncode
                except subprocess.TimeoutExpired:
                    record['returncode'] = None; record['timed_out'] = True
                    raise
            require(record['returncode'] == 0, 'Scientific stage failed; previous completed stage outputs preserved: '+name)
            write(output/(name+'.execution.json'), record)
    finally:
        write(output/'child_execution.json', records)
    _, final_proof = validated_inputs()
    require(proof == final_proof, 'Human training/input provenance changed during analysis')
    products = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    receipt = {'status': 'full_population_system_completed', 'utc': datetime.now(timezone.utc).isoformat(),
               'job_id': os.environ['SLURM_JOB_ID'], 'stages': records, 'outputs': products,
               'training_gate': proof, 'elapsed_seconds': time.monotonic()-started,
               'current_behavioral_target': 42, 'behavioral_split': [26, 8, 8],
               'both_human_data_finetunings_before_comparison': True, 'result_direction_gate': False,
               'retrospective_development': True, 'independent_human_replication_completed': False,
               'new_simulations': False, 'new_downloads': False, 'completed_work_repeated': False}
    write(output/'receipt.json', receipt); print(json.dumps(receipt, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-check', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline-seconds', type=int, default=72000)
    parser.add_argument('--branch', choices=STAGES)
    args = parser.parse_args()
    if args.runtime_check:
        print(json.dumps(runtime_check()))
    elif args.branch:
        branch(args.branch, args.output, args.deadline_seconds)
    else:
        analysis(args.output, args.deadline_seconds)
