"""Once-only full-population extension; successful parent jobs are adopted."""
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import stat
import subprocess
import sys
import time
import types

ROOT=Path(__file__).resolve().parent
REMOTE=Path('/scratch/pwa209/adversarial-study/maintenance/cross-mechanistic-20261024-v2')
CONTROL=Path('/scratch/pwa209/adversarial-study/runs/cross-mechanistic-20261024-v2')
RUN=CONTROL
RESULT=Path('/scratch/pwa209/adversarial-study/results/cross-mechanistic-20261024-v2')
PYTHON=Path('/scratch/pwa209/adversarial-study/environments/20261004T193709Z-545c74b08e71/bin/python')
UID=3164768
MANIFEST='FOLLOWUP_MANIFEST.json'
DREAM_PARENT=Path('/scratch/pwa209/adversarial-study/maintenance/dream-startup-repair-20261021-v4')
DREAM_CONTROL=Path('/scratch/pwa209/adversarial-study/runs/dream-startup-repair-20261021-v4')
DREAM_RESULT=Path('/scratch/pwa209/adversarial-study/results/dream-startup-repair-20261021-v4/dream_extension')
DREAM_PARENT_PIN='a95bc34c0876a9ed03ae5670885a41500294cc912f85d8183153920c4cc209d5'
DREAM_QUALIFIER_JOB='6672041'
DREAM_JOB='6672118'
DREAM_QUALIFICATION_PIN='412c82c4099943f5e2a5b32eeb58d11eefb0d60cf6f0ec8920de925724f9a5b2'
DREAM_EXECUTION_PIN='4da78341be24796dc28f592833c3696ea5427addd1fcc8d0cbd7a283a0c62219'
DREAM_CHILD_PIN='967569d07c16f29cb9f8520e16b7f3607d87db167d12638d4f9a722f2e28ba1a'
DREAM_SCIENTIFIC_PIN='236672b4a456fdc1be71e84857bd1214f7d7294ade78b643899130815c890863'
DREAM_METADATA_PIN='81ec750d2c3d6c153a35ffa7a9b0eab8ffa5ce9cfd69dec40fd862a2abe62a9f'
DREAM_STORAGE_PINS={
 'qualify.storage.qualification.json':'a2941d43f03b1c64eb9d1be2d1aae0ef903dabde8d416c2892787068ded1f42f',
 'dream.storage.before.json':'332a236723fb66027bbdc1097c1de0702119aef7361318ef430a6097ac34fcc4',
 'dream.storage.after.json':'9ffe23d3336a0d879360508f45a21c7f828aec875878df23ecf68da1515c9a11'}
FROZEN_DISCOVERY_PIN='1b2fe2a189184cbb846187034ba7355e7af264080b11ab7778240a42e7eabbf8'
PARENT=Path('/home/pwa209/adversarial-study/maintenance/scientific-results-audit-20261021-v4')
PARENT_RUN=Path('/home/pwa209/adversarial-study/runs/scientific-results-audit-20261021-v4')
PARENT_RESULT=Path('/scratch/pwa209/adversarial-study/results/scientific-results-audit-20261021-v4/numerical_audit')
PARENT_PIN='d1843d53d74a62b39e84695cc92014d6fa5555cca8de4234378422387c0ae9ea'
PARENT_QUALIFIER_JOB='6639721'
PARENT_NUMERICAL_JOB='6639953'
PARENT_QUALIFICATION_PIN='617e74479d163f9858a0891f60dec52e764736c5ddac6da7942991d7229830e3'
PARENT_NUMERICAL_PIN='e09d30e64ea904d5826495543b7e3e38bd4f8c374a994df6a77a231d8ba9d52c'
STARTUP_CONFIG_PIN='7a32dd862ad78dae02fb502f836bb4693671ccda94cb90ca58f12cc203e7de51'
PARENT_STORAGE_PINS={
 'qualify.storage.qualification.json':'43a588dd38b710de945e100777f9511d5e0b51f5d53b12e1ecf2cf896ea50018',
 'analysis.storage.before.json':'8c9a45ee18293ac11716b5b7f2b754f258b44ed42444bbad2e5f001dd49080ce',
 'analysis.storage.after.json':'7dc0481c9cbc07e8be0b1c885849cf3625fd5646dde7bfb1a6c6e95b5216467c'}
PRODUCTS=('behavior_paired','discovery_clipping','threshold_direction','mechanism_adequacy',
 'native_recovery','temporal_controls','generative_support','existing_summaries')
A4=Path('/home/pwa209/adversarial-study/maintenance/project-byte-accounting-20261018-v4')
A4_PIN='be077ca5e83630111a369114188d9e098f2eb778d3604f7b4fa9ab9521b9bc80'
READER_ROOT=Path('/home/pwa209/adversarial-study/maintenance/remaining-stages-repair-20261020-v9')
READER_PIN='df5e97b8adbde59e5841d3c3ae1f78b8424cfd44e5013e6d083da42ce356d72d'
EXTENSION_PIN='01ec02536857d4ce36334016ec732c30ebdba207493fb017f2119b9e05744428'
BRIDGE=Path('/home/pwa209/adversarial-study/maintenance/bridge-retry-20261008')
BRIDGE_PIN='e6787130643b72eee7bdb5c235555371d98fe142996a521c33d4913185f48cf3'
BRIDGE_COMMON_PIN='8bb989b5e5dfdc431069060a78f4dd25014a182a9bd7f47dd71f517db6121ec1'
POLICY={'qualify':{'cpus':4,'memory':'8G','time':'02:00:00','seconds':7200},
        'analysis':{'cpus':16,'memory':'64G','time':'1-00:00:00','seconds':86400}}
RESERVES={'qualify':4096,'analysis':4096}
FORMAT='JobIDRaw,User,UID,Account,JobName%80,State%40,ExitCode,AllocCPUS,ReqCPUS,Elapsed'
_OBSERVER=None
_DEADLINE=None


def require(condition,message):
    if not condition:raise ValueError('DREAM existing-data follow-up: '+message)


def utc():return datetime.now(timezone.utc).isoformat()
def digest(raw):return hashlib.sha256(raw).hexdigest()


def raw_file(path,maximum=2097152):
    path=Path(path);before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink() and path.resolve()==path
        and before.st_uid==UID and before.st_nlink==1 and before.st_size<=maximum,'Owned canonical bounded file required: '+str(path))
    with path.open('rb') as stream:
        opened=os.fstat(stream.fileno())
        require((opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns)==
            (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns),'File changed before open')
        raw=stream.read(before.st_size)
        ended=os.fstat(stream.fileno())
    after=path.lstat()
    require(len(raw)==before.st_size and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)==
        (ended.st_dev,ended.st_ino,ended.st_size,ended.st_mtime_ns),'File changed while reading')
    return raw


def read(path,maximum=2097152):return json.loads(raw_file(path,maximum))
def sha(path):return digest(raw_file(path,8388608))


def load_pinned(name,path,pin):
    raw=raw_file(path,8388608);require(digest(raw)==pin,'Pinned Python source differs: '+str(path))
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(raw,str(path),'exec'),module.__dict__);return module


def verify_tree(root,filename,pin):
    manifest_raw=raw_file(root/filename);require(digest(manifest_raw)==pin,'External manifest pin differs')
    value=json.loads(manifest_raw);files=value['files'];require(isinstance(files,dict) and 1<=len(files)<=128,'Bounded static file map required')
    total=0
    for name,expected in files.items():
        relative=PurePosixPath(name)
        require(str(relative)==name and relative.parts and not relative.is_absolute()
            and all(re.fullmatch('[A-Za-z0-9_.-]+',part) and part not in {'.','..'} for part in relative.parts),'Unsafe static member')
        raw=raw_file(root/name,8388608);total+=len(raw)
        require(total<=67108864 and digest(raw)==expected,'Static source pin or packet bound differs: '+name)
    return value


def verify_packet(pin,*,full_upstream=True):
    require(type(full_upstream) is bool,'Explicit full-upstream verification mode required')
    require(ROOT==REMOTE and not ROOT.is_symlink() and os.getuid()==UID,'Own canonical audit release required')
    value=verify_tree(ROOT,MANIFEST,pin)
    require(value['remote_root']==str(REMOTE) and value['run_root']==str(RUN) and value['result_root']==str(RESULT),
        'Fixed audit namespaces differ')
    require(type(value['minimum_qualification_tests']) is int and value['minimum_qualification_tests']>0,'Qualification test budget required')
    require(value['resources']==POLICY and value['entry_reserves']==RESERVES,'Fixed phase resources/peak entry reserves differ')
    require(value['parent_audit_manifest_sha256']==PARENT_PIN and value['parent_dream_manifest_sha256']==DREAM_PARENT_PIN
        and value['parent_dream_job']==DREAM_JOB and value['current_behavioral_target']==42
        and value['behavioral_split']==[26,8,8] and value['numerical_analysis_resubmission_permitted'] is False,
        'Follow-up identity or original human contract differs')
    require(value['dream_conversion_resubmission_permitted'] is False,
        'Completed DREAM conversion/scoring may not be resubmitted')
    analysis_driver(value)
    frozen_discovery_member(value)
    parent=verify_tree(PARENT,'AUDIT_MANIFEST.json',PARENT_PIN)
    dream=verify_tree(DREAM_PARENT,'DREAM_REPAIR_MANIFEST.json',DREAM_PARENT_PIN)
    upstream=value.get('upstream_metadata',[]);require(isinstance(upstream,list) and len(upstream)<=128,'Bounded upstream metadata required')
    total=0
    for item in upstream:
        path=Path(item['path']);require(path.is_absolute() and path.suffix in {'.json','.py','.csv'}
            and isinstance(item.get('sha256'),str) and re.fullmatch('[0-9a-f]{64}',item['sha256']),
            'Explicit upstream metadata/source and SHA pin required')
        if full_upstream:
            raw=raw_file(path);total+=len(raw);require(total<=16777216 and digest(raw)==item['sha256'],'Upstream metadata pin differs')
    training=value['required_human_training_receipts'];require(training==parent['required_human_training_receipts']==dream['required_human_training_receipts']
        and len(training)==2 and {item['model'] for item in training}=={'centaur','base'},
        'Both actual human-data fine-tunings must be pinned')
    records={}
    for item in training:
        raw=raw_file(Path(item['path']));require(digest(raw)==item['sha256'],'Training receipt pin differs')
        record=json.loads(raw);require(record['model']==item['model'] and record['status']=='human_finetuning_completed'
            and all(record.get(key) is True for key in ('native_training_executed','human_data','adaptation_completed','weights_changed'))
            and record.get('test_events_consumed')==0,'Actual human fine-tuning has not qualified')
        split=record['split'];groups=[split[key] for key in ('train_participant_ids','validation_participant_ids','test_participant_ids')]
        require(list(map(len,groups))==[26,8,8] and len(set().union(*map(set,groups)))==42
            and record['actual_optimizer_steps']==record['optimizer_steps']>0
            and record['initial_trainable_weights_sha256']!=record['final_trainable_weights_sha256'],
            'Actual 42-person human adaptation/split proof differs')
        records[item['model']]=record
    keys=('configuration_sha256','train_events_sha256','validation_events_sha256','actual_training_order_sha256','optimizer_steps',
        'actual_optimizer_steps','epochs','trainable_parameters','split_files_sha256','assay_id','split')
    require(all(records['centaur'][key]==records['base'][key] for key in keys),'Human training data/split/budgets are not matched')
    return value


def analysis_driver(manifest):
    name=manifest['analysis_driver']
    require(isinstance(name,str) and re.fullmatch('[A-Za-z0-9_.-]+[.]py',name)
        and name not in {'audit_ops.py','guard_runner.py'} and name in manifest['files']
        and 'inputs.json' in manifest['files'],'One own source-pinned analysis entry point and inputs manifest are required')
    require(manifest['runtime_probe_status']=='cross_mechanistic_system_runtime_qualified',
        'Actual analysis ABI qualification status differs')
    return ROOT/name


def frozen_discovery_member(manifest):
    name=manifest['frozen_discovery_member'];relative=PurePosixPath(name)
    require(isinstance(name,str) and str(relative)==name and not relative.is_absolute()
        and relative.suffix=='.py' and all(re.fullmatch('[A-Za-z0-9_.-]+',part) and part not in {'.','..'} for part in relative.parts)
        and manifest['files'].get(name)==FROZEN_DISCOVERY_PIN,'Exact own copied frozen-discovery role differs')
    return ROOT/name


def qualification_test_environment(manifest):
    source=frozen_discovery_member(manifest)
    require(sha(source)==FROZEN_DISCOVERY_PIN,'Frozen descriptive discovery source differs')
    os.environ['DREAM_AUDIT_FROZEN_DISCOVERY_SOURCE']=str(source)
    return {'frozen_discovery_source':str(source),'sha256':FROZEN_DISCOVERY_PIN,
        'source_imported':False,'scientific_arrays_opened':False}


def scheduled_deadline(phase):
    global _DEADLINE
    seconds=POLICY[phase]['seconds'];end=os.environ.get('SLURM_JOB_END_TIME')
    if end is not None:
        require(re.fullmatch('[0-9]+',end),'Invalid Slurm end time');seconds=min(seconds,int(end)-time.time())
    require(seconds>120,'Actual phase allocation has expired');_DEADLINE=time.monotonic()+seconds-120


def observer():
    global _OBSERVER,_DEADLINE
    if _OBSERVER is None:
        if _DEADLINE is None:_DEADLINE=time.monotonic()+120
        base=load_pinned('_audit_pristine_scheduler_reader',READER_ROOT/'scheduler_reads.py',READER_PIN)
        extension=load_pinned('_audit_bounded_read_extension',READER_ROOT/'read_resilience.py',EXTENSION_PIN)
        _OBSERVER=extension.make_adapter(base,remaining=lambda:max(0,_DEADLINE-time.monotonic()),raise_exhausted=True)
    return _OBSERVER


def observe(argv):
    result=observer().run(argv,text=True,capture_output=True,timeout=30,stdin=subprocess.DEVNULL)
    require(type(result.returncode) is int and result.returncode==0 and len(result.stdout.encode())<=131072
        and len(result.stderr.encode())<=8192,'Actual bounded scheduler observation failed')
    return result.stdout


def fields(raw):
    pairs=[(m.group(1),m.group(2).strip()) for m in re.finditer(r'([^\s=]+)=(.*?)(?=\s+[^\s=]+=|$)',raw,re.S)]
    require(len(pairs)==len({key for key,_ in pairs}),'Duplicate scheduler field');return dict(pairs)


def validate_allocation(raw,job,phase,dependency=None,*,held=False,fulfilled=False):
    p=POLICY[phase];value=fields(raw);memory={p['memory'],str(int(p['memory'][:-1])*1024)+'M'}
    wanted={'JobId':str(job),'JobName':'acs-cross-mechanistic-'+phase,'UserId':'pwa209(3164768)',
        'Account':'def-ptewarie_cpu','NumCPUs':str(p['cpus']),'NumTasks':'1','CPUs/Task':str(p['cpus']),
        'Requeue':'0','TimeLimit':p['time'],'Command':str(REMOTE/'worker.sh'),'WorkDir':str(REMOTE),
        'Comment':'acs:cross-mechanistic-20261024-v2:'+phase,'StdOut':str(CONTROL/(phase+'.out'))}
    require(all(value.get(k)==v for k,v in wanted.items()) and value.get('NumNodes') in {'1','1-1'}
        and value.get('MinMemoryNode') in memory,'Actual audit identity/resources differ')
    require(value.get('JobState')==('PENDING' if held else 'RUNNING') and (not held or value.get('Reason')=='JobHeldUser'),
        'Actual held/running allocation differs')
    allowed={None,'(null)'} if dependency is None else {dependency,dependency+'(unfulfilled)'}
    if fulfilled:allowed|={None,'(null)'}
    require(value.get('Dependency') in allowed,'Actual serialized dependency differs')
    if held:require(value.get('AllocTRES') in {None,'(null)'},'Held audit already allocated')
    for key in ('ReqTRES',) if held else ('ReqTRES','AllocTRES'):
        pairs=[item.split('=',1) for item in value.get(key,'').split(',')]
        require(all(len(item)==2 for item in pairs) and len(pairs)==len({item[0] for item in pairs}),'Malformed audit TRES')
        tres=dict(pairs);require(set(tres)<={'cpu','mem','node','billing'} and tres.get('cpu')==str(p['cpus'])
            and tres.get('mem') in memory and tres.get('node')=='1','Audit GPU/CPU/memory TRES differs')
    return value


def queue_guard(allowed=()):
    raw=observe(['squeue','-h','-u','pwa209','-o','%i|%j|%T|%C']);seen=set();requested=0;active=0;permitted=set(map(str,allowed))
    for line in raw.splitlines():
        row=[part.strip() for part in line.split('|')]
        require(len(row)==4 and re.fullmatch(r'\d+(?:\+\d+)?(?:_(?:\d+|\[[\d,%:-]+\]))?',row[0])
            and row[1] and row[2] and row[3].isdigit() and row[0] not in seen,'Malformed actual own queue')
        seen.add(row[0])
        if row[1].startswith('acs-'):
            require(row[0] in permitted,'Other study jobs require explicit reconciliation')
            require(row[2] in {'PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED','RESIZING'},'Unknown live study allocation state')
            requested+=int(row[3])
            if row[2]!='PENDING':active+=int(row[3])
    require(active<=16 and requested<=50,'Original active CPU16 or requested CPU50 ceiling exceeded')
    return {'raw':raw,'requested_study_cpus':requested,'active_study_cpus':active,'original_cap':16,'user_cap':50}


@contextmanager
def locked(shared=False):
    import fcntl
    if not CONTROL.exists():
        require(not shared and not CONTROL.is_symlink() and CONTROL.parent.parent.is_dir()
            and CONTROL.parent.parent.resolve()==CONTROL.parent.parent,
            'Canonical fresh scratch control parent required')
        CONTROL.parent.mkdir(exist_ok=True)
        require(CONTROL.parent.is_dir() and not CONTROL.parent.is_symlink()
            and CONTROL.parent.resolve()==CONTROL.parent and CONTROL.parent.stat().st_uid==UID,
            'Owned scratch control parent required')
        CONTROL.mkdir()
    require(CONTROL.is_dir() and CONTROL.resolve()==CONTROL and not CONTROL.is_symlink(),
        'Canonical scratch-only control directory required')
    path=CONTROL/'operations.lock'
    if not path.exists():
        with path.open('x') as stream:stream.flush();os.fsync(stream.fileno())
    raw_file(path)
    with path.open('r+') as stream:
        fcntl.flock(stream,fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def slot_paths():
    paths=[CONTROL/(phase+'.'+name+'.json') for phase in POLICY
        for name in ('intent','acceptance','held','release-intent','released')]
    return paths+[CONTROL/'qualification.json',CONTROL/'analysis_receipt.json',RUN/'qualify.runtime_probe.json',
        RUN/'qualify.guard_context.json',RUN/'qualify.parent_completion.json',RUN/'qualify.native_schema.json']+[
        CONTROL/name for name in ('analysis.child.stdout.log','analysis.child.stderr.log','analysis.child.execution.json')]+[
        CONTROL/(phase+'.out') for phase in POLICY]+[
        RUN/(phase+'.'+name+'.json') for phase in POLICY for name in ('started','failure','wrapper_failure')]


def state(create=False):
    path=CONTROL/'state.creation.json'
    if not path.exists():
        require(create and not RESULT.exists() and not RESULT.is_symlink()
            and set(CONTROL.iterdir())=={CONTROL/'operations.lock'},'Fresh DREAM scratch state required')
        RESULT.mkdir();paths=slot_paths();require(not any(p.exists() or p.is_symlink() for p in paths),'Existing audit custody; never replay')
        identities={}
        for target in paths:
            with target.open('x') as stream:stream.flush();os.fsync(stream.fileno())
            info=target.stat();identities[str(target)]=[info.st_dev,info.st_ino]
        with path.open('x') as stream:json.dump({'utc':utc(),'identities':identities},stream);stream.flush();os.fsync(stream.fileno())
    value=read(path)
    for target in slot_paths():
        info=target.lstat();require(stat.S_ISREG(info.st_mode) and not target.is_symlink() and target.resolve()==target
            and info.st_uid==UID and info.st_nlink==1 and [info.st_dev,info.st_ino]==value['identities'][str(target)],'Precreated audit inode changed')
    return value


def write_slot(path,value):
    payload=json.dumps(value,indent=2,allow_nan=False)+'\n'
    known=state();info=path.stat();require([info.st_dev,info.st_ino]==known['identities'][str(path)] and info.st_size==0,
        'Audit receipt already written; never replay')
    with path.open('r+',encoding='utf8',newline='\n') as stream:
        opened=os.fstat(stream.fileno());require([opened.st_dev,opened.st_ino]==known['identities'][str(path)]
            and stat.S_ISREG(opened.st_mode) and opened.st_nlink==1 and opened.st_uid==UID and opened.st_size==0,
            'Audit receipt inode changed before writing')
        stream.write(payload);stream.truncate();stream.flush();os.fsync(stream.fileno())
        after=path.lstat();require([after.st_dev,after.st_ino]==known['identities'][str(path)] and not path.is_symlink(),
            'Audit receipt inode changed while writing')


@contextmanager
def log_slot(path):
    known=state();require(path in {CONTROL/'analysis.child.stdout.log',CONTROL/'analysis.child.stderr.log'},
        'Only explicit precreated DREAM child streams may be written')
    descriptor=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
    with os.fdopen(descriptor,'r+b',buffering=0) as stream:
        info=os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid==UID and info.st_nlink==1 and info.st_size==0
            and [info.st_dev,info.st_ino]==known['identities'][str(path)],'Existing or replaced child stream; never replay')
        try:yield stream
        finally:stream.flush();os.fsync(stream.fileno())


def log_capture(path,maximum=32768):
    """Bounded actual bytes; a large stream is a labelled tail, never a full hash."""
    require(path in {CONTROL/'analysis.child.stdout.log',CONTROL/'analysis.child.stderr.log'}
        and type(maximum) is int and 0<maximum<=32768,'Fixed bounded child log required')
    known=state();descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(descriptor,'rb') as stream:
        before=os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_uid==UID and before.st_nlink==1
            and [before.st_dev,before.st_ino]==known['identities'][str(path)],'Child stream identity differs')
        offset=max(0,before.st_size-maximum);stream.seek(offset);raw=stream.read(before.st_size-offset)
        after=os.fstat(stream.fileno())
    identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    require(len(raw)==before.st_size-offset and identity(before)==identity(after)==identity(path.lstat()),
        'Child stream changed during bounded capture')
    return {'path':str(path),'file_bytes':before.st_size,'offset':offset,'captured_bytes':len(raw),
        'complete_file':offset==0,'sha256_of_captured_bytes':digest(raw),'text':raw.decode('utf8',errors='replace')}


def maybe(path):
    raw=raw_file(path);return json.loads(raw) if raw else None


def submit_argv(phase,pin,dependency=None):
    p=POLICY[phase]
    argv=['sbatch','--parsable','--hold','--nodes=1','--ntasks=1','--no-requeue','--account=def-ptewarie_cpu',
        '--job-name=acs-cross-mechanistic-'+phase,'--comment=acs:cross-mechanistic-20261024-v2:'+phase,
        '--cpus-per-task='+str(p['cpus']),'--mem='+p['memory'],'--time='+p['time'],
        '--chdir='+str(ROOT),'--output='+str(CONTROL/(phase+'.out')),'--open-mode=append']
    if dependency:argv.append('--dependency='+dependency)
    return argv+[str(REMOTE/'worker.sh'),phase,'--manifest-sha256',pin]


def custody(phase,pin):
    require(phase in POLICY,'Only DREAM repair phases have custody')
    values={name:maybe(CONTROL/(phase+'.'+name+'.json')) for name in ('intent','acceptance','held','release-intent','released')}
    if not any(values.values()):return None
    require(all(isinstance(item,dict) for item in values.values()),'Partial mutation custody requires reconciliation')
    accepted=values['acceptance'];require(type(accepted.get('returncode')) is int and accepted['returncode']==0
        and re.fullmatch(r'[1-9][0-9]*(;[\w.-]+)?',accepted.get('stdout','').strip()),'Ambiguous actual audit acceptance')
    job=accepted['stdout'].strip().split(';')[0];dependency=values['release-intent'].get('dependency')
    require((phase=='qualify' and dependency is None) or (phase!='qualify' and re.fullmatch(r'afterok:[1-9][0-9]*',dependency or '')),
        'Unexpected phase dependency')
    require(values['intent']['argv']==submit_argv(phase,pin,dependency) and all(values[name].get('manifest_sha256')==pin
        for name in ('intent','release-intent','released')),'Original audit intent/source differs')
    require(all(values[name].get('job_id')==job for name in ('held','release-intent','released'))
        and values['released'].get('returncode')==0 and values['released'].get('release_attempted_exactly_once') is True,
        'Actual audit held/released identity differs')
    require(values['release-intent']['verified_held_record_sha256']==sha(CONTROL/(phase+'.held.json')),'Held receipt pin differs')
    if phase!='qualify':
        qualifier=custody('qualify',pin)
        require(qualifier is not None and dependency=='afterok:'+qualifier['job_id']
            and values['intent'].get('qualifier_job_id')==qualifier['job_id'],'Serial custody chain differs')
    validate_allocation(values['held']['raw'],job,phase,dependency,held=True,fulfilled=values['held'].get('dependency_already_qualified') is True)
    return {'job_id':job,'dependency':dependency,'values':values}


def accounting(job,phase):
    raw=observe(['sacct','-X','-n','-P','-j',str(job),'--format='+FORMAT]);rows=[line.split('|') for line in raw.splitlines() if line.strip()]
    require(len(rows)<=1,'Duplicate actual accounting row')
    if not rows:return None
    row=rows[0];require(len(row)==10 and row[:5]==[str(job),'pwa209',str(UID),'def-ptewarie_cpu','acs-cross-mechanistic-'+phase]
        and row[8]==str(POLICY[phase]['cpus']),'Actual audit accounting identity differs')
    return {'raw':raw,'row':row}


def guard_module():
    bridge=verify_tree(BRIDGE,'BRIDGE_RETRY_MANIFEST.json',BRIDGE_PIN)
    require(bridge['files']['bridge_common.py']==BRIDGE_COMMON_PIN,'Frozen canonical guard dependency differs')
    load_pinned('bridge_common',BRIDGE/'bridge_common.py',BRIDGE_COMMON_PIN)
    value=verify_tree(A4,'PROJECT_BYTE_ACCOUNTING_MANIFEST.json',A4_PIN)
    return load_pinned('_audit_original_A4_guard',A4/'project_bytes.py',value['files']['project_bytes.py'])


def qualification_proof(pin,job):
    proof=custody('qualify',pin);require(proof is not None and proof['job_id']==str(job),'Exact released qualifier required')
    actual=accounting(job,'qualify');require(actual is not None and actual['row'][5:9]==['COMPLETED','0:0','4','4'],
        'Actual qualifier has not completed successfully')
    value=read(CONTROL/'qualification.json');manifest=read(ROOT/MANIFEST);checked=value.get('tests',{})
    require(value.get('job_id')==str(job) and value.get('manifest_sha256')==pin
        and value.get('status')=='cross_mechanistic_system_qualified' and type(checked.get('passed')) is int
        and checked['passed']>=manifest['minimum_qualification_tests'] and checked.get('run')==checked['passed']
        and all(checked.get(key)==0 for key in ('failed','errors','skipped','expected_failures','unexpected_successes')),
        'Actual follow-up qualification receipt differs')
    probe=value.get('runtime_probe',{});capture=RUN/'qualify.runtime_probe.json';observed=read(capture)
    require(probe.get('capture_path')==str(capture) and probe.get('capture_sha256')==sha(capture)
        and all(probe.get(key)==item for key,item in observed.items()) and probe.get('returncode')==0
        and probe.get('error') is None and probe.get('truncated') is False and probe.get('command_retried') is False
        and probe.get('argv')==[str(PYTHON),'-B',str(analysis_driver(manifest)),'--runtime-check']
        and probe.get('proof')==json.loads(probe['stdout']),'Actual output-free analysis ABI probe differs')
    validate_runtime_proof(probe['proof'],manifest)
    test_context=value.get('qualification_source_context',{})
    require(test_context.get('frozen_discovery_source')==str(frozen_discovery_member(manifest))
        and test_context.get('sha256')==FROZEN_DISCOVERY_PIN and test_context.get('source_imported') is False
        and test_context.get('scientific_arrays_opened') is False,'Exact frozen descriptive test-source qualification differs')
    context_capture=RUN/'qualify.guard_context.json';context=value.get('guard_runtime_context',{})
    require(context.get('capture_path')==str(context_capture) and context.get('capture_sha256')==sha(context_capture)
        and context.get('proof')==read(context_capture) and context.get('proof',{}).get('status')=='original_A4_runtime_context_qualified'
        and context['proof'].get('bridge_manifest_sha256')==BRIDGE_PIN and context['proof'].get('canonical_bridge_source_sha256')==BRIDGE_COMMON_PIN,
        'Actual original guard runtime context did not qualify')
    schema_capture=RUN/'qualify.native_schema.json';schema=value.get('native_schema_probe',{})
    require(schema.get('capture_path')==str(schema_capture) and schema.get('capture_sha256')==sha(schema_capture)
        and schema.get('proof')==read(schema_capture)
        and schema['proof'].get('status')=='cross_mechanistic_actual_inputs_qualified'
        and schema['proof'].get('job_id')==str(job)
        and schema['proof'].get('source_sha256')==manifest['files']['native_probe.py']
        and schema['proof'].get('input_manifest_sha256')==manifest['files']['inputs.json']
        and schema['proof'].get('raw_labels_preserved') is True,
        'Actual saved native clock/pair qualification is absent or differs')
    capture=RUN/'qualify.storage.qualification.json'
    require(value['storage']==read(capture) and value['storage_capture_sha256']==sha(capture)
        and value['storage_capture_path']==str(capture),'Actual qualification storage capture differs')
    guard=guard_module();require(guard.sha(guard.PLAN)==guard.PLAN_SHA,'Original storage plan differs')
    guard.validate_snapshot(value['storage'],guard.read(guard.PLAN),RESERVES['qualify'])
    require(value['storage'].get('guard_admission_passed') is True and value['storage'].get('guard_admission_error') is None,
        'Original A4 full admission did not pass')
    prior_capture=RUN/'qualify.parent_completion.json';prior=value.get('parent_completion',{})
    require(prior.get('capture_path')==str(prior_capture) and prior.get('capture_sha256')==sha(prior_capture)
        and prior.get('proof')==read(prior_capture) and prior['proof'].get('status')=='completed_dream_parent_adopted'
        and prior['proof'].get('parent_manifest_sha256')==DREAM_PARENT_PIN
        and prior['proof'].get('dream_job_id')==DREAM_JOB and prior['proof'].get('qualifier_job_id')==DREAM_QUALIFIER_JOB
        and prior['proof'].get('scientific_receipt_sha256')==DREAM_SCIENTIFIC_PIN,
        'Actual completed DREAM prerequisite proof differs')
    return {'job_id':str(job),'actual_accounting':actual,'qualification_sha256':sha(CONTROL/'qualification.json')}


def validate_runtime_proof(proof,manifest):
    require(isinstance(proof,dict) and proof.get('status')==manifest['runtime_probe_status']
        and proof.get('driver_sha256')==manifest['files'][manifest['analysis_driver']]
        and proof.get('input_manifest_sha256')==manifest['files']['inputs.json']
        and all(proof.get(key) is False for key in ('scientific_arrays_opened','output_directories_created',
            'scientific_analysis_executed')),'Actual analysis source/input/output-free ABI qualification differs')
    return proof


def mutate_once(argv):
    env={key:value for key,value in os.environ.items() if not key.startswith(('SLURM_','SBATCH_')) and key not in {'TMPDIR','SLURM_TMPDIR'}}
    try:
        result=subprocess.run(argv,text=True,capture_output=True,timeout=30,stdin=subprocess.DEVNULL,env=env)
        return {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    except Exception as error:return {'returncode':None,'error':type(error).__name__+': '+str(error),'ambiguous':True}


def submit(phase,pin,qualifier_job=None):
    require(phase in POLICY,'Only fixed audit phases may be submitted');verify_packet(pin,full_upstream=False)
    with locked():
        state(create=phase=='qualify');require(maybe(CONTROL/(phase+'.intent.json')) is None,'Audit submission already attempted; never replay')
        dependency=None;allowed=[item['job_id'] for name in POLICY if (item:=custody(name,pin)) is not None];fulfilled=False
        if phase!='qualify':
            parent=custody('qualify',pin);require(parent is not None and parent['job_id']==str(qualifier_job),'Exact source-bound qualifier release required')
            dependency='afterok:'+parent['job_id']
            actual=accounting(qualifier_job,'qualify')
            require(actual is not None,'Qualifier accounting is unavailable; do not infer admission')
            if actual is not None and actual['row'][5]=='COMPLETED':qualification_proof(pin,qualifier_job);fulfilled=True
            elif actual is not None:require(actual['row'][5] in {'PENDING','RUNNING','COMPLETING','CONFIGURING'},'Qualifier already failed')
        before=queue_guard(allowed);argv=submit_argv(phase,pin,dependency)
        write_slot(CONTROL/(phase+'.intent.json'),{'utc':utc(),'manifest_sha256':pin,'argv':argv,'queue_before':before,
            'qualifier_job_id':str(qualifier_job) if phase!='qualify' else None,'prior_intents_replayed':False})
        accepted=mutate_once(argv);write_slot(CONTROL/(phase+'.acceptance.json'),{'utc':utc(),**accepted})
        require(type(accepted.get('returncode')) is int and accepted['returncode']==0
            and re.fullmatch(r'[1-9][0-9]*(;[\w.-]+)?',accepted.get('stdout','').strip()),'Ambiguous submission; preserve custody, never replay')
        job=accepted['stdout'].strip().split(';')[0];raw=observe(['scontrol','show','job',job,'-o'])
        allocation=validate_allocation(raw,job,phase,dependency,held=True,fulfilled=fulfilled)
        write_slot(CONTROL/(phase+'.held.json'),{'utc':utc(),'job_id':job,'raw':raw,'actual':allocation,'dependency_already_qualified':fulfilled})
        queue_guard([*allowed,job]);verify_packet(pin,full_upstream=False)
        write_slot(CONTROL/(phase+'.release-intent.json'),{'utc':utc(),'job_id':job,'manifest_sha256':pin,
            'dependency':dependency,'verified_held_record_sha256':sha(CONTROL/(phase+'.held.json'))})
        released=mutate_once(['scontrol','release',job])
        write_slot(CONTROL/(phase+'.released.json'),{'utc':utc(),'job_id':job,'manifest_sha256':pin,'dependency':dependency,
            'release_attempted_exactly_once':True,**released})
        require(type(released.get('returncode')) is int and released['returncode']==0,'Ambiguous release; retain original custody, never replay')
        return job


def worker_context(phase,pin,job):
    with locked(shared=True):
        state();proof=custody(phase,pin);require(proof is not None and proof['job_id']==job,'Own released audit custody required')
        known=[item['job_id'] for name in POLICY if (item:=custody(name,pin)) is not None]
        allocation=validate_allocation(observe(['scontrol','show','job',job,'-o']),job,phase,proof['dependency'],fulfilled=phase!='qualify')
        live=queue_guard(known)
    return {'allocation':allocation,'active_guard':live,'custody':proof}


def parent_accounting(job,phase,cpus):
    raw=observe(['sacct','-X','-n','-P','-j',str(job),'--format='+FORMAT])
    rows=[[part.strip() for part in line.split('|')] for line in raw.splitlines() if line.strip()]
    require(len(rows)==1 and len(rows[0])==10 and rows[0][:5]==
        [str(job),'pwa209',str(UID),'def-ptewarie_cpu','acs-results-audit-'+phase]
        and rows[0][5:9]==['COMPLETED','0:0',str(cpus),str(cpus)],
        'Actual original '+phase+' job has not completed successfully')
    return {'raw':raw,'row':rows[0]}


def validate_numerical_metadata(value,inputs,parent):
    require(value.get('status')=='existing_data_numerical_audit_completed'
        and value.get('analysis_source_sha256')==parent['files']['audit_analysis.py']
        and value.get('input_manifest_file_sha256')==parent['files']['inputs.json']
        and value.get('input_manifest_canonical_sha256')==digest(json.dumps(inputs,sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()),'Pinned numerical source/input identity differs')
    expected=inputs['files'];actual=value.get('inputs',{})
    require(len(expected)==34 and set(actual)==set(expected),'All34 numerical input roles must be preserved')
    for role,spec in expected.items():
        item=actual[role]
        require(item.get('path')==spec['path'] and item.get('sha256')==spec['sha256']
            and type(item.get('bytes')) is int and 0<=item['bytes']<=spec['max_bytes']
            and (spec['captured_bytes'] is None or item['bytes']==spec['captured_bytes']),
            'Numerical input role differs: '+role)
    helpers=('paired_inference','discovery_sensitivity','mechanism_sensitivity_v2')
    require(value.get('pure_helper_source_sha256')=={name:parent['files'][name+'.py'] for name in helpers}
        and all(value.get(key) is False for key in ('new_model_fitting_or_split_changes',
            'scientific_arrays_or_network_or_scheduler_used','new_independent_human_or_physiological_qualification')),
        'Original fixed-model numerical scope differs')
    products=value.get('products',{});require(set(products)==set(PRODUCTS),'All8 existing numerical outputs must be preserved')
    for name,item in products.items():
        require(item.get('path')==str(PARENT_RESULT/(name+'.json'))
            and re.fullmatch('[0-9a-f]{64}',item.get('sha256','')) and type(item.get('bytes')) is int
            and 0<item['bytes']<=2097152,'Exact small numerical product boundary differs: '+name)
    require(sum(item['bytes'] for item in products.values())<=4194304,'Numerical summary retrieval bound exceeded')
    return products


def verify_numerical_products(products):
    verified={}
    for name,item in products.items():
        raw=raw_file(Path(item['path']),item['bytes'])
        require(len(raw)==item['bytes'] and digest(raw)==item['sha256'],'Actual numerical output bytes differ: '+name)
        json.loads(raw)
        verified[name]={'path':item['path'],'sha256':digest(raw),'bytes':len(raw),'actual_bytes_verified':True}
    return verified


def existing_data_proof():
    """Scheduled metadata-only adoption; no original analysis phase can be submitted."""
    parent=verify_tree(PARENT,'AUDIT_MANIFEST.json',PARENT_PIN)
    oldq=parent_accounting(PARENT_QUALIFIER_JOB,'qualify',4)
    oldn=parent_accounting(PARENT_NUMERICAL_JOB,'analysis',8)
    qraw=raw_file(PARENT/'qualification.json');require(digest(qraw)==PARENT_QUALIFICATION_PIN,'Original qualification receipt bytes differ')
    q=json.loads(qraw);tests=q.get('tests',{})
    require(q.get('status')=='scientific_results_audit_qualified' and q.get('job_id')==PARENT_QUALIFIER_JOB
        and q.get('manifest_sha256')==PARENT_PIN and tests=={'run':91,'passed':91,'failed':0,'errors':0,
            'skipped':0,'expected_failures':0,'unexpected_successes':0},'Actual91-check original qualification differs')
    guard=guard_module();require(guard.sha(guard.PLAN)==guard.PLAN_SHA,'Frozen original1TB plan differs')
    plan=guard.read(guard.PLAN);guards={}
    for name,pin in PARENT_STORAGE_PINS.items():
        raw=raw_file(PARENT_RUN/name);require(digest(raw)==pin,'Preserved original storage capture differs: '+name)
        snapshot=json.loads(raw);guard.validate_snapshot(snapshot,plan,30128)
        require(snapshot.get('guard_admission_passed') is True and snapshot.get('guard_admission_error') is None
            and snapshot.get('project_byte_guard_manifest_sha256')==A4_PIN,'Original full storage admission differs')
        guards[name]={'sha256':pin,'project_entries':snapshot['project_entries'],'validated_by_original_A4':True}
        if name=='qualify.storage.qualification.json':
            require(q['storage']==snapshot and q['storage_capture_path']==str(PARENT_RUN/name)
                and q['storage_capture_sha256']==pin,'Original qualifier storage binding differs')
    inputs_raw=raw_file(PARENT/'inputs.json');require(digest(inputs_raw)==parent['files']['inputs.json'],'Original inputs manifest differs')
    numerical_raw=raw_file(PARENT_RESULT/'receipt.json');require(digest(numerical_raw)==PARENT_NUMERICAL_PIN,
        'Actual completed numerical scientific receipt differs')
    numerical=json.loads(numerical_raw);products=validate_numerical_metadata(numerical,json.loads(inputs_raw),parent)
    verified=verify_numerical_products(products)
    return {'status':'prior_qualified_numerical_outputs_verified','parent_manifest_sha256':PARENT_PIN,
        'qualifier_job_id':PARENT_QUALIFIER_JOB,'qualifier_actual_accounting':oldq,'qualifier_receipt_sha256':digest(qraw),
        'numerical_job_id':PARENT_NUMERICAL_JOB,'numerical_actual_accounting':oldn,
        'numerical_scientific_receipt_sha256':digest(numerical_raw),'input_roles_verified':34,'products':verified,
        'preserved_storage_captures':guards,'prior_home_execution_receipt_read':False,
        'prior_home_execution_receipt_unavailable_reason':'Observed original HOME analysis_receipt inode EPIPE108',
        'original_numerical_job_resubmitted':False,'new_independent_human_evidence_claimed':False}


def pinned_json(path,pin):
    raw=raw_file(path);require(digest(raw)==pin,'Pinned completed parent metadata differs: '+str(path))
    return json.loads(raw)


def completed_dream_proof():
    """Adopt completed parent custody and metadata; no conversion or array is opened."""
    manifest=verify_tree(DREAM_PARENT,'DREAM_REPAIR_MANIFEST.json',DREAM_PARENT_PIN)
    parent=load_pinned('_followup_completed_parent_ops',DREAM_PARENT/'audit_ops.py',manifest['files']['audit_ops.py'])
    parent.verify_packet(DREAM_PARENT_PIN)
    qualified=parent.qualification_proof(DREAM_PARENT_PIN,DREAM_QUALIFIER_JOB)
    released=parent.custody('dream',DREAM_PARENT_PIN)
    require(released is not None and released['job_id']==DREAM_JOB
        and released['dependency']=='afterok:'+DREAM_QUALIFIER_JOB,'Exact completed DREAM parent custody differs')
    actual=parent.accounting(DREAM_JOB,'dream')
    require(actual is not None and actual['row'][5:9]==['COMPLETED','0:0','8','8'],
        'Actual completed DREAM parent accounting differs')
    q=pinned_json(DREAM_CONTROL/'qualification.json',DREAM_QUALIFICATION_PIN)
    execution=pinned_json(DREAM_CONTROL/'dream_receipt.json',DREAM_EXECUTION_PIN)
    child=pinned_json(DREAM_CONTROL/'dream.child.execution.json',DREAM_CHILD_PIN)
    science=pinned_json(DREAM_RESULT/'receipt.json',DREAM_SCIENTIFIC_PIN)
    require(q['job_id']==DREAM_QUALIFIER_JOB and q['manifest_sha256']==DREAM_PARENT_PIN
        and qualified['qualification_sha256']==DREAM_QUALIFICATION_PIN
        and execution['status']=='dream_startup_repair_completed' and execution['job_id']==DREAM_JOB
        and execution['manifest_sha256']==DREAM_PARENT_PIN and execution['output_root']==str(DREAM_RESULT)
        and execution['qualifier']==qualified,'Completed parent qualification/execution identity differs')
    command=execution['command']
    require(command['capture_path']==str(DREAM_CONTROL/'dream.child.execution.json') and command['capture_sha256']==DREAM_CHILD_PIN
        and all(command.get(key)==value for key,value in child.items())
        and type(child.get('returncode')) is int and child['returncode']==0 and child.get('error') is None
        and child.get('traceback') is None and child.get('command_retried') is False
        and child.get('scheduled_child_jobs_submitted') is False,
        'Completed parent scientific child did not qualify')
    stdout=child['stdout']
    require(stdout.get('complete_file') is True and stdout.get('offset')==0
        and digest(stdout['text'].encode())==stdout['sha256_of_captured_bytes']
        and len(stdout['text'].encode())==stdout['file_bytes']==stdout['captured_bytes']
        and json.loads(stdout['text'])==science,'Completed DREAM scientific receipt/actual stdout binding differs')
    require(science.get('status')=='dream_status_aux_extension_completed' and science.get('slurm_job_id')==DREAM_JOB
        and science.get('headers_audited')==287 and science.get('headers_qualified')==256
        and science.get('headers_not_qualified')==31 and science.get('original_candidate_episodes')==287
        and science.get('original_source_people')==39 and science.get('converted_recordings')==256
        and science.get('converted_participants')==35 and science.get('eligible_source_people')==35
        and science.get('metadata_dispositions_sha256')==DREAM_METADATA_PIN,
        'Completed DREAM case/person/disposition counts differ')
    require(all(science.get(key) is True for key in ('all_metadata_dispositions_bound','all_metadata_eligible_headers_qualified',
        'all_original_headers_audited','archive_extracted_all287_headers_reaudited','original_schema_exclusions_only',
        'native_frozen_clinical_instrument','source_stage_aware_discovery'))
        and all(science.get(key) is False for key in ('all_original_headers_qualified','human_functional_mapping_qualified',
            'human_intervention_qualified','new_downloads','new_independent_human_evidence','prior_outputs_modified','result_direction_gate'))
        and execution.get('results_or_hypotheses_used_as_execution_gate') is False
        and execution.get('previous_outputs_modified') is False,'Completed parent scientific scope differs')
    stages=science['stages']
    require([stage['stage'] for stage in stages]==['convert','prepare','native_score','discover','trajectories','source_stage_discovery']
        and all(type(stage.get('returncode')) is int and stage['returncode']==0 for stage in stages),
        'All six original DREAM stages must already have completed')
    guard=guard_module();require(guard.sha(guard.PLAN)==guard.PLAN_SHA,'Frozen original1TB plan differs')
    plan=guard.read(guard.PLAN);verified={}
    for name,pin in DREAM_STORAGE_PINS.items():
        snapshot=pinned_json(DREAM_CONTROL/name,pin);guard.validate_snapshot(snapshot,plan,30128)
        require(snapshot.get('guard_admission_passed') is True and snapshot.get('guard_admission_error') is None
            and snapshot.get('project_byte_guard_manifest_sha256')==A4_PIN,'Completed DREAM original A4 admission differs')
        if name.startswith('qualify.'):
            require(q['storage']==snapshot and q['storage_capture_path']==str(DREAM_CONTROL/name)
                and q['storage_capture_sha256']==pin,'Parent qualifier storage binding differs')
        else:
            key='before_storage' if '.before.' in name else 'after_storage';embedded=execution[key]
            require(embedded['snapshot']==snapshot and embedded['capture']==str(DREAM_CONTROL/name)
                and embedded['sha256']==pin,'Parent DREAM before/after storage binding differs')
        verified[name]={'sha256':pin,'validated_by_original_A4':True,'historical_admission_only':True}
    numerical=existing_data_proof()
    return {'status':'completed_dream_parent_adopted','parent_manifest_sha256':DREAM_PARENT_PIN,
        'qualifier_job_id':DREAM_QUALIFIER_JOB,'qualification_receipt_sha256':DREAM_QUALIFICATION_PIN,
        'dream_job_id':DREAM_JOB,'actual_accounting':actual,'execution_receipt_sha256':DREAM_EXECUTION_PIN,
        'child_execution_sha256':DREAM_CHILD_PIN,'scientific_receipt_sha256':DREAM_SCIENTIFIC_PIN,
        'historical_guard_captures':verified,'original_numerical_prerequisite':numerical,
        'parent_jobs_resubmitted':False,'scientific_arrays_opened':False}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=POLICY);parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--qualifier-job');args=parser.parse_args()
    print(json.dumps({'phase':args.phase,'job_id':submit(args.phase,args.manifest_sha256,args.qualifier_job)}),flush=True)
