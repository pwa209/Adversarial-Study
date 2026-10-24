"""Create a source-hashed archive and explicit authorized Narval deployment request.

The request is submitted through the established authenticated file bridge. It
never includes login credentials or modifies other research projects.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def package(root, output):
    files = {}
    pipeline = root / "study_pipeline"
    for folder in ("adversarial_study", "native_tools", "matlab", "configs", "tests", "cluster", "docs", "data_templates", "tools"):
        for path in (pipeline / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".m", ".json", ".md", ".sh", ".csv"}:
                files[path.relative_to(root).as_posix()] = path.read_bytes().replace(b"\r\n", b"\n")
    for name in ("run_study.py", "README.md", "pyproject.toml", "BUILD_MANIFEST.json"):
        path = pipeline / name
        if path.exists(): files[path.relative_to(root).as_posix()] = path.read_bytes().replace(b"\r\n", b"\n")
    for name in ("Adversarial_Consciousness_Research_Proposal.md", "Study_Design_and_Decisions.md", "Scientific_Revision_20261004.md", "Research_Question_and_Scope.md", "Original_Study_Protocol.md"):
        path = root / name
        if path.exists(): files[name] = path.read_bytes().replace(b"\r\n", b"\n")
    for path in (root / "source_records/extensions").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".md", ".txt", ".bin"}:
            payload = path.read_bytes()
            files[path.relative_to(root).as_posix()] = payload if path.suffix == ".bin" else payload.replace(b"\r\n", b"\n")
    # Preserve the exact original released bytes, including line endings.
    for name in ("figshare_v2_metadata.json", "trained_consciousness_detector.mat", "predict_consciousness_level.m",
                 "remap_gcs_scale.m", "BoundedOutputLayer.m", "conscious_brain_configuration_file.conf"):
        path = root / "source_records/toker_v2" / name
        if not path.is_file(): raise ValueError("Missing pinned release " + name)
        files[path.relative_to(root).as_posix()] = path.read_bytes()
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "git_repository": False,
        "files": {name: hashlib.sha256(value).hexdigest() for name, value in sorted(files.items())}}
    files["SOURCE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    archive = output / "study-source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, payload in sorted(files.items()):
            entry = tarfile.TarInfo(name); entry.size = len(payload)
            entry.mode = 0o755 if name.endswith(".sh") else 0o644
            tar.addfile(entry, io.BytesIO(payload))
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + archive_sha[:12]
    payload = base64.b64encode(archive.read_bytes()).decode()
    # Controlled absolute roots and immutable release names; no arbitrary path deletion.
    request = f'''set -euo pipefail
[[ "$(id -un)" == pwa209 ]] || exit 9
case "$(hostname -f)" in *.narval.calcul.quebec) ;; *) exit 10;; esac
export PYTHONDONTWRITEBYTECODE=1
project=/home/pwa209/adversarial-study
scratch=/scratch/pwa209/adversarial-study
tag={tag}
release="$project/releases/$tag"
run="$project/runs/$tag"
test ! -e "$release"
test ! -e "$run"
mkdir -p "$project/incoming" "$project/releases" "$project/runs" "$scratch/data" "$scratch/normalized" "$scratch/native" "$scratch/environments" "$scratch/results/$tag"
archive="$project/incoming/$tag.tar.gz"
test ! -e "$archive"
base64 -d > "$archive" <<'ACS_SOURCE_BASE64'
{payload}
ACS_SOURCE_BASE64
printf '%s  %s\\n' '{archive_sha}' "$archive" | sha256sum --check
mkdir "$release" "$run"
mkdir "$run/logs" "$run/receipts" "$run/qualification"
python3 - "$archive" "$release" <<'PYACS_EXTRACT'
import pathlib,sys,tarfile
archive, destination = sys.argv[1:]
with tarfile.open(archive) as stream:
    members = stream.getmembers()
    for item in members:
        path = pathlib.PurePosixPath(item.name)
        if not item.isfile() or path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe source archive member')
    stream.extractall(destination, members=members)
PYACS_EXTRACT
module load StdEnv/2023 python/3.11.5
python "$release/study_pipeline/cluster/make_plan.py" --source "$release" --tag "$tag" --output "$run/execution_plan.json"
python - "$release" "$run" <<'PYACS_VERIFY'
import pathlib,sys,hashlib,json,importlib.util
source, run = map(pathlib.Path,sys.argv[1:])
manifest = json.loads((source/'SOURCE_MANIFEST.json').read_text())
for name, digest in manifest['files'].items():
    if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
        raise ValueError('Changed source '+name)
spec=importlib.util.spec_from_file_location('controller',source/'study_pipeline/cluster/controller.py')
controller=importlib.util.module_from_spec(spec); spec.loader.exec_module(controller)
controller.validate_plan(json.loads((run/'execution_plan.json').read_text()))
with (run/'controller_submission.intent.json').open('x') as handle:
    json.dump({{'tag':run.name,'plan_sha256':hashlib.sha256((run/'execution_plan.json').read_bytes()).hexdigest()}},handle)
PYACS_VERIFY
set +e
answer=$(sbatch --parsable --nodes=1 --ntasks=1 --no-requeue --account=def-ptewarie_cpu --cpus-per-task=1 --mem=4G --time=48:00:00 --job-name=acs-controller --comment="acs-controller-$tag" --output="$run/logs/controller-%j.out" --chdir="$release" "$release/study_pipeline/cluster/controller.sh" "$run/execution_plan.json" 2> "$run/controller_submission.stderr")
submit_status=$?
set -e
printf '%s\\n' "$answer" > "$run/controller_submission.stdout"
python - "$answer" "$submit_status" "$run" <<'PYACS_ACK'
import json,sys,re,pathlib,datetime
answer,status,root=sys.argv[1:]
accepted=int(status)==0 and re.fullmatch(r'\\d+(;[\\w.-]+)?',answer) is not None
record={{'status':'ACCEPTED' if accepted else 'OUTCOME_UNKNOWN_RECONCILE_BEFORE_RETRY',
         'job_id':answer.split(';')[0] if accepted else None,'stdout':answer,'returncode':int(status),
         'run_root':root,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}
with (pathlib.Path(root)/'controller_submission.receipt.json').open('x') as handle: json.dump(record,handle,indent=2)
print(json.dumps(record)); sys.exit(0 if accepted else 2)
PYACS_ACK
python - "$run" <<'PYACS_TRANSFER_CLAIM'
import pathlib,sys,json
run=pathlib.Path(sys.argv[1]); (run/'transfer').mkdir(exist_ok=True)
with (run/'transfer_launch.intent.json').open('x') as handle:
    json.dump({{'run':str(run),'execution_kind':'bounded HTTPS file management only; numerical/controller work remains scheduled'}},handle)
PYACS_TRANSFER_CLAIM
export PYTHONPATH="$release/study_pipeline"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup nice -n 19 python "$release/study_pipeline/cluster/transfer.py" run --plan "$run/execution_plan.json" --wait-timeout-s 28800 > "$run/logs/transfer.log" 2>&1 < /dev/null &
transfer_pid=$!
python - "$transfer_pid" "$run" <<'PYACS_TRANSFER_ACK'
import pathlib,sys,json,datetime
pid,root=sys.argv[1:]
record={{'pid':int(pid),'status':'STARTED_CHECK_WORKFLOW_RECEIPT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'kind':'bounded_https_file_management','niceness':19,'run':root}}
with (pathlib.Path(root)/'transfer_launch.receipt.json').open('x') as handle:json.dump(record,handle,indent=2)
print(json.dumps(record))
PYACS_TRANSFER_ACK
'''
    request_path = output / ("adversarial_deploy_" + tag.replace("-", "_") + ".sh")
    request_path.write_text(request, encoding="utf-8", newline="\n")
    receipt = {"tag": tag, "archive": str(archive.resolve()), "archive_sha256": archive_sha,
        "source_manifest_sha256": hashlib.sha256(files["SOURCE_MANIFEST.json"]).hexdigest(),
        "files": len(manifest["files"]), "bytes": archive.stat().st_size,
        "request": str(request_path.resolve()), "request_id": request_path.stem,
        "remote_source": "/home/pwa209/adversarial-study/releases/" + tag,
        "remote_run": "/home/pwa209/adversarial-study/runs/" + tag,
        "remote_scratch": "/scratch/pwa209/adversarial-study"}
    (output / "package_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    package(args.root.resolve(), args.output.resolve())


if __name__ == "__main__": main()
