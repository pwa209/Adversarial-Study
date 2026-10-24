"""Execute one source-pinned Slurm stage and retain its actual completion receipt."""
from __future__ import annotations
import argparse
import getpass
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_source(source):
    manifest = json.loads((source / "SOURCE_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or digest(path) != expected:
            raise ValueError(f"Changed deployed source: {name}")


def parse_personal_quota(stdout, username, personal_root):
    """Parse only the requested user's exact scratch summary, never human units.

    Lustre's default kbytes are KiB and files are quota-counted inodes. A long
    filesystem pathname can appear on its own line. OST/MDT breakdowns, missing
    counts, rounded human-readable quantities and ambiguous rows are rejected.
    """
    if not re.search(r"Disk quotas for (?:usr|user)\s+"+re.escape(username)+r"\s+\(uid\s*\d+\)",stdout):
        raise ValueError("Personal quota output does not identify the actual user")
    if not re.search(r"Filesystem\s+kbytes\s+quota\s+limit\s+grace\s+files\s+quota\s+limit\s+grace",stdout):
        raise ValueError("Unexpected personal Lustre quota columns/units")
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    rows=[]
    for index,line in enumerate(lines):
        fields=line.split()
        if fields[0] not in (personal_root,"/scratch"):
            continue
        values=fields[1:]
        if not values:
            if index+1>=len(lines):
                raise ValueError("Wrapped quota row has no counts")
            values=lines[index+1].split()
        if len(values)!=8:
            raise ValueError("Personal Lustre quota row has unexpected field count")
        parsed={}
        for column,name in ((0,"used_kib"),(1,"soft_kib"),(2,"hard_kib"),(4,"used_inodes"),(5,"soft_inodes"),(6,"hard_inodes")):
            pattern=r"[0-9]+\*?" if column in (0,4) else r"[0-9]+"
            if not re.fullmatch(pattern,values[column]):
                raise ValueError("Personal quota count is missing, rounded or malformed")
            value=int(values[column].rstrip("*"))
            if value>2**63-1:
                raise ValueError("Personal quota count exceeds supported integer range")
            parsed[name]=value
        if any(not re.fullmatch(r"-|none|0|[0-9]+[A-Za-z0-9:.]*",values[i],re.I) for i in (3,7)):
            raise ValueError("Personal quota grace field is malformed")
        parsed.update(personal_allocated_bytes=parsed["used_kib"]*1024,
                      personal_inode_count=parsed["used_inodes"],
                      quota_usage_marked_over_limit=values[0].endswith("*") or values[4].endswith("*"))
        rows.append(parsed)
    if len(rows)!=1:
        raise ValueError("Personal scratch quota summary is missing or ambiguous")
    return rows[0]


def storage_snapshot(root, *, timeout_s=30):
    """Bounded personal quota guard, without walking old study environments.

    Whole-user allocated storage/inodes conservatively guard this study. They
    are not measurements of this study's logical file bytes/directory entries.
    UNKNOWN is never converted into zero or an allowance pass. Temporary output
    files avoid an unbounded post-timeout communicate() waiting for pipe EOF.
    """
    snapshot={"status":"UNKNOWN","root":str(root),"checked_utc":stamp(),
              "method":"bounded Lustre personal scratch allocated-byte/inode quota; no recursive traversal",
              "scope":"personal_scratch_all_projects_not_exact_own_study",
              "personal_allocated_bytes":None,"personal_inode_count":None,"timeout_s":timeout_s}
    process=None
    try:
        if not re.fullmatch(r"[0-9]+",os.environ.get("SLURM_JOB_ID","")):
            raise ValueError("Storage audit requires an actual scheduled Slurm job")
        if not isinstance(timeout_s,(int,float)) or not 0<timeout_s<=30:
            raise ValueError("Storage quota timeout must be at most30seconds")
        username=getpass.getuser()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+",username):
            raise ValueError("Unsupported actual username for scratch quota")
        personal_root="/scratch/"+username
        declared=PurePosixPath(str(root))
        if not declared.is_absolute() or ".." in declared.parts or not declared.is_relative_to(PurePosixPath(personal_root)):
            raise ValueError("Study scratch root is outside the actual user's scratch")
        # POSIX resolve checks any local path aliases before targeting quota.
        # Unit parser tests run on Windows without pretending it is Narval.
        if os.name=="posix" and not Path(root).resolve().is_relative_to(Path(personal_root).resolve()):
            raise ValueError("Resolved study scratch root escapes personal scratch")
        command=["lfs","quota","-u",username,"/scratch"]
        snapshot.update(username=username,personal_root=personal_root,command=command)
        with tempfile.TemporaryFile(mode="w+b") as stdout_file,tempfile.TemporaryFile(mode="w+b") as stderr_file:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout_file,stderr=stderr_file,
                                     start_new_session=True)
            snapshot["quota_pid"]=process.pid
            try:
                exit_code=process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                snapshot.update(error="Personal Lustre quota command exceeded bounded timeout",
                                quota_process_kill_requested=True)
                return snapshot
            stdout_file.seek(0);stderr_file.seek(0)
            stdout=stdout_file.read(65537);stderr=stderr_file.read(65537)
            if len(stdout)>65536 or len(stderr)>65536:
                raise ValueError("Unexpectedly large personal quota output")
            stdout=stdout.decode("utf-8",errors="strict");stderr=stderr.decode("utf-8",errors="strict")
        snapshot.update(exit_code=exit_code,stdout=stdout,stderr=stderr)
        if exit_code!=0:
            raise ValueError("Personal Lustre quota command failed")
        if stderr.strip():
            raise ValueError("Personal Lustre quota returned a warning/error; counts not certified")
        snapshot.update(parse_personal_quota(stdout,username,personal_root),status="VERIFIED_PERSONAL_QUOTA")
    except Exception as error:
        snapshot["error"]=f"{type(error).__name__}: {error}"
    return snapshot


def guard_storage(snapshot,plan):
    if snapshot.get("status")!="VERIFIED_PERSONAL_QUOTA":
        raise ValueError("Storage accounting UNKNOWN; stage is not admitted")
    if (snapshot["personal_inode_count"]>=plan["entry_allowance"] or
            snapshot["personal_allocated_bytes"]>=plan["storage_allowance_bytes"]):
        raise ValueError("Conservative personal scratch storage/inode guard reached study allowance; exact own-study allocation is not asserted")


def run(plan_path, stage_id):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Research stages require a Slurm allocation")
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    source = Path(plan["source"]).resolve()
    stage = next(item for item in plan["stages"] if item["id"] == stage_id)
    state_root = Path(plan["run_root"]) / "receipts"
    receipt_path = state_root / (stage_id + ".json")
    if receipt_path.exists():
        raise RuntimeError("Stage has a completion receipt; preserve it rather than rerun")
    state_root.mkdir(parents=True, exist_ok=True)
    claim_path = state_root / (stage_id + ".claim.json")
    with claim_path.open("x", encoding="utf-8") as claim:
        json.dump({"job_id": os.environ["SLURM_JOB_ID"], "utc": stamp()}, claim)
    values = {"source": str(source), "pipeline": str(source / "study_pipeline"),
              "project": plan["project"], "scratch": plan["scratch"],
              "run": plan["run_root"], "python": sys.executable}
    command = []
    for argument in stage["command"]:
        for key, value in values.items():
            argument = argument.replace("{" + key + "}", value)
        command.append(argument)
    receipt = {"stage": stage_id, "job_id": os.environ["SLURM_JOB_ID"],
        "started_utc": stamp(), "plan_sha256": digest(plan_path),
        "source_manifest_sha256": digest(source / "SOURCE_MANIFEST.json"),
        "command": command, "host": platform.node(), "requested_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "evidence_role": stage.get("evidence_role", "runtime_qualification")}
    save(state_root / (stage_id + ".started.json"), receipt)
    try:
        if digest(source / "SOURCE_MANIFEST.json") != plan["source_manifest_sha256"]:
            raise ValueError("Source manifest differs from plan")
        verify_source(source)
        receipt["storage_before"] = storage_snapshot(plan["scratch"])
        guard_storage(receipt["storage_before"],plan)
        save(state_root / (stage_id + ".started.json"),receipt)
        journal = Path(plan["run_root"]) / "execution_state.json"
        for attempt in range(60):
            journal_record = json.loads(journal.read_text())
            if journal_record["plan_sha256"] != receipt["plan_sha256"]:
                raise ValueError("Plan differs from the accepted controller journal")
            accepted = journal_record["stages"][stage_id]
            if accepted.get("job_id") == os.environ["SLURM_JOB_ID"] and accepted["state"] == "ACCEPTED":
                break
            if accepted["state"] != "SUBMISSION_INTENT":
                raise ValueError("Allocation does not match accepted submission journal")
            time.sleep(.5)
        else:
            raise ValueError("Controller acceptance was not committed after 30 seconds")
        result = subprocess.run(command, cwd=source / "study_pipeline", check=False)
        receipt.update(status="completed" if result.returncode == 0 else "failed", exit_code=result.returncode)
    except Exception as error:
        receipt.update(status="failed", exit_code=2, error=f"{type(error).__name__}: {error}")
    import resource
    receipt.update(finished_utc=stamp(), max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    receipt["storage_after"] = storage_snapshot(plan["scratch"])
    if receipt["storage_after"].get("status")!="VERIFIED_PERSONAL_QUOTA":
        receipt["storage_audit_limitation"]="Post-execution storage accounting UNKNOWN; actual execution exit/status retained, next stage admission requires a fresh verified quota"
    else:
        try:
            guard_storage(receipt["storage_after"],plan)
        except ValueError as error:
            receipt["storage_audit_limitation"]=str(error)
    save(receipt_path, receipt)
    return receipt["exit_code"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.plan, args.stage))
