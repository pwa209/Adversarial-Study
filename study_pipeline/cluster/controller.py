"""Finite scheduled DAG controller with submission journals and no result gates."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def inspect_job(job_id):
    result = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"], text=True, capture_output=True)
    states = result.stdout.strip().splitlines()
    if result.returncode == 0 and states:
        return states[0]
    result = subprocess.run(["sacct", "-X", "-n", "-P", "-j", str(job_id), "--format=State%40,ExitCode"], text=True, capture_output=True)
    if result.returncode != 0 or not result.stdout.strip():
        return "WAITING_ACCOUNTING"
    return result.stdout.strip().splitlines()[0].split("|")[0].split()[0]


def validate_plan(plan):
    budget = plan["max_allocated_cpus"]
    controller_cpus = plan.get("controller_cpus", 1)
    if not isinstance(budget, int) or not isinstance(controller_cpus, int) or not 0 < controller_cpus < budget:
        raise ValueError("Invalid CPU budget")
    stages = plan["stages"]
    gpu_budget = plan.get("max_allocated_gpus", 0)
    if not isinstance(gpu_budget, int) or not 0 <= gpu_budget <= 2:
        raise ValueError("Invalid GPU budget")
    ids = [stage["id"] for stage in stages]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z][a-z0-9_]{0,60}", name) for name in ids):
        raise ValueError("Stage IDs must be unique and safe")
    dependencies = {stage["id"]: set(stage.get("depends_on", [])) for stage in stages}
    for stage in stages:
        if dependencies[stage["id"]] - set(ids):
            raise ValueError("Unknown dependency")
        if stage.get("external_input"):
            continue
        if (not isinstance(stage["cpus"], int) or not 0 < stage["cpus"] <= budget - controller_cpus
                or not isinstance(stage.get("command"), list) or not stage["command"]
                or not all(isinstance(value, str) for value in stage["command"])):
            raise ValueError("Invalid executable stage resources or argv")
        gpus = stage.get("gpus", 0)
        if not isinstance(gpus, int) or not 0 <= gpus <= gpu_budget:
            raise ValueError("Stage exceeds GPU budget")
        if gpus and (stage.get("gpu_type") != "a100" or stage.get("account") != plan.get("gpu_account")):
            raise ValueError("GPU stage must use declared A100 allocation/account")
    done, ancestors = set(), {}
    while len(done) < len(ids):
        ready = {name for name, parents in dependencies.items() if name not in done and parents <= done}
        if not ready:
            raise ValueError("Cyclic stage dependencies")
        for name in ready:
            ancestors[name]=set(dependencies[name])
            for parent in dependencies[name]:
                ancestors[name].update(ancestors[parent])
        done.update(ready)
    policy=plan.get("cross_system_policy",{})
    if not isinstance(policy,dict):
        raise ValueError("Cross-system policy must be a structured object")
    if policy.get("human_finetuning_before_comparison") is True:
        by_id={stage["id"]:stage for stage in stages}
        protected_roles={"human_finetuned_cross_system_comparison","post_finetuning_protected_human_prediction"}
        for stage in stages:
            if stage.get("evidence_role") not in protected_roles:
                continue
            required=stage.get("required_human_training_stages")
            if (not isinstance(required,list) or not required or
                    any(not isinstance(name,str) or not name for name in required) or len(set(required))!=len(required)):
                raise ValueError("Protected prediction/comparison must declare distinct required human weight-training stages")
            for name in required:
                if name not in by_id:
                    raise ValueError("Required human weight-training stage is absent from the plan")
                training=by_id[name]
                if training.get("evidence_role")!="native_human_data_weight_finetuning" or training.get("external_input"):
                    raise ValueError("Comparison preconditions require actual native human-data weight fine-tuning stages")
                if name not in ancestors[stage["id"]]:
                    raise ValueError("Every required human weight-training stage must precede protected prediction/comparison through the DAG")
    source = Path(plan["source"])
    if hashlib.sha256((source / "SOURCE_MANIFEST.json").read_bytes()).hexdigest() != plan["source_manifest_sha256"]:
        raise ValueError("Deployed source manifest differs from the execution plan")


def cycle(plan_path, state_path, state):
    plan = json.loads(plan_path.read_text())
    if state["plan_sha256"] != hashlib.sha256(plan_path.read_bytes()).hexdigest():
        raise RuntimeError("Execution plan changed; deploy a new version")
    run_root = Path(plan["run_root"])
    for stage in plan["stages"]:
        item = state["stages"][stage["id"]]
        if item["state"] != "ACCEPTED":
            continue
        receipt_path = run_root / "receipts" / (stage["id"] + ".json")
        scheduler = inspect_job(item["job_id"])
        item["scheduler_state"] = scheduler
        terminal_failures = {"FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (receipt["job_id"] != item["job_id"] or receipt["plan_sha256"] != state["plan_sha256"]):
                raise RuntimeError("Stage receipt does not match the accepted allocation")
            if scheduler == "COMPLETED" or scheduler in terminal_failures:
                item.update(state="COMPLETED" if receipt["exit_code"] == 0 and scheduler == "COMPLETED" else "FAILED", receipt=str(receipt_path))
        else:
            if scheduler in terminal_failures:
                item.update(state="FAILED", reason=f"Slurm {scheduler}; retain the allocation and logs")
            elif scheduler == "COMPLETED":
                item["missing_receipt_polls"] = item.get("missing_receipt_polls", 0) + 1
                if item["missing_receipt_polls"] >= 3:
                    item.update(state="FAILED", reason="Slurm completed without a matching worker receipt")
    reserved_cpus = plan.get("controller_cpus", 1) + sum(stage["cpus"] for stage in plan["stages"] if state["stages"][stage["id"]]["state"] == "ACCEPTED")
    reserved_gpus = sum(stage.get("gpus", 0) for stage in plan["stages"] if state["stages"][stage["id"]]["state"] == "ACCEPTED")
    for stage in plan["stages"]:
        item = state["stages"][stage["id"]]
        if item["state"] != "WAITING":
            continue
        parents = [state["stages"][key]["state"] for key in stage.get("depends_on", [])]
        if any(parent in {"FAILED", "BLOCKED_UPSTREAM"} for parent in parents):
            item.update(state="BLOCKED_UPSTREAM", reason="An executed prerequisite failed; no automatic replay")
            continue
        if not all(parent == "COMPLETED" for parent in parents):
            continue
        if stage.get("transfer_group"):
            transfer_root = run_root / "transfer"
            transfer_receipt = transfer_root / (stage["transfer_group"] + ".complete.json")
            if not transfer_receipt.exists():
                failures = []
                for path in transfer_root.glob("*.failed.*.json"):
                    failure = json.loads(path.read_text())
                    if (failure.get("group") == stage["transfer_group"] and failure.get("plan_sha256") == state["plan_sha256"] and failure.get("source_manifest_sha256") == plan["source_manifest_sha256"]):
                        failures.append(str(path))
                if failures:
                    item.update(state="BLOCKED_UPSTREAM", reason="Bounded transfer stopped before this cache was completed", transfer_failure_receipts=failures)
                else:
                    item["transfer_state"] = "WAITING_BOUNDED_HTTPS_TRANSFER"
                continue
            transfer = json.loads(transfer_receipt.read_text())
            if (transfer.get("status") != "complete_verified" or transfer.get("group") != stage["transfer_group"]
                    or transfer.get("plan_sha256") != state["plan_sha256"]
                    or transfer.get("source_manifest_sha256") != plan["source_manifest_sha256"]):
                raise RuntimeError("Transfer receipt does not match the immutable source and plan")
            item["transfer_receipt"] = str(transfer_receipt)
        if reserved_cpus + stage["cpus"] > plan["max_allocated_cpus"]:
            continue
        if reserved_gpus + stage.get("gpus", 0) > plan.get("max_allocated_gpus", 0):
            continue
        comment = "acs-" + state["plan_sha256"][:12] + "-" + stage["id"]
        item.update(state="SUBMISSION_INTENT", comment=comment, intended_utc=stamp())
        save(state_path, state)
        command = ["sbatch", "--parsable", "--account=" + stage.get("account", plan["account"]),
            "--nodes=1", "--ntasks=1", "--no-requeue",
            "--job-name=acs-" + stage["id"], "--comment=" + comment,
            "--cpus-per-task=" + str(stage["cpus"]), "--mem=" + stage["memory"],
            "--time=" + stage["walltime"], "--output=" + str(run_root / "logs" / (stage["id"] + "-%j.out")),
            "--chdir=" + plan["source"]]
        if stage.get("gpus", 0):
            command += ["--gres=gpu:" + stage["gpu_type"] + ":" + str(stage["gpus"])]
        command += [plan["source"] + "/study_pipeline/cluster/worker.sh", str(plan_path), stage["id"]]
        result = subprocess.run(command, text=True, capture_output=True)
        item.update(submission_stdout=result.stdout.strip(), submission_stderr=result.stderr.strip(), submission_returncode=result.returncode)
        if result.returncode == 0 and re.fullmatch(r"\d+(;[\w.-]+)?", result.stdout.strip()):
            item.update(state="ACCEPTED", job_id=result.stdout.strip().split(";")[0], accepted_utc=stamp())
            reserved_cpus += stage["cpus"]
            reserved_gpus += stage.get("gpus", 0)
        else:
            item.update(state="SUBMISSION_OUTCOME_UNKNOWN", reason="Inspect scheduler by unique comment before recovery; never replay automatically")
            save(state_path, state)
            raise RuntimeError("Submission outcome requires reconciliation")
        save(state_path, state)
    statuses = [item["state"] for item in state["stages"].values()]
    state.update(updated_utc=stamp(), status="RUNNING")
    if any(value in {"SUBMISSION_INTENT", "SUBMISSION_OUTCOME_UNKNOWN"} for value in statuses):
        state["status"] = "REQUIRES_RECONCILIATION"
    elif all(value in {"COMPLETED", "EXTERNAL_INPUT", "FAILED", "BLOCKED_UPSTREAM"} for value in statuses):
        state["status"] = "AVAILABLE_DATA_STAGES_TERMINAL" if not any(value in {"FAILED", "BLOCKED_UPSTREAM"} for value in statuses) else "FAILED_STAGES_RETAINED"
    save(state_path, state)
    return state["status"]


def main():
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Controllers require a scheduled CPU allocation")
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) != plan.get("controller_cpus", 1):
        raise RuntimeError("Controller allocation differs from budgeted CPU count")
    run_root = Path(plan["run_root"])
    for name in ("logs", "receipts"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    with (run_root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = run_root / "execution_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if any(item["state"] in {"SUBMISSION_INTENT", "SUBMISSION_OUTCOME_UNKNOWN"} for item in state["stages"].values()):
                raise RuntimeError("Unreconciled submission journal; no automatic restart")
            state.setdefault("controller_history", []).append({"job_id": os.environ["SLURM_JOB_ID"], "resumed_utc": stamp()})
        else:
            state = {"plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(), "created_utc": stamp(),
                "scientific_completion": False, "controller_job_id": os.environ["SLURM_JOB_ID"],
                "stages": {stage["id"]: {"state": "EXTERNAL_INPUT" if stage.get("external_input") else "WAITING",
                    "reason": stage.get("external_input")} for stage in plan["stages"]}}
            save(state_path, state)
        while True:
            status = cycle(plan_path, state_path, state)
            print(json.dumps({"utc": stamp(), "status": status, "controller_job": os.environ["SLURM_JOB_ID"]}), flush=True)
            if args.once or status != "RUNNING":
                return 0 if status not in {"REQUIRES_RECONCILIATION", "FAILED_STAGES_RETAINED"} else 2
            time.sleep(45)


if __name__ == "__main__":
    raise SystemExit(main())
