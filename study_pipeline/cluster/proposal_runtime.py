"""Scheduled additive proposal analyses, with immutable parent and bounded storage.

This job follows the existing parent controller. It does not reset that graph,
replace successful receipts, or open the protected independent-human archive.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

PARENT_TAG = "20261004T210516Z-53593b22ca18"
PARENT_SOURCE_SHA = "1a73ac518fa37ce9c891a640c5b1a6bdb0411f3e374986f7e29ac75fcf0b3f6e"
PARENT_PLAN_SHA = "d50c90ec64ba3c79f689f497f7faa951e9635a41ce5c0a4a7df222f14cf6d3e2"
PROJECT = Path("/home/pwa209/adversarial-study")
SCRATCH = Path("/scratch/pwa209/adversarial-study")
POLICY = PROJECT / "maintenance/bridge-retry-20261008"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            h.update(chunk)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_source(source):
    source = Path(source).resolve()
    manifest = json.loads((source / "SOURCE_MANIFEST.json").read_text())
    if manifest.get("parent_source_manifest_sha256") != PARENT_SOURCE_SHA:
        raise ValueError("Additive release belongs to a different parent")
    for name, expected in manifest["files"].items():
        original_path = source / name
        path = original_path.resolve()
        if not path.is_relative_to(source) or original_path.is_symlink() or sha(path) != expected:
            raise ValueError("Immutable additive source mismatch: " + name)
    if sha(source / "PARENT_SOURCE_MANIFEST.json") != PARENT_SOURCE_SHA:
        raise ValueError("Parent manifest bytes changed")
    return manifest


def guard(source, reserve=0):
    """Reuse the qualified strict250000 policy and the actual personal quota."""
    common = load_module("proposal_qualified_policy_common", POLICY / "bridge_common.py")
    common.qualified()
    policy = common.policy()
    parent_plan = json.loads((PROJECT / "runs" / PARENT_TAG / "execution_plan.json").read_text())
    frozen_worker = load_module("proposal_frozen_quota_worker", PROJECT / "releases" / PARENT_TAG / "study_pipeline/cluster/stage_worker.py")
    snapshot = policy.snapshot(str(SCRATCH), frozen_worker.storage_snapshot)
    policy.guard(snapshot, parent_plan)
    if snapshot["project_entries"] + reserve >= 250000:
        raise ValueError("Insufficient project entry headroom for bounded new artifacts")
    return snapshot


def verify_parent():
    source = PROJECT / "releases" / PARENT_TAG
    run = PROJECT / "runs" / PARENT_TAG
    if sha(source / "SOURCE_MANIFEST.json") != PARENT_SOURCE_SHA or sha(run / "execution_plan.json") != PARENT_PLAN_SHA:
        raise ValueError("Frozen parent source/plan changed")
    required = ("cohort_discovery", "paris_discovery", "tactile_discovery", "dream_cohort_discovery",
                "human_neural_state_space", "human_representation_bridge", "manifold_mechanistic_bridge", "cross_system_training_audit")
    pins = {}
    for name in required:
        path = run / "receipts" / (name + ".json")
        value = json.loads(path.read_text())
        if value.get("status") != "completed" or value.get("exit_code") != 0:
            raise ValueError("Required parent stage not completed: " + name)
        if value.get("plan_sha256") != PARENT_PLAN_SHA or value.get("source_manifest_sha256") != PARENT_SOURCE_SHA:
            raise ValueError("Parent execution lineage mismatch: " + name)
        pins[name] = {"path": str(path), "sha256": sha(path)}
    return pins


def step(run, name, arguments):
    record = run / "receipts" / (name + ".json")
    if record.exists():
        raise ValueError("Never replay an existing additive stage: " + name)
    value = {"stage": name, "status": "started", "utc": utc(), "job_id": os.environ["SLURM_JOB_ID"],
             "command": [sys.executable, *map(str, arguments)]}
    write(record, value)
    with (run / "logs" / (name + ".log")).open("x") as stream:
        completed = subprocess.run(value["command"], stdout=stream, stderr=subprocess.STDOUT, check=False)
    value.update(status="completed" if completed.returncode == 0 else "failed", exit_code=completed.returncode, finished_utc=utc())
    write(record, value)
    if completed.returncode:
        raise RuntimeError("Additive stage failed: " + name)
    return value


def threshold_resources(manifest):
    """Select only the original healthy threshold recordings plus their channels."""
    resources = [r for r in manifest["resources"] if r["source_id"] == "openneuro_ds001785" and
                 any("task-" + task + "_" in r["relative_path"] for task in ("thrdown", "thrup"))]
    prefixes = {str(PurePosixPath(r["relative_path"]).parent) + "/" + "_".join(PurePosixPath(r["relative_path"]).name.split("_")[:2]) + "_channels.tsv"
                for r in resources if r["relative_path"].endswith(".set")}
    channels = [r for r in manifest["resources"] if r["relative_path"] in prefixes]
    if {r["relative_path"] for r in channels} != prefixes:
        raise ValueError("Original threshold recordings lack their pinned channel tables")
    resources += channels
    if len([r for r in resources if r["relative_path"].endswith(".set")]) != 36:
        raise ValueError("Expected exactly36 original healthy threshold runs")
    return resources


def acquire_threshold_sources(manifest, cache):
    """Explicitly acquire every selected original, despite pilot defaults."""
    from adversarial_study import acquisition
    resources = threshold_resources(manifest)
    return acquisition.acquire_manifest({"schema_version": 1, "resources": resources}, cache,
        resource_ids=[r["resource_id"] for r in resources], max_bytes_per_second=8 * 1024 ** 2,
        max_bytes=sum(r["expected_bytes"] for r in resources) + 1024 ** 2, max_files=1000)


def tactile_runtime():
    # Dependency qualification runs here. The pinned serialization helper is
    # explicitly installed by each conversion child, not this controller.
    base = PROJECT / "maintenance/conversion-repair-20261005"
    sys.path.insert(0, str(base / "paris_runtime"))
    from qualify_fif import dependencies
    dependencies(Path(os.environ["ACS_REPAIR_OVERLAY"]))


def launch_breadth(source, run, prepared):
    receipt = json.loads((prepared / "prepare_receipt.json").read_text())
    if receipt.get("status") != "qualified_brightness_human_split_prepared":
        write(run / "breadth_schedule.json", {"status": "BLOCKED_NO_QUALIFIED_BEHAVIOURAL_SPLIT", "prepare_receipt_sha256": sha(prepared / "prepare_receipt.json")})
        return
    intent = run / "breadth_schedule.json"
    if intent.exists():
        raise ValueError("Behavioural submission already attempted; reconcile before retry")
    jobs = {}
    record = {"status": "submitting", "utc": utc(), "parent_cpu_job": os.environ["SLURM_JOB_ID"], "jobs": jobs,
              "prepare_receipt_sha256": sha(prepared / "prepare_receipt.json")}
    write(intent, record)
    prior = os.environ["SLURM_JOB_ID"]
    for name in ("centaur_train", "base_train", "breadth_evaluate", "breadth_compare"):
        gpu = name != "breadth_compare"
        command = ["sbatch", "--parsable", "--hold", "--dependency=afterok:" + prior,
                   "--account=" + ("def-ptewarie_gpu" if gpu else "def-ptewarie_cpu"),
                   "--job-name=acs-proposal-" + name, "--comment=acs-proposal:" + source.name + ":" + name,
                   "--cpus-per-task=4", "--mem=64G" if gpu else "--mem=16G", "--time=12:00:00" if gpu else "--time=02:00:00",
                   "--output=" + str(run / "logs" / (name + "-%j.log"))]
        if gpu:
            command += ["--gpus-per-node=a100:2", "--nodes=1"]
        command += [str(source / "study_pipeline/cluster/proposal_run.sh"), str(source), str(run), name]
        record.update(inflight_stage=name, submission_command=command)
        write(intent, record)
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        record["last_submission"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        write(intent, record)
        job = result.stdout.strip().split(";")[0]
        if result.returncode or not job.isdigit():
            raise RuntimeError("Ambiguous behavioural submission; never automatically replay")
        jobs[name] = job
        write(intent, record)
        check = subprocess.run(["scontrol", "show", "job", job], text=True, capture_output=True, check=True).stdout
        if "UserId=pwa209(3164768)" not in check or "NumCPUs=4" not in check or "Comment=acs-proposal:" + source.name + ":" + name not in check:
            raise RuntimeError("Unexpected held behavioural allocation; leave it held")
        if gpu and "gres/gpu=2" not in check:
            raise RuntimeError("Unexpected GPU count; leave allocation held")
        subprocess.run(["scontrol", "release", job], check=True)
        prior = job
    record.update(status="queued_serial_after_human_split", finished_utc=utc(), maximum_concurrent_cpus=4, maximum_concurrent_gpus=2)
    write(intent, record)


def cpu(source, run):
    manifest = verify_source(source)
    parent = verify_parent()
    before = guard(source, reserve=32000)
    write(run / "qualification.started.json", {"utc": utc(), "parent_pins": parent, "source_manifest_sha256": sha(source / "SOURCE_MANIFEST.json"), "storage_before": before})
    step(run, "additive_tests", ["-m", "unittest", "discover", "-s", source / "study_pipeline/tests", "-p", "test_proposal_*.py"])
    write(run / "qualification.json", {"status": "qualified_additive_proposal_analysis", "scope": "software_source_runtime_only",
        "independent_replication_qualified": False, "human_intervention_qualified": False,
        "source_manifest_sha256": sha(source / "SOURCE_MANIFEST.json"), "parent_pins": parent,
        "tests_receipt_sha256": sha(run / "receipts/additive_tests.json"), "utc": utc()})
    result = SCRATCH / "results" / source.name
    result.mkdir(parents=True, exist_ok=False)
    old = SCRATCH / "results" / PARENT_TAG
    discovery = result / "proposal_discovery"
    arguments = ["-m", "adversarial_study.proposal_discovery", "--config", source / "study_pipeline/configs/proposal_discovery.json", "--output", discovery]
    for name in ("cohort_discovery", "paris_discovery", "tactile_discovery", "dream_cohort_discovery"):
        arguments += ["--source", old / name]
    step(run, "proposal_discovery", arguments)
    step(run, "proposal_mechanisms", ["-m", "adversarial_study.proposal_mechanisms",
        "--states", old / "human_neural_state_space", "--representations", old / "human_representation_bridge",
        "--bridge", old / "manifold_mechanistic_bridge", "--fit", old / "joint_mechanistic_fit", "--discovery", discovery,
        "--config", source / "study_pipeline/configs/proposal_mechanisms.json", "--output", result / "proposal_mechanisms"])
    for kind in ("replication", "experiment"):
        step(run, kind + "_contract", ["-m", "adversarial_study.proposal_validation", "prepare-template", "--kind", kind, "--output", result / (kind + "_contract")])
    prepared = result / "proposal_breadth_prepare"
    step(run, "proposal_breadth_prepare", ["-m", "adversarial_study.proposal_breadth", "prepare", "--discovery", discovery,
        "--source-root", SCRATCH / "data/extensions-20261004", "--manifest", source / "study_pipeline/configs/sources.extensions.json",
        "--generator", source / "study_pipeline/docs/proposal_sources/hellmann_generate_prompts.py",
        "--config", source / "study_pipeline/configs/proposal_breadth.json", "--output", prepared])
    # Serial, bounded public file acquisition runs in this compute allocation.
    from adversarial_study import acquisition
    public = json.loads((source / "study_pipeline/configs/sources.public.json").read_text())
    resources = threshold_resources(public)
    cache = SCRATCH / "data" / ("thresholds-" + source.name)
    acquire_record = run / "threshold_acquisition.json"
    try:
        answer = acquire_threshold_sources(public, cache)
        write(acquire_record, {"status": "acquired", "receipt": answer, "utc": utc()})
    except acquisition.urllib.error.URLError as error:
        write(acquire_record, {"status": "BLOCKED_TRANSFER_NETWORK", "reason": str(error), "utc": utc(),
            "required_original_resources": len(resources), "source_manifest_sha256": sha(source / "study_pipeline/configs/sources.public.json")})
    if json.loads(acquire_record.read_text())["status"] == "acquired":
        tactile_runtime()
        parent_run = PROJECT / "runs" / PARENT_TAG
        native = SCRATCH / "native/20261004T165416Z-b6b22b33a4ba"
        for task in ("thrdown", "thrup"):
            guard(source, reserve=16000)
            norm = result / ("threshold_" + task)
            step(run, task + "_convert", ["-m", "adversarial_study.proposal_tactile", "convert", "--source-root", cache,
                "--manifest", source / "study_pipeline/configs/sources.public.json", "--task", task, "--output", norm,
                "--release-dir", native, "--qualification", parent_run / "qualification/native_detector.json",
                "--integrity-helper", source / "study_pipeline/qualified_tools/tactile_adapter_repair.py",
                "--serialization-helper", source / "study_pipeline/qualified_tools/serialization_repair.py",
                "--serialization-helper-sha256", manifest["files"]["study_pipeline/qualified_tools/serialization_repair.py"]])
            config = norm / "study.config.json"; pointer = run / ("tactile_" + task + ".prepared.json")
            score = result / (task + ".scores.csv")
            # Four original scalp channels emit one input and at most one
            # retained activation file each per eligible window. Bound the
            # export from all source episodes before any native score exists.
            with (norm / "episodes_development.csv").open(newline="") as stream:
                episode_count = sum(1 for _ in csv.DictReader(stream))
            # Includes per-episode channel-quality metadata and extra margin.
            guard(source, reserve=12 * episode_count + 256)
            for action in ("prepare", "batch"):
                args = [source / "study_pipeline/cluster/pilot.py", action, "--config", config, "--pointer", pointer]
                if action == "batch":
                    prepared_pointer = json.loads(pointer.read_text())
                    with Path(prepared_pointer["jobs"]).open(newline="") as stream:
                        native_jobs = sum(1 for _ in csv.DictReader(stream))
                    guard(source, reserve=native_jobs + 128)
                    args += ["--scores", score]
                step(run, task + "_" + action, args)
            stage = result / (task + "_discovery")
            step(run, task + "_neural_observations", ["-m", "adversarial_study.model_guided", "discover", "--pointer", pointer,
                "--scores", score, "--qualification", parent_run / "qualification/native_detector.json", "--output", stage])
            step(run, task + "_staircase", ["-m", "adversarial_study.proposal_tactile", "analyze", "--discovery", stage, "--output", result / (task + "_staircase")])
    after = guard(source)
    verify_source(source)
    write(run / "cpu_phase.json", {"status": "additive_existing_data_analyses_completed", "utc": utc(),
        "source_manifest_sha256": sha(source / "SOURCE_MANIFEST.json"), "source_files": len(manifest["files"]),
        "storage_after": after, "independent_human_data_collected": False,
        "independent_replication_completed": False, "randomized_human_experiment_completed": False,
        "behavioural_prepare_status": json.loads((prepared / "prepare_receipt.json").read_text())["status"],
        "threshold_acquisition_status": json.loads(acquire_record.read_text())["status"]})
    launch_breadth(source, run, prepared)


def breadth(source, run, phase):
    verify_source(source)
    if json.loads((run / "cpu_phase.json").read_text())["status"] != "additive_existing_data_analyses_completed":
        raise ValueError("Additional analyses were not qualified")
    guard(source, reserve=500)
    result = SCRATCH / "results" / source.name
    prepared = result / "proposal_breadth_prepare"
    base = ["-m", "adversarial_study.proposal_breadth"]
    if phase in ("centaur_train", "base_train"):
        model = phase.split("_")[0]
        step(run, phase, base + ["train", "--prepared", prepared, "--model", model, "--output", result / phase])
    elif phase == "breadth_evaluate":
        for model in ("centaur", "base"):
            step(run, model + "_evaluate", base + ["evaluate", "--prepared", prepared, "--model", model,
                "--centaur-trained", result / "centaur_train",
                "--base-trained", result / "base_train", "--output", result / (model + "_evaluate")])
    elif phase == "breadth_compare":
        step(run, phase, base + ["compare", "--prepared", prepared, "--centaur-train", result / "centaur_train",
            "--base-train", result / "base_train", "--centaur-eval", result / "centaur_evaluate",
            "--base-eval", result / "base_evaluate", "--output", result / "breadth_comparison"])
    else:
        raise ValueError("Unknown scheduled proposal phase")
    write(run / (phase + ".storage.json"), guard(source))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--source", type=Path, required=True); p.add_argument("--run", type=Path, required=True)
    p.add_argument("--phase", required=True); args = p.parse_args()
    if os.getuid() != 3164768 or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise ValueError("Own scheduled Narval allocation required")
    source, run = args.source.resolve(), args.run.resolve()
    if source.parent != PROJECT / "releases" or run.parent != PROJECT / "runs" or source.name != run.name:
        raise ValueError("Unexpected proposal source/run identity")
    queue = subprocess.run(["squeue", "-h", "-u", "pwa209", "-o", "%i|%j|%T|%C"], text=True, capture_output=True, check=True).stdout
    allocated_cpus = allocated_gpus = 0
    for line in queue.splitlines():
        job, name, state, cpus = line.split("|")
        if name.startswith("acs-") and state in ("RUNNING", "COMPLETING", "CONFIGURING"):
            allocated_cpus += int(cpus)
            text = subprocess.run(["scontrol", "show", "job", job, "-o"], text=True, capture_output=True, check=True).stdout
            import re
            allocation = re.search(r"(?:^|\s)AllocTRES=(\S+)", text)
            if allocation:
                count = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", allocation.group(1))
                allocated_gpus += int(count.group(1)) if count else 0
    if allocated_cpus > 16 or allocated_gpus > 2:
        raise ValueError("Current study allocations exceed the user's CPU/GPU allowance")
    write(run / (args.phase + ".allocation.json"), {"utc": utc(), "own_study_queue": queue,
        "running_study_cpus": allocated_cpus, "running_study_gpus": allocated_gpus,
        "maximum_study_cpus": 16, "maximum_study_gpus": 2})
    sys.path.insert(0, str(source / "study_pipeline"))
    if args.phase == "cpu": cpu(source, run)
    else: breadth(source, run, args.phase)


if __name__ == "__main__":
    main()
