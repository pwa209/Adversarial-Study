"""Once-only dependent submission when Slurm accounting trails live state.

This adapter never changes the sealed release and never synthesizes an sacct
record. It requires the original released qualifier custody plus exact current
scontrol identity/resources, then uses the unchanged held-submission transaction.
"""
import re


def _tres(ops, raw):
    parts = [item.split("=", 1) for item in raw.split(",")]
    ops.require(parts and all(len(p) == 2 for p in parts)
                and len(parts) == len({p[0] for p in parts}), "Malformed live-parent requested TRES")
    values = dict(parts)
    ops.require(set(values) <= {"cpu", "mem", "node", "billing"}, "Unknown live-parent resources")
    if values.get("mem", "").endswith("G"):
        values["mem"] = str(int(values["mem"][:-1]) * 1024) + "M"
    return values


def _live_parent_proof(ops, pin, parent, accounting):
    job = parent["job_id"]
    # The original custody API has already checked the source-pinned held and
    # released records. A current record supplements them; it cannot replace them.
    held = ops.fields(parent["values"]["held"]["raw"])
    observed = ops.observe(["scontrol", "show", "job", job, "-o"])
    current = ops.fields(observed)
    immutable = ("JobId", "JobName", "UserId", "Account", "NumCPUs", "NumTasks", "CPUs/Task",
                 "Requeue", "TimeLimit", "Command", "WorkDir", "Comment", "StdOut")
    ops.require(all(k in held and current.get(k) == held[k] for k in immutable),
                "Current qualifier identity/resources differ from actual frozen held record")
    ops.require(current.get("JobId") == job and current.get("NumNodes") in {"1", "1-1"}
                and held.get("NumNodes") in {"1", "1-1"}, "Current qualifier node/job identity differs")
    memory = {ops.POLICY["qualify"]["memory"], str(int(ops.POLICY["qualify"]["memory"][:-1]) * 1024) + "M"}
    ops.require(current.get("MinMemoryNode") in memory and held.get("MinMemoryNode") in memory,
                "Current qualifier memory differs")
    ops.require(_tres(ops, current.get("ReqTRES", "")) == _tres(ops, held.get("ReqTRES", "")),
                "Current qualifier requested resources changed")
    ops.require(current.get("Dependency") in {None, "(null)"}
                and held.get("Dependency") in {None, "(null)"}, "Qualifier dependency changed")
    ops.require(current.get("Restarts", "0") == held.get("Restarts", "0") == "0", "Qualifier was restarted")
    status = current.get("JobState")
    if status == "COMPLETED":
        # Accounting may have caught up between the two observations. Only the
        # original completed qualification proof can authorize this case.
        completed = ops.qualification_proof(pin, job)
        return {"mode": "completed_original_qualification_proof", "scontrol_raw": observed,
                "accounting_observation": accounting, "qualification_proof": completed,
                "accounting_row_synthesized": False}, True
    ops.require(status in {"PENDING", "RUNNING"}, "Qualifier is failed, cancelled, ambiguous or unsupported live state")
    if status == "RUNNING":
        ops.validate_allocation(observed, job, "qualify")
    else:
        # Slurm's literal Reason=None is a normal pending state before backfill
        # consideration. A missing Python None remains unavailable evidence.
        ops.require(current.get("Reason") not in {None, "", "(null)", "Unknown", "JobHeldUser", "JobHeldAdmin"},
                    "Qualifier is held or its pending reason is unknown")
        ops.require(current.get("AllocTRES") in {None, "(null)"}, "Pending qualifier has unknown allocation")
        ops.require(current.get("NodeList") in {None, "(null)"}, "Pending qualifier unexpectedly names allocated nodes")
    return {"mode": "actual_current_scontrol_parent", "job_id": job, "state": status,
            "scontrol_raw": observed, "verified_immutable_fields": {k: current[k] for k in immutable},
            "actual_held_record_sha256": ops.digest(parent["values"]["held"]["raw"].encode()),
            "accounting_observation": accounting, "accounting_unavailable": accounting is None,
            "accounting_row_synthesized": False, "completed_qualification_claimed": False}, False


def submit_live_parent(ops, pin, qualifier_job):
    """Return one actual analysis job ID; preserve all ambiguous mutation evidence."""
    phase = "analysis"
    ops.verify_packet(pin, full_upstream=False)
    with ops.locked():
        ops.state(create=False)
        ops.require(ops.maybe(ops.CONTROL / (phase + ".intent.json")) is None,
                    "Audit submission already attempted; never replay")
        allowed = [item["job_id"] for name in ops.POLICY if (item := ops.custody(name, pin)) is not None]
        parent = ops.custody("qualify", pin)
        ops.require(parent is not None and parent["job_id"] == str(qualifier_job),
                    "Exact source-bound qualifier release required")
        dependency = "afterok:" + parent["job_id"]
        actual = ops.accounting(qualifier_job, "qualify")
        if actual is not None:
            ops.require(actual["row"][5] in {"COMPLETED", "PENDING", "RUNNING"}, "Qualifier already failed or has unsupported accounting state")
        if actual is not None and actual["row"][5] == "COMPLETED":
            qualified = ops.qualification_proof(pin, qualifier_job)
            proof = {"mode": "completed_original_qualification_proof", "accounting_observation": actual,
                     "qualification_proof": qualified, "accounting_row_synthesized": False}
            fulfilled = True
        else:
            proof, fulfilled = _live_parent_proof(ops, pin, parent, actual)
        before = ops.queue_guard(allowed)
        argv = ops.submit_argv(phase, pin, dependency)
        ops.write_slot(ops.CONTROL / (phase + ".intent.json"), {
            "utc": ops.utc(), "manifest_sha256": pin, "argv": argv, "queue_before": before,
            "qualifier_job_id": str(qualifier_job), "prior_intents_replayed": False,
            "live_parent_evidence": proof})
        accepted = ops.mutate_once(argv)
        ops.write_slot(ops.CONTROL / (phase + ".acceptance.json"), {"utc": ops.utc(), **accepted})
        ops.require(type(accepted.get("returncode")) is int and accepted["returncode"] == 0
                    and re.fullmatch(r"[1-9][0-9]*(;[\w.-]+)?", accepted.get("stdout", "").strip()),
                    "Ambiguous submission; preserve custody, never replay")
        job = accepted["stdout"].strip().split(";")[0]
        raw = ops.observe(["scontrol", "show", "job", job, "-o"])
        allocation = ops.validate_allocation(raw, job, phase, dependency, held=True, fulfilled=fulfilled)
        ops.write_slot(ops.CONTROL / (phase + ".held.json"), {"utc": ops.utc(), "job_id": job,
            "raw": raw, "actual": allocation, "dependency_already_qualified": fulfilled})
        ops.queue_guard([*allowed, job])
        ops.verify_packet(pin, full_upstream=False)
        ops.write_slot(ops.CONTROL / (phase + ".release-intent.json"), {
            "utc": ops.utc(), "job_id": job, "manifest_sha256": pin, "dependency": dependency,
            "verified_held_record_sha256": ops.sha(ops.CONTROL / (phase + ".held.json"))})
        released = ops.mutate_once(["scontrol", "release", job])
        ops.write_slot(ops.CONTROL / (phase + ".released.json"), {
            "utc": ops.utc(), "job_id": job, "manifest_sha256": pin, "dependency": dependency,
            "release_attempted_exactly_once": True, **released})
        ops.require(type(released.get("returncode")) is int and released["returncode"] == 0,
                    "Ambiguous release; retain original custody, never replay")
        return job
