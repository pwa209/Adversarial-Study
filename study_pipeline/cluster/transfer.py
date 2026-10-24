"""Finite Narval login-host HTTPS transfer; scientific work stays in Slurm.

Adapted from the read-only Perturbational Consciousness Manifold transfer
precedent. This helper never submits jobs, preprocesses EEG, fits a model, or
evaluates a scientific result. Login-mode network-body and checksum reads share
an8MiB/s serial limit and nice19. Scheduled `verify` rehashes every cached input.
Only public, source-pinned manifests may enter the declared finite transfer groups.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adversarial_study import acquisition as acq
from native_tools import figshare_source as figshare

PROJECT = Path("/home/pwa209/adversarial-study")
SCRATCH = Path("/scratch/pwa209/adversarial-study")
RATE_CAP = 8*1024*1024
BYTE_CAP = 1_000_000_000_000
FILE_CAP = 150_000
GROUP_ORDER = ("native_sources", "acquire_monash_pilot", "acquire_psych", "acquire_dream_pilot",
               "acquire_monash_cohort", "acquire_paris_archive", "acquire_tactile_pilot",
               "acquire_tactile_cohort", "acquire_dream_cohort", "acquire_dream_sparse",
               "acquire_centaur")
NEURAL_EXPANSION = {"acquire_monash_cohort", "acquire_paris_archive", "acquire_tactile_pilot",
                    "acquire_tactile_cohort", "acquire_dream_cohort", "acquire_dream_sparse"}
STATUS_PUBLICATION_LOCK = threading.Lock()


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return result


def save(path: Path, value: dict) -> None:
    # Windows can reject overlapping replacements of the same destination even
    # with distinct temporary files. One process-wide lock serializes main and
    # heartbeat publication; the project-wide kernel lock excludes other writers.
    with STATUS_PUBLICATION_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+f".{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp")
        with temporary.open("x",encoding="utf-8") as stream:
            json.dump(value,stream,indent=2,allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)


def publish(path: Path, value: dict) -> None:
    """Publish an immutable completion marker without replacing earlier evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def login_guard(plan: dict) -> None:
    """Only the authorized account/login hosts and this study's owned roots."""
    host = socket.gethostname().split(".")[0].lower()
    if getpass.getuser() != "pwa209" or not re.fullmatch(r"narval\d*", host):
        raise RuntimeError("transfer requires pwa209 on an authenticated Narval login host")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("login transfer must run outside compute allocations")
    if Path(plan["project"]) != PROJECT or Path(plan["scratch"]) != SCRATCH:
        raise ValueError("only the authorized adversarial-study project/scratch roots are supported")
    source = Path(plan["source"]).resolve(strict=True)
    run = Path(plan["run_root"]).resolve(strict=True)
    if not source.is_relative_to(PROJECT.resolve(strict=True)) or not run.is_relative_to((PROJECT/"runs").resolve(strict=True)):
        raise ValueError("source/run path escaped the owned study project")
    if Path(__file__).resolve() != source/"study_pipeline/cluster/transfer.py":
        raise ValueError("run the transfer helper from the source pinned in the plan")
    source_manifest = load(source/"SOURCE_MANIFEST.json")
    for name in ("study_pipeline/cluster/transfer.py", "study_pipeline/adversarial_study/acquisition.py",
                 "study_pipeline/adversarial_study/__init__.py", "study_pipeline/native_tools/figshare_source.py",
                 "study_pipeline/native_tools/__init__.py"):
        if digest(source/name) != source_manifest.get("files", {}).get(name):
            raise ValueError("changed transfer implementation in deployed source: "+name)
    current = os.getpriority(os.PRIO_PROCESS, 0)
    os.nice(max(0, 19-current))
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        raise RuntimeError("failed to establish nice19")


def _resolve(value: str, plan: dict) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("transfer path must be a nonempty string")
    replacements = {"source": plan["source"], "pipeline": str(Path(plan["source"])/"study_pipeline"),
                    "run": plan["run_root"], "project": plan["project"], "scratch": plan["scratch"]}
    for key, replacement in replacements.items():
        value = value.replace("{"+key+"}", replacement)
    if "{" in value or "}" in value or not Path(value).is_absolute():
        raise ValueError("unresolved or nonabsolute transfer path")
    return Path(value).resolve()


def _pinned_json(path: Path, plan: dict, source_manifest: dict) -> dict:
    source = Path(plan["source"]).resolve()
    try:
        relative = path.relative_to(source).as_posix()
    except ValueError as error:
        raise ValueError("transfer metadata must be inside immutable deployed source") from error
    expected = source_manifest.get("files", {}).get(relative)
    if not expected or digest(path) != expected:
        raise ValueError("transfer metadata differs from the deployed source pin: "+relative)
    return load(path)


def load_context(plan_path: str | Path) -> tuple[dict, dict, list[dict]]:
    """Resolve the explicit transfer descriptors and freeze their resource sets."""
    path = Path(plan_path).resolve()
    plan = load(path)
    source = Path(plan["source"]).resolve()
    source_path = source/"SOURCE_MANIFEST.json"
    if digest(source_path) != plan["source_manifest_sha256"]:
        raise ValueError("source manifest differs from execution plan")
    source_manifest = load(source_path)
    pins = {"plan_sha256": digest(path), "source_manifest_sha256": plan["source_manifest_sha256"]}
    transfer = plan.get("transfer", {})
    descriptors = transfer.get("groups", [])
    if not isinstance(descriptors, list) or not descriptors or len(descriptors) > len(GROUP_ORDER):
        raise ValueError("plan requires declared explicit finite transfer groups")
    ids = [descriptor.get("id") for descriptor in descriptors]
    if len(ids) != len(set(ids)) or set(ids)-set(GROUP_ORDER):
        raise ValueError("unknown or duplicate transfer group")
    groups = []
    scratch = Path(plan["scratch"]).resolve()
    for descriptor in sorted(descriptors, key=lambda d: GROUP_ORDER.index(d["id"])):
        group = dict(descriptor)
        destination = _resolve(group["destination"], plan)
        if not destination.is_relative_to(scratch) or destination == scratch:
            raise ValueError("transfer destination must stay in owned study scratch")
        group["destination"] = str(destination)
        kind = group.get("kind")
        if kind == "figshare_v2":
            if group["id"] != "native_sources":
                raise ValueError("Figshare release belongs to native_sources group")
            metadata_path = _resolve(group["metadata"], plan)
            metadata = _pinned_json(metadata_path, plan, source_manifest)
            items = figshare.selected_files(metadata, group.get("include_detector", True), group.get("include_config_bank", True))
            resources = [{"resource_id": f"figshare_v2:{item['id']}", "source_id": "figshare_v2_native",
                          "relative_path": relative, "url": item["download_url"], "expected_bytes": item["size"],
                          "checksum": {"algorithm": "md5", "value": item["computed_md5"].lower()},
                          "provider_article_id": 30906557, "provider_version": 2,
                          "provider_file_id": item["id"]} for item, relative in items]
            group["metadata_value"] = metadata
            group["metadata_sha256"] = digest(metadata_path)
        elif kind == "public_manifest":
            manifest_path = _resolve(group["manifest"], plan)
            manifest = _pinned_json(manifest_path, plan, source_manifest)
            requested = set(group.get("resource_ids", []))
            sources = set(group.get("source_ids", []))
            available = manifest.get("resources", [])
            if not requested and not sources:
                raise ValueError("transfer group must explicitly select resources or source IDs")
            if requested-{r["resource_id"] for r in available} or sources-{r["source_id"] for r in available}:
                raise ValueError("transfer selector is absent from the pinned public manifest")
            resources = [r for r in available if r["resource_id"] in requested or r["source_id"] in sources]
            group["metadata_sha256"] = digest(manifest_path)
        else:
            raise ValueError("unsupported transfer kind")
        if not resources:
            raise ValueError("empty transfer resource set")
        for resource in resources:
            acq.validate_resource(resource)
            acq._target(destination, resource["relative_path"])
        if len({r["resource_id"] for r in resources}) != len(resources) or len({r["relative_path"] for r in resources}) != len(resources):
            raise ValueError("duplicate transfer resource ID/path")
        if group["id"] in NEURAL_EXPANSION:
            if group.get("requires_stage", "qualify_detector") != "qualify_detector":
                raise ValueError("cohort transfer prerequisite must be native integrity qualification")
            group["requires_stage"] = "qualify_detector"
        elif group.get("requires_stage"):
            raise ValueError("initial native/pilot transfers must not depend on scientific results")
        group["resource_manifest"] = {"schema_version": 1, "resources": resources,
                                      "metadata_file_sha256": group["metadata_sha256"]}
        group["transfer_manifest_sha256"] = acq.canonical_hash(group["resource_manifest"])
        group["expected_bytes"] = sum(r["expected_bytes"] for r in resources)
        groups.append(group)
    max_bytes = transfer.get("max_bytes", plan.get("storage_allowance_bytes", BYTE_CAP))
    max_files = transfer.get("max_files", plan.get("entry_allowance", FILE_CAP))
    rate = transfer.get("max_rate_bytes_per_second", RATE_CAP)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 0 < max_bytes <= BYTE_CAP:
        raise ValueError("transfer byte budget exceeds authorized finite allowance")
    if isinstance(max_files, bool) or not isinstance(max_files, int) or not 0 < max_files <= FILE_CAP:
        raise ValueError("transfer file budget exceeds authorized finite allowance")
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 < rate <= RATE_CAP:
        raise ValueError("transfer rate must not exceed8MiB/s")
    union = {}
    for group in groups:
        for resource in group["resource_manifest"]["resources"]:
            key = (group["destination"], resource["relative_path"])
            if key in union and acq.canonical_hash(union[key]) != acq.canonical_hash(resource):
                raise ValueError("two groups assign conflicting identities to one destination")
            union[key] = resource
    if len(union) > max_files or sum(r["expected_bytes"] for r in union.values()) > max_bytes:
        raise ValueError("selected transfer footprint exceeds finite byte/file budget")
    plan["_transfer_limits"] = {"max_bytes": max_bytes, "max_files": max_files, "rate": rate,
                                "unique_selected_files": len(union),
                                "unique_selected_bytes": sum(r["expected_bytes"] for r in union.values())}
    return plan, pins, groups


class SharedReadLimit:
    """One leaky-bucket limit for serial network-body and checksum reads."""
    def __init__(self, rate: float, max_network_bytes: int = BYTE_CAP):
        self.rate = rate
        self.max_network_bytes = max_network_bytes
        self.next_time = time.monotonic()
        self.network_bytes = 0
        self.checksum_bytes = 0

    def consume(self, size: int, kind: str) -> None:
        now = time.monotonic()
        self.next_time = max(self.next_time, now) + size/self.rate
        if kind == "network":
            self.network_bytes += size
            if self.network_bytes > self.max_network_bytes:
                raise ValueError("shared transferred-byte budget exhausted")
        else:
            self.checksum_bytes += size
        wait = self.next_time-time.monotonic()
        if wait > 0:
            time.sleep(wait)


@contextmanager
def limited_acquisition(limiter: SharedReadLimit):
    """Wrap only acquisition IO; no model or scientific modules are imported."""
    original_open = acq.urllib.request.urlopen
    original_checksums = acq.file_checksums
    class Response:
        def __init__(self, response): self.response = response
        def __getattr__(self, name): return getattr(self.response, name)
        def __enter__(self): self.response.__enter__(); return self
        def __exit__(self, *args): return self.response.__exit__(*args)
        def read(self, size=-1):
            remaining = limiter.max_network_bytes-limiter.network_bytes
            # At most one look-ahead byte distinguishes EOF from budget overflow.
            size = min(size if size >= 0 else 1024*1024, max(0, remaining)+1)
            block = self.response.read(size)
            limiter.consume(len(block), "network")
            return block
    def open_limited(*args, **kwargs): return Response(original_open(*args, **kwargs))
    def checksums_limited(path, resource):
        algorithm = resource["checksum"]["algorithm"]
        provider = hashlib.new("sha1" if algorithm == "git_blob_sha1" else algorithm)
        if algorithm == "git_blob_sha1": provider.update(f"blob {path.stat().st_size}\0".encode())
        own = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                provider.update(block); own.update(block)
                limiter.consume(len(block), "checksum")
        return {"provider": provider.hexdigest(), "sha256": own.hexdigest()}
    acq.urllib.request.urlopen = open_limited
    acq.file_checksums = checksums_limited
    try:
        yield
    finally:
        acq.urllib.request.urlopen = original_open
        acq.file_checksums = original_checksums


def _paths(plan: dict, group_id: str) -> tuple[Path, Path]:
    root = Path(plan["run_root"])/"transfer"
    return root/(group_id+".complete.json"), root/(group_id+".receipt.json")


def validate_marker(plan: dict, pins: dict, group: dict) -> dict:
    marker_path, receipt_path = _paths(plan, group["id"])
    marker = load(marker_path)
    required = {**pins, "status": "complete_verified", "group": group["id"],
                "transfer_manifest_sha256": group["transfer_manifest_sha256"],
                "resource_count": len(group["resource_manifest"]["resources"]),
                "expected_bytes": group["expected_bytes"]}
    if any(marker.get(key) != value for key, value in required.items()):
        raise ValueError("transfer completion marker differs from group/plan/source pins")
    if digest(receipt_path) != marker.get("receipt_sha256"):
        raise ValueError("transfer receipt changed after completion marker")
    receipt = load(receipt_path)
    if marker.get("resources") != receipt.get("resources"):
        raise ValueError("completion marker resource list differs from hashed transfer receipt")
    expected = {r["resource_id"]: r for r in group["resource_manifest"]["resources"]}
    records = receipt.get("resources", [])
    if len(records) != len(expected) or len({r.get("resource_id") for r in records}) != len(expected):
        raise ValueError("transfer receipt has missing/duplicate resources")
    for record in records:
        item = expected.get(record.get("resource_id"))
        if item is None or record.get("resource_manifest_sha256") != acq.canonical_hash(item):
            raise ValueError("transfer receipt resource pin differs from source")
        if record.get("bytes") != item["expected_bytes"] or record.get("provider_checksum") != item["checksum"]:
            raise ValueError("transfer receipt byte count/provider checksum differs")
        if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
            raise ValueError("transfer receipt lacks file SHA256")
    return marker


def _wait_prerequisite(plan: dict, pins: dict, group: dict, timeout_s: float) -> None:
    stage = group.get("requires_stage")
    if not stage:
        return
    path = Path(plan["run_root"])/"receipts"/(stage+".json")
    deadline = time.monotonic()+timeout_s
    while True:
        journal_path = Path(plan["run_root"])/"execution_state.json"
        if journal_path.exists() and not path.exists():
            journal = load(journal_path)
            if journal.get("plan_sha256") == pins["plan_sha256"] and journal.get("stages", {}).get(stage, {}).get("state") in {"FAILED", "BLOCKED_UPSTREAM"}:
                raise ValueError("scheduled integrity prerequisite is terminal/blocked; retain evidence and continue independent groups")
        if path.exists():
            receipt = load(path)
            if receipt.get("stage") != stage or any(receipt.get(k) != v for k, v in pins.items()):
                raise ValueError("qualification receipt differs from scheduled stage/plan/source pins")
            if receipt.get("status") != "completed" or receipt.get("exit_code") != 0:
                raise ValueError("scheduled native integrity qualification failed; retain prior stages")
            journal = load(Path(plan["run_root"])/"execution_state.json")
            accepted = journal.get("stages", {}).get(stage, {})
            if journal.get("plan_sha256") != pins["plan_sha256"] or accepted.get("job_id") != receipt.get("job_id"):
                raise ValueError("qualification receipt does not match accepted scheduled allocation")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("native integrity qualification did not finish within bounded wait")
        time.sleep(min(30, max(0, deadline-time.monotonic())))


def _native_source_manifest(group: dict, receipt: dict) -> dict:
    """Publish the layout expected by the existing Figshare/native-runtime API."""
    root = Path(group["destination"])
    metadata = group["metadata_value"]
    entries = figshare.selected_files(metadata, group.get("include_detector", True), group.get("include_config_bank", True))
    verified = {r["relative_path"]: r for r in receipt["resources"]}
    records = [{"path": relative, "file_id": item["id"], "size": item["size"],
                "computed_md5": item["computed_md5"].lower(), "sha256": verified[relative]["sha256"],
                "download_url": item["download_url"]} for item, relative in entries]
    result = {"status": "pinned_payload_verified", "article_id": 30906557, "version": 2,
              "doi": figshare.DOI, "metadata_api": figshare.API_URL,
              "metadata_sha256": hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
              "acquired_at_utc": stamp(), "files": records,
              "source_layout": "published flat C++ files reconstructed under src; bytes unchanged",
              "licenses": ["LICENSE","license.txt"], "clinical_validation": "not_established", "native_execution": "not_yet_tested"}
    path = root/"SOURCE_MANIFEST.json"
    if path.exists():
        previous = load(path)
        if any(previous.get(k) != result[k] for k in ("status","article_id","version","doi","metadata_sha256","files")):
            raise ValueError("existing native source manifest differs; use a new destination")
        return previous
    publish(path, result)
    return result


def run_transfer(plan_path: str | Path, *, group_ids: list[str] | None = None,
                 wait_timeout_s: float = 28800) -> dict:
    if not math.isfinite(wait_timeout_s) or not 0 < wait_timeout_s <= 86400:
        raise ValueError("wait timeout must be finite and at most24hours")
    plan, pins, groups = load_context(plan_path)
    login_guard(plan)
    wanted = set(group_ids or [])
    if wanted-set(g["id"] for g in groups):
        raise ValueError("unknown selected transfer group")
    groups = [g for g in groups if not wanted or g["id"] in wanted]
    state_root = Path(plan["run_root"])/"transfer"
    state_root.mkdir(parents=True, exist_ok=True)
    # Project-wide singleton prevents two deployed versions sharing a network cap.
    lock_path = Path(plan["project"])/"transfer.lock"
    limiter = SharedReadLimit(plan["_transfer_limits"]["rate"],plan["_transfer_limits"]["max_bytes"])
    state = {"schema_version":1,"status":"starting",**pins,"pid":os.getpid(),"host":socket.gethostname(),
             "execution_kind":"bounded_https_file_management_only","limits":plan["_transfer_limits"],
             "groups":[g["id"] for g in groups],"started_utc":stamp(),"completed_groups":[]}
    stop = threading.Event()
    def progress():
        while not stop.wait(30):
            state.update(updated_utc=stamp(), network_body_bytes=limiter.network_bytes,checksum_read_bytes=limiter.checksum_bytes)
            save(state_root/"status.json",dict(state))
            print(json.dumps({k:state.get(k) for k in ("status","group","updated_utc","network_body_bytes","checksum_read_bytes")}),flush=True)
    with acq._resource_lock(lock_path), limited_acquisition(limiter):
        monitor = threading.Thread(target=progress,daemon=True,name="bounded-transfer-status")
        monitor.start()
        current = None
        try:
            for group in groups:
                current = group
                state.update(group=group["id"],status="waiting_integrity_prerequisite",updated_utc=stamp())
                save(state_root/"status.json",dict(state))
                try:
                    marker_path, receipt_path = _paths(plan,group["id"])
                    if marker_path.exists():
                        validate_marker(plan,pins,group)
                        state["completed_groups"].append(group["id"])
                        continue
                    _wait_prerequisite(plan,pins,group,wait_timeout_s)
                    state.update(status="acquiring_public_files",updated_utc=stamp())
                    save(state_root/"status.json",dict(state))
                    receipt = acq.acquire_manifest(group["resource_manifest"],group["destination"],
                        resource_ids=[r["resource_id"] for r in group["resource_manifest"]["resources"]],
                        max_bytes_per_second=plan["_transfer_limits"]["rate"],
                        max_bytes=plan["_transfer_limits"]["max_bytes"],max_files=plan["_transfer_limits"]["max_files"])
                    native = _native_source_manifest(group,receipt) if group["kind"]=="figshare_v2" else None
                    save(receipt_path,receipt)
                    marker = {"schema_version":1,"status":"complete_verified","group":group["id"],**pins,
                        "transfer_manifest_sha256":group["transfer_manifest_sha256"],"metadata_sha256":group["metadata_sha256"],
                        "receipt_sha256":digest(receipt_path),"resource_count":len(receipt["resources"]),
                        "expected_bytes":group["expected_bytes"],"destination":group["destination"],
                        "resources":receipt["resources"],"native_source_manifest_sha256":digest(Path(group["destination"])/"SOURCE_MANIFEST.json") if native else None,
                        "completed_utc":stamp(),"qualification":"source byte identity; no scientific result gate"}
                    publish(marker_path,marker)
                    validate_marker(plan,pins,group)
                    state["completed_groups"].append(group["id"])
                    print(json.dumps({"group":group["id"],"status":"complete_verified","resource_count":marker["resource_count"],"expected_bytes":marker["expected_bytes"]}),flush=True)
                except Exception as error:
                    failure = {**state, "status": "failed", "error": f"{type(error).__name__}: {error}",
                               "group": group["id"], "finished_utc": stamp(),
                               "transfer_manifest_sha256": group["transfer_manifest_sha256"]}
                    publish(state_root/(group["id"]+f".failed.{time.time_ns()}.json"), failure)
                    state.setdefault("failed_groups", []).append(group["id"])
                    if not plan.get("transfer", {}).get("continue_independent_groups"):
                        raise
                    print(json.dumps({"group": group["id"], "status": "failed_retained_continue_independent_groups", "error": failure["error"]}), flush=True)
            state.update(status="partial_failed_groups_retained" if state.get("failed_groups") else "complete_verified",finished_utc=stamp(),network_body_bytes=limiter.network_bytes,
                         checksum_read_bytes=limiter.checksum_bytes)
            return state
        except BaseException as error:
            state.update(status="failed",error=f"{type(error).__name__}: {error}",finished_utc=stamp())
            if current and current["id"] not in state.get("failed_groups", []):
                publish(state_root/(current["id"]+f".failed.{time.time_ns()}.json"),{**state,"group":current["id"],
                    "transfer_manifest_sha256":current["transfer_manifest_sha256"]})
            raise
        finally:
            stop.set()
            monitor.join(timeout=5)
            save(state_root/"status.json",dict(state))


def verify_group(plan_path: str | Path, group_id: str) -> dict:
    """Scheduled, network-free full checksum verification of a staged group."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("cache byte verification requires a Slurm allocation")
    plan,pins,groups=load_context(plan_path)
    group=next((g for g in groups if g["id"]==group_id),None)
    if group is None: raise ValueError("unknown transfer group")
    marker=validate_marker(plan,pins,group)
    records=[acq.verify_acquired(group["destination"],r) for r in group["resource_manifest"]["resources"]]
    originals={r["resource_id"]:r for r in marker["resources"]}
    if any(r["sha256"]!=originals[r["resource_id"]]["sha256"] for r in records):
        raise ValueError("cached bytes differ from login transfer receipts")
    if group["kind"]=="figshare_v2":
        if digest(Path(group["destination"])/"SOURCE_MANIFEST.json")!=marker["native_source_manifest_sha256"]:
            raise ValueError("native source manifest changed after transfer")
    result={"status":"scheduled_cache_verified","group":group_id,**pins,
            "transfer_manifest_sha256":group["transfer_manifest_sha256"],"job_id":os.environ["SLURM_JOB_ID"],
            "resource_count":len(records),"verified_bytes":sum(r["bytes"] for r in records),
            "resources":records,"completed_utc":stamp(),"network_access":False}
    publish(Path(plan["run_root"])/"transfer"/(group_id+f".scheduled_verify.{os.environ['SLURM_JOB_ID']}.json"),result)
    return result


def wait_group(plan_path: str | Path, group_id: str, timeout_s: float=900) -> dict:
    """Bounded marker wait only, without opening signals or accessing a network."""
    if not math.isfinite(timeout_s) or not 0 < timeout_s <= 86400: raise ValueError("invalid bounded wait")
    plan,pins,groups=load_context(plan_path)
    group=next((g for g in groups if g["id"]==group_id),None)
    if group is None: raise ValueError("unknown transfer group")
    deadline=time.monotonic()+timeout_s
    marker,_=_paths(plan,group_id)
    while True:
        if marker.exists(): return validate_marker(plan,pins,group)
        for path in (Path(plan["run_root"])/"transfer").glob(group_id+".failed.*.json"):
            failure=load(path)
            if all(failure.get(k)==v for k,v in pins.items()): raise RuntimeError("transfer failed; inspect retained group receipt")
        if time.monotonic()>=deadline: raise TimeoutError("transfer marker wait budget exhausted")
        time.sleep(min(30,max(0,deadline-time.monotonic())))


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=["run","verify","wait"])
    parser.add_argument("--plan",required=True,type=Path)
    parser.add_argument("--group",action="append")
    parser.add_argument("--wait-timeout-s",type=float,default=28800)
    args=parser.parse_args(argv)
    if args.operation=="run": result=run_transfer(args.plan,group_ids=args.group,wait_timeout_s=args.wait_timeout_s)
    else:
        if not args.group or len(args.group)!=1: parser.error("verify/wait requires exactly one --group")
        result=verify_group(args.plan,args.group[0]) if args.operation=="verify" else wait_group(args.plan,args.group[0],args.wait_timeout_s)
    print(json.dumps({k:v for k,v in result.items() if k!="resources"},indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
