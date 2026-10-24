"""Reuse only a pinned dependency environment; never reuse scientific outputs.

This stdlib-only helper runs before activation (link), then inside the activated
cached environment (recheck). Missing or changed evidence fails closed. It does
not install, copy, repair, or delete environments, and does not certify tests for
the current release: the environment stage must still run those tests.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys


FILES = {
    "cpu": ("environment", "environment-resolved.txt", "environment-runtime.json"),
    "centaur": ("centaur_dependencies", "centaur-environment-resolved.txt", "centaur-environment-runtime.json"),
}
CPU_RUNTIME = ("python", "numpy", "scipy", "mne", "h5py", "pyarrow", "arrow_module")
GPU_PACKAGES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors", "sentencepiece")


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _inside(path: Path, roots: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _scoped(raw: str | Path, roots: list[Path], *, exists: bool = True) -> Path:
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts or not _inside(path, roots):
        raise ValueError(f"Path outside configured adversarial-study roots: {path}")
    resolved = path.resolve(strict=exists)
    if not _inside(resolved, roots):
        raise ValueError(f"Resolved path escapes adversarial-study roots: {path}")
    return path


def _pin(value: dict, roots: list[Path]) -> Path:
    if not isinstance(value, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))):
        raise ValueError("Cache evidence requires an explicit SHA256 pin")
    path = _scoped(value["path"], roots)
    if not path.is_file() or _sha(path) != value["sha256"]:
        raise ValueError(f"Cache evidence missing or changed: {path}")
    return path


def _runtime_identity(runtime: dict, kind: str) -> dict:
    if kind == "cpu":
        if any(not isinstance(runtime.get(key), str) or not runtime[key] for key in CPU_RUNTIME):
            raise ValueError("Incomplete CPU runtime versions")
        return {key: runtime[key] for key in CPU_RUNTIME}
    versions = runtime.get("versions", {})
    if any(not isinstance(versions.get(key), str) or not versions[key] for key in GPU_PACKAGES):
        raise ValueError("Incomplete Centaur dependency versions")
    if not isinstance(runtime.get("cuda"), str) or not runtime["cuda"]:
        raise ValueError("Missing actual CUDA runtime version")
    if runtime.get("status") != "native_dependencies_available":
        raise ValueError("Centaur runtime did not report actual allocated native devices")
    devices = runtime.get("devices", [])
    if (not isinstance(devices, list) or len(devices) < 2
            or any(not isinstance(device, dict) or not isinstance(device.get("total_memory_bytes"), int)
                   or device["total_memory_bytes"] <= 0 for device in devices)
            or sum(device["total_memory_bytes"] for device in devices) < 76 * 1024**3):
        raise ValueError("Centaur runtime requires fresh allocated GPU memory >=76 GiB over >=2 devices")
    return {"versions": {key: versions[key] for key in GPU_PACKAGES}, "cuda": runtime["cuda"]}


def _context(kind: str, destination: Path, run: Path, config: Path) -> dict:
    if kind not in FILES:
        raise ValueError("Unknown dependency environment kind")
    specification = _json(config)
    if specification.get("schema_version") != 1:
        raise ValueError("Unsupported cache schema")
    roots = [Path(raw) for raw in specification.get("project_roots", [])]
    if not roots or any(not root.is_absolute() or root.name != "adversarial-study"
                        or root.resolve(strict=True) != root for root in roots):
        raise ValueError("Project roots must be existing, absolute adversarial-study directories without aliases")
    entry = specification.get("caches", {}).get(kind)
    if not isinstance(entry, dict):
        raise ValueError(f"No pinned {kind} cache; refusing duplicate environment installation")
    tag = entry.get("source_release", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", tag):
        raise ValueError("Source release must be a release tag")
    stage, freeze_name, runtime_name = FILES[kind]
    if entry.get("stage") != stage:
        raise ValueError("Pinned stage does not match the dependency kind")
    source = _scoped(entry["environment"], roots)
    suffix = "-centaur" if kind == "centaur" else ""
    if (source.parent.name != "environments" or source.parent.parent not in roots
            or source.name != tag + suffix or not source.is_dir() or source.resolve() != source):
        raise ValueError("Cache source must be the original release environment directory")
    pins = {key: _pin(entry[key], roots) for key in ("receipt", "resolved", "runtime")}
    source_run = pins["resolved"].parent
    if (source_run.parent.name != "runs" or source_run.parent.parent not in roots or source_run.name != tag
            or pins["resolved"].name != freeze_name
            or pins["runtime"] != source_run / runtime_name
            or pins["receipt"] != source_run / "receipts" / (stage + ".json")):
        raise ValueError("Cache evidence must belong to the original source release run")
    if any(path.resolve() != path for path in pins.values()):
        raise ValueError("Pinned evidence must be original files rather than aliases to another run")
    previous = _json(pins["receipt"])
    if previous.get("stage") != stage or previous.get("status") != "completed" or previous.get("exit_code") != 0:
        raise ValueError("Previous dependency stage did not complete successfully")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(previous.get(key, "")))
           for key in ("source_manifest_sha256", "plan_sha256")):
        raise ValueError("Previous dependency stage lacks its actual source/plan pins")
    prior_identity = _runtime_identity(_json(pins["runtime"]), kind)
    run = _scoped(run, roots)
    destination = _scoped(destination, roots, exists=False)
    if (not run.is_dir() or run.parent.name != "runs" or run.parent.parent not in roots
            or run.resolve() != run or run.name == tag
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", run.name)
            or destination.parent.name != "environments" or destination.parent.parent not in roots
            or destination.parent.resolve() != destination.parent or destination.name != run.name + suffix):
        raise ValueError("Destination and run must describe the same new release")
    if not destination.parent.is_dir():
        raise ValueError("Destination environment parent must already exist")
    probes = {}
    for relative in ("pyvenv.cfg", "bin/activate"):
        path = _scoped(source / relative, roots)
        if not path.is_file() or source not in path.resolve().parents:
            raise ValueError("Cache metadata escapes its environment")
        probes[relative] = _sha(path)
    # Python in Alliance virtual environments commonly links into CVMFS.
    if not (source / "bin/python").is_file():
        raise ValueError("Cached Python interpreter is unavailable")
    return {"kind": kind, "source_release": tag, "source_environment": str(source),
            "destination_environment": str(destination), "new_run": str(run),
            "cache_config": {"path": str(config.resolve()), "sha256": _sha(config)},
            "pinned_inputs": {key: {"path": str(path), "sha256": _sha(path)} for key, path in pins.items()},
            "source_environment_metadata_sha256": probes, "runtime_identity": prior_identity,
            "previous_source_manifest_sha256": previous.get("source_manifest_sha256"),
            "previous_plan_sha256": previous.get("plan_sha256"),
            "scientific_results_reused": False, "trained_weights_reused": False,
            "current_release_tests_required": True}


def _write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def reuse(kind: str, destination: str | Path, run: str | Path, config: str | Path, *, phase: str = "link") -> dict:
    destination, run, config = Path(destination), Path(run), Path(config)
    context = _context(kind, destination, run, config)
    reuse_path = run / f"{kind}-environment-cache-reuse.json"
    verify_path = run / f"{kind}-environment-cache-verification.json"
    if phase == "link":
        if os.path.lexists(destination) or reuse_path.exists() or verify_path.exists():
            raise ValueError("Refusing to overwrite an existing environment or reuse receipt")
        os.symlink(context["source_environment"], destination, target_is_directory=True)
        context.update(status="pinned_cache_linked_runtime_recheck_required", phase="link")
        context["linked_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _write_new(reuse_path, context)
        return context
    if phase != "recheck":
        raise ValueError("Unknown cache verification phase")
    prior = _json(reuse_path)
    if any(prior.get(key) != value for key, value in context.items()):
        raise ValueError("Cache source, metadata, configuration, or destination changed after link")
    if (not destination.is_symlink() or destination.resolve(strict=True) != Path(context["source_environment"])
            or os.readlink(destination) != context["source_environment"]):
        raise ValueError("Destination no longer links to the pinned source environment")
    if Path(sys.prefix).resolve() != Path(context["source_environment"]).resolve():
        raise ValueError("Runtime recheck must execute inside the activated cached Python environment")
    _, freeze_name, runtime_name = FILES[kind]
    active_freeze, active_runtime = run / freeze_name, run / runtime_name
    if active_freeze.is_symlink() or active_runtime.is_symlink():
        raise ValueError("Current runtime evidence must be newly generated regular files")
    if _sha(active_freeze) != context["pinned_inputs"]["resolved"]["sha256"]:
        raise ValueError("Actual activated pip freeze differs from the pinned source environment")
    fresh = _json(active_runtime)
    if _runtime_identity(fresh, kind) != context["runtime_identity"]:
        raise ValueError("Actual activated runtime versions differ from the pinned source environment")
    context.update(status="active_cached_environment_verified_current_release_tests_required", phase="recheck",
                   actual_active_prefix=sys.prefix, actual_python_executable=sys.executable,
                   active_resolved={"path": str(active_freeze), "sha256": _sha(active_freeze)},
                   active_runtime={"path": str(active_runtime), "sha256": _sha(active_runtime)},
                   reuse_receipt_sha256=_sha(reuse_path),
                   verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    _write_new(verify_path, context)
    return context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=FILES, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("link", "recheck"), default="link")
    args = parser.parse_args()
    try:
        value = reuse(args.kind, args.destination, args.run, args.config, phase=args.phase)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"Dependency cache verification failed closed: {error}\n")
    print(json.dumps({"status": value["status"], "kind": args.kind, "destination": args.destination}))


if __name__ == "__main__":
    main()
