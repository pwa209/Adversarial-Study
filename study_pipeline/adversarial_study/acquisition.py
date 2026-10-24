"""Checksum-pinned public source acquisition; no analysis or synthetic data.

Run on an approved transfer host or a scheduled, low-priority worker. Downloads
are serial, rate-limited, resumable and published only after exact byte-count and
provider checksum verification. A resource's manifest identity, rather than an
HTTP ETag (which may be multipart), defines its scientific source identity.
No network access occurs at import. Public availability does not imply a
redistribution licence; source-specific licence status stays in the manifest.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def _target(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if not relative or parts.is_absolute() or ".." in parts.parts or "\\" in relative:
        raise ValueError("resource path must be a contained relative POSIX path")
    if any(":" in part or not part for part in parts.parts):
        raise ValueError("unsafe resource path")
    target = (root / Path(*parts.parts)).resolve()
    if target == root or root not in target.parents:
        raise ValueError("resource path escapes destination")
    return target


def validate_resource(resource: dict[str, Any]) -> None:
    for key in ("resource_id", "source_id", "relative_path", "url"):
        if not isinstance(resource.get(key), str) or not resource[key]:
            raise ValueError(f"resource requires {key}")
    url = urllib.parse.urlsplit(resource["url"])
    if url.scheme != "https" or not url.netloc or url.username or url.password:
        raise ValueError("only public HTTPS resources without URL credentials are supported")
    size = resource.get("expected_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("expected_bytes must be a positive integer")
    checksum = resource.get("checksum", {})
    algorithm = checksum.get("algorithm")
    length = {"sha256": 64, "md5": 32, "git_blob_sha1": 40}.get(algorithm)
    value = checksum.get("value", "")
    if not length or not isinstance(value, str) or len(value) != length:
        raise ValueError("provider sha256, md5, or git_blob_sha1 checksum is required")
    if any(c not in "0123456789abcdef" for c in value):
        raise ValueError("checksum must be lowercase hexadecimal")


def file_checksums(path: Path, resource: dict[str, Any]) -> dict[str, str]:
    algorithm = resource["checksum"]["algorithm"]
    provider = hashlib.new("sha1" if algorithm == "git_blob_sha1" else algorithm)
    if algorithm == "git_blob_sha1":
        provider.update(f"blob {path.stat().st_size}\0".encode())
    own = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            provider.update(chunk)
            own.update(chunk)
    return {"provider": provider.hexdigest(), "sha256": own.hexdigest()}


def verify_acquired(root: str | Path, resource: dict[str, Any]) -> dict[str, Any]:
    """Rehash a published input; a previous receipt alone is never trusted."""
    validate_resource(resource)
    root = Path(root).resolve()
    target = _target(root, resource["relative_path"])
    if not target.is_file() or target.stat().st_size != resource["expected_bytes"]:
        raise ValueError(f"missing or wrong-sized acquired input: {target}")
    hashes = file_checksums(target, resource)
    if hashes["provider"] != resource["checksum"]["value"]:
        raise ValueError(f"acquired input checksum mismatch: {target}")
    return {"resource_id": resource["resource_id"], "source_id": resource["source_id"],
            "relative_path": resource["relative_path"], "bytes": target.stat().st_size,
            "provider_checksum": resource["checksum"], "sha256": hashes["sha256"],
            "resource_manifest_sha256": canonical_hash(resource)}


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def _resource_lock(path: Path):
    """Kernel-held lock released on worker death, allowing interrupted resume.

    The lock inode remains on disk: unlinking a live lock file would let another
    process lock a new inode for the same resource. Its presence alone never
    signifies an active worker. POSIX scheduled workers use flock; Windows uses
    a one-byte nonblocking file lock.
    """
    stream = path.open("a+b")
    if path.stat().st_size == 0:
        stream.write(b"\0")
        stream.flush()
    stream.seek(0)
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def acquire_manifest(manifest: str | Path | dict[str, Any], destination: str | Path,
                     *, resource_ids: list[str] | None = None,
                     source_ids: list[str] | None = None,
                     max_bytes_per_second: float = 8 * 1024 * 1024,
                     max_bytes: int = 1_000_000_000_000,
                     max_files: int = 150_000, request_interval_s: float = 1.0,
                     retries: int = 3, timeout_s: float = 90.0) -> dict[str, Any]:
    """Acquire selected immutable resources, or the manifest's pilot defaults.

    Byte/file budgets apply to the declared selected footprint, including
    previously acquired files, and separately bound newly transferred bytes.
    Selection by source ID acquires every listed available file for that source;
    it imposes no participant or favourable-outcome gate. Incomplete `.part`
    files survive interruptions; conflicting identities/ETags fail explicitly.
    """
    for name, value in (("rate", max_bytes_per_second), ("request interval", request_interval_s),
                        ("timeout", timeout_s)):
        if not math.isfinite(value) or value < 0 or (name != "request interval" and value == 0):
            raise ValueError(f"invalid {name}")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (max_bytes, max_files)):
        raise ValueError("positive integer resource budgets are required")
    if not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a nonnegative integer")
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    resources = manifest.get("resources", [])
    if not isinstance(resources, list):
        raise ValueError("manifest resources must be a list")
    known_ids, known_paths = set(), set()
    root = Path(destination).resolve()
    for resource in resources:
        validate_resource(resource)
        path = _target(root, resource["relative_path"])
        if resource["resource_id"] in known_ids or path in known_paths:
            raise ValueError("duplicate resource ID or destination path")
        known_ids.add(resource["resource_id"])
        known_paths.add(path)
    requested = set(resource_ids or [])
    requested_sources = set(source_ids or [])
    if requested - known_ids or requested_sources - {r["source_id"] for r in resources}:
        raise ValueError("requested resource/source IDs are absent from manifest")
    selected = [r for r in resources if r["resource_id"] in requested or
                r["source_id"] in requested_sources or
                (not requested and not requested_sources and r.get("default_acquire") is True)]
    if not selected:
        raise ValueError("selection contains no resources")
    total = sum(r["expected_bytes"] for r in selected)
    if len(selected) > max_files or total > max_bytes:
        raise ValueError(f"selected footprint exceeds budget: {len(selected)} files, {total} bytes")
    root.mkdir(parents=True, exist_ok=True)
    transferred = 0
    receipts = []
    last_request = -math.inf
    for resource in selected:
        target = _target(root, resource["relative_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = target.with_name(target.name + ".acquisition.lock")
        ownership = _resource_lock(lock)
        ownership.__enter__()
        try:
            if target.exists():
                receipt = verify_acquired(root, resource)
                receipt["status"] = "verified_existing"
                receipts.append(receipt)
                continue
            part = target.with_name(target.name + ".part")
            identity_path = target.with_name(target.name + ".part.json")
            identity = {"resource_manifest_sha256": canonical_hash(resource), "etag": None}
            if part.exists():
                if not identity_path.exists():
                    raise ValueError("partial file has no pinned identity; manual review required")
                previous = json.loads(identity_path.read_text(encoding="utf-8"))
                if previous.get("resource_manifest_sha256") != identity["resource_manifest_sha256"]:
                    raise ValueError("partial download identity changed")
                identity = previous
            else:
                _write_json_atomic(identity_path, identity)
            if part.exists() and part.stat().st_size > resource["expected_bytes"]:
                raise ValueError("partial file exceeds expected resource size")
            for attempt in range(retries + 1):
                start = part.stat().st_size if part.exists() else 0
                if start == resource["expected_bytes"]:
                    break
                wait = request_interval_s - (time.monotonic() - last_request)
                if wait > 0:
                    time.sleep(wait)
                headers = {"User-Agent": "adversarial-study-acquisition/1", "Accept-Encoding": "identity"}
                if start:
                    headers["Range"] = f"bytes={start}-"
                    if identity.get("etag"):
                        headers["If-Range"] = identity["etag"]
                request = urllib.request.Request(resource["url"], headers=headers)
                last_request = time.monotonic()
                try:
                    with urllib.request.urlopen(request, timeout=timeout_s) as response:
                        status = response.status
                        etag = response.headers.get("ETag")
                        if identity.get("etag") and etag and identity["etag"] != etag:
                            raise ValueError("server identity changed during partial download")
                        if response.headers.get("Content-Encoding", "identity") not in ("identity", ""):
                            raise ValueError("encoded response cannot preserve source byte identity")
                        if status == 206:
                            import re
                            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                                 response.headers.get("Content-Range", ""))
                            if not match or int(match[1]) != start or int(match[3]) != resource["expected_bytes"]:
                                raise ValueError("invalid resumed Content-Range")
                            if int(match[2]) < start or int(match[2]) >= resource["expected_bytes"]:
                                raise ValueError("invalid resumed byte endpoint")
                        elif status == 200:
                            start = 0  # Range ignored: replace partial, never append full response.
                        else:
                            raise ValueError(f"unexpected download status {status}")
                        length = response.headers.get("Content-Length")
                        if length and (int(length) < 0 or int(length) > resource["expected_bytes"] - start):
                            raise ValueError("server response exceeds expected byte count")
                        identity["etag"] = etag or identity.get("etag")
                        _write_json_atomic(identity_path, identity)
                        beginning = time.monotonic()
                        response_bytes = 0
                        with part.open("ab" if start else "wb") as stream:
                            while True:
                                chunk = response.read(1024 * 1024)
                                if not chunk:
                                    break
                                if start + response_bytes + len(chunk) > resource["expected_bytes"]:
                                    raise ValueError("response exceeds pinned resource size")
                                if transferred + len(chunk) > max_bytes:
                                    raise ValueError("actual transferred-byte budget exhausted")
                                stream.write(chunk)
                                transferred += len(chunk)
                                response_bytes += len(chunk)
                                delay = response_bytes / max_bytes_per_second - (time.monotonic() - beginning)
                                if delay > 0:
                                    time.sleep(delay)
                            stream.flush()
                            os.fsync(stream.fileno())
                    if part.stat().st_size != resource["expected_bytes"]:
                        raise OSError("truncated source response; partial retained")
                    break
                except (urllib.error.URLError, OSError) as error:
                    if attempt == retries:
                        raise
                    delay = min(2 ** attempt, 30)
                    if isinstance(error, urllib.error.HTTPError):
                        if error.code not in (408, 429, 500, 502, 503, 504):
                            raise
                        retry_after = error.headers.get("Retry-After", "")
                        if retry_after.isdigit():
                            delay = max(delay, min(int(retry_after), 60))
                    time.sleep(delay)
            hashes = file_checksums(part, resource)
            if part.stat().st_size != resource["expected_bytes"] or hashes["provider"] != resource["checksum"]["value"]:
                raise ValueError("download checksum/size mismatch; partial retained for review")
            # Exclusive hardlink publication prevents an accidental concurrent overwrite.
            os.link(part, target)
            part.unlink()
            identity_path.unlink()
            receipt = verify_acquired(root, resource)
            receipt["status"] = "published_verified"
            _write_json_atomic(target.with_name(target.name + ".acquisition.json"), receipt)
            receipts.append(receipt)
        finally:
            ownership.__exit__(None, None, None)
    result = {"schema_version": 1, "manifest_sha256": canonical_hash(manifest),
              "selected_files": len(selected), "selected_bytes": total,
              "transferred_bytes": transferred, "resources": receipts,
              "synthetic": False, "qualification": "source byte identity only; signal/event qualification is separate"}
    _write_json_atomic(root / f"acquisition_receipt_{time.time_ns()}.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--resource", action="append", default=[])
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--max-bytes-per-second", type=float, default=8 * 1024 * 1024)
    parser.add_argument("--max-bytes", type=int, default=1_000_000_000_000)
    parser.add_argument("--max-files", type=int, default=150_000)
    parser.add_argument("--request-interval-s", type=float, default=1.0)
    args = parser.parse_args(argv)
    result = acquire_manifest(args.manifest, args.destination, resource_ids=args.resource,
                              source_ids=args.source_id, max_bytes_per_second=args.max_bytes_per_second,
                              max_bytes=args.max_bytes, max_files=args.max_files,
                              request_interval_s=args.request_interval_s)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
