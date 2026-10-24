"""Acquire the pinned Figshare v2 release without altering published sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API_URL = "https://api.figshare.com/v2/articles/30906557/versions/2"
DOI = "10.6084/m9.figshare.30906557.v2"
DETECTOR_FILES = {
    "trained_consciousness_detector.mat", "predict_consciousness_level.m",
    "remap_gcs_scale.m", "BoundedOutputLayer.m",
}
REQUIRED_SOURCE = {"main.cpp", "solver.cpp", "dendrite.cpp", "dendrite.h",
                   "dendrite_scaled.cpp", "dendrite_scaled.h"}
BASELINE = "conscious_brain_configuration_file.conf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _read_url(url: str, limit: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "AdversarialStudy-native-provenance/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        result = response.read(limit + 1)
    if len(result) > limit:
        raise ValueError("Remote payload exceeds declared size/limit")
    return result


def selected_files(metadata: dict, include_detector: bool = False,
                   include_config_bank: bool = False) -> list[tuple[dict, str]]:
    if metadata.get("id") != 30906557 or metadata.get("doi") != DOI:
        raise ValueError("Metadata must identify pinned article 30906557 version 2")
    selected, seen = [], set()
    for item in metadata.get("files", []):
        name = item.get("name", "")
        if not isinstance(name, str) or not name or Path(name).name != name or "\\" in name or "/" in name:
            raise ValueError("Release filenames must be safe flat basenames")
        target = None
        if name.endswith((".cpp", ".h")):
            target = "src/" + name
        elif name in {"Makefile", "README.md", "README.txt", "LICENSE", "license.txt"}:
            target = name
        elif name == BASELINE or (include_config_bank and re.fullmatch(r"doc_config\d+\.conf", name)):
            target = "configuration_files/" + name
        elif include_detector and name in DETECTOR_FILES:
            target = name
        if target is None:
            continue
        if target in seen:
            raise ValueError("Duplicate selected release filename")
        seen.add(target)
        file_id, size = item.get("id"), item.get("size")
        digest = item.get("computed_md5")
        if isinstance(file_id, bool) or not isinstance(file_id, int) or file_id <= 0:
            raise ValueError("Missing numeric Figshare file ID")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= 10_000_000:
            raise ValueError("Invalid or unexpectedly large selected file")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{32}", digest):
            raise ValueError("Missing Figshare computed MD5")
        # Never follow an arbitrary URL supplied by metadata.
        expected = f"https://ndownloader.figshare.com/files/{file_id}"
        if item.get("download_url") != expected or item.get("is_link_only", False):
            raise ValueError("Selected file must be an actual Figshare download")
        selected.append((item, target))
    basenames = {Path(target).name for _, target in selected}
    required = REQUIRED_SOURCE | {BASELINE, "LICENSE", "license.txt"}
    if include_detector:
        required |= DETECTOR_FILES
    if not required <= basenames:
        raise ValueError("Pinned release lacks required sources, baseline, licenses, or detector")
    return sorted(selected, key=lambda pair: pair[1])


def acquire_release(destination: Path, metadata: dict, *, include_detector: bool = False,
                    include_config_bank: bool = False, reader=_read_url) -> dict:
    entries = selected_files(metadata, include_detector, include_config_bank)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "SOURCE_MANIFEST.json"
    if manifest_path.exists():
        raise ValueError("Preserve existing source manifest; use a new destination")
    records = []
    for item, relative in entries:
        path = destination / relative
        if path.is_symlink() or not path.resolve().is_relative_to(destination):
            raise ValueError("Refuse symlink in release destination")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            payload = path.read_bytes()
        else:
            payload = reader(item["download_url"], item["size"])
        md5 = hashlib.md5(payload).hexdigest()
        if len(payload) != item["size"] or md5 != item["computed_md5"].lower():
            raise ValueError("File size/MD5 differs from pinned metadata: " + relative)
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(payload)
        records.append({"path": relative, "file_id": item["id"], "size": len(payload),
                        "computed_md5": md5, "sha256": hashlib.sha256(payload).hexdigest(),
                        "download_url": item["download_url"]})
    manifest = {"status": "pinned_payload_verified", "article_id": 30906557, "version": 2,
                "doi": DOI, "metadata_api": API_URL,
                "metadata_sha256": hashlib.sha256(json.dumps(metadata, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest(),
                "acquired_at_utc": datetime.now(timezone.utc).isoformat(), "files": records,
                "source_layout": "published flat C++ files reconstructed under src; bytes unchanged",
                "licenses": ["LICENSE", "license.txt"],
                "clinical_validation": "not_established", "native_execution": "not_yet_tested"}
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, help="Pinned local API metadata; otherwise fetch v2 API")
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--include-detector", action="store_true")
    parser.add_argument("--include-config-bank", action="store_true")
    args = parser.parse_args(argv)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig")) if args.metadata else json.loads(_read_url(API_URL, 2_000_000))
    result = acquire_release(args.destination, metadata, include_detector=args.include_detector,
                             include_config_bank=args.include_config_bank)
    print(json.dumps({"status": result["status"], "files": len(result["files"]),
                      "manifest": str(args.destination.resolve() / "SOURCE_MANIFEST.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
