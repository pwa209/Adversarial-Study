"""Distances between qualified native clinical-model activation vectors.

This is detector representation geometry, not neural rank, anatomy or human
consciousness geometry. Every within-participant/source/SAME-CHANNEL job pair
is retained for every qualified layer, including undefined/missing metrics.
No score, distance, experience or favourable result selects a pair.

Actual MATLAB -v7 files are read lazily through scipy.io.loadmat; SciPy is an
optional runtime dependency. Files must have the batch_predict naming contract
``scores.csv.<job_id>.activations.mat`` and one activation_values struct. The
native batch receipt must bind scores, jobs, qualification, signal hashes and
the exact activation output paths/hashes. Legacy unbound activation outputs
remain unavailable for geometry rather than being certified retroactively.

Usage: python -m adversarial_study.activation_geometry --jobs native_jobs.csv
       --scores scores.csv --qualification qualification.json --output-dir NEW
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from . import native
from .core import StudyError, read_csv, read_json, sha256_file, write_csv, write_json


INTERPRETATION = (
    "Cosine distance describes flattened activations of the frozen clinical detector at one "
    "qualified layer and input channel. It is not neural rank, a neural manifold, biological "
    "coupling, experiential distance or independent evidence of consciousness. Pairs share people "
    "and episodes and are dependent descriptive observations. Saturation and pre-clipping outputs "
    "are reported separately. Each computed activation distance requires an exact native-batch "
    "output path/hash binding and a qualified layer-set match; runtime qualification does not "
    "establish clinical calibration or checkpoint variant identity."
)


def matlab_layer_field_name(name: str) -> str:
    """Conservative ASCII subset of makeValidName's default underscore mapping.

    Ordinary released layer names and punctuation substitutions are supported.
    Whitespace/case transformations, keywords, non-ASCII and truncation require
    an authoritative mapping and are refused rather than guessed. Collisions
    are checked separately because the MATLAB helper would overwrite a field.
    """
    keywords = {"break", "case", "catch", "classdef", "continue", "else", "elseif", "end", "for",
                "function", "global", "if", "otherwise", "parfor", "persistent", "return", "spmd",
                "switch", "try", "while"}
    if not isinstance(name, str) or not name or not name.isascii() or re.search(r"\s", name) or name in keywords:
        raise StudyError(f"Unsupported MATLAB layer-name transformation; authoritative field mapping needed: {name!r}")
    field = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not re.match(r"^[A-Za-z]", field):
        field = "x" + field
    if len(field) > 63:
        raise StudyError("MATLAB layer-name truncation is not inferred")
    return field


def _load_mat(path: Path) -> dict[str, np.ndarray]:
    try:
        scipy_io = importlib.import_module("scipy.io")
    except ImportError as error:
        raise StudyError("Reading native activation .mat files requires SciPy; load scipy-stack on the qualified cluster runtime") from error
    document = scipy_io.loadmat(str(path), struct_as_record=False, squeeze_me=False)
    if "activation_values" not in document:
        raise ValueError("MAT file has no activation_values variable")
    root = document["activation_values"]
    if not isinstance(root, np.ndarray) or root.size != 1:
        raise ValueError("activation_values must be one MATLAB struct")
    struct = root.reshape(-1)[0]
    fields = getattr(struct, "_fieldnames", None)
    if fields is None:
        raise ValueError("activation_values is not a MATLAB struct")
    return {name: np.asarray(getattr(struct, name)) for name in fields}


def _layer_names(qualification: Mapping[str, Any]) -> list[str]:
    names = qualification.get("qualified_layers", [])
    if isinstance(names, str):
        names = [names] if names else []
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise StudyError("qualified_layers must contain unique actual layer names")
    return names


def _vector(value: Any) -> tuple[np.ndarray | None, dict]:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not array.size or not np.isfinite(array).all():
        return None, {"status": "invalid_activation_values", "shape": list(array.shape), "dimension": int(array.size)}
    vector = np.asarray(array, dtype=np.float64).ravel(order="F")
    if not np.isfinite(vector).all():
        return None, {"status": "invalid_activation_values", "shape": list(array.shape), "dimension": int(array.size)}
    return vector, {"status": "available", "shape": list(array.shape), "dimension": int(array.size),
                    "flatten_order": "MATLAB_column_major", "zero_norm": bool(np.max(np.abs(vector)) == 0)}


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    # Scaling before the norm avoids overflow for finite large activations.
    sa, sb = float(np.max(np.abs(a))), float(np.max(np.abs(b)))
    if sa == 0 or sb == 0:
        return None
    an, bn = a / sa, b / sb
    similarity = float(np.dot(an / np.linalg.norm(an), bn / np.linalg.norm(bn)))
    return float(1. - np.clip(similarity, -1., 1.))


def _boundary(gcs: float) -> str:
    return "floor" if gcs <= 3 + 1e-7 else ("ceiling" if gcs >= 15 - 1e-7 else "interior")


def _verified_context(jobs_path: Path, scores_path: Path, qualification_path: Path) -> dict:
    """Verify the COMPLETE native batch once, before any participant partition."""
    jobs_path, scores_path, qualification_path = map(Path, (jobs_path, scores_path, qualification_path))
    jobs = native.import_scores(scores_path, jobs_path, qualification_path)
    qualification = read_json(qualification_path)
    layers = _layer_names(qualification)
    mapping = {name: matlab_layer_field_name(name) for name in layers}
    if len(set(mapping.values())) != len(mapping):
        raise StudyError("Qualified layer names collide under MATLAB makeValidName; extraction is ambiguous")
    scores = {row["job_id"]: row for row in read_csv(scores_path, ("job_id",))}
    batch_receipt = read_json(Path(str(scores_path) + ".receipt.json"))
    outputs = batch_receipt.get("activation_outputs", [])
    if isinstance(outputs, dict):
        outputs = [outputs]  # MATLAB jsonencode's singleton struct-array form.
    if not isinstance(outputs, list):
        raise StudyError("Native activation_outputs must be a struct/list registry")
    bindings = {}
    for output in outputs:
        if not isinstance(output, dict) or not all(output.get(key) for key in ("job_id", "path", "sha256")):
            raise StudyError("Malformed native activation output binding")
        if output["job_id"] in bindings or output["job_id"] not in scores:
            raise StudyError("Duplicate or unknown job in native activation output registry")
        if scores[output["job_id"]].get("extraction_status") != "forward_correspondence_verified":
            raise StudyError("Native activation binding belongs to unverified job extraction")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", str(output["sha256"])):
            raise StudyError("Invalid native activation output digest")
        bindings[output["job_id"]] = output
    for job in jobs:
        for field in ("participant_id", "source_id", "channel_name", "episode_id", "job_id"):
            if not str(job.get(field, "")).strip():
                raise StudyError(f"Activation pairs require nonempty {field}")
    qualified = (qualification.get("extraction_status") == "forward_correspondence_verified"
                 and qualification.get("activation_repeats_verified") is True and bool(layers))
    return {"jobs_path": jobs_path, "scores_path": scores_path, "qualification_path": qualification_path,
            "jobs": jobs, "qualification": qualification, "layers": layers, "mapping": mapping,
            "scores": scores, "bindings": bindings, "qualified": qualified,
            "inputs": {"jobs_path": str(jobs_path.resolve()), "jobs_sha256": sha256_file(jobs_path),
                       "scores_path": str(scores_path.resolve()), "scores_sha256": sha256_file(scores_path),
                       "qualification_path": str(qualification_path.resolve()), "qualification_sha256": sha256_file(qualification_path),
                       "batch_receipt_sha256": sha256_file(Path(str(scores_path) + ".receipt.json"))}}


def _geometry_for_jobs(context: Mapping[str, Any], jobs: list[dict]) -> dict:
    """Construct geometry for verified rows, keeping only this partition in RAM."""
    scores_path = context["scores_path"]
    qualification, layers, mapping = context["qualification"], context["layers"], context["mapping"]
    scores, bindings, qualified = context["scores"], context["bindings"], context["qualified"]
    inventory, vectors = [], {}
    for job in sorted(jobs, key=lambda row: row["job_id"]):
        ident = job["job_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
            raise StudyError("Unsafe native activation job identifier")
        path = Path(str(scores_path) + "." + ident + ".activations.mat")
        entry = {"job_id": ident, "episode_id": job["episode_id"], "participant_id": job["participant_id"],
                 "source_id": job["source_id"], "channel_name": job["channel_name"],
                 "activation_path": str(path.resolve()), "exists": path.is_file(), "layers": {}}
        if path.is_file():
            entry["observed_activation_sha256"] = sha256_file(path)
        reason = None
        if not qualified:
            reason = "extraction_not_qualified"
        elif scores[ident].get("extraction_status") != "forward_correspondence_verified":
            reason = "job_extraction_not_verified"
        elif not path.is_file():
            reason = "missing_activation_file"
        elif ident not in bindings:
            reason = "missing_native_activation_binding"
        elif Path(bindings[ident]["path"]).resolve() != path.resolve():
            reason = "native_activation_path_mismatch"
        elif bindings[ident]["sha256"].lower() != entry["observed_activation_sha256"]:
            reason = "native_activation_hash_mismatch"
        if reason is not None:
            entry["status"] = reason
            entry["layers"] = {name: {"status": reason} for name in layers}
        else:
            try:
                values = _load_mat(path)
            except StudyError:
                # Missing optional dependency is a runtime gate, not missing human data.
                raise
            except (ValueError, OSError, TypeError, NotImplementedError) as error:
                entry["status"] = "unreadable_activation_file"
                entry["error"] = str(error)
                entry["layers"] = {name: {"status": entry["status"]} for name in layers}
            else:
                entry["actual_mat_fields"] = sorted(values)
                if set(values) != set(mapping.values()):
                    entry["status"] = "qualified_layer_set_mismatch"
                    entry["layers"] = {name: {"status": entry["status"]} for name in layers}
                else:
                    entry["status"] = "qualified_layer_set_verified"
                    for name, field in mapping.items():
                        vector, descriptor = _vector(values[field])
                        entry["layers"][name] = descriptor
                        if vector is not None:
                            vectors[(ident, name)] = vector
        inventory.append(entry)
    by_id = {item["job_id"]: item for item in inventory}
    groups = defaultdict(list)
    for job in jobs:
        groups[(job["participant_id"], job["source_id"], job["channel_name"])].append(job)
    pairs = []
    eligible_job_pairs = 0
    for (participant, source, channel), group in sorted(groups.items()):
        for a, b in combinations(sorted(group, key=lambda row: (row["episode_id"], row["job_id"])), 2):
            eligible_job_pairs += 1
            for layer in layers:
                ia, ib = by_id[a["job_id"]]["layers"][layer], by_id[b["job_id"]]["layers"][layer]
                distance, reason = None, ""
                if ia["status"] != "available" or ib["status"] != "available":
                    status = "activation_unavailable"
                    reason = f"a:{ia['status']};b:{ib['status']}"
                elif ia["shape"] != ib["shape"] or ia["dimension"] != ib["dimension"]:
                    status = "shape_mismatch"
                    reason = "Layer arrays must have identical shape and dimension; no padding or remapping"
                else:
                    distance = _cosine(vectors[(a["job_id"], layer)], vectors[(b["job_id"], layer)])
                    status = "computed" if distance is not None else "undefined_zero_norm"
                pre_a, pre_b = a["preclip_gcs"], b["preclip_gcs"]
                pairs.append({"participant_id": participant, "source_id": source, "channel_name": channel,
                              "layer_name": layer, "matlab_field_name": mapping[layer],
                              "job_id_a": a["job_id"], "job_id_b": b["job_id"],
                              "episode_id_a": a["episode_id"], "episode_id_b": b["episode_id"],
                              "gcs_a": a["gcs"], "gcs_b": b["gcs"], "score_a": a["score"], "score_b": b["score"],
                              "absolute_score_difference": abs(a["score"] - b["score"]),
                              "absolute_gcs_difference": abs(a["gcs"] - b["gcs"]),
                              "preclip_gcs_a": pre_a, "preclip_gcs_b": pre_b,
                              "absolute_preclip_gcs_difference": abs(pre_a - pre_b) if pre_a is not None and pre_b is not None else None,
                              "preclip_pair_status": "both_available" if pre_a is not None and pre_b is not None else "missing_preclip",
                              "saturated_a": a["saturated"], "saturated_b": b["saturated"],
                              "boundary_a": _boundary(a["gcs"]), "boundary_b": _boundary(b["gcs"]),
                              "dimension_a": ia.get("dimension"), "dimension_b": ib.get("dimension"),
                              "cosine_distance": distance, "metric_status": status, "metric_reason": reason})
    status_counts = {name: dict(Counter(row["metric_status"] for row in pairs if row["layer_name"] == name)) for name in layers}
    computed = sum(row["metric_status"] == "computed" for row in pairs)
    result = {"schema_version": 1, "status": "activation_geometry_computed" if computed else "activation_geometry_unavailable",
              "evidence_status": "DESCRIPTIVE_DETECTOR_REPRESENTATION_MEASUREMENT", "interpretation": INTERPRETATION,
              "qualified_extraction": qualified, "qualified_layers": layers, "layer_field_mapping": mapping,
              "pair_selection": "Every unordered pair in the full input job registry within participant/source/channel; deterministic lexical order, no score/distance/outcome filtering",
              "inputs": dict(context["inputs"]),
              "complete_native_batch_job_count": len(context["jobs"]),
              "native_provenance_verification": "Complete batch imported before partitioning; per-participant fitting does not reclassify channel rows as independent humans",
              "checkpoint_sha256": qualification["checkpoint_sha256"], "extraction_helper_sha256": qualification.get("helper_sha256"),
              "source_code_sha256": sha256_file(__file__), "activation_inventory": inventory, "pairs": pairs,
              "counts": {"jobs": len(jobs), "within_person_source_channel_job_pairs": eligible_job_pairs,
                         "layer_pair_rows": len(pairs), "computed_distances": computed, "participants": len({row["participant_id"] for row in jobs})},
              "metric_status_counts_by_layer": status_counts,
              "claim_status": "No neural-rank, biological, consciousness or independent-validation claim is made"}
    json.dumps(result, allow_nan=False)
    return result


def compute_activation_geometry(jobs_path: Path, scores_path: Path, qualification_path: Path) -> dict:
    """Small-batch API: inspect actual files and retain the full pair registry.

    Missing files/extraction, missing or extra layers, nonfinite arrays, shape
    mismatch and zero norms remain typed rows with cosine_distance=None. They
    are never padded, zero-filled, silently dropped or replaced by a score.
    For large cohorts use the participant-streaming CLI, not this in-RAM API.
    """
    context = _verified_context(jobs_path, scores_path, qualification_path)
    return _geometry_for_jobs(context, context["jobs"])


def _write_geometry_report(report: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True)
    pairs = report.pop("pairs")
    write_csv(output_dir / "layer_distances.csv", pairs,
              None if pairs else ["participant_id", "source_id", "channel_name", "layer_name", "job_id_a", "job_id_b", "cosine_distance", "metric_status"])
    inventory = report.pop("activation_inventory")
    write_json(output_dir / "activation_inventory.json", inventory)
    report["distance_table_sha256"] = sha256_file(output_dir / "layer_distances.csv")
    report["activation_inventory_sha256"] = sha256_file(output_dir / "activation_inventory.json")
    write_json(output_dir / "receipt.json", report)
    return report


def export_participant_geometry(jobs_path: Path, scores_path: Path, qualification_path: Path,
                                output_dir: Path, *, pointer_path: Path | None = None) -> dict:
    """Write one person's vectors/pairs at a time after one full native import.

    Metadata participant IDs define partitions, retaining all source/channel
    strata and all pairs within each. No rows are selected by scores or geometry.
    Signal hashes are checked once for the complete batch, not reread for every
    participant. A root receipt indexes each hashed participant directory.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise StudyError("Participant geometry output directory already exists")
    context = _verified_context(jobs_path, scores_path, qualification_path)
    if pointer_path:
        context["inputs"].update({"pointer_path": str(Path(pointer_path).resolve()), "pointer_sha256": sha256_file(pointer_path)})
    partitions = defaultdict(list)
    for job in context["jobs"]:
        partitions[job["participant_id"]].append(job)
    output_dir.mkdir(parents=True)
    index = []
    totals = Counter()
    statuses = {layer: Counter() for layer in context["layers"]}
    for participant, jobs in sorted(partitions.items()):
        directory = output_dir / "participants" / hashlib.sha256(participant.encode("utf-8")).hexdigest()
        report = _geometry_for_jobs(context, jobs)
        report["partition_participant_id"] = participant
        report = _write_geometry_report(report, directory)
        totals.update(report["counts"])
        for layer, counts in report["metric_status_counts_by_layer"].items():
            statuses[layer].update(counts)
        index.append({"participant_id": participant, "directory": str(directory.resolve()),
                      "receipt_sha256": sha256_file(directory / "receipt.json"), "counts": report["counts"], "status": report["status"]})
    result = {"schema_version": 1, "mode": "participant_streaming", "status": "activation_geometry_computed" if totals["computed_distances"] else "activation_geometry_unavailable",
              "evidence_status": "DESCRIPTIVE_DETECTOR_REPRESENTATION_MEASUREMENT", "interpretation": INTERPRETATION,
              "qualified_layers": context["layers"], "qualified_extraction": context["qualified"],
              "inputs": context["inputs"], "counts": dict(totals), "participant_partitions": index,
              "metric_status_counts_by_layer": {layer: dict(counts) for layer, counts in statuses.items()},
              "complete_native_batch_job_count": len(context["jobs"]),
              "native_provenance_verification": "Complete native batch imported once before participant partitioning; all input signals verified once",
              "pair_selection": "Full participant/source/channel registry; no score, activation distance or outcome selection",
              "source_code_sha256": sha256_file(__file__)}
    write_json(output_dir / "receipt.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--jobs", type=Path)
    inputs.add_argument("--pointer", type=Path, help="Prepared JSON pointer containing jobs (absolute or relative to pointer)")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; previous results are preserved")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--by-participant", action="store_true", help="Default: stream one participant at a time into hashed subdirectories")
    mode.add_argument("--single-table", action="store_true", help="Small-batch compatibility: keep every activation/pair in RAM")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output-dir exists; use a new directory")
    jobs = args.jobs
    if args.pointer:
        pointer = read_json(args.pointer)
        if not isinstance(pointer.get("jobs"), str) or not pointer["jobs"]:
            parser.error("pointer must contain a nonempty jobs CSV path")
        jobs = Path(pointer["jobs"])
        if not jobs.is_absolute():
            jobs = args.pointer.parent / jobs
    if not args.single_table:
        export_participant_geometry(jobs, args.scores, args.qualification, args.output_dir, pointer_path=args.pointer)
        return 0
    report = compute_activation_geometry(jobs, args.scores, args.qualification)
    if args.pointer:
        report["inputs"].update({"pointer_path": str(args.pointer.resolve()), "pointer_sha256": sha256_file(args.pointer)})
    _write_geometry_report(report, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
