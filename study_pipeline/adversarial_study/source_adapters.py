"""Audited development-source adapters, without filtering or re-referencing.

Monash converts original checksum-pinned BrainVision float32 multiplexed files.
Tactile converts checksum-pinned continuous EEGLAB using optional MNE. EEG units
are microvolts; auxiliary channels remain separate, with explicit source units.
Ten-second EEG predictor intervals never become ten-second awareness reports.
Original codes, missingness, timing uncertainty and every candidate disposition
remain visible. These are development data, not protected participant validation.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import re
from typing import Any
import zipfile

import numpy as np

from .acquisition import canonical_hash, verify_acquired
from .core import RECORDING_COLUMNS, sha256_file, write_csv, write_json

FIXED_INSTRUMENT_CHANNELS = ["Cz", "Pz", "Fz", "Oz"]
MONASH_CODE_COMMIT = "176724bde9258d3a28d7ceea5aa0843928239130"
MONASH_CODE_URL = ("https://github.com/andrillon/wanderIM/blob/" + MONASH_CODE_COMMIT +
                   "/eeg/wanderIM_preproc_eeg_parfor_v3.m")
REPORT_INSTRUCTION_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12771569/"
EPISODE_COLUMNS = ["episode_id", "recording_id", "start_s", "stop_s", "report_start_s",
                   "report_stop_s", "report_category", "capacity_value"]


def _read_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        result = list(reader)
    if any(None in row or None in row.values() for row in result):
        raise ValueError("malformed source TSV")
    return result


def _number(value: Any, *, missing: bool = False) -> float | None:
    if missing and (value is None or str(value).strip().lower() in ("", "n/a", "nan")):
        return None
    number = float(value)
    if not math.isfinite(number):
        if missing and math.isnan(number):
            return None
        raise ValueError("source field must be finite")
    return number


def _decode_header(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        # Public MWI306 declares UTF-8 but contains CP1252 0xB5 micro signs.
        return data.decode("cp1252"), "cp1252_fallback_despite_declared_codepage"


def parse_brainvision_header(path: str | Path) -> dict[str, Any]:
    """Strictly qualify the supported original binary schema and channel gains.

    BrainVision resolutions convert stored values to stated physical units.
    This adapter supports only multiplexed IEEE_FLOAT_32 and explicit V/mV/uV
    gains. The original acquisition reference must occur in source comments;
    absence cannot silently become average reference or an assumed AFz.
    """
    path = Path(path)
    text, encoding = _decode_header(path)
    sections: dict[str, dict[str, str]] = {}
    section = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            sections.setdefault(section, {})
        elif section and "=" in line and not line.startswith(";"):
            key, value = line.split("=", 1)
            sections[section][key.strip()] = value.strip()
    common = sections.get("Common Infos", {})
    if common.get("DataFormat") != "BINARY" or common.get("DataOrientation") != "MULTIPLEXED":
        raise ValueError("unsupported BrainVision data format/orientation")
    if sections.get("Binary Infos", {}).get("BinaryFormat") != "IEEE_FLOAT_32":
        raise ValueError("unsupported BrainVision binary precision")
    count = int(common["NumberOfChannels"])
    interval = _number(common["SamplingInterval"])
    if count < 1 or interval <= 0:
        raise ValueError("invalid BrainVision sampling dimensions")
    reference = re.search(r"^\s*Reference Channel Name\s*=\s*(\S+)\s*$", text, re.MULTILINE)
    if not reference:
        raise ValueError("acquisition reference absent from original header; explicit qualification required")
    names, gains, units = [], [], []
    for i in range(1, count + 1):
        fields = sections.get("Channel Infos", {}).get(f"Ch{i}", "").split(",")
        if len(fields) < 4:
            raise ValueError("incomplete BrainVision channel description")
        name, _, resolution, unit = fields[:4]
        name = name.replace("\\1", ",")
        unit = unit.replace("µ", "u").replace("μ", "u")
        conversion = {"uV": 1.0, "mV": 1000.0, "V": 1e6}.get(unit)
        gain = _number(resolution)
        if not name or conversion is None or gain <= 0:
            raise ValueError("channel physical units/gain cannot be qualified")
        names.append(name)
        gains.append(gain * conversion)
        units.append(unit)
    if len(set(names)) != count:
        raise ValueError("duplicate source channel labels")
    for key in ("DataFile", "MarkerFile"):
        filename = common.get(key, "")
        if Path(filename).name != filename or "\\" in filename or ":" in filename or not filename:
            raise ValueError("BrainVision dependencies must be contained sibling filenames")
    return {"sfreq_hz": 1e6 / interval, "channel_names": names, "gain_uv": gains,
            "source_units": units, "data_file": common["DataFile"],
            "marker_file": common["MarkerFile"], "reference": reference[1],
            "header_encoding": encoding, "binary_format": "IEEE_FLOAT_32",
            "orientation": "MULTIPLEXED", "source_channel_count": count}


def parse_brainvision_markers(path: str | Path, sfreq_hz: float,
                             sample_count: int) -> list[dict[str, Any]]:
    """Keep original source marker codes; convert one-based sample positions."""
    text, _ = _decode_header(Path(path))
    rows = []
    for line in text.splitlines():
        if not re.match(r"^Mk\d+=", line):
            continue
        fields = line.split("=", 1)[1].split(",")
        if len(fields) < 5:
            raise ValueError("invalid BrainVision marker")
        index, size = int(fields[2]) - 1, int(fields[3])
        if index < 0 or index >= sample_count or size < 1:
            raise ValueError("marker position outside genuine recording")
        rows.append({"marker_type": fields[0].replace("\\1", ","),
                     "description": fields[1].replace("\\1", ","),
                     "sample_index": index, "size_samples": size,
                     "onset_s": index / sfreq_hz, "original_sample_position": index + 1})
    if not rows or any(b["sample_index"] < a["sample_index"] for a, b in zip(rows, rows[1:])):
        raise ValueError("missing or unsorted BrainVision markers")
    return rows


def monash_probe_episodes(probe_res: np.ndarray, markers: list[dict], *,
                          recording_id: str, sfreq_hz: float, duration_s: float,
                          max_clock_residual_s: float = 0.1,
                          max_clock_drift_ppm: float = 500.0,
                          require_end_markers: bool = True,
                          full_recording_duration_s: float | None = None,
                          eeg_anchor_basis: str = "original_P_marker_uncorrected_for_photodiode_lag") -> tuple[list[dict], list[dict], dict]:
    """Map original rows to P markers with count/block/clock checks.

    The P marker anchors the task interruption; author photodiode correction is
    NOT reproduced. Ordinal alignment must satisfy identical probe counts,
    original block/probe sequencing, and an affine clock residual/drift bound.
    Ten seconds before that marker is an analyst EEG interval. Reports remain
    point-linked retrospective state reports, with no asserted duration.
    Original state4 stays XX (no-recall wording is separately source-defined).
    Original Q4 awareness rating1fully..4not-at-all stays an ordinal self-rating.
    Capacity stays missing pending qualification of a separate behavioural assay.
    """
    matrix = np.asarray(probe_res, dtype=float)
    if not math.isfinite(sfreq_hz) or sfreq_hz <= 0 or abs(10*sfreq_hz-round(10*sfreq_hz)) > 1e-8:
        raise ValueError("ten-second EEG interval is not on the source sample grid")
    for name, value in (("duration", duration_s), ("clock residual bound", max_clock_residual_s),
                        ("clock drift bound", max_clock_drift_ppm)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid {name}")
    if matrix.ndim != 2 or matrix.shape[1] != 38 or matrix.shape[0] < 2:
        raise ValueError("Monash original probe_res must have38columns and at least2rows")
    starts = [m for m in markers if m["marker_type"] == "P"]
    ends = [m for m in markers if m["marker_type"] == "C"]
    if len(starts) != matrix.shape[0] or (require_end_markers and len(ends) != len(starts)):
        raise ValueError("probe/response table/marker counts differ; no ordinal guessing")
    if not require_end_markers and ends:
        raise ValueError("optional end-marker mode expects no observed end markers")
    if full_recording_duration_s is not None and (not math.isfinite(full_recording_duration_s) or
                                                 full_recording_duration_s < duration_s):
        raise ValueError("invalid full source recording duration")
    # Bounds are explicit recording bookkeeping when source FIF has no C markers;
    # these are never emitted as invented source probe-end events.
    end_bounds = [m["onset_s"] for m in ends] if require_end_markers else [
        m["onset_s"] for m in starts[1:]] + [full_recording_duration_s or duration_s]
    for i, row in enumerate(matrix):
        block, number = row[3], row[0]
        if not np.isfinite([block, number, row[1], row[2]]).all():
            raise ValueError("missing original block/probe clock fields")
        if block < 1 or block != int(block) or number < 1 or number != int(number):
            raise ValueError("invalid original block/probe index")
        if i == 0 and number != 1:
            raise ValueError("first original probe must be1")
        if i and not ((block == matrix[i-1, 3] and number == matrix[i-1, 0] + 1) or
                      (block == matrix[i-1, 3] + 1 and number == 1)):
            raise ValueError("original probe sequence has omissions/reordering")
        if not starts[i]["onset_s"] < end_bounds[i]:
            raise ValueError("probe end precedes its source onset")
        if require_end_markers and i + 1 < len(starts) and end_bounds[i] >= starts[i+1]["onset_s"]:
            raise ValueError("overlapping source probes")
    source_clock = matrix[:, 1]
    eeg_clock = np.array([m["onset_s"] for m in starts])
    if np.any(np.diff(source_clock) <= 0):
        raise ValueError("original behaviour probe clock is not increasing")
    design = np.column_stack([source_clock - source_clock[0], np.ones(len(source_clock))])
    slope, intercept = np.linalg.lstsq(design, eeg_clock, rcond=None)[0]
    residual = eeg_clock - design @ np.array([slope, intercept])
    maximum = float(np.max(np.abs(residual)))
    if maximum > max_clock_residual_s or abs(slope - 1) * 1e6 > max_clock_drift_ppm:
        raise ValueError("source behaviour/EEG clocks fail alignment bounds")
    episodes, dispositions = [], []
    labels = {1: "ON", 2: "MW", 3: "MB", 4: "XX_no_recall_original_code4"}
    for i, (row, marker, end_bound) in enumerate(zip(matrix, starts, end_bounds)):
        point = marker["onset_s"]
        state = _number(row[31], missing=True)
        if state is not None and (state != int(state) or int(state) not in labels):
            raise ValueError("unknown original Monash mind-state code")
        episode_id = f"{recording_id}_probe_{i+1:03d}"
        disposition = {"episode_id": episode_id, "source_probe_index": i+1,
                       "source_point_s": point, "status": "included"}
        if point < 10.0 or point > duration_s:
            disposition["status"] = "outside_selected_continuous_extent"
            dispositions.append(disposition)
            continue
        report_answer = _number(row[15], missing=True)  # original Q2 answer GetSecs
        if report_answer is not None:
            report_answer = float(slope * (report_answer-source_clock[0]) + intercept)
            if report_answer < point - max_clock_residual_s or report_answer > end_bound + max_clock_residual_s:
                raise ValueError("Q2 answer lies outside matched source probe")
        awareness = _number(row[33], missing=True)
        if awareness is not None and (awareness != int(awareness) or awareness not in (1, 2, 3, 4)):
            raise ValueError("invalid source Q4 ordinal awareness rating")
        episode = {"episode_id": episode_id, "recording_id": recording_id,
                   "start_s": (marker["sample_index"] - round(10*sfreq_hz))/sfreq_hz,
                   "stop_s": point, "report_start_s": point, "report_stop_s": point,
                   "report_point_s": point, "report_answer_s": report_answer,
                   "report_category": labels[int(state)] if state is not None else "missing_original_response",
                   "capacity_value": None, "capacity_construct": "not_supplied_separate_assay_required",
                   "capacity_units": "not_supplied",
                   "awareness_rating_raw": awareness,
                   "awareness_rating_construct": "original_Q4_self_rated_awareness",
                   "awareness_rating_units": "ordinal1fully..4not_at_all",
                   "original_state_code": state, "original_state_label": "XX" if state == 4 else labels.get(state, "missing"),
                   "source_block": int(row[3]), "source_probe_within_block": int(row[0]),
                   "source_task_code": _number(row[4]),
                   "report_timing_basis": "point-linked_retrospection_before_source_P_marker_duration_unspecified",
                   "report_instruction_reference": REPORT_INSTRUCTION_URL,
                   "eeg_anchor_basis": eeg_anchor_basis,
                   "original_annotation_point_s": marker.get("original_annotation_onset_s", point),
                   "event_sample_grid_offset_s": point-marker.get("original_annotation_onset_s", point),
                   "analysis_window_basis": "analyst_selected10s_pre_probe_not_experience_duration",
                   "original_look_rating_Q1": _number(row[30], missing=True),
                   "original_origin_rating_Q3": _number(row[32], missing=True),
                   "original_intention_rating_Q5": _number(row[34], missing=True),
                   "original_engagement_rating_Q6": _number(row[35], missing=True),
                   "original_performance_rating_Q7": _number(row[36], missing=True),
                   "original_alertness_rating_Q8": _number(row[37], missing=True)}
        episodes.append(episode)
        dispositions.append(disposition)
    return episodes, dispositions, {"clock_slope": float(slope), "clock_drift_ppm": float((slope-1)*1e6),
                                   "clock_intercept_at_first_behaviour_probe_s": float(intercept),
                                   "max_clock_residual_s": maximum, "matched_probe_count": len(starts),
                                   "alignment_reference": MONASH_CODE_URL,
                                   "photodiode_correction_applied": False,
                                   "observed_probe_end_markers_checked": require_end_markers}


def paris_annotation_markers(raw: Any) -> list[dict]:
    """Map pinned author Paris probe annotation S3 to nearest resampled grid.

    The original annotation time stays visible. Author resampling to250Hz may
    place a former500Hz event between samples; nearest-sample offset is recorded
    explicitly (at most half one current sample), not silently treated as exact
    physical probe timing. Other annotation descriptions never become probes.
    """
    sfreq = float(raw.info["sfreq"])
    first = float(getattr(raw, "first_time", 0.0))
    result = []
    for onset, description in zip(raw.annotations.onset, raw.annotations.description):
        if not re.fullmatch(r"Stimulus/S\s+3", str(description)):
            continue
        original = float(onset)-first
        index = round(original*sfreq)
        if index < 0 or index >= raw.n_times or abs(index/sfreq-original) > .500001/sfreq:
            raise ValueError("Paris original annotation cannot be mapped to sample grid")
        result.append({"marker_type": "P", "description": str(description), "sample_index": index,
                       "onset_s": index/sfreq, "original_annotation_onset_s": original})
    if not result or any(b["sample_index"] <= a["sample_index"] for a, b in zip(result, result[1:])):
        raise ValueError("Paris S3 annotations missing, duplicated, or unsorted")
    return result


def _extract_member(archive: Path, member: str, destination: Path,
                    max_member_bytes: int = 4_000_000_000) -> dict:
    """Extract one named member of an already checksum-verified source ZIP.

    Output is a caller-chosen contained filename, never an archive-supplied path.
    Size, CRC (zipfile), duplicate member names, and total received bytes are
    checked. No hidden Mac resource fork or unsupported split FIF is promoted.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        matches = [info for info in source.infolist() if info.filename == member]
        if len(matches) != 1:
            raise ValueError("missing or duplicate source ZIP member")
        info = matches[0]
        if info.is_dir() or info.file_size < 1 or info.file_size > max_member_bytes:
            raise ValueError("source ZIP member exceeds declared extraction bound")
        if destination.exists():
            raise ValueError("extracted source filename already exists")
        received = 0
        with source.open(info) as stream, destination.open("xb") as output:
            for block in iter(lambda: stream.read(1024*1024), b""):
                received += len(block)
                if received > info.file_size:
                    raise ValueError("ZIP member exceeds central-directory size")
                output.write(block)
        if received != info.file_size:
            raise ValueError("truncated ZIP member")
    return {"archive_member": member, "uncompressed_bytes": received, "zip_crc32": f"{info.CRC:08x}",
            "extracted_sha256": sha256_file(destination), "archive_member_pin": "CRC and SHA256 within checksum-pinned source archive"}


def _paris_one(candidate: dict, *, output: Path, maximum: float | None,
                archive_receipts: list[dict]) -> tuple[dict, list[dict], dict, list[dict]]:
    subject = candidate["subject"]
    folder = output/"_verified_archive_members"/subject
    fif = folder/("sub-"+subject+"_ica_raw.fif")
    behaviour = folder/("wanderIM_behavres_s"+subject+".mat")
    member_receipts = [_extract_member(Path(candidate["eeg_archive"]), candidate["eeg_member"], fif),
                       _extract_member(Path(candidate["behaviour_archive"]), candidate["behaviour_member"], behaviour)]
    mne = importlib.import_module("mne")
    raw = mne.io.read_raw_fif(str(fif), preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    if abs(sfreq-250.0) > 1e-8:
        raise ValueError("Paris archive does not match pinned author250Hz derivative schema")
    if not raw.info.get("custom_ref_applied", False):
        raise ValueError("Paris average-reference assertion not supported by embedded custom-reference flag")
    types = raw.get_channel_types()
    eeg_indices = [i for i, (name, kind) in enumerate(zip(raw.ch_names, types))
                   if kind == "eeg" and name not in ("EOG", "ECG", "D1") and not name.startswith("BIP")]
    aux_indices = [i for i, kind in enumerate(types) if i not in eeg_indices and kind in ("eeg", "eog", "ecg")]
    unknown_indices = [i for i in range(len(types)) if i not in eeg_indices and i not in aux_indices]
    selected = _extent(raw.n_times, sfreq, maximum)
    eeg = raw.get_data(picks=eeg_indices, start=0, stop=selected)*1e6
    aux = raw.get_data(picks=aux_indices, start=0, stop=selected)*1e6
    names = [raw.ch_names[i] for i in eeg_indices]
    aux_names = [raw.ch_names[i] for i in aux_indices]
    markers = paris_annotation_markers(raw)
    scipy_io = importlib.import_module("scipy.io")
    original = scipy_io.loadmat(behaviour, squeeze_me=True, struct_as_record=False)
    recording_id = "mindblanking_paris_"+subject
    episodes, dispositions, alignment = monash_probe_episodes(
        original["probe_res"], markers, recording_id=recording_id, sfreq_hz=sfreq,
        duration_s=selected/sfreq, require_end_markers=False,
        full_recording_duration_s=raw.n_times/sfreq,
        eeg_anchor_basis="source_Stimulus_S3_annotation_author_resampled_grid_offset_recorded")
    gaps = []
    for onset, duration, description in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if any(term in str(description).lower() for term in ("bad", "edge", "boundary")):
            onset = float(onset)-float(getattr(raw, "first_time", 0.0))
            if onset < selected/sfreq:
                gaps.append([max(0.0, onset), min(selected/sfreq, onset+max(float(duration), 1/sfreq))])
    reference = "author_pipeline_average_reference_continuous_ICA_derivative"
    bad_names = sorted(set(raw.info.get("bads", [])) & set(names))
    path = _write_npz(output, recording_id, eeg, names, sfreq, reference, aux, aux_names,
                      np.asarray(gaps, dtype=float).reshape(-1, 2), bad_names)
    unknown_path = None
    if unknown_indices:
        unknown_path = output/"signals"/f"{recording_id}.unqualified_aux.npz"
        unknown = raw.get_data(picks=unknown_indices, start=0, stop=selected)
        if not np.isfinite(unknown).all():
            raise ValueError("nonfinite original Paris auxiliary samples")
        np.savez(unknown_path, reader_values=unknown,
                 channel_names=np.asarray([raw.ch_names[i] for i in unknown_indices], dtype="U"),
                 reader_channel_types=np.asarray([types[i] for i in unknown_indices], dtype="U"),
                 units_qualification=np.asarray("physical_units_not_qualified_MNE_reader_values_preserved"))
    record = _recording_row(recording_id, recording_id, "mindblanking_paris", path, output,
                            sfreq, eeg.shape, "original_session", reference)
    record["site_id"] = "Paris"
    record["source_conversion"] = "gain_normalization_only_author_preprocessing_retained"
    audit = {"recording_id": recording_id, "source_archive_inputs": archive_receipts,
             "archive_member_inputs": member_receipts, "source_signal_kind": "continuous_preprocessed_FIF_not_original_raw",
             "source_author_preprocessing": "250Hz_1to60Hz_notch50and100_FastICA95percent_average_reference",
             "source_author_code_commit": "f498ee9d39c3f1ffbbdd0a3c608fe2ae763843bf",
             "embedded_custom_ref_applied": bool(raw.info.get("custom_ref_applied")),
             "embedded_projector_count": len(raw.info.get("projs", [])),
             "reference_claim_basis": "pinned_author_pipeline_plus_embedded_custom_reference_flag; flag alone does not identify electrode scheme",
             "adapter_filtering_applied": False, "adapter_reference_changed": False,
             "adapter_resampling_applied": False, "source_license": "CC-BY-4.0",
             "alignment": alignment, "original_sample_count": raw.n_times,
             "selected_sample_count": selected, "scalp_channels": names, "auxiliary_channels": aux_names,
             "unqualified_aux_file": str(unknown_path) if unknown_path else None,
             "unqualified_aux_sha256": sha256_file(unknown_path) if unknown_path else None,
             "output_sha256": sha256_file(path), "synthetic": False}
    if hasattr(raw, "close"):
        raw.close()
    return record, episodes, audit, dispositions


def tactile_trial_episodes(events: list[dict], *, recording_id: str, sfreq_hz: float,
                           duration_s: float, capacity: str = "confidence") -> tuple[list[dict], list[dict]]:
    """Predict subsequent per-event detection/confidence from10s before stimulus.

    The source stim-adapt marker is the anchor; actual physical/acoustic onset
    alignment remains unqualified. First-order response and later confidence
    report times remain separate. Hit/miss/fa/cr retain their original labels.
    Metadata says confidence units percent while released numeric values may
    be fractions: retain source values and units discrepancy, without scaling.
    """
    if capacity not in ("confidence", "detection"):
        raise ValueError("capacity must be confidence or detection")
    if (not math.isfinite(sfreq_hz) or sfreq_hz <= 0 or
            abs(10*sfreq_hz-round(10*sfreq_hz)) > 1e-8 or
            not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("invalid tactile duration/sampling grid")
    required = {"onset", "trial_type", "event_sample", "stimamp", "stimon", "confidence",
                "response_time", "response_time2", "sdt"}
    if not events or any(not required <= set(row) for row in events):
        raise ValueError("original tactile event schema is incomplete")
    onsets = [_number(row["onset"]) for row in events]
    if any(b < a for a, b in zip(onsets, onsets[1:])):
        raise ValueError("tactile source events are unsorted")
    episodes, dispositions = [], []
    indices = [i for i, row in enumerate(events) if row["trial_type"] == "stim-adapt"]
    for trial, index in enumerate(indices, start=1):
        stop_index = indices[trial] if trial < len(indices) else len(events)
        group = events[index:stop_index]
        stimulus = group[0]
        point = _number(stimulus["onset"])
        sample = _number(stimulus["event_sample"])
        if sample != int(sample) or abs(sample / sfreq_hz - point) > 1.01 / sfreq_hz:
            raise ValueError("source zero-based event_sample/seconds disagree")
        first_order = [row for row in group if row["trial_type"] in ("hit", "miss", "fa", "cr")]
        confidence_rows = [row for row in group if row["trial_type"] == "conf-resp"]
        if len(first_order) > 1 or len(confidence_rows) > 1:
            raise ValueError("duplicate source responses within tactile trial")
        response = first_order[0] if first_order else None
        confidence_response = confidence_rows[0] if confidence_rows else None
        episode_id = f"{recording_id}_trial_{trial:04d}"
        disposition = {"episode_id": episode_id, "source_trial_index": trial,
                       "source_point_s": point, "status": "included"}
        if point < 10 or point > duration_s:
            disposition["status"] = "outside_selected_continuous_extent"
            dispositions.append(disposition)
            continue
        category = response["trial_type"] if response else "missing_first_order_response"
        detection = 1.0 if category in ("hit", "fa") else 0.0 if category in ("miss", "cr") else None
        confidence = _number(confidence_response["confidence"], missing=True) if confidence_response else None
        response_s = _number(response["onset"]) if response else None
        confidence_s = _number(confidence_response["onset"]) if confidence_response else None
        for answer in (response_s, confidence_s):
            if answer is not None and answer < point:
                raise ValueError("tactile response precedes its source stimulus")
        report_s = confidence_s if capacity == "confidence" else response_s
        episodes.append({"episode_id": episode_id, "recording_id": recording_id,
                         "start_s": (int(sample) - round(10*sfreq_hz))/sfreq_hz,
                         "stop_s": int(sample)/sfreq_hz,
                         "report_start_s": report_s, "report_stop_s": report_s,
                         "report_point_s": report_s, "report_category": category,
                         "capacity_value": confidence if capacity == "confidence" else detection,
                         "capacity_construct": "subsequent_tactile_" + capacity,
                         "capacity_units": "source_numeric_confidence_metadata_percent_unscaled" if capacity == "confidence" else "binary_felt_yes_no",
                         "report_timing_basis": "subsequent_per_event_point_report",
                         "report_instruction_reference": "https://openneuro.org/datasets/ds001785/versions/1.1.1",
                         "eeg_anchor_basis": "source_stim_adapt_marker_physical_trigger_alignment_unqualified",
                         "analysis_window_basis": "10s_pre_stimulus_predictor",
                         "source_trial_type": stimulus["trial_type"],
                         "source_stimulus_intensity": _number(stimulus["stimamp"], missing=True),
                         "source_stimon": _number(stimulus["stimon"], missing=True),
                         "original_sdt": stimulus["sdt"], "original_detection": detection,
                         "original_confidence": confidence,
                         "confidence": confidence, "detection": detection,
                         "stim_intensity": _number(stimulus["stimamp"], missing=True),
                         "first_order_response_s": response_s, "confidence_response_s": confidence_s,
                         "original_response_time": _number(stimulus["response_time"], missing=True),
                         "original_response_time2": _number(stimulus["response_time2"], missing=True)})
        dispositions.append(disposition)
    return episodes, dispositions


def _pinned_inputs(root: Path, manifest: dict, paths: list[Path]) -> list[dict]:
    root = root.resolve()
    by_path = {r["relative_path"]: r for r in manifest["resources"]}
    receipts = []
    for path in paths:
        relative = path.resolve().relative_to(root).as_posix()
        if relative not in by_path:
            raise ValueError(f"input absent from immutable source manifest: {relative}")
        receipts.append(verify_acquired(root, by_path[relative]))
    return receipts


def _tactile_sidecar(set_path: Path, root: Path, manifest: dict) -> tuple[Path, dict]:
    """Resolve this cohort's run- or task-level EEG JSON without inventing pins.

    ds001785 stores its EEG metadata in a same-directory task sidecar whose
    filename omits the run entity. This is BIDS inheritance, not missing source
    metadata. Restrict resolution to the exact filename and the filename with
    only its run entity removed; subject, session, task and suffix stay fixed.
    This cohort's bounded resolver rejects an ambiguous immutable manifest
    rather than guessing how competing same-directory metadata should merge.
    Every selected input is still checksum-verified by _pinned_inputs.
    """
    root = root.resolve()
    exact = set_path.with_suffix(".json")
    if not set_path.name.endswith("_eeg.set"):
        raise ValueError("unsupported tactile EEG filename suffix")
    parts = exact.stem.split("_")
    runs = [part for part in parts if part.startswith("run-")]
    if len(runs) > 1 or any(not re.fullmatch(r"run-\d+", run) for run in runs):
        raise ValueError("ambiguous/invalid tactile run entity")
    paths = [exact]
    if runs:
        paths.append(exact.with_name("_".join(part for part in parts if part != runs[0])+".json"))
    by_relative = {path.resolve().relative_to(root).as_posix(): path for path in paths}
    matches = [r for r in manifest["resources"] if r["relative_path"] in by_relative]
    if not matches:
        raise ValueError("EEG sidecar absent from immutable source manifest: "+", ".join(by_relative))
    if len(matches) != 1:
        raise ValueError("ambiguous same-directory BIDS EEG sidecars in immutable source manifest")
    relative = matches[0]["relative_path"]
    selected = by_relative[relative]
    return selected, {"policy": "same_directory_BIDS_run_or_task_sidecar",
                      "selected_relative_path": relative,
                      "run_entity_omitted": selected != exact,
                      "inheritance_rule_url": "https://bids-specification.readthedocs.io/en/stable/common-principles.html#the-inheritance-principle"}


def _extent(samples: int, sfreq: float, maximum: float | None) -> int:
    if not math.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("invalid source sampling frequency")
    if maximum is None or maximum == 0:
        return samples
    if not math.isfinite(maximum) or maximum < 10:
        raise ValueError("selected continuous extent must be at least10s")
    rounded = round(maximum * sfreq)
    if abs(rounded/sfreq-maximum) > 1e-9:
        raise ValueError("selected continuous extent is not on source sample grid")
    return min(samples, rounded)


def _write_npz(output: Path, recording_id: str, eeg: np.ndarray, names: list[str],
               sfreq: float, reference: str, aux: np.ndarray,
               aux_names: list[str], gaps: np.ndarray, bad_channels: list[str] | None = None) -> Path:
    if not np.isfinite(eeg).all() or not np.isfinite(aux).all():
        raise ValueError("nonfinite original acquisition; no repair performed")
    if not set(FIXED_INSTRUMENT_CHANNELS) <= set(names):
        raise ValueError("source lacks prespecified Cz/Pz/Fz/Oz instrument channels")
    path = output / "signals" / f"{recording_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, data=eeg, sfreq_hz=sfreq, channel_names=np.asarray(names, dtype="U"),
             units=np.asarray("uV"), reference=np.asarray(reference), gap_intervals_s=gaps,
             bad_channels=np.asarray(bad_channels or [], dtype="U"),
             auxiliary_data=aux, auxiliary_channel_names=np.asarray(aux_names, dtype="U"),
             auxiliary_units=np.asarray(["uV"] * len(aux_names), dtype="U"))
    return path


def _recording_row(recording_id: str, participant_id: str, source_id: str,
                   path: Path, output: Path, sfreq: float, shape: tuple[int, int],
                   session: str, reference: str) -> dict:
    return {"recording_id": recording_id, "participant_id": participant_id, "source_id": source_id,
            "cohort_id": source_id, "site_id": "Monash" if source_id == "monash_wanderim" else "source_tactile_site",
            "session_id": session, "role": "development", "signal_path": path.relative_to(output).as_posix(),
            "signal_format": "npz", "modality": "EEG", "units": "uV",
            "duration_s": shape[1]/sfreq, "sfreq_hz": sfreq, "channel_count": shape[0],
            "detector_exposure": "unknown", "analyst_exposure": "yes", "synthetic": False,
            "reference": reference, "source_conversion": "gain_normalization_only_no_filter_rereference_resample"}


def _monash_one(header_path: Path, behaviour: Path, *, source_root: Path,
                 output: Path, manifest: dict, maximum: float | None) -> tuple[dict, list[dict], dict, list[dict]]:
    header = parse_brainvision_header(header_path)
    data_path = header_path.parent / header["data_file"]
    marker_path = header_path.parent / header["marker_file"]
    inputs = _pinned_inputs(source_root, manifest, [header_path, data_path, marker_path, behaviour])
    count = len(header["channel_names"])
    size = data_path.stat().st_size
    if size % (4*count):
        raise ValueError("BrainVision binary length is not an integral sample count")
    samples = size//(4*count)
    sfreq = header["sfreq_hz"]
    selected = _extent(samples, sfreq, maximum)
    original = np.memmap(data_path, dtype="<f4", mode="r", shape=(samples, count))
    aux_indices = [i for i, name in enumerate(header["channel_names"]) if name in ("EOG", "ECG", "D1") or name.startswith("BIP")]
    eeg_indices = [i for i in range(count) if i not in aux_indices]
    names = [header["channel_names"][i] for i in eeg_indices]
    aux_names = [header["channel_names"][i] for i in aux_indices]
    gains = np.asarray(header["gain_uv"], dtype=np.float64)
    eeg = np.empty((len(eeg_indices), selected), dtype=np.float32)
    aux = np.empty((len(aux_indices), selected), dtype=np.float32)
    for start in range(0, selected, 250000):
        stop = min(selected, start+250000)
        block = np.asarray(original[start:stop], dtype=np.float64)*gains
        if not np.isfinite(block).all():
            raise ValueError("nonfinite original BrainVision samples")
        eeg[:, start:stop] = block[:, eeg_indices].T
        aux[:, start:stop] = block[:, aux_indices].T
    markers = parse_brainvision_markers(marker_path, sfreq, samples)
    gaps = np.array([[m["onset_s"], m["onset_s"]+1/sfreq] for m in markers
                     if m["marker_type"] == "New Segment" and m["sample_index"] > 0 and m["sample_index"] < selected], dtype=float).reshape(-1, 2)
    scipy_io = importlib.import_module("scipy.io")
    source_behaviour = scipy_io.loadmat(behaviour, squeeze_me=True, struct_as_record=False)
    subject = re.search(r"MWI(\d{3})", header_path.name)[1]
    recording_id = "monash_wanderim_" + header_path.stem
    episodes, dispositions, alignment = monash_probe_episodes(source_behaviour["probe_res"], markers,
                                                             recording_id=recording_id, sfreq_hz=sfreq,
                                                             duration_s=selected/sfreq)
    path = _write_npz(output, recording_id, eeg, names, sfreq, header["reference"], aux, aux_names, gaps)
    record = _recording_row(recording_id, "monash_wanderim_"+subject, "monash_wanderim",
                            path, output, sfreq, eeg.shape, "original_session", header["reference"])
    audit = {"recording_id": recording_id, "source_inputs": inputs, "header": header,
             "alignment": alignment, "original_sample_count": samples, "selected_sample_count": selected,
             "scalp_channels": names, "auxiliary_channels": aux_names,
             "output_sha256": sha256_file(path), "reference_changed": False,
             "filtering_applied": False, "resampling_applied": False,
             "original_state_code4_preserved": True, "license": None,
             "signal_status": "original_raw_gain_normalized", "synthetic": False}
    return record, episodes, audit, dispositions


def _tactile_one(set_path: Path, *, source_root: Path, output: Path, manifest: dict,
                  maximum: float | None, capacity: str) -> tuple[dict, list[dict], dict, list[dict]]:
    prefix = set_path.name.removesuffix("_eeg.set")
    eeg_folder = set_path.parent
    fdt = set_path.with_suffix(".fdt")
    sidecar, sidecar_resolution = _tactile_sidecar(set_path, source_root, manifest)
    events_path = eeg_folder / (prefix + "_events.tsv")
    channels_path = eeg_folder / ("_".join(prefix.split("_")[:2]) + "_channels.tsv")
    inputs = _pinned_inputs(source_root, manifest, [set_path, fdt, sidecar, events_path, channels_path])
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("RecordingType") != "continuous" or not metadata.get("EEGReference"):
        raise ValueError("continuous acquisition/reference not explicit in EEG sidecar")
    channel_rows = _read_table(channels_path)
    mne = importlib.import_module("mne")
    raw = mne.io.read_raw_eeglab(str(set_path), preload=False, verbose="ERROR")
    if raw.ch_names != [r["name"] for r in channel_rows]:
        raise ValueError("EEGLAB order/labels do not correspond exactly to original channels.tsv")
    sfreq = float(raw.info["sfreq"])
    if abs(sfreq-float(metadata["SamplingFrequency"])) > 1e-8:
        raise ValueError("sampling frequency disagrees with original sidecar")
    selected = _extent(raw.n_times, sfreq, maximum)
    eeg_indices = [i for i, row in enumerate(channel_rows) if row["type"] == "EEG"
                   and row["name"] != "EOG" and not row["name"].startswith("BIP")]
    aux_indices = [i for i, row in enumerate(channel_rows) if i not in eeg_indices and row.get("units") in ("uV", "µV", "mV", "V")]
    unknown_indices = [i for i in range(len(channel_rows)) if i not in eeg_indices and i not in aux_indices]
    if any(channel_rows[i].get("units") not in ("uV", "µV", "mV", "V") for i in eeg_indices):
        raise ValueError("scalp EEG source units are unqualified")
    # MNE returns SI volts for physical EEG/EOG; BIP3 AUDIO has no source physical unit.
    eeg = raw.get_data(picks=eeg_indices, start=0, stop=selected)*1e6
    aux = raw.get_data(picks=aux_indices, start=0, stop=selected)*1e6
    names = [raw.ch_names[i] for i in eeg_indices]
    aux_names = [raw.ch_names[i] for i in aux_indices]
    bad_names = list(raw.info.get("bads", [])) + [row["name"] for row in channel_rows if row.get("status") == "bad"]
    bad_names = sorted(set(bad_names) & set(names))
    gaps = []
    for onset, duration, description in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if any(term in str(description).lower() for term in ("bad", "edge", "boundary")):
            onset = float(onset)-float(getattr(raw, "first_time", 0.0))
            if onset < selected/sfreq:
                gaps.append([max(0.0, onset), min(selected/sfreq, onset+max(float(duration), 1/sfreq))])
    events = _read_table(events_path)
    recording_id = "openneuro_ds001785_"+prefix
    episodes, dispositions = tactile_trial_episodes(events, recording_id=recording_id, sfreq_hz=sfreq,
                                                    duration_s=selected/sfreq, capacity=capacity)
    reference = metadata["EEGReference"]
    path = _write_npz(output, recording_id, eeg, names, sfreq, reference, aux, aux_names,
                      np.asarray(gaps, dtype=float).reshape(-1, 2), bad_names)
    unknown_path = None
    if unknown_indices:
        unknown_path = output / "signals" / f"{recording_id}.unqualified_aux.npz"
        unknown = raw.get_data(picks=unknown_indices, start=0, stop=selected)
        if not np.isfinite(unknown).all():
            raise ValueError("nonfinite auxiliary samples")
        np.savez(unknown_path, reader_values=unknown,
                 channel_names=np.asarray([raw.ch_names[i] for i in unknown_indices], dtype="U"),
                 source_units=np.asarray([channel_rows[i].get("units", "unknown") for i in unknown_indices], dtype="U"),
                 units_qualification=np.asarray("physical_units_unknown_MNE_reader_values_preserved"))
    participant, session = prefix.split("_")[:2]
    record = _recording_row(recording_id, "openneuro_ds001785_"+participant,
                            "openneuro_ds001785", path, output, sfreq, eeg.shape, session, reference)
    audit = {"recording_id": recording_id, "source_inputs": inputs, "source_sidecar": metadata,
             "source_sidecar_resolution": sidecar_resolution,
             "original_channel_table": channel_rows, "scalp_channels": names, "auxiliary_channels": aux_names,
             "excluded_from_scalp_geometry": [r["name"] for i, r in enumerate(channel_rows) if i not in eeg_indices],
             "unqualified_aux_file": str(unknown_path) if unknown_path else None,
             "unqualified_aux_sha256": sha256_file(unknown_path) if unknown_path else None,
             "source_channel_count_discrepancy": int(metadata.get("EEGChannelCount", len(names))) != len(names),
             "original_sample_count": raw.n_times, "selected_sample_count": selected,
             "output_sha256": sha256_file(path), "reference_changed": False, "filtering_applied": False,
             "resampling_applied": False, "physical_stimulus_trigger_alignment_qualified": False,
             "confidence_units": "source numeric values retained; events.json says percent",
             "source_license": "CC0", "synthetic": False}
    if hasattr(raw, "close"):
        raw.close()
    return record, episodes, audit, dispositions


def convert_sources(adapter: str, source_root: str | Path, manifest_path: str | Path,
                    output_dir: str | Path, *, subjects: list[str] | None = None,
                    task: str = "adapt", maximum_duration_s: float | None = None,
                    capacity: str = "confidence", template: str | Path | None = None,
                    release_dir: str | Path | None = None,
                    qualification: str | Path | None = None) -> dict[str, Any]:
    """Convert every selected available candidate, recording all dispositions.

    Output directory must be new, keeping prior conversions immutable. File
    availability, source schema and window compatibility determine eligibility;
    outcomes never select subjects. Missing/unmatched originals stay in the
    source census. Ordinary exceptions are logged per candidate; zero compatible
    recordings raises after saving the explicit qualification report.
    """
    if adapter not in ("monash", "paris", "tactile"):
        raise ValueError("unsupported adapter")
    root = Path(source_root).resolve()
    output = Path(output_dir).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    source_id = {"monash": "monash_wanderim", "paris": "mindblanking_paris", "tactile": "openneuro_ds001785"}[adapter]
    acquisition_source_id = "mindblanking_zenodo17649277" if adapter == "paris" else source_id
    expected = [r for r in manifest["resources"] if r["source_id"] == acquisition_source_id]
    archive_receipts = []
    if adapter == "monash":
        candidates = [r for r in expected if r["relative_path"].endswith(".vhdr")]
        if subjects:
            candidates = [r for r in candidates if re.search(r"MWI(\d{3})", r["relative_path"])[1] in subjects]
    elif adapter == "paris":
        archives = {}
        for name in ("Paris dataset EEG ICA cleaned.zip", "Paris dataset behaviour.zip"):
            matches = [r for r in expected if Path(r["relative_path"]).name == name]
            if len(matches) != 1:
                raise ValueError("missing/duplicate pinned Paris source archive")
            archive_receipts.append(verify_acquired(root, matches[0]))
            archives[name] = root/matches[0]["relative_path"]
        with zipfile.ZipFile(archives["Paris dataset behaviour.zip"]) as behaviour_archive:
            behaviour_members = [x.filename for x in behaviour_archive.infolist()
                                 if x.filename.endswith(".mat") and not x.filename.startswith("__MACOSX/")]
        candidates = []
        with zipfile.ZipFile(archives["Paris dataset EEG ICA cleaned.zip"]) as eeg_archive:
            for info in eeg_archive.infolist():
                if info.filename.startswith("__MACOSX/"):
                    continue
                match = re.fullmatch(r"sub[-_](\d{3})_ica_raw\.fif", Path(info.filename).name)
                if not match or (subjects and match[1] not in subjects):
                    continue
                subject = match[1]
                behaviours = [name for name in behaviour_members if f"_s{subject}_" in Path(name).name]
                if len(behaviours) != 1:
                    behaviour_member = None
                else:
                    behaviour_member = behaviours[0]
                candidates.append({"resource_id": "zenodo17649277:Paris:"+subject,
                                   "relative_path": info.filename, "subject": subject,
                                   "eeg_archive": str(archives["Paris dataset EEG ICA cleaned.zip"]),
                                   "eeg_member": info.filename,
                                   "behaviour_archive": str(archives["Paris dataset behaviour.zip"]),
                                   "behaviour_member": behaviour_member})
        candidate_ids = [r["resource_id"] for r in candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("duplicate Paris subject FIF member; source qualification required")
    else:
        candidates = [r for r in expected if r["relative_path"].endswith(".set") and f"task-{task}_" in r["relative_path"]]
        if subjects:
            candidates = [r for r in candidates if any(f"/{s}/" in r["relative_path"] for s in subjects)]
    if not candidates:
        raise ValueError("no candidates for requested original source selection")
    output.mkdir(parents=True, exist_ok=False)
    records, episodes, audits, eligibility, dispositions = [], [], [], [], []
    for resource in candidates:
        path = root / resource["relative_path"]
        state = {"source_resource_id": resource["resource_id"], "source_path": resource["relative_path"],
                 "status": "pending", "reason": "", "outcome_selection_applied": False}
        try:
            if adapter == "monash":
                subject = re.search(r"MWI(\d{3})", path.name)[1]
                matches = [root/r["relative_path"] for r in expected
                           if r.get("kind") == "monash_behaviour" and f"_s{subject}_" in r["relative_path"]]
                if len(matches) != 1:
                    raise ValueError("no unique original matched behavioural file")
                record, eps, audit, disp = _monash_one(path, matches[0], source_root=root, output=output,
                                                       manifest=manifest, maximum=maximum_duration_s)
            elif adapter == "paris":
                if resource["behaviour_member"] is None:
                    raise ValueError("no unique original matched Paris behavioural file")
                record, eps, audit, disp = _paris_one(resource, output=output, maximum=maximum_duration_s,
                                                      archive_receipts=archive_receipts)
            else:
                record, eps, audit, disp = _tactile_one(path, source_root=root, output=output,
                                                       manifest=manifest, maximum=maximum_duration_s, capacity=capacity)
            records.append(record)
            episodes.extend(eps)
            audits.append(audit)
            dispositions.extend(disp)
            state.update(status="converted", recording_id=record["recording_id"], episodes_in_selected_extent=len(eps))
        except (ValueError, OSError, KeyError, ImportError) as error:
            state.update(status="not_qualified", reason=f"{type(error).__name__}: {error}")
        eligibility.append(state)
    write_csv(output/"recordings.csv", records, list(RECORDING_COLUMNS)+["reference", "source_conversion"])
    names = sorted(set(EPISODE_COLUMNS) | {k for row in episodes for k in row})
    write_csv(output/"episodes_development.csv", episodes, names)
    write_csv(output/"source_eligibility.csv", eligibility)
    write_csv(output/"episode_eligibility.csv", dispositions, sorted({k for r in dispositions for k in r}) or
              ["episode_id", "source_point_s", "status"])
    if template is None:
        template = Path(__file__).resolve().parent.parent/"configs"/"study.template.json"
    cfg = json.loads(Path(template).read_text(encoding="utf-8"))
    cfg.update(study_id=f"adversarial-consciousness-{source_id}-development", mode="empirical", path_base=".")
    cfg["paths"].update(recordings="recordings.csv", episodes_development="episodes_development.csv",
                         data_root=".", output_dir="runs")
    if release_dir:
        cfg["paths"]["release_dir"] = str(Path(release_dir).resolve())
    if qualification:
        cfg["paths"]["qualification"] = str(Path(qualification).resolve())
    cfg["instrument"].update(channel_names=FIXED_INSTRUMENT_CHANNELS, channel_aggregation="median")
    # Fixed signal-only policy before any native score: source raw EEG contains
    # electrode DC offsets. Measure deviations rather than these offsets, keep
    # original native input bytes, and retain per-window scalp-channel audits.
    cfg["quality"].update(amplitude_reference="channel_median")
    cfg["channel_quality"] = {"enabled": True, "minimum_channels": 16,
        "max_abs_uv": 250.0, "flat_std_uv": 0.01, "amplitude_reference": "channel_median"}
    cfg["analysis"].update(construct="development source qualification only",
                            timing_definition="analyst10s_pre_probe" if adapter in ("monash", "paris") else "10s_pre_stimulus_predictor",
                            enable_prediction=False)
    cfg["discovery"]["relationship_columns"] = ["awareness_rating_raw"] if adapter in ("monash", "paris") else [
        "capacity_value", "confidence", "detection", "stim_intensity"]
    cfg["source_conversion"] = {"adapter": adapter, "manifest_sha256": canonical_hash(manifest),
                                "qualification_report": "source_qualification.json", "synthetic": False,
                                "maximum_duration_s": maximum_duration_s, "subjects_requested": subjects or [],
                                "interpretation": "development acquisition/runtime qualification; no independent confirmatory claim"}
    write_json(output/"study.config.json", cfg)
    report = {"schema_version": 1, "source_id": source_id, "manifest_sha256": canonical_hash(manifest),
              "candidate_recordings": len(candidates), "converted_recordings": len(records),
              "converted_participants": len({r["participant_id"] for r in records}),
              "selected_episodes": len(episodes), "source_eligibility": eligibility,
              "recordings": audits, "synthetic": False, "role": "development",
              "protected_evaluation_qualification": False,
              "no_favourable_result_gate": True, "qualification_status": "compatible_development_inputs" if records else "blocked_no_compatible_inputs",
              "output_tables_sha256": {name: sha256_file(output/name) for name in
                                       ("recordings.csv", "episodes_development.csv", "source_eligibility.csv", "episode_eligibility.csv", "study.config.json")}}
    write_json(output/"source_qualification.json", report)
    if not records:
        raise ValueError(f"no compatible inputs; inspect {output/'source_qualification.json'}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", choices=["monash", "paris", "tactile"])
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", action="append")
    parser.add_argument("--task", default="adapt")
    parser.add_argument("--max-duration-s", type=float, default=None)
    parser.add_argument("--capacity", choices=["confidence", "detection"], default="confidence")
    parser.add_argument("--template")
    parser.add_argument("--release-dir")
    parser.add_argument("--qualification")
    args = parser.parse_args(argv)
    report = convert_sources(args.adapter, args.source_root, args.manifest, args.output,
                             subjects=args.subject, task=args.task, maximum_duration_s=args.max_duration_s,
                             capacity=args.capacity, template=args.template,
                             release_dir=args.release_dir, qualification=args.qualification)
    print(json.dumps({k: v for k, v in report.items() if k not in ("recordings", "source_eligibility")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
