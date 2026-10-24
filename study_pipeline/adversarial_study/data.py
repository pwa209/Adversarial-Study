"""Strict recording adapters for descriptive EEG analysis.

All public arrays are channels x samples in microvolts.  NPZ is the portable
interchange format: ``data``, ``sfreq_hz`` and Unicode ``channel_names`` are
required. Optional ``sample_times_s`` (or ``time_s``) exposes discontinuities;
optional ``gap_intervals_s`` is an N x 2 array of invalid intervals in seconds
relative to recording onset.  A recording without timestamps/intervals is an
explicit assertion by its producer that its samples are continuous.

These adapters neither reproduce the clinical MATLAB instrument preprocessing
nor filter, interpolate, pad, resample, select a reference, or repair recordings.
Unit/reference conversion and source-specific event alignment remain audited
preprocessing responsibilities.  MNE is an optional, lazily imported reader.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _validate_arrays(
    data: Any, sfreq_hz: Any, channel_names: Any, *,
    require_finite: bool = True, normalize_data: bool = True,
) -> tuple[np.ndarray, float, list[str]]:
    array = np.asarray(data)
    if array.ndim != 2 or min(array.shape) < 1 or array.dtype.kind not in "fiu":
        raise ValueError("data must be a nonempty real channels x samples array")
    if normalize_data:
        array = np.asarray(array, dtype=np.float64)
    if require_finite and not np.isfinite(array).all():
        raise ValueError("recording data contain non-finite values")
    frequency = np.asarray(sfreq_hz)
    if frequency.size != 1:
        raise ValueError("sfreq_hz must be one finite positive scalar")
    sfreq = float(frequency.reshape(-1)[0])
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq_hz must be one finite positive scalar")
    labels = np.asarray(channel_names)
    if labels.ndim != 1 or labels.size != array.shape[0] or labels.dtype.kind != "U":
        raise ValueError("channel_names must contain one Unicode label per channel")
    names = [str(label) for label in labels]
    if any(not label.strip() for label in names) or len(set(names)) != len(names):
        raise ValueError("channel_names must be nonempty and unique")
    return array, sfreq, names


def _validated_intervals(value: Any) -> np.ndarray:
    intervals = np.asarray(value, dtype=np.float64)
    if intervals.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("gap_intervals_s must be an N x 2 array")
    if not np.isfinite(intervals).all() or np.any(intervals[:, 1] < intervals[:, 0]):
        raise ValueError("gap intervals must have finite, ordered endpoints")
    return intervals


def read_recording(path: str | Path, format: str = "npz") -> dict[str, Any]:
    """Read a recording with explicit labels and amplitudes in microvolts.

    NPZ inputs use ``allow_pickle=False``; object arrays are rejected. Optional
    ``units`` must be ``uV`` or ``microvolts``. Its absence means the documented
    NPZ contract (microvolts), not automatic unit inference. Optional
    ``reference`` is retained as provenance and never applied.

    ``format='mne'`` chooses a known reader by extension; explicit ``edf``,
    ``bdf``, ``fif``, ``brainvision`` and ``eeglab`` are also supported when MNE
    is installed. Only channels tagged EEG are returned, including flagged bad
    EEG channels (recorded in ``bad_channels``); MNE volts are multiplied by
    1e6. BAD/EDGE/boundary annotations become invalid intervals. No reference
    changes or projectors are applied by this code. Site-specific units,
    channel types, references and annotations require source qualification.
    """
    recording_path = Path(path)
    if not recording_path.is_file():
        raise FileNotFoundError(recording_path)
    kind = str(format).lower()
    if kind == "npz":
        loaded = np.load(recording_path, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError("NPZ format requires a .npz archive")
        with loaded as archive:
            missing = {"data", "sfreq_hz", "channel_names"} - set(archive.files)
            if missing:
                raise ValueError(f"missing NPZ keys: {', '.join(sorted(missing))}")
            data, sfreq, names = _validate_arrays(
                archive["data"], archive["sfreq_hz"], archive["channel_names"]
            )
            if "units" in archive:
                unit = np.asarray(archive["units"])
                if unit.size != 1 or str(unit.reshape(-1)[0]) not in {"uV", "microvolts"}:
                    raise ValueError("NPZ data must explicitly use microvolts (uV)")
            result: dict[str, Any] = {
                "data": data,
                "sfreq_hz": sfreq,
                "channel_names": names,
                "units": "uV",
                "reference": "as_stored",
                "source_path": str(recording_path.resolve()),
                "bad_channels": [],
            }
            if "reference" in archive:
                reference = np.asarray(archive["reference"])
                if reference.size != 1 or reference.dtype.kind != "U":
                    raise ValueError("reference must be one Unicode provenance string")
                result["reference"] = str(reference.reshape(-1)[0])
            if "sample_times_s" in archive and "time_s" in archive:
                raise ValueError("supply sample_times_s or time_s, not both")
            for key in ("sample_times_s", "time_s"):
                if key in archive:
                    times = np.asarray(archive[key], dtype=np.float64)
                    if times.shape != (data.shape[1],) or not np.isfinite(times).all():
                        raise ValueError("sample timestamps must be finite and match sample count")
                    if np.any(np.diff(times) <= 0):
                        raise ValueError("sample timestamps must increase strictly")
                    result["sample_times_s"] = times.copy()
            if "gap_intervals_s" in archive:
                result["gap_intervals_s"] = _validated_intervals(archive["gap_intervals_s"])
            if "bad_channels" in archive:
                bad = np.asarray(archive["bad_channels"])
                if bad.ndim != 1 or bad.dtype.kind != "U" or not set(bad.tolist()) <= set(names):
                    raise ValueError("bad_channels must be Unicode labels present in channel_names")
                result["bad_channels"] = bad.tolist()
        return result

    if kind == "mne":
        name = recording_path.name.lower()
        if name.endswith((".fif", ".fif.gz")):
            kind = "fif"
        else:
            kind = {".edf": "edf", ".bdf": "bdf", ".vhdr": "brainvision", ".set": "eeglab"}.get(
                recording_path.suffix.lower(), ""
            )
    reader_names = {
        "edf": "read_raw_edf", "bdf": "read_raw_bdf", "fif": "read_raw_fif",
        "brainvision": "read_raw_brainvision", "eeglab": "read_raw_eeglab",
    }
    if kind not in reader_names:
        raise ValueError(f"unsupported recording format: {format!r}")
    try:
        mne = importlib.import_module("mne")
    except ImportError as exc:
        raise RuntimeError("MNE is required for this format; use the qualified NPZ adapter otherwise") from exc
    raw = getattr(mne.io, reader_names[kind])(str(recording_path), preload=True, verbose="ERROR")
    picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    if len(picks) == 0:
        raise ValueError("source has no channels explicitly tagged EEG")
    names = [raw.ch_names[int(index)] for index in picks]
    data, sfreq, names = _validate_arrays(
        raw.get_data(picks=picks) * 1e6, raw.info["sfreq"], np.asarray(names, dtype="U")
    )
    invalid, reasons = [], []
    for onset, duration, description in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        label = str(description).lower()
        if label.startswith(("bad", "edge")) or "boundary" in label:
            start = float(onset) - float(raw.first_time)
            invalid.append([start, start + float(duration)])
            reasons.append(str(description))
    return {
        "data": data, "sfreq_hz": sfreq, "channel_names": names, "units": "uV",
        "reference": "as_stored", "source_path": str(recording_path.resolve()),
        "bad_channels": [name for name in raw.info.get("bads", []) if name in names],
        "gap_intervals_s": _validated_intervals(invalid),
        "gap_interval_reasons": reasons,
    }


def extract_window(
    recording: Mapping[str, Any], start_s: float, stop_s: float,
    duration_s: float = 10.0, tolerance_s: float = 0.01,
) -> np.ndarray:
    """Copy one exact, contiguous window in microvolts without padding.

    Times are relative to recording onset; stop is exclusive. Both endpoints
    and requested duration must lie on the sample grid. ``tolerance_s`` checks
    requested versus actual duration only; it does not authorize rounding an
    off-grid endpoint. When supplied, timestamps must match their expected
    acquisition times to within 1e-6 sample, and every included interval must
    be contiguous. Windows overlapping invalid intervals or a zero-duration
    boundary are rejected. Flagged bad channels require explicit source
    handling before extraction and cannot silently qualify as usable EEG.
    """
    data, sfreq, _ = _validate_arrays(
        recording["data"], recording["sfreq_hz"], recording["channel_names"],
        require_finite=False, normalize_data=False,
    )
    values = np.asarray([start_s, stop_s, duration_s, tolerance_s], dtype=np.float64)
    if not np.isfinite(values).all() or duration_s <= 0 or tolerance_s < 0:
        raise ValueError("window times must be finite, duration positive and tolerance nonnegative")
    if start_s < 0 or stop_s <= start_s:
        raise ValueError("window must have nonnegative start and positive extent")
    if recording.get("units", "uV") not in {"uV", "microvolts"}:
        raise ValueError("extract_window requires data in microvolts")
    if recording.get("bad_channels"):
        raise ValueError("source contains flagged bad channels; handle them explicitly before extraction")
    samples = np.asarray([start_s, stop_s, duration_s]) * sfreq
    rounded = np.rint(samples).astype(np.int64)
    if not np.all(np.abs(samples - rounded) <= 1e-6):
        raise ValueError("window endpoints and duration must lie on the sample grid")
    first, last, expected = [int(value) for value in rounded]
    if first < 0 or last > data.shape[1]:
        raise ValueError("window is out of recording bounds; no padding is performed")
    if abs((stop_s - start_s) - duration_s) > tolerance_s or last - first != expected:
        raise ValueError("window must contain exactly the requested duration in samples")
    intervals = _validated_intervals(recording.get("gap_intervals_s", []))
    for left, right in intervals:
        overlaps = left < stop_s and right > start_s
        crosses_boundary = left == right and start_s < left < stop_s
        if overlaps or crosses_boundary:
            raise ValueError("window overlaps an annotated gap or invalid interval")
    if "sample_times_s" in recording or "time_s" in recording:
        times = np.asarray(recording.get("sample_times_s", recording.get("time_s")), dtype=np.float64)
        if times.shape != (data.shape[1],):
            raise ValueError("sample timestamps must match sample count")
        actual = times[first:last] - times[0]
        expected_times = np.arange(first, last, dtype=np.float64) / sfreq
        if not np.all(np.abs(actual - expected_times) <= 1e-6 / sfreq):
            raise ValueError("window has discontinuous or off-grid acquisition timestamps")
    window = np.asarray(data[:, first:last], dtype=np.float64)
    if not np.isfinite(window).all():
        raise ValueError("selected window data contain non-finite values; no repair is performed")
    return window.copy()
