"""Transparent descriptive comparators for human EEG windows.

These features describe scalp data in microvolts. They are neither the native
clinical MATLAB preprocessing nor a substitute for its frozen prediction.
No learned cross-person embedding, clinical calibration or mechanistic claim
is produced. Defaults are development choices to freeze after source and
measurement qualification; they are not validated clinical artifact rules.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _as_signal(data: Any, sfreq_hz: float, *, allow_nonfinite: bool = False) -> tuple[np.ndarray, float]:
    array = np.asarray(data)
    if array.ndim != 2 or min(array.shape) < 1 or array.dtype.kind not in "fiu":
        raise ValueError("data must be a nonempty real channels x samples array in microvolts")
    array = np.asarray(array, dtype=np.float64)
    sfreq = float(sfreq_hz)
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq_hz must be finite and positive")
    if not allow_nonfinite and not np.isfinite(array).all():
        raise ValueError("features require finite signals; artifacts are not imputed")
    return array, sfreq


def quality_metrics(
    data: Any, sfreq_hz: float, max_abs_uv: float = 250.0, flat_std_uv: float = 0.01,
    amplitude_reference: str = "as_stored",
) -> dict[str, Any]:
    """Screen amplitude and flat channels using configurable development rules.

    ``max_abs_uv`` screens as-stored amplitude by default. Explicit source
    policies may screen deviations from each channel's window median to avoid
    confusing electrode DC offsets with fluctuating artifacts. This only
    changes the quality measurement; returned native inputs remain untouched.
    ``flat_std_uv`` is the minimum per-channel population standard deviation.
    A single non-finite value or violating channel disqualifies a window. This
    limited screen does not detect all ocular, muscle, motion or stimulation
    artifacts. Data are never repaired. For non-finite windows amplitude and
    SD are ``None`` (to permit strict JSON), with ``finite=False``.
    """
    array, _ = _as_signal(data, sfreq_hz, allow_nonfinite=True)
    if amplitude_reference not in {"as_stored", "channel_median"}:
        raise ValueError("Unknown quality amplitude reference")
    if not np.isfinite(max_abs_uv) or max_abs_uv <= 0:
        raise ValueError("max_abs_uv must be finite and positive")
    if not np.isfinite(flat_std_uv) or flat_std_uv < 0:
        raise ValueError("flat_std_uv must be finite and nonnegative")
    finite = bool(np.isfinite(array).all())
    reasons = []
    maximum, minimum_std = None, None
    if not finite:
        reasons.append("nonfinite_signal")
    else:
        measured = array if amplitude_reference == "as_stored" else array - np.median(array, axis=1, keepdims=True)
        maximum = float(np.max(np.abs(measured)))
        minimum_std = float(np.min(np.std(array, axis=1, ddof=0)))
        if maximum > max_abs_uv:
            reasons.append("amplitude_exceeds_development_threshold")
        if minimum_std <= flat_std_uv:
            reasons.append("flat_channel_at_development_threshold")
    return {
        "qualified": not reasons, "reasons": reasons, "finite": finite,
        "max_abs_uv": maximum, "min_std_uv": minimum_std,
        "configured_max_abs_uv": float(max_abs_uv), "configured_flat_std_uv": float(flat_std_uv),
        "amplitude_reference": amplitude_reference,
    }


def channel_quality_selection(data, sfreq_hz, channel_names, settings=None):
    """Select measured scalp channels using a fixed signal-only policy.

    Retain every channel's measurement/reason. No reference change, replacement,
    interpolation or editing of signal samples occurs. Caller handles minimum
    channel eligibility separately from clinical detector-channel eligibility.
    """
    array, _ = _as_signal(data, sfreq_hz, allow_nonfinite=True)
    if len(channel_names) != array.shape[0]:
        raise ValueError("Quality channel names differ from array")
    settings = settings or {}
    if not settings.get("enabled", False):
        return list(range(array.shape[0])), []
    policy = {key: settings[key] for key in ("max_abs_uv", "flat_std_uv", "amplitude_reference") if key in settings}
    indices, audit = [], []
    for index, name in enumerate(channel_names):
        measured = quality_metrics(array[index:index + 1], sfreq_hz, **policy)
        audit.append({"channel_name": name, **measured})
        if measured["qualified"]: indices.append(index)
    return indices, audit


def _welch_density(
    data: np.ndarray, sfreq: float, segment_s: float, overlap_fraction: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    # Explicit Hann Welch: segmentwise mean removal, window energy scaling,
    # one-sided doubling, equally weighted periodograms, no zero padding.
    exact_length = segment_s * sfreq
    segment_n = int(round(exact_length))
    if not np.isclose(exact_length, segment_n, rtol=0, atol=1e-6) or segment_n < 4:
        raise ValueError("Welch segment must be at least four samples on the sample grid")
    if data.shape[1] < segment_n:
        raise ValueError("window shorter than Welch segment; no padding is performed")
    overlap_n = int(np.floor(segment_n * overlap_fraction))
    step = segment_n - overlap_n
    starts = range(0, data.shape[1] - segment_n + 1, step)
    window = np.hanning(segment_n)
    density = np.zeros((data.shape[0], segment_n // 2 + 1), dtype=np.float64)
    count = 0
    for first in starts:
        segment = data[:, first:first + segment_n]
        segment = segment - segment.mean(axis=1, keepdims=True)
        fft = np.fft.rfft(segment * window, axis=1)
        periodogram = (fft.real ** 2 + fft.imag ** 2) / (sfreq * np.dot(window, window))
        if segment_n % 2 == 0:
            periodogram[:, 1:-1] *= 2.0
        else:
            periodogram[:, 1:] *= 2.0
        density += periodogram
        count += 1
    density /= count
    return np.fft.rfftfreq(segment_n, 1.0 / sfreq), density, count


def _var1_diagnostics(data: np.ndarray, sfreq: float, train_fraction: float, ridge: float) -> dict[str, float]:
    """Fit standardized VAR(1) on an early segment; score later transitions.

    Scaling and intercept are fitted only to training samples. The transition
    straddling the split is excluded. All channels are predictors and targets;
    ridge acts on slope coefficients only. The temporal holdout is still part
    of the SAME human window and supplies a descriptive predictability
    comparator, never independent participant/model validation. Stability is
    the spectral radius of the fitted discrete-time transition matrix below 1.
    """
    split = int(np.floor(data.shape[1] * train_fraction))
    if split < 5 or data.shape[1] - split < 3:
        raise ValueError("VAR(1) requires at least five training and three holdout samples")
    mean = data[:, :split].mean(axis=1, keepdims=True)
    scale = data[:, :split].std(axis=1, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (data - mean) / scale
    x_train, y_train = standardized[:, :split - 1].T, standardized[:, 1:split].T
    x_test, y_test = standardized[:, split:-1].T, standardized[:, split + 1:].T
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    x_test = np.column_stack((np.ones(len(x_test)), x_test))
    penalty = np.eye(x_train.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(
        x_train.T @ x_train / len(x_train) + penalty,
        x_train.T @ y_train / len(x_train),
    )
    mse = float(np.mean((y_test - x_test @ coefficient) ** 2))
    persistence = float(np.mean((y_test - x_test[:, 1:]) ** 2))
    radius = float(np.max(np.abs(np.linalg.eigvals(coefficient[1:, :].T))))
    return {
        "var1_holdout_mse_standardized": mse,
        "var1_persistence_holdout_mse_standardized": persistence,
        "var1_improvement_over_persistence": (persistence - mse) / max(persistence, 1e-12),
        "var1_spectral_radius": radius,
        "var1_stable": float(radius < 1.0),
        "var1_training_transitions": float(len(x_train)),
        "var1_holdout_transitions": float(len(x_test)),
        "var1_step_s": 1.0 / sfreq,
        "var1_ridge": ridge,
    }


def compute_features(
    data: Any, sfreq_hz: float, settings: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Compute numeric spectral, covariance, autocorrelation and VAR features.

    Inputs are channels x samples in microvolts; no filtering/re-referencing is
    performed. Defaults: two-second symmetric Hann segments, 50% overlap,
    half-open bands delta [0.5,4), theta [4,8), alpha [8,13), beta [13,30) and
    gamma [30,45) Hz. Densities (uV^2/Hz) are averaged over segments/channels;
    band power is a frequency-bin sum times frequency resolution. Relative
    powers divide by mean-channel power in [0.5,45) Hz. This retains voltage
    weighting across channels. Flat/all-zero input gives zero band fractions
    and rank, and must separately fail quality screening.

    Effective rank is exp(entropy of nonnegative covariance eigenvalues
    normalized by their sum); participation rank is the squared sum divided
    by squared-eigenvalue sum. These are spatial covariance descriptors,
    dependent on montage, reference and observation length, not manifold or
    intrinsic dimensionality estimates.

    Autocorrelation is mean(x[t]*x[t+k])/mean(x^2) after channel demeaning.
    Lag defaults are 40, 100 and 500 ms, rounded to nearest available sample;
    actual lags and each channel's value are retained. Zero-variance channels
    contribute zero. VAR(1) uses 70% of the window for fitting and a ridge of
    0.001; its holdout error/stability are descriptive within-window quantities.
    They cannot establish participant generalization or a biological mechanism.

    Configurable keys: ``welch_segment_s``, ``welch_overlap_fraction``,
    ``bands_hz`` (name -> [low,high]), ``power_range_hz``, ``acf_lags_s``,
    ``var_train_fraction``, ``var_ridge``. Unknown keys fail rather than being
    ignored. Settings must be frozen outside discovery before evaluation.
    """
    array, sfreq = _as_signal(data, sfreq_hz)
    options: dict[str, Any] = {
        "welch_segment_s": 2.0, "welch_overlap_fraction": 0.5,
        "bands_hz": {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
                     "beta": (13.0, 30.0), "gamma": (30.0, 45.0)},
        "power_range_hz": (0.5, 45.0), "acf_lags_s": (0.04, 0.1, 0.5),
        "var_train_fraction": 0.7, "var_ridge": 0.001,
    }
    supplied = dict(settings or {})
    unknown = set(supplied) - set(options)
    if unknown:
        raise ValueError(f"unknown feature settings: {', '.join(sorted(unknown))}")
    options.update(supplied)
    segment_s, overlap = float(options["welch_segment_s"]), float(options["welch_overlap_fraction"])
    train_fraction, ridge = float(options["var_train_fraction"]), float(options["var_ridge"])
    if not np.isfinite([segment_s, overlap, train_fraction, ridge]).all():
        raise ValueError("feature settings must be finite")
    if segment_s <= 0 or not 0 <= overlap < 1 or not 0 < train_fraction < 1 or ridge <= 0:
        raise ValueError("invalid Welch overlap, segment duration, VAR fraction or positive ridge")
    bounds = np.asarray(options["power_range_hz"], dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 <= bounds[0] < bounds[1] <= sfreq / 2:
        raise ValueError("power_range_hz must be ordered finite frequencies at/below Nyquist")
    bands = options["bands_hz"]
    if not isinstance(bands, Mapping) or not bands:
        raise ValueError("bands_hz must be a nonempty mapping")
    validated_bands = {}
    for name, interval in bands.items():
        if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
            raise ValueError("band names must be nonempty alphanumeric/underscore labels")
        interval = np.asarray(interval, dtype=float)
        if interval.shape != (2,) or not np.isfinite(interval).all() or not bounds[0] <= interval[0] < interval[1] <= bounds[1]:
            raise ValueError("each band must lie within power_range_hz")
        validated_bands[name] = interval
    frequencies, density, segments = _welch_density(array, sfreq, segment_s, overlap)
    mean_density = density.mean(axis=0)
    df = float(frequencies[1] - frequencies[0])
    power_mask = (frequencies >= bounds[0]) & (frequencies < bounds[1])
    if not np.any(power_mask):
        raise ValueError("power_range_hz contains no Welch frequency bins")
    total = float(mean_density[power_mask].sum() * df)
    features = {
        "n_channels": float(array.shape[0]), "n_samples": float(array.shape[1]),
        "duration_s": array.shape[1] / sfreq, "sfreq_hz": sfreq,
        "welch_segments": float(segments), "welch_frequency_resolution_hz": df,
        "total_band_power_uv2": total,
    }
    for name, (low, high) in validated_bands.items():
        selected = (frequencies >= low) & (frequencies < high)
        if not np.any(selected):
            raise ValueError(f"band {name!r} contains no Welch frequency bins")
        power = float(mean_density[selected].sum() * df)
        features[f"power_{name}_uv2"] = power
        features[f"relative_{name}"] = power / total if total > 0 else 0.0
    centered = array - array.mean(axis=1, keepdims=True)
    covariance = centered @ centered.T / max(array.shape[1] - 1, 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    covariance_total = float(eigenvalues.sum())
    if covariance_total > 0:
        probabilities = eigenvalues[eigenvalues > 0] / covariance_total
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        participation_rank = covariance_total ** 2 / float(np.dot(eigenvalues, eigenvalues))
    else:
        effective_rank, participation_rank = 0.0, 0.0
    features.update({"effective_rank": effective_rank,
                     "covariance_participation_rank": participation_rank})
    lags = np.asarray(options["acf_lags_s"], dtype=np.float64)
    if lags.ndim != 1 or not len(lags) or not np.isfinite(lags).all() or np.any(lags <= 0):
        raise ValueError("acf_lags_s must contain positive finite lag durations")
    variance = np.mean(centered ** 2, axis=1)
    for index, lag_s in enumerate(lags):
        lag_n = int(round(float(lag_s) * sfreq))
        if lag_n < 1 or lag_n >= array.shape[1]:
            raise ValueError("autocorrelation lags must span at least one sample and be shorter than the window")
        numerator = np.mean(centered[:, :-lag_n] * centered[:, lag_n:], axis=1)
        correlation = np.divide(numerator, variance, out=np.zeros_like(numerator), where=variance > 0)
        key = f"acf_lag{index:02d}"
        features[f"{key}_s"] = lag_n / sfreq
        features[f"{key}_mean"] = float(correlation.mean())
        features[f"{key}_std"] = float(correlation.std())
        # Familiar default labels retain actual sampled lag metadata above.
        milliseconds = float(lag_s) * 1000.0
        if np.isclose(milliseconds, round(milliseconds), rtol=0, atol=1e-8):
            features[f"acf_{int(round(milliseconds))}ms"] = float(correlation.mean())
        for channel, value in enumerate(correlation):
            features[f"{key}_ch{channel:03d}"] = float(value)
    features.update(_var1_diagnostics(array, sfreq, train_fraction, ridge))
    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError("feature computation produced non-finite values; inspect signal scale and conditioning")
    return features
