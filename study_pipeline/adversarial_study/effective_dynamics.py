"""Observation-space temporal models for individual continuous EEG windows.

The four prespecified families are covariance-only/no history, diagonal AR(1),
coupled VAR(1), and coupled VAR(2). Models fit the early part of EACH window;
identical later targets measure one-step prediction and genuinely recursive
multi-step forecasts. This fitting constructs a window-level measurement. It
is not held-out-human validation, biological parameter recovery, or evidence
that a fitted cross-channel coefficient is a causal neural connection.

No filtering, resampling, interpolation, latent-state embedding, artifact
repair, or concatenation of separate episodes takes place here. The NPZ/window
helper uses the project's strict sample-grid and discontinuity checks.

Standalone usage::

    python -m adversarial_study.effective_dynamics --recording eeg.npz \
        --windows windows.csv --output-dir new-output
    python -m adversarial_study.effective_dynamics --synthetic-recovery \
        --output-dir new-synthetic-output

Windows CSV requires episode_id,start_s,stop_s. Recording/participant/source/
cohort/role metadata, when present, are preserved; behavioural outcomes and
clinical scores are never used to fit, split, select channels or choose models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .core import read_csv, sha256_file, write_csv, write_json
from .data import extract_window, read_recording


FAMILIES = ("covariance_only", "diagonal_ar1", "coupled_var1", "coupled_var2")
INTERPRETATION = (
    "Fitted matrices describe linear prediction between selected observed EEG channels at the "
    "recorded sampling interval. They do not identify anatomical pathways, synaptic parameters, "
    "causal coupling, consciousness, or an intervention effect. Common input, reference, volume "
    "conduction and measurement noise can contribute to cross-channel coefficients. Early/late "
    "prediction within the same window is measurement construction, not independent human validation."
)


def _options(settings: Mapping[str, Any] | None) -> dict:
    defaults = dict(train_fraction=0.65, ridge=0.001, covariance_ridge=0.0001,
                    max_channels=8, channel_selection=None, forecast_steps=[1, 5, 20],
                    temporal_controls=True, seed=20261004)
    if settings is not None and not isinstance(settings, Mapping):
        raise ValueError("Effective-dynamics settings must be a mapping")
    if settings:
        unknown = set(settings) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown effective-dynamics settings: {sorted(unknown)}")
        defaults.update(settings)
    for name in ("ridge", "covariance_ridge"):
        value = defaults[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive")
        defaults[name] = float(value)
    fraction = defaults["train_fraction"]
    if isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, (int, float, np.integer, np.floating)) or not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("train_fraction must lie strictly between zero and one")
    defaults["train_fraction"] = float(fraction)
    for name in ("max_channels", "seed"):
        value = defaults[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < (2 if name == "max_channels" else 0):
            raise ValueError(f"{name} must be an integer in its valid range")
        defaults[name] = int(value)
    if not isinstance(defaults["temporal_controls"], bool):
        raise ValueError("temporal_controls must be a boolean")
    steps = defaults["forecast_steps"]
    if not isinstance(steps, (list, tuple)) or not steps:
        raise ValueError("forecast_steps must be a nonempty list of positive integers")
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1 for x in steps):
        raise ValueError("forecast_steps must contain positive integer sample counts")
    if len(set(steps)) != len(steps):
        raise ValueError("forecast_steps must be unique")
    defaults["forecast_steps"] = sorted(int(x) for x in steps)
    selection = defaults["channel_selection"]
    if selection is not None:
        if not isinstance(selection, (list, tuple)) or not selection:
            raise ValueError("channel_selection must be an explicit list of channel labels or indices")
        defaults["channel_selection"] = [int(x) if isinstance(x, np.integer) else x for x in selection]
    return defaults


def _signal(data: Any, sfreq_hz: Any, channel_names: Sequence[str] | None,
            sample_times_s: Any, options: dict) -> tuple[np.ndarray, float, list[str], list[int]]:
    array = np.asarray(data)
    if array.ndim != 2 or array.dtype.kind not in "fiu" or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("Effective dynamics require a real multichannel channels x samples array")
    frequency = np.asarray(sfreq_hz)
    if frequency.size != 1 or frequency.dtype.kind not in "fiu":
        raise ValueError("sfreq_hz must be one finite positive scalar")
    sfreq = float(frequency.reshape(-1)[0])
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq_hz must be one finite positive scalar")
    if isinstance(channel_names, str):
        raise ValueError("channel_names must be a sequence of labels, not one string")
    names = list(channel_names) if channel_names is not None else [f"ch{i:03d}" for i in range(array.shape[0])]
    if len(names) != array.shape[0] or any(not isinstance(x, str) or not x.strip() for x in names) or len(set(names)) != len(names):
        raise ValueError("Channel labels must be nonempty unique strings matching the channel count")
    if sample_times_s is not None:
        times = np.asarray(sample_times_s, dtype=float)
        if times.shape != (array.shape[1],) or not np.isfinite(times).all():
            raise ValueError("Sample timestamps must be finite and match the sample count")
        if not np.all(np.abs(times - times[0] - np.arange(array.shape[1]) / sfreq) <= 1e-6 / sfreq):
            raise ValueError("Discontinuous or off-grid samples cannot form an effective-dynamics window")
    selection = options["channel_selection"]
    if selection is None:
        indices = list(range(min(options["max_channels"], array.shape[0])))
    else:
        if not isinstance(selection, (list, tuple)) or not selection:
            raise ValueError("channel_selection must be an explicit list of channel labels or indices")
        indices = []
        for value in selection:
            if isinstance(value, str) and value in names:
                index = names.index(value)
            elif not isinstance(value, bool) and isinstance(value, (int, np.integer)) and 0 <= value < len(names):
                index = int(value)
            else:
                raise ValueError(f"Unavailable selected channel: {value!r}")
            indices.append(index)
        if len(indices) != len(set(indices)):
            raise ValueError("Selected channels must be unique")
        if not 2 <= len(indices) <= options["max_channels"]:
            raise ValueError("Select at least two and no more than max_channels observed channels")
    selected = np.asarray(array[indices], dtype=np.float64)
    if not np.isfinite(selected).all():
        raise ValueError("Selected channels contain non-finite values; no repair is performed")
    return selected, sfreq, [names[i] for i in indices], indices


def _fit(z: np.ndarray, split: int, family: str, ridge: float, covariance_ridge: float) -> dict:
    # Common training targets start at sample two for fairness across lag orders.
    d = z.shape[0]
    y = z[:, 2:split].T
    matrices = []
    if family == "covariance_only":
        intercept = y.mean(axis=0)
        prediction = np.broadcast_to(intercept, y.shape)
        slope_count = 0
    elif family == "diagonal_ar1":
        x = z[:, 1:split - 1].T
        xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
        slopes = np.mean(xc * yc, axis=0) / (np.mean(xc * xc, axis=0) + ridge)
        matrices = [np.diag(slopes)]
        intercept = y.mean(axis=0) - x.mean(axis=0) * slopes
        prediction = intercept + x * slopes
        slope_count = d
    else:
        lag = 1 if family == "coupled_var1" else 2
        x = np.concatenate([z[:, 2 - k:split - k].T for k in range(1, lag + 1)], axis=1)
        xm, ym = x.mean(axis=0), y.mean(axis=0)
        xc, yc = x - xm, y - ym
        coefficient = np.linalg.solve(xc.T @ xc / len(x) + ridge * np.eye(x.shape[1]),
                                      xc.T @ yc / len(x))
        intercept = ym - xm @ coefficient
        matrices = [coefficient[k * d:(k + 1) * d].T for k in range(lag)]
        prediction = intercept + x @ coefficient
        slope_count = lag * d * d
    residual = y - prediction
    covariance = residual.T @ residual / len(residual) + covariance_ridge * np.eye(d)
    return {"intercept": intercept, "matrices": matrices, "innovation_covariance": covariance,
            "slope_parameter_count": slope_count, "training_residual_mse": float(np.mean(residual ** 2))}


def _descriptors(matrices: list[np.ndarray], sfreq: float, d: int) -> dict:
    if not matrices:
        return {"eigenvalues": [], "spectral_radius": None, "stable": None,
                "dominant_decay_time_s": None, "off_diagonal_frobenius_fraction": None}
    companion = matrices[0] if len(matrices) == 1 else np.block([
        [matrices[0], matrices[1]], [np.eye(d), np.zeros((d, d))]])
    eigenvalues = np.linalg.eigvals(companion)
    radius = float(np.max(np.abs(eigenvalues)))
    entries = []
    for value in eigenvalues:
        modulus = float(abs(value))
        decay = float(-1. / (sfreq * np.log(modulus))) if 0 < modulus < 1 else (0. if modulus == 0 else None)
        entries.append({"real": float(value.real), "imaginary": float(value.imag), "modulus": modulus,
                        "decay_time_s_if_stable": decay,
                        "oscillation_frequency_hz": float(abs(np.angle(value)) * sfreq / (2 * np.pi))})
    total = sum(float(np.sum(a ** 2)) for a in matrices)
    offdiag = sum(float(np.sum((a - np.diag(np.diag(a))) ** 2)) for a in matrices)
    return {"eigenvalues": entries, "spectral_radius": radius, "stable": bool(radius < 1),
            "dominant_decay_time_s": float(-1 / (sfreq * np.log(radius))) if 0 < radius < 1 else (0. if radius == 0 else None),
            "off_diagonal_frobenius_fraction": float(np.sqrt(offdiag / total)) if total else 0.}


def _predict(fit: dict, previous: np.ndarray, previous2: np.ndarray) -> np.ndarray:
    predicted = np.broadcast_to(fit["intercept"], previous.shape).copy()
    if fit["matrices"]:
        predicted += previous @ fit["matrices"][0].T
    if len(fit["matrices"]) > 1:
        predicted += previous2 @ fit["matrices"][1].T
    return predicted


def _forecast(z: np.ndarray, split: int, fit: dict, steps: int, sfreq: float, scale: np.ndarray) -> dict:
    origins = np.arange(split + 2, z.shape[1] - steps + 1)
    previous, previous2 = z[:, origins - 1].T, z[:, origins - 2].T
    total, raw_total, terminal = 0., 0., None
    with np.errstate(over="ignore", invalid="ignore"):
        for step in range(steps):
            prediction = _predict(fit, previous, previous2)
            residual = z[:, origins + step].T - prediction
            if not np.isfinite(residual ** 2).all():
                return {"steps": steps, "horizon_s": steps / sfreq, "origins": len(origins),
                        "trajectory_mse_standardized": None, "terminal_mse_standardized": None,
                        "trajectory_mse_input_units_squared": None,
                        "numerical_status": "nonfinite_unconstrained_forecast"}
            terminal = float(np.mean(residual ** 2))
            total += terminal
            raw_total += float(np.mean((residual * scale) ** 2))
            previous2, previous = previous, prediction
    return {"steps": steps, "horizon_s": steps / sfreq, "origins": len(origins),
            "trajectory_mse_standardized": total / steps, "terminal_mse_standardized": terminal,
            "trajectory_mse_input_units_squared": raw_total / steps, "numerical_status": "finite"}


def _compare(array: np.ndarray, sfreq: float, options: dict) -> dict:
    split = int(np.floor(array.shape[1] * options["train_fraction"]))
    d, n = array.shape
    if split - 2 < max(30, 4 * (2 * d + 1)) or n - split - 2 < max(options["forecast_steps"]):
        raise ValueError("Window needs enough early samples for VAR(2) and later samples for every forecast horizon")
    mean = array[:, :split].mean(axis=1)
    scale = array[:, :split].std(axis=1, ddof=0)
    if np.any(scale <= 1e-12):
        raise ValueError("A selected channel has insufficient training variance; channels are not silently replaced")
    z = (array - mean[:, None]) / scale[:, None]
    if not np.isfinite(z).all():
        raise ValueError("Training-standardized data are nonfinite")
    target = z[:, split + 2:].T
    previous, previous2 = z[:, split + 1:-1].T, z[:, split:-2].T
    persistence_mse = float(np.mean((target - previous) ** 2))
    models = {}
    for family in FAMILIES:
        fit = _fit(z, split, family, options["ridge"], options["covariance_ridge"])
        residual = target - _predict(fit, previous, previous2)
        covariance = fit["innovation_covariance"]
        logdet = float(np.linalg.slogdet(covariance)[1])
        quadratic = float(np.mean(np.sum(residual * np.linalg.solve(covariance, residual.T).T, axis=1)))
        models[family] = {
            "family": family, "lag_order": len(fit["matrices"]),
            "equation": "z[t] = intercept + sum(A[k] @ z[t-k-1]) + innovation",
            "intercept_standardized": fit["intercept"].tolist(),
            "lag_matrices_standardized": [a.tolist() for a in fit["matrices"]],
            "innovation_covariance_standardized_regularized": covariance.tolist(),
            "slope_parameter_count": fit["slope_parameter_count"],
            "intercept_parameter_count": d, "innovation_covariance_parameter_count": d * (d + 1) // 2,
            "training_residual_mse_standardized": fit["training_residual_mse"],
            "one_step_mse_standardized": float(np.mean(residual ** 2)),
            "one_step_mse_by_channel_standardized": np.mean(residual ** 2, axis=0).tolist(),
            "one_step_mse_input_units_squared": float(np.mean((residual * scale) ** 2)),
            "mean_one_step_gaussian_nll": 0.5 * (d * np.log(2 * np.pi) + logdet + quadratic),
            "forecasts": [_forecast(z, split, fit, steps, sfreq, scale) for steps in options["forecast_steps"]],
            "descriptors": _descriptors(fit["matrices"], sfreq, d),
        }
    return {"split_sample": split, "training_sample_count": split, "training_transition_count": split - 2,
            "heldout_transition_count": n - split - 2, "first_heldout_target_sample": split + 2,
            "training_mean_input_units": mean.tolist(), "training_scale_input_units": scale.tolist(),
            "persistence_one_step_mse_standardized": persistence_mse, "models": models}


def temporal_surrogate(data: np.ndarray, split_sample: int, kind: str, seed: int) -> np.ndarray:
    """Transform early/later blocks separately so control training uses no test data.

    joint_shuffle preserves each block's contemporaneous multichannel samples
    and covariance exactly; it destroys ordering. independent_phase preserves
    each channel/block FFT magnitudes and circular autocovariance, disrupting
    cross-channel phase relations. Neither is a causal or biological null.
    Phase controls introduce block-edge effects in noncircular forecasting.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or not 1 < split_sample < x.shape[1] - 1:
        raise ValueError("Surrogate requires finite data and valid separate train/test blocks")
    if kind not in {"joint_shuffle", "independent_phase"}:
        raise ValueError("Unknown temporal surrogate")
    rng = np.random.default_rng(seed)
    pieces = []
    for block in (x[:, :split_sample], x[:, split_sample:]):
        if kind == "joint_shuffle":
            pieces.append(block[:, rng.permutation(block.shape[1])])
        else:
            spectrum = np.fft.rfft(block, axis=1)
            phase = rng.uniform(-np.pi, np.pi, spectrum.shape)
            phase[:, 0] = 0.
            if block.shape[1] % 2 == 0:
                phase[:, -1] = 0.
            pieces.append(np.fft.irfft(spectrum * np.exp(1j * phase), n=block.shape[1], axis=1))
    return np.concatenate(pieces, axis=1)


def compare_effective_dynamics(data: Any, sfreq_hz: Any, *, channel_names: Sequence[str] | None = None,
                               sample_times_s: Any = None, settings: Mapping[str, Any] | None = None) -> dict:
    """Fit all families without score/outcome selection and return strict-JSON data.

    Default channel selection is the first at most eight STORED observed
    channels, with names/indices reported. Explicit selection is preferable
    once frozen. Train/test boundaries exclude cross-boundary transitions.
    Test one-step predictions use observed held-out histories; recursive
    forecasts use only each origin's initial observed history, never subsequent
    observations. Origins overlap, so neither samples nor origins are treated
    as independent people. No family is required to win or to be stable.

    Without sample timestamps the caller asserts genuine continuous sampling;
    use analyze_recording_windows for annotated recordings. Input artifact and
    source qualification are separate responsibilities.
    """
    options = _options(settings)
    array, sfreq, selected, indices = _signal(data, sfreq_hz, channel_names, sample_times_s, options)
    result = _compare(array, sfreq, options)
    result.update({"schema_version": 1, "evidence_status": "WITHIN_WINDOW_MEASUREMENT_CONSTRUCTION",
                   "independent_human_validation": False, "interpretation": INTERPRETATION,
                   "selected_channel_names": selected, "selected_channel_indices": indices,
                   "n_channels": len(selected), "n_samples": array.shape[1], "sfreq_hz": sfreq,
                   "sample_step_s": 1. / sfreq, "settings": options,
                   "artifact_qualification": "not assessed by this model",
                   "selection_rule": "explicit channels" if options["channel_selection"] is not None else "first stored channels up to max_channels",
                   "scoring_note": "Common early targets start at sample 2; common later targets start two samples after the split. Recursive forecast origins overlap.",
                   "controls": {}})
    if options["temporal_controls"]:
        for index, kind in enumerate(("joint_shuffle", "independent_phase")):
            seed = options["seed"] + index
            surrogate = temporal_surrogate(array, result["split_sample"], kind, seed)
            comparison = _compare(surrogate, sfreq, options)
            result["controls"][kind] = {
                "seed": seed, "blockwise_transformation": True,
                "preserves": ("Exact contemporaneous channel vectors, marginal values and block covariance; not temporal ordering."
                              if kind == "joint_shuffle" else
                              "Per-channel block Fourier magnitudes and circular autocovariance; not original between-channel phase relations or boundary behaviour."),
                "models": comparison["models"],
            }
    # Reject silently unreportable results rather than writing nonstandard JSON.
    json.dumps(result, allow_nan=False)
    return result


def episode_features(result: Mapping[str, Any]) -> dict[str, float | None]:
    """Flatten explicitly named measurements; do not select a biological account."""
    features: dict[str, float | None] = {}
    for family, report in result["models"].items():
        prefix = f"effective_{family}"
        features[prefix + "_one_step_mse"] = report["one_step_mse_standardized"]
        features[prefix + "_gaussian_nll"] = report["mean_one_step_gaussian_nll"]
        for name in ("spectral_radius", "dominant_decay_time_s", "off_diagonal_frobenius_fraction"):
            features[prefix + "_" + name] = report["descriptors"][name]
        for horizon in report["forecasts"]:
            tag = prefix + f"_forecast_{horizon['steps']}samples"
            features[tag + "_horizon_s"] = horizon["horizon_s"]
            features[tag + "_trajectory_mse"] = horizon["trajectory_mse_standardized"]
            features[tag + "_terminal_mse"] = horizon["terminal_mse_standardized"]
        for kind, control in result.get("controls", {}).items():
            features[prefix + "_" + kind + "_one_step_mse"] = control["models"][family]["one_step_mse_standardized"]
    return features


def analyze_recording_windows(recording: Mapping[str, Any], windows: Sequence[Mapping[str, Any]], *,
                              settings: Mapping[str, Any] | None = None,
                              recording_id: str | None = None) -> dict:
    """Extract and fit each episode independently; gaps never become transitions."""
    options = _options(settings)
    reports, features, exclusions = [], [], []
    seen = set()
    metadata_keys = ("episode_id", "participant_id", "source_id", "cohort_id", "recording_id", "role")
    for row in windows:
        ident = str(row.get("episode_id", "")).strip()
        if not ident or ident in seen:
            raise ValueError("Windows require unique nonempty episode_id values")
        seen.add(ident)
        metadata = {key: row[key] for key in metadata_keys if key in row}
        try:
            if recording_id is not None and row.get("recording_id", recording_id) != recording_id:
                raise ValueError("Window recording_id does not match the supplied recording")
            start, stop = float(row["start_s"]), float(row["stop_s"])
            window = extract_window(recording, start, stop, duration_s=stop - start, tolerance_s=0.)
            result = compare_effective_dynamics(window, recording["sfreq_hz"],
                                               channel_names=recording["channel_names"], settings=options)
            result.update({"metadata": metadata, "start_s": start, "stop_s": stop})
            reports.append(result)
            features.append({**metadata, "start_s": start, "stop_s": stop, **episode_features(result)})
        except (ValueError, KeyError, TypeError) as error:
            exclusions.append({**metadata, "reason": str(error)})
    return {"evidence_status": "WITHIN_WINDOW_MEASUREMENT_CONSTRUCTION", "interpretation": INTERPRETATION,
            "settings": options, "episode_reports": reports, "features": features, "exclusions": exclusions,
            "counts": {"input_windows": len(windows), "fitted_windows": len(reports), "excluded_windows": len(exclusions)}}


def _generate(matrices: list[np.ndarray], covariance: np.ndarray, steps: int, rng: np.random.Generator) -> np.ndarray:
    d = covariance.shape[0]
    noise = rng.multivariate_normal(np.zeros(d), covariance, size=steps + 256)
    series = np.zeros((steps + 256, d))
    for t in range(2, len(series)):
        series[t] = noise[t]
        for lag, a in enumerate(matrices, start=1):
            series[t] += a @ series[t - lag]
    return series[256:].T


def synthetic_recovery_report(*, replicates: int = 4, samples: int = 1400, seed: int = 20261004,
                              ridge: float = 0.001) -> dict:
    """Known observation-space generators; report confusion without pass/fail gates.

    Correct-family coefficient errors are calculated in TRAINING-standardized
    coordinates, not confused with physical coupling. Nested families can win
    noisily and a richer family can approximate a simpler generator. The
    minimum held-out error is diagnostic simulation bookkeeping, not empirical
    selection or validation of a biological hypothesis.
    """
    for name, value, minimum in (("replicates", replicates, 1), ("samples", samples, 200), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    generators = {
        "covariance_only": [],
        "diagonal_ar1": [np.diag([0.7, 0.5, -0.3])],
        "coupled_var1": [np.array([[0.55, 0.25, 0.], [0., 0.45, 0.25], [0.15, 0., 0.35]])],
        "coupled_var2": [np.array([[0.4, 0.2, 0.], [0., 0.25, 0.2], [0.2, 0., 0.3]]), np.diag([0.3, -0.25, 0.2])],
    }
    covariance = np.array([[1., 0.2, 0.], [0.2, 0.8, 0.1], [0., 0.1, 0.6]])
    rng = np.random.default_rng(seed)
    rows = []
    confusion = {truth: {fitted: 0 for fitted in FAMILIES} for truth in FAMILIES}
    for truth, matrices in generators.items():
        for replicate in range(replicates):
            data = _generate(matrices, covariance, samples, rng)
            report = compare_effective_dynamics(data, 100., settings={"temporal_controls": False, "ridge": ridge})
            errors = {family: report["models"][family]["one_step_mse_standardized"] for family in FAMILIES}
            winner = min(errors, key=errors.get)
            confusion[truth][winner] += 1
            scale = np.asarray(report["training_scale_input_units"])
            expected = [a * scale[None, :] / scale[:, None] for a in matrices]
            fitted = [np.asarray(a) for a in report["models"][truth]["lag_matrices_standardized"]]
            matrix_rmse = float(np.sqrt(np.mean(np.concatenate([(a - b).ravel() for a, b in zip(expected, fitted)]) ** 2))) if expected else None
            rows.append({"generator": truth, "replicate": replicate, "minimum_error_family": winner,
                         "one_step_mse_by_family": errors, "correct_family_lag_matrix_rmse_standardized": matrix_rmse,
                         "expected_standardized_lag_matrices": [a.tolist() for a in expected],
                         "fitted_standardized_lag_matrices": [a.tolist() for a in fitted],
                         "correct_family_forecasts": report["models"][truth]["forecasts"]})
    result = {"evidence_status": "ASSUMPTIONS_ONLY_SYNTHETIC", "empirical_inference": False,
              "interpretation": "Recovery/confusion under these specified linear observation-space generators, not NFTsim, biological pathways or empirical human findings.",
              "settings": {"replicates": replicates, "samples": samples, "seed": seed, "ridge": ridge},
              "true_input_space_lag_matrices": {name: [a.tolist() for a in matrices] for name, matrices in generators.items()},
              "true_innovation_covariance": covariance.tolist(), "confusion_counts": confusion, "replicates": rows}
    json.dumps(result, allow_nan=False)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, help="Genuine continuous NPZ EEG; audited metadata and annotations required")
    parser.add_argument("--windows", type=Path, help="CSV episode_id,start_s,stop_s; episodes never concatenated")
    parser.add_argument("--recording-id", help="Check every supplied recording_id against this identity")
    parser.add_argument("--settings-json", type=Path, help="Prespecified settings; channel selection cannot use outcomes")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing directories are refused")
    parser.add_argument("--synthetic-recovery", action="store_true", help="Known-generator diagnostic, no empirical interpretation")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output-dir already exists; use a new path to preserve prior results")
    if args.synthetic_recovery:
        if args.recording or args.windows or args.settings_json or args.recording_id:
            parser.error("synthetic-recovery does not consume empirical inputs or custom settings")
        args.output_dir.mkdir(parents=True)
        write_json(args.output_dir / "synthetic_recovery.json", synthetic_recovery_report())
        return 0
    if not args.recording or not args.windows:
        parser.error("empirical measurement requires --recording and --windows")
    settings = None
    if args.settings_json:
        settings = json.loads(args.settings_json.read_text(encoding="utf-8-sig"))
        if not isinstance(settings, dict):
            parser.error("settings-json must contain an object")
    recording = read_recording(args.recording, "npz")
    windows = read_csv(args.windows, ("episode_id", "start_s", "stop_s"))
    report = analyze_recording_windows(recording, windows, settings=settings, recording_id=args.recording_id)
    args.output_dir.mkdir(parents=True)
    paths = []
    for item in report.pop("episode_reports"):
        key = hashlib.sha256(str(item["metadata"]["episode_id"]).encode()).hexdigest()[:24]
        path = args.output_dir / "episodes" / f"{key}.json"
        write_json(path, item)
        paths.append({"episode_id": item["metadata"]["episode_id"], "path": str(path.resolve()), "sha256": sha256_file(path)})
    write_csv(args.output_dir / "episode_features.csv", report.pop("features"),
              None if paths else ["episode_id", "start_s", "stop_s"])
    exclusions = report.pop("exclusions")
    write_csv(args.output_dir / "exclusions.csv", exclusions, None if exclusions else ["episode_id", "reason"])
    report.update({"inputs": {"recording_path": str(args.recording.resolve()), "recording_sha256": sha256_file(args.recording),
                              "windows_path": str(args.windows.resolve()), "windows_sha256": sha256_file(args.windows)},
                   "source_code_sha256": sha256_file(__file__), "python_version": platform.python_version(),
                   "numpy_version": np.__version__, "episode_receipts": paths,
                   "features_sha256": sha256_file(args.output_dir / "episode_features.csv"),
                   "recording_reference": recording.get("reference", "as_stored"), "recording_units": recording.get("units", "uV")})
    write_json(args.output_dir / "receipt.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
