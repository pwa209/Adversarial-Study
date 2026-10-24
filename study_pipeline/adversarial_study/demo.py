"""Synthetic fixture. These signals, labels and mock scores are not study data."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from . import core, models, statistics
from .cli import analyze, prepare


def create_demo(directory: Path, participants: int = 16, seed: int = 20261004) -> Path:
    directory = directory.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise core.StudyError("Synthetic fixture destination must be empty; preserve previous runs")
    signal_dir = directory / "data"
    signal_dir.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(core.read_json(Path(__file__).parent.parent / "configs" / "study.template.json"))
    cfg.update(mode="development", study_id="SYNTHETIC_SOFTWARE_CHECK_NOT_HUMAN_EVIDENCE", path_base=".", seed=seed)
    cfg["paths"].update(recordings="data/recordings.csv", episodes_development="data/episodes_development.csv", data_root="data", output_dir="runs", release_dir=str(Path(__file__).parent.parent.parent / "source_records" / "toker_v2"))
    cfg["instrument"]["channel_names"] = ["Cz"]
    cfg["analysis"].update(claim_id="DEMO_ONLY", primary_contrast_name="synthetic_A_minus_B",
        construct="synthetic_signal_condition", timing_definition="synthetic exact ten-second label interval",
        value_column="effective_rank", target_value="demo_A", comparator_value="demo_B", bootstrap_reps=300,
        equivalence_margin=.1, folds=4)
    cfg["precision"].update(eligible_n_grid=[40, 80], replicates=30)
    rng = np.random.default_rng(seed)
    recordings, episodes = [], []
    fs, episodes_per_person, channels = 250.0, 6, ["Cz", "Fz", "Pz", "Oz"]
    time = np.arange(int(10 * fs)) / fs
    for index in range(participants):
        pid, rid = f"SYNTHETIC_P{index:03d}", f"SYNTHETIC_R{index:03d}"
        source = f"SYNTHETIC_SITE_{index % 2}"
        segments = []
        for episode in range(episodes_per_person):
            condition = "demo_A" if episode % 2 == 0 else "demo_B"
            amplitude = 8 + rng.normal(0, .5)
            common = amplitude * np.sin(2 * np.pi * (10 if condition == "demo_A" else 5) * time)
            noise_sd = 1.5 if condition == "demo_A" else 4.0
            signal = np.vstack([common + noise_sd * rng.normal(size=len(time)) +
                    (channel + 1) * np.sin(2 * np.pi * (16 + channel) * time) for channel in range(4)])
            segments.append(signal)
            start, stop = episode * 10, (episode + 1) * 10
            episodes.append({"episode_id": f"{rid}_E{episode:02d}", "recording_id": rid,
                "start_s": start, "stop_s": stop, "report_start_s": start, "report_stop_s": stop,
                "report_category": condition, "capacity_value": (1 if condition == "demo_A" else .4) + rng.normal(0, .1)})
        path = signal_dir / f"{rid}.npz"
        np.savez_compressed(path, data=np.concatenate(segments, axis=1), sfreq_hz=fs,
                            channel_names=np.array(channels), units=np.array("uV"), reference=np.array("synthetic"))
        recordings.append({"recording_id": rid, "participant_id": pid, "source_id": source,
            "cohort_id": source, "site_id": source, "session_id": "synthetic_session",
            "role": "development", "signal_path": path.name, "signal_format": "npz", "modality": "EEG",
            "units": "uV", "duration_s": 60, "sfreq_hz": fs, "channel_count": 4,
            "detector_exposure": "no", "analyst_exposure": "yes", "synthetic": True})
    core.write_csv(signal_dir / "recordings.csv", recordings, list(core.RECORDING_COLUMNS))
    core.write_csv(signal_dir / "episodes_development.csv", episodes)
    config_path = directory / "demo.json"
    core.write_json(config_path, cfg)
    return config_path


def run_demo(directory: Path) -> dict:
    config_path = create_demo(directory)
    cfg = core.load_config(config_path)
    prepared = prepare(cfg)
    rows = core.read_csv(prepared / "features.csv")
    for row in rows:
        # Arbitrary mock readout for exercising aggregation/reporting. No weights
        # from the real checkpoint are loaded and no clinical inference occurs.
        gcs = float(np.clip(13 - 8 * float(row["relative_delta"]) - 4 * float(row["relative_theta"]), 3, 15))
        row.update(gcs=gcs, score=(gcs - 3) / 12, preclip_gcs="", saturated_fraction=0,
                   scored_channels=1, channel_aggregation="synthetic_mock", score_origin="synthetic_mock_not_toker")
    run = core.new_run(cfg, "synthetic_mock_scores", inputs=[prepared / "features.csv"])
    table = run / "analysis.csv"
    core.write_csv(table, rows)
    core.write_json(run / "analysis_table.json", {"analysis_sha256": core.sha256_file(table),
        "native_receipts_verified": False, "evidence_status": "synthetic_software_check_only"})
    core.finish_run(run)
    contrast = analyze(cfg, table)
    prediction = analyze(cfg, table, predict=True)
    diagnostics = core.new_run(cfg, "synthetic_diagnostics")
    core.write_json(diagnostics / "precision.json", {"evidence_status": "assumption_based_simulation_only",
                                                    "result": statistics.simulate_precision(cfg["precision"])})
    core.write_json(diagnostics / "recovery.json", {"evidence_status": "linear_simulator_test_only",
                                                   "result": models.effective_recovery({"replicates": 8, "seed": cfg["seed"]})})
    core.finish_run(diagnostics)
    summary = {"status": "synthetic_pipeline_completed", "human_evidence": False,
        "native_matlab_executed": False, "config": str(config_path), "prepared": str(prepared),
        "contrast": str(contrast), "grouped_prediction": str(prediction), "diagnostics": str(diagnostics)}
    core.write_json(directory / "demo_summary.json", summary)
    return summary
