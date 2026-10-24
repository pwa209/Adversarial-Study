"""Participant-disjoint native human-task fine-tuning of a clinical-model COPY.

The published clinical instrument remains fixed. A new task head and actual
pretrained backbone parameters are optimized in MATLAB on genuine human EEG and
original detection/report labels. This is neither GCS retraining nor a post-hoc
ridge calibration. No cross-system interpretation precedes actual adaptation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from adversarial_study import core, model_guided, native
from .detector_worker import matlab_string, run_matlab
from .nftsim_runtime import require_scheduled_job

EXPORT_FIELDS = ("example_id", "episode_id", "participant_id", "source_id", "recording_id",
                 "channel_name", "input_path", "sfreq_hz", "signal_sha256", "label", "split")
ASSAYS = {
    "detection": {"assay_id": "ds001785_tactile_detection_v1", "target_field": "detection",
                  "source_ids": ["openneuro_ds001785"],
                  "label_semantics": "Original subsequent tactile detection response, not awareness or GCS",
                  "label_vocabulary": ["not_detected", "detected"]},
    "report_category": {"assay_id": "sart_immediate_original_report_v1", "target_field": "report_category",
                        "source_ids": ["monash_wanderim", "mindblanking_paris"],
                        "label_semantics": "Original nominal just-before-interruption task-focus reports; no-recall remains separate; not a consciousness diagnosis",
                        "label_vocabulary": None},
}


def participant_split(people: list[str], seed: int = 20261004) -> dict:
    """Assign people label-blindly once; windows/channels never cross partitions."""
    identities = sorted(set(people))
    if len(identities) != len(people) or len(identities) < 6 or any(not p for p in identities):
        raise core.StudyError("Fine-tuning needs at least six unique human participants")
    ordered = sorted(identities, key=lambda p: hashlib.sha256((str(seed)+"\0"+p).encode()).hexdigest())
    validation_count = max(1, math.ceil(len(ordered)*.15))
    test_count = max(1, math.ceil(len(ordered)*.15))
    return {"seed": seed, "rule": "Label-blind SHA256 participant ordering; final ceil(15%) validation and preceding ceil(15%) test; remainder training",
            "train_participant_ids": sorted(ordered[:-(validation_count+test_count)]),
            "validation_participant_ids": sorted(ordered[-validation_count:]),
            "test_participant_ids": sorted(ordered[-(validation_count+test_count):-validation_count]),
            "interpretation": "Disjoint adaptation/development evaluation; previously analysed public data are not prospective independent replication"}


def validate_split(split: dict) -> dict[str, str]:
    names = {"train": "train_participant_ids", "validation": "validation_participant_ids", "test": "test_participant_ids"}
    mapping = {}
    for partition, key in names.items():
        values = split.get(key, [])
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise core.StudyError("Empty or duplicate human split identities: " + key)
        for value in values:
            if not isinstance(value, str) or not value or value in mapping:
                raise core.StudyError("Human participants overlap fine-tuning partitions")
            mapping[value] = partition
    return mapping


def _label(row: dict, assay: str) -> str | None:
    if assay == "detection":
        value = model_guided._finite(row.get("detection"))
        if value is None:
            return None
        if value not in (0., 1.):
            raise core.StudyError("Detection target must preserve an original binary response")
        return "detected" if value == 1 else "not_detected"
    value = row.get("report_category", "")
    return str(value) if value else None


def build_examples(observations: list[dict], jobs: list[dict], assay: str, split: dict,
                   *, maximum_examples: int = 12000) -> tuple[list[dict], list[dict]]:
    if assay not in ASSAYS or not 1 <= maximum_examples <= 50000:
        raise core.StudyError("Unknown human assay or unbounded example budget")
    people = validate_split(split)
    original = {(r["participant_id"], r["episode_id"]): r for r in observations}
    if len(original) != len(observations):
        raise core.StudyError("Duplicate human EEG outcome identity")
    examples, excluded, used = [], [], set()
    for job in jobs:
        key = job["participant_id"], job["episode_id"]
        if key not in original:
            raise core.StudyError("Export job lacks its actual human outcome")
        row = original[key]
        if row["source_id"] != job["source_id"] or row["source_id"] not in ASSAYS[assay]["source_ids"]:
            raise core.StudyError("Human data source does not measure the selected assay")
        identity = job["job_id"]
        if identity in used:
            raise core.StudyError("Repeated native EEG example identity")
        used.add(identity)
        label = _label(row, assay)
        if label is None:
            excluded.append({"example_id": identity, "participant_id": row["participant_id"], "episode_id": row["episode_id"], "reason": "missing_original_human_label"})
            continue
        if job["participant_id"] not in people:
            raise core.StudyError("Labelled human participant is absent from the fixed split")
        examples.append({"example_id": identity, "episode_id": row["episode_id"], "participant_id": row["participant_id"],
                         "source_id": row["source_id"], "recording_id": row["recording_id"], "label": label,
                         "split": people[row["participant_id"]], **{key: job[key] for key in ("channel_name", "input_path", "sfreq_hz", "signal_sha256")}})
    if len(examples) > maximum_examples:
        raise core.StudyError("Fine-tuning example budget exceeded; use a declared new budget rather than outcome-selecting samples")
    if {r["participant_id"] for r in examples} != set(people):
        raise core.StudyError("Assigned split identities differ from actual labelled human inputs")
    for partition in ("train", "validation", "test"):
        if not any(row["split"] == partition for row in examples):
            raise core.StudyError("No labelled human examples in " + partition)
    train_labels = {r["label"] for r in examples if r["split"] == "train"}
    if len(train_labels) < 2:
        raise core.StudyError("Training people do not provide at least two actual human classes")
    unexpected = {r["label"] for r in examples} - train_labels
    if unexpected:
        raise core.StudyError("Held-out labels have no training support; preserve split rather than relabel: " + ",".join(sorted(unexpected)))
    return examples, excluded


def export(pointer: Path, scores: Path, qualification: Path, output_dir: Path,
           *, assay: str = "detection", seed: int = 20261004, maximum_examples: int = 12000,
           split_path: Path | None = None) -> dict:
    require_scheduled_job()
    cfg, prepared, rows, inputs = model_guided._human(pointer, scores, qualification)
    if not rows:
        raise core.StudyError("No genuine scored human episodes for fine-tuning")
    if assay not in ASSAYS:
        raise core.StudyError("Unknown original human assay")
    split = core.read_json(split_path) if split_path else participant_split(sorted({row["participant_id"] for row in rows if _label(row, assay) is not None}), seed)
    jobs_path = prepared / "native/native_jobs.csv"
    jobs = core.read_csv(jobs_path)
    examples, excluded = build_examples(rows, jobs, assay, split, maximum_examples=maximum_examples)
    output = model_guided._new_output(output_dir)
    core.write_json(output / "dataset_split.json", split)
    core.write_csv(output / "human_examples.csv", examples, list(EXPORT_FIELDS))
    core.write_csv(output / "excluded_labels.csv", excluded, ["example_id", "participant_id", "episode_id", "reason"])
    classes = sorted({r["label"] for r in examples if r["split"] == "train"})
    assay_record = ASSAYS[assay] | {"source_ids": sorted({r["source_id"] for r in rows}),
                                   "input_modality": "genuine_single_channel_continuous_10s_human_scalp_EEG",
                                   "label_vocabulary": classes}
    exposures = sorted({str(row.get("detector_exposure", "unknown")) for row in rows})
    split_hash = core.sha256_file(output / "dataset_split.json")
    receipt = {"schema_version": 1, "status": "human_finetuning_inputs_verified_not_trained", "system_id": "clinical_human_task_copy",
               "human_data": True, "human_assay": assay_record, "assay_id": assay_record["assay_id"], "split": split | {"split_sha256": split_hash},
               "data_split_sha256": split_hash, "examples_sha256": core.sha256_file(output / "human_examples.csv"),
               "train_examples_sha256": hashlib.sha256(json.dumps([r for r in core.read_csv(output / "human_examples.csv") if r["split"] == "train"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "label_exclusions": len(excluded), "example_counts": dict(Counter(r["split"] for r in examples)),
               "episode_counts": {s: len({r["episode_id"] for r in examples if r["split"] == s}) for s in ("train", "validation", "test")},
               "test_observation_ids": sorted({r["episode_id"] for r in examples if r["split"] == "test"}),
               "pretrained_exposure": {"status": "unknown", "recorded_values": exposures,
                                      "released_variant_and_training_manifest": "unresolved; disjoint adaptation partitions do not resolve pretrained exposure"},
               "inputs": [{"path": str(Path(p).resolve()), "sha256": core.sha256_file(p)} for p in inputs],
               "original_instrument": native.verify_release(core.resolve_path(cfg, "release_dir", True)),
               "readout_semantics": "Separate human-task classifier; original released clinical GCS predictor unchanged",
               "native_training_executed": False, "adaptation_completed": False,
               "test_access": "Labels exported as held-out evaluation input; native trainer does not use test arrays for optimization or validation"}
    if split_path:
        receipt["inputs"].append({"path": str(Path(split_path).resolve()), "sha256": core.sha256_file(split_path)})
    core.write_json(output / "export_receipt.json", receipt)
    return receipt


def verify_export(export_dir: Path) -> dict:
    directory = Path(export_dir).resolve()
    receipt = core.read_json(directory / "export_receipt.json")
    if receipt.get("status") != "human_finetuning_inputs_verified_not_trained" or receipt.get("human_data") is not True:
        raise core.StudyError("Fine-tuning requires genuine human input provenance")
    mapping = validate_split(receipt["split"])
    for name, key in (("dataset_split.json", "data_split_sha256"), ("human_examples.csv", "examples_sha256")):
        if core.sha256_file(directory / name) != receipt[key]:
            raise core.StudyError("Human fine-tuning split/examples changed")
    for item in receipt["inputs"]:
        if core.sha256_file(item["path"]) != item["sha256"]:
            raise core.StudyError("Human fine-tuning source input changed")
    examples = core.read_csv(directory / "human_examples.csv")
    for row in examples:
        if mapping.get(row["participant_id"]) != row["split"] or row["source_id"] not in receipt["human_assay"]["source_ids"] or row["label"] not in receipt["human_assay"]["label_vocabulary"]:
            raise core.StudyError("Human example does not match its actual assay/fixed partition")
        if core.sha256_file(row["input_path"]) != row["signal_sha256"]:
            raise core.StudyError("Actual human EEG fine-tuning signal changed")
    canonical_training = hashlib.sha256(json.dumps([r for r in examples if r["split"] == "train"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # CSV numeric fields are textual. Hash the stable written representation.
    if canonical_training != receipt["train_examples_sha256"]:
        raise core.StudyError("Actual training example identities/labels differ")
    return receipt


def verify_completed(output_dir: Path, export_dir: Path) -> dict:
    """Reject head-only/planned adaptation or unmatched held-person predictions."""
    receipt = verify_export(export_dir)
    result = core.read_json(Path(output_dir) / "training_receipt.json")
    if result.get("status") != "completed_actual_human_finetune" or result.get("native_training_executed") is not True or result.get("backbone_training") is not True or result.get("optimizer_steps", 0) <= 0 or result.get("human_data") is not True or result.get("adaptation_completed") is not True:
        raise core.StudyError("Native receipt does not establish actual human backbone fine-tuning")
    if result.get("data_split_sha256") != receipt["data_split_sha256"] or result.get("original_checkpoint_sha256") != native.CHECKPOINT_SHA256 or result.get("human_assay") != receipt["human_assay"]:
        raise core.StudyError("Fine-tuning instrument/split/actual human assay differs")
    for key in ("train_participant_ids", "validation_participant_ids", "test_participant_ids"):
        if sorted(result.get(key, [])) != sorted(receipt["split"][key]):
            raise core.StudyError("Fine-tuning used different actual human participant partitions")
    if core.sha256_file(result["trained_checkpoint"]["path"]) != result["trained_checkpoint"]["sha256"] or result.get("adapted_checkpoint_sha256") != result["trained_checkpoint"]["sha256"] or result["trained_checkpoint"]["sha256"] == native.CHECKPOINT_SHA256:
        raise core.StudyError("Adapted native checkpoint changed or is the original")
    updates = result["backbone_updates"]
    if core.sha256_file(updates["path"]) != updates["sha256"]:
        raise core.StudyError("Backbone update evidence changed")
    changes = json.loads(Path(updates["path"]).read_text(encoding="utf-8"))
    if isinstance(changes, dict):  # MATLAB encodes a scalar struct as one object.
        changes = [changes]
    if not changes or len(changes) != result.get("changed_backbone_parameter_count") or any(r["before_sha256"] == r["after_sha256"] or r["layer"] not in result.get("explicit_trained_backbone_layers", []) for r in changes):
        raise core.StudyError("No actual returned pretrained backbone update evidence")
    evaluation = core.read_json(Path(output_dir) / "evaluation_receipt.json")
    if evaluation.get("status") != "completed_actual_heldout_human_evaluation" or evaluation.get("data_split_sha256") != receipt["data_split_sha256"] or evaluation.get("human_assay") != receipt["human_assay"]:
        raise core.StudyError("Held-out human evaluation has not completed for this assay/split")
    predictions = evaluation["test_predictions"]
    if core.sha256_file(predictions["path"]) != predictions["sha256"] or result.get("test_predictions") != predictions:
        raise core.StudyError("Held-out native predictions changed")
    expected = {}
    for row in core.read_csv(Path(export_dir) / "human_examples.csv"):
        if row["split"] == "test":
            identity = row["episode_id"]
            attributes = (row["participant_id"], row["source_id"], row["label"])
            if identity in expected and expected[identity][0] != attributes:
                raise core.StudyError("Actual held-out episode identity conflicts")
            old_count = expected.get(identity, (None, 0))[1]
            expected[identity] = attributes, old_count + 1
    rows = core.read_csv(predictions["path"])
    if len(rows) != len(expected) or {r["episode_id"] for r in rows} != set(expected):
        raise core.StudyError("Actual predictions omit/add held-out human episodes")
    for row in rows:
        attributes, channel_count = expected[row["episode_id"]]
        if (row["participant_id"], row["source_id"], row["label"]) != attributes or int(row["channel_count"]) != channel_count or row["prediction"] not in receipt["human_assay"]["label_vocabulary"]:
            raise core.StudyError("Held-out native prediction labels/identities differ")
        probabilities = [model_guided._finite(v) for k, v in row.items() if k.startswith("probability_")]
        if len(probabilities) != len(receipt["human_assay"]["label_vocabulary"]) or any(v is None or not 0 <= v <= 1 for v in probabilities) or abs(sum(probabilities) - 1) > 1e-5:
            raise core.StudyError("Actual held-out class probabilities are invalid")
    return result


def verify_adaptation(directory: Path) -> dict:
    """Public artifact-only audit for cross-system readiness; no MATLAB execution."""
    output = Path(directory).resolve()
    result = core.read_json(output / "training_receipt.json")
    export_dir = Path(result["export_dir"]).resolve()
    original_receipt = result["export_receipt"]
    if Path(original_receipt["path"]).resolve() != export_dir / "export_receipt.json" or core.sha256_file(original_receipt["path"]) != original_receipt["sha256"]:
        raise core.StudyError("Clinical adaptation export receipt changed")
    for key, name in (("trained_checkpoint", "adapted_human_task_net.mat"), ("backbone_updates", "backbone_updates.json"), ("test_predictions", "heldout_predictions.csv")):
        if Path(result[key]["path"]).resolve() != output / name:
            raise core.StudyError("Clinical adaptation artifacts are not bound to their native output directory")
    result = verify_completed(output, export_dir)
    native.verify_release(Path(result["original_release_dir"]))
    evaluation = core.read_json(output / "evaluation_receipt.json")
    if evaluation.get("trained_checkpoint") != result["trained_checkpoint"] or evaluation.get("split") != result["split"]:
        raise core.StudyError("Clinical held-out evaluation does not use the adapted checkpoint/fixed split")
    observed = {r["episode_id"] for r in core.read_csv(result["test_predictions"]["path"])}
    if set(result.get("test_observation_ids", [])) != observed or set(evaluation.get("test_observation_ids", [])) != observed:
        raise core.StudyError("Actual clinical test prediction identities differ from evaluation receipts")
    return result


def train(export_dir: Path, release_dir: Path, output_dir: Path, *, epochs: int = 8,
          minibatch: int = 32, learning_rate: float = .0001, backbone_factor: float = .1,
          matlab: str = "matlab", timeout_s: int = 14400) -> dict:
    require_scheduled_job()
    receipt = verify_export(export_dir)
    native.verify_release(release_dir)
    if not 1 <= epochs <= 30 or not 1 <= minibatch <= 128 or not 0 < learning_rate <= .01 or not 0 < backbone_factor <= 1 or not 1 <= timeout_s <= 43200:
        raise core.StudyError("Human fine-tuning parameters exceed bounded native design")
    output = model_guided._new_output(output_dir)
    native_dir = Path(__file__).resolve().parent
    command = "addpath(" + matlab_string(native_dir) + "); addpath(" + matlab_string(native_dir.parent / "matlab") + "); human_finetune_native(" + ",".join([
        matlab_string(Path(export_dir).resolve()), matlab_string(Path(release_dir).resolve()), matlab_string(output),
        str(epochs), str(minibatch), format(learning_rate, ".17g"), format(backbone_factor, ".17g")]) + ");"
    worker = run_matlab(command, output / "matlab_training.log", matlab=matlab, timeout=timeout_s)
    worker["export_receipt_sha256"] = core.sha256_file(Path(export_dir) / "export_receipt.json")
    worker["task"] = receipt["human_assay"]
    core.write_json(output / "worker_receipt.json", worker)
    result_path = output / "training_receipt.json"
    if worker["returncode"] != 0 or not result_path.is_file():
        raise core.StudyError("Native human-task fine-tuning did not complete; retain MATLAB log/worker receipt")
    result = verify_adaptation(output)
    native.verify_release(release_dir)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("export")
    for key in ("pointer", "scores", "qualification", "output-dir"):
        command.add_argument("--" + key, type=Path, required=True)
    command.add_argument("--assay", choices=sorted(ASSAYS), default="detection")
    command.add_argument("--seed", type=int, default=20261004)
    command.add_argument("--maximum-examples", type=int, default=12000)
    command.add_argument("--split-path", type=Path)
    command = commands.add_parser("train")
    for key in ("export-dir", "release-dir", "output-dir"):
        command.add_argument("--" + key, type=Path, required=True)
    command.add_argument("--epochs", type=int, default=8)
    command.add_argument("--minibatch", type=int, default=32)
    command.add_argument("--learning-rate", type=float, default=.0001)
    command.add_argument("--backbone-factor", type=float, default=.1)
    command.add_argument("--matlab", default="matlab")
    command.add_argument("--timeout-s", type=int, default=14400)
    args = vars(parser.parse_args(argv))
    name = args.pop("command")
    try:
        result = globals()[name](**args)
        print(json.dumps({"status": result["status"], "assay_id": result.get("assay_id")}))
    except (core.StudyError, OSError, KeyError, ValueError) as exc:
        parser.exit(2, "Native human-data adaptation failed: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
