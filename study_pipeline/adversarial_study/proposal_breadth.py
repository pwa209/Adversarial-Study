"""Discovery-selected, source-role-qualified native behavioural fine-tuning.

The source choice alphabet is observed, never invented. Confidence markers stay
in the original transcript but are not choice endpoints. Both native models must
actually learn from the same human participants before any held-person comparison.
This task analogy does not certify a neural-to-behavioural mechanism or calibration.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import sys

from . import core, extension_finetune as native, extension_sources, proposal_discovery
from .acquisition import canonical_hash

DEFAULT_CONFIG = Path(__file__).parents[1]/"configs/proposal_breadth.json"
DEFAULT_GENERATOR = Path(__file__).parents[1]/"docs/proposal_sources/hellmann_generate_prompts.py"
STUDY = "hellmann_unpublished_brightness"
SOURCE = "psych201"
ASSAY = "psych201:hellmann_unpublished_brightness:source_original_brightness_choice"
READY = "qualified_brightness_human_split_prepared"
GENERATOR_SHA = "89be6d1eb31e06d132587b753b1aa5c9d2b3da66de37b9bb809356a61e9dca66"
COMMIT = "c63dea18897393b706227b50024b27fa8d8907d8"
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
BRIGHTNESS = re.compile(r"Left disc has brightness ("+NUMBER+r"), right disc has brightness ("+NUMBER+r")\. You press $")
CONFIDENCE_GAP = re.compile(r" after ("+NUMBER+r")ms\. You rate your confidence with $")
ORIGINAL_MARKERS = extension_sources.marked_responses
ORIGINAL_ROWS = extension_sources._behaviour_rows


def _record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": core.sha256_file(path)}


def _code_records():
    return [_record(module.__file__) for module in
            (sys.modules[__name__], native, extension_sources, core, proposal_discovery,
             sys.modules[native.resource_audit.__module__], sys.modules[canonical_hash.__module__])]


def load_config(path=DEFAULT_CONFIG):
    cfg = core.read_json(path)
    if (cfg.get("schema_version"), cfg.get("study"), cfg.get("source_id"), cfg.get("generator_commit"), cfg.get("generator_sha256")) != (1, STUDY, SOURCE, COMMIT, GENERATOR_SHA):
        raise core.StudyError("Require the exact pinned brightness source and generator specification")
    training = cfg.get("training", {})
    if set(training.get("people", {})) != {"train", "validation", "test"} or any(type(v) is not int or v < 1 for v in training["people"].values()):
        raise core.StudyError("Positive, explicit participant counts required")
    if training.get("require_psych101_unexposed") is not True:
        raise core.StudyError("Explicit false provider exposure flags are required")
    for name in ("max_responses_per_session", "max_context_tokens", "epochs", "max_optimizer_steps", "validation_events_per_person", "test_events_per_person", "gradient_accumulation"):
        if type(training.get(name)) is not int or training[name] < 1:
            raise core.StudyError("Positive native training budget required: "+name)
    for name in ("maximum_source_rows", "maximum_target_sessions"):
        if type(cfg.get(name)) is not int or cfg[name] < 1:
            raise core.StudyError("Positive bounded census limit required: "+name)
    return cfg


def source_identity(row):
    """Accept exact generator experiment identities; never substring task names."""
    experiment = row.get("experiment")
    study = row.get("study")
    if not isinstance(experiment, str) or experiment.split("/")[0] != STUDY:
        return None
    if "study" in row and (not isinstance(study, str) or study.split("/")[0] != STUDY):
        return None
    participant = row.get("participant")
    if not isinstance(participant, str) or not participant.strip():
        raise core.StudyError("Original source participant must be a nonempty string")
    return STUDY, experiment, participant


def qualify_markers(text):
    """Keep original indices and source labels; verify paired generator roles."""
    if not isinstance(text, str) or not text.startswith("You see two discs in each trial, varying in brightness."):
        raise core.StudyError("Original pinned brightness instructions missing")
    events = ORIGINAL_MARKERS(text)
    if not events or len(events) % 2:
        raise core.StudyError("Pinned trial requires one original choice and one confidence marker")
    choices, missing_confidence = [], 0
    for ordinal in range(0, len(events), 2):
        choice, confidence = events[ordinal:ordinal+2]
        choice_prefix = choice["prefix"][:-2]
        brightness = BRIGHTNESS.search(choice_prefix)
        if not brightness or not all(math.isfinite(float(x)) for x in brightness.groups()):
            raise core.StudyError("Choice marker lacks exact finite source brightness trial/prefix")
        if not confidence["prefix"][:-2].endswith("You rate your confidence with "):
            raise core.StudyError("Confidence marker cannot be relabelled as a choice")
        gap = text[choice["stop"]+2:confidence["start"]-2]
        rt = CONFIDENCE_GAP.fullmatch(gap)
        if not rt or not math.isfinite(float(rt.group(1))) or float(rt.group(1)) < 0:
            raise core.StudyError("Original choice/confidence timing grammar incompatible")
        value = confidence["response"].strip()
        if value.lower() in {"nan", "none", "na", "n/a", "missing"}:
            missing_confidence += 1
        else:
            try:
                rating = float(value)
            except ValueError as exc:
                raise core.StudyError("Confidence is not source numeric/missing metadata") from exc
            if not math.isfinite(rating) or not -1 <= rating <= 1:
                raise core.StudyError("Confidence violates the original -1 to 1 source scale")
        choices.append(choice)
    return choices, {"choice_events": len(choices), "confidence_events": len(choices),
                     "missing_confidence_events": missing_confidence, "original_marker_count": len(events)}


def choice_responses(text):
    return qualify_markers(text)[0]


def construct_selection(discovery_dir):
    receipt = proposal_discovery.verify_outputs(discovery_dir)
    rows = core.read_csv(Path(discovery_dir)/"observations.csv", ("source_id", "participant_id", "report_eligibility"))
    groups = defaultdict(lambda: {"rows": 0, "observed_report_rows": 0, "measured_rows": 0, "people": set()})
    for row in rows:
        # Source semantics are explicit; numeric unrelated capacities are not mapped.
        if row["source_id"] != "openneuro_ds001785":
            continue
        for target in ("detection", "confidence"):
            value = row.get(target)
            if value in (None, "") and row.get("capacity_construct") == "subsequent_tactile_"+target:
                value = row.get("capacity_value")
            group = groups[(row["source_id"], target)]
            group["rows"] += 1
            if row["report_eligibility"] == "observed_original_report":
                group["observed_report_rows"] += 1
            try:
                numeric = float(value)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(numeric) or (target == "detection" and numeric not in (0, 1)):
                continue
            group["measured_rows"] += 1
            group["people"].add(row["participant_id"])
    census = [{"source_id": source, "target": target, **{k: v for k, v in group.items() if k != "people"},
               "independent_participants": len(group["people"]), "source_construct": "subsequent_tactile_"+target}
              for (source, target), group in sorted(groups.items())]
    return {"status": "prespecified_construct_candidate_selected" if any(r["measured_rows"] for r in census) else "BLOCKED_NO_MEASURED_RELEVANT_CONSTRUCT",
            "candidate_study": STUDY, "candidate_family": "perceptual_discrimination_and_confidence",
            "human_construct_census": census, "discovery_receipt": _record(Path(discovery_dir)/"receipt.json"),
            "discovery_status": receipt["status"], "selection_rule": "Any actual source-qualified measured detection/confidence; no observed association, effect, sign, significance or favorable-yield threshold",
            "result_direction_gate": False, "neural_to_behavioural_mapping_qualified": False,
            "required_mapping": "Independent paired human neural and behavioural measures with calibrated tactile/visual assay mapping and shared computational prediction"}


def source_census(source_root, manifest_path, cfg):
    root, manifest = Path(source_root).resolve(), core.read_json(manifest_path)
    resources = [r for r in manifest["resources"] if r["source_id"] == SOURCE]
    if not resources:
        raise core.StudyError("Actual Psych-201 source resources absent")
    resource_receipts, sessions, person_rows = {}, [], defaultdict(list)
    all_studies, exposures, alphabet, all_exposures = Counter(), Counter(), Counter(), Counter()
    rows_count = 0
    for resource, receipt, row in ORIGINAL_ROWS(resources, root):
        rows_count += 1
        if rows_count > cfg["maximum_source_rows"]:
            raise core.StudyError("Whole-source census exceeds declared row budget; no outcome-selected truncation")
        resource_receipts[resource["resource_id"]] = {**receipt, "path": str((root/resource["relative_path"]).resolve())}
        label = str(row.get("study", row.get("experiment", ""))).split("/")[0]
        all_studies[label] += 1
        flag_pattern = json.dumps({k: row.get(k) for k in ("is_psych101", "is_psych101_test", "in_eval")}, sort_keys=True, allow_nan=False)
        all_exposures[(label, flag_pattern)] += 1
        try:
            identity = source_identity(row)
        except core.StudyError as exc:
            identity = None
            if str(row.get("experiment", "")).split("/")[0] == STUDY:
                if len(sessions) >= cfg["maximum_target_sessions"]:
                    raise core.StudyError("Target session census exceeds declared bound")
                sessions.append({"source_resource_id": resource["resource_id"], "participant": "", "experiment": row.get("experiment", ""), "schema_status": "invalid", "reason": str(exc), "eligible": False})
        if identity is None:
            continue
        if len(sessions) >= cfg["maximum_target_sessions"]:
            raise core.StudyError("Target session census exceeds declared bound")
        study, experiment, participant = identity
        flags = {k: row.get(k) for k in ("is_psych101", "is_psych101_test", "in_eval")}
        exposures[flag_pattern] += 1
        qualified_exposure = flags["is_psych101"] is False and flags["is_psych101_test"] is False
        entry = {"source_resource_id": resource["resource_id"], "resource_sha256": receipt["sha256"],
                 "participant": participant, "experiment": experiment, "study": study,
                 "exposure_flags": flags, "provider_exposure_explicit_false": qualified_exposure,
                 "text_sha256": hashlib.sha256(row.get("text", "").encode()).hexdigest() if isinstance(row.get("text"), str) else "",
                 "schema_status": "invalid", "reason": "", "eligible": False}
        try:
            choices, count = qualify_markers(row["text"])
            entry.update(count, schema_status="qualified", choice_alphabet=sorted({e["response"].strip() for e in choices}))
            alphabet.update(e["response"].strip() for e in choices)
        except (ValueError, KeyError) as exc:
            entry["reason"] = str(exc)
        sessions.append(entry)
        person_rows[study+"\0"+participant].append(entry)
    eligible_people = set()
    for person, entries in person_rows.items():
        # A clean session cannot erase affirmative/unknown exposure in another.
        unexposed = all(e["provider_exposure_explicit_false"] for e in entries)
        if unexposed and any(e["schema_status"] == "qualified" for e in entries):
            eligible_people.add(person)
        for entry in entries:
            entry["eligible"] = unexposed and entry["schema_status"] == "qualified"
            if not unexposed:
                entry["reason"] = "Person has affirmative or unknown prior provider exposure in at least one target session"
    unique_sessions = [(e["study"], e["experiment"], e["participant"]) for e in sessions if e.get("eligible")]
    duplicate_sessions = len(unique_sessions)-len(set(unique_sessions))
    return {"status": "whole_psych201_source_census_completed", "all_source_rows": rows_count,
            "all_study_session_counts": dict(sorted(all_studies.items())), "target_study": STUDY,
            "all_source_exposure_census": [{"study": study, "exposure_flags": json.loads(pattern), "sessions": n} for (study, pattern), n in sorted(all_exposures.items())],
            "target_sessions": len(sessions), "target_original_people": len(person_rows),
            "eligible_unexposed_people": len(eligible_people), "eligible_sessions": sum(e.get("eligible", False) for e in sessions),
            "duplicate_eligible_session_identities": duplicate_sessions,
            "original_choice_alphabet": sorted(alphabet), "original_choice_label_counts": dict(sorted(alphabet.items())),
            "exposure_flag_pattern_counts": dict(exposures), "source_resources": list(resource_receipts.values()),
            "sessions": sessions, "eligible_people": sorted(eligible_people),
            "confidence_role": "Raw original confidence markers retained, not choice endpoints; missingness and source -1..1 scale explicitly audited",
            "response_semantics": "Observed rawCSV response labels; no inferred L/R recoding or correctness mapping",
            "provider_prior_exposure": "All target sessions must have is_psych101=False and is_psych101_test=False; other historical model exposure unknown"}


def _qualified_rows(census):
    allowed = {(e["source_resource_id"], e["experiment"], e["participant"], e["text_sha256"])
               for e in census["sessions"] if e.get("eligible")}
    def rows(resources, root):
        for resource, receipt, row in ORIGINAL_ROWS(resources, root):
            try:
                identity = source_identity(row)
            except core.StudyError:
                continue
            if identity is None:
                continue
            _, experiment, participant = identity
            if not isinstance(row.get("text"), str):
                continue
            text_hash = hashlib.sha256(row["text"].encode()).hexdigest()
            if (resource["resource_id"], experiment, participant, text_hash) in allowed:
                yield resource, receipt, row
    return rows


def human_assay(training_cfg):
    return {"assay_id": ASSAY, "target_field": "source_original_brightness_choice", "source_ids": [SOURCE],
            "study": STUDY, "input_modality": "behavioural_task_transcript",
            "label_semantics": "Original rawCSV .response for brighter-disc arrow-key choice; confidence/experience markers are not choice labels; physical side/correctness mapping not inferred",
            "response_alphabet": training_cfg["response_alphabet"], "response_role": "You press <<original_response>>",
            "original_response_indices_preserved": True, "confidence_fitted_as_choice": False,
            "neural_to_behavioural_mapping_qualified": False}


def optimizer_observations(events, cfg):
    """Reconstruct the unchanged native fixed-budget shuffled observation order."""
    generator, updates, seen = random.Random(cfg["seed"]), 0, []
    for epoch in range(cfg["epochs"]):
        order = list(range(len(events)))
        generator.shuffle(order)
        for first in range(0, len(order), cfg["gradient_accumulation"]):
            if updates >= cfg["max_optimizer_steps"]:
                break
            indices = order[first:first+cfg["gradient_accumulation"]]
            seen.extend((epoch, events[index]["session_id"], events[index]["response_index"]) for index in indices)
            updates += 1
        if updates >= cfg["max_optimizer_steps"]:
            break
    return seen, updates


@contextmanager
def role_context(*, census=None, training_cfg=None, prepared_record=None, comparison_config=None):
    """Scoped adapter around unchanged native operations; restore on every exit."""
    saved = native.marked_responses, native._behaviour_rows, native.write_json, native._json, native._token_events
    captured = {}
    def token_events(data, tokenizer, cfg):
        events, excluded = saved[4](data, tokenizer, cfg)
        roles = {row.get("fine_tune_role") for row in data}
        if training_cfg is not None and len(roles) == 1 and next(iter(roles)) in ("train", "validation"):
            role = next(iter(roles))
            selected = native._person_event_limit(events, cfg["validation_events_per_person"]) if role == "validation" else events
            identities = [{k: event[k] for k in ("session_id", "person_id", "response_index", "response")} for event in selected]
            captured[role] = {"observations": identities, "native_token_events_sha256": native.event_fingerprint(selected),
                              "original_human_observations_sha256": native.human_observation_fingerprint(data, identities),
                              "tokenizer_exclusions": excluded}
        return events, excluded
    def writer(path, value):
        if Path(path).name == "training_receipt.json" and value.get("status") == "human_finetuning_completed":
            if training_cfg is None or prepared_record is None:
                raise core.StudyError("Native brightness receipt needs source-qualified prepared provenance")
            value = deepcopy(value)
            value["assay_id"], value["human_assay"] = ASSAY, human_assay(training_cfg)
            value["proposal_breadth_prepared"] = prepared_record
            value["proposal_breadth_code_sha256"] = core.sha256_file(__file__)
            value["source_role_adapter"] = "Choice-marker filtering preserves source text/labels/marker indices; truthful assay metadata written on first artifact creation"
            if set(captured) != {"train", "validation"} or captured["train"]["native_token_events_sha256"] != value["train_events_sha256"] or captured["validation"]["native_token_events_sha256"] != value["validation_events_sha256"]:
                raise core.StudyError("Actual native event/tokenizer selection differs from captured source roles")
            order, updates = optimizer_observations(captured["train"]["observations"], training_cfg)
            if canonical_hash(order) != value["actual_training_order_sha256"] or updates != value["actual_optimizer_steps"]:
                raise core.StudyError("Actual native optimizer observation order/budget differs")
            value["source_role_training_events"] = captured
            value["actual_optimizer_observation_order"] = order
        return saved[2](path, value)
    def reader(path):
        value = saved[3](path)
        if comparison_config is not None and Path(path).resolve() == Path(comparison_config).resolve():
            # Legacy index check confuses marker index with selected choice ordinal.
            # The additive comparator enforces the actual frozen ordinal budget.
            value = dict(value)
            value.pop("max_responses_per_session", None)
        return value
    try:
        native.marked_responses = choice_responses
        native._token_events = token_events
        if census is not None:
            native._behaviour_rows = _qualified_rows(census)
        native.write_json, native._json = writer, reader
        yield
    finally:
        native.marked_responses, native._behaviour_rows, native.write_json, native._json, native._token_events = saved


def prepare(discovery_dir, source_root, manifest_path, config_path, generator_path, output_dir):
    cfg = load_config(config_path)
    generator = _record(generator_path)
    if generator["sha256"] != cfg["generator_sha256"]:
        raise core.StudyError("Pinned source generator bytes differ; do not improvise a task schema")
    provenance = {"config": _record(config_path), "manifest": _record(manifest_path), "generator": generator,
                  "code": _code_records(), "discovery_dir": str(Path(discovery_dir).resolve()),
                  "source_root": str(Path(source_root).resolve())}
    selection = construct_selection(discovery_dir)
    census = source_census(source_root, manifest_path, cfg)
    status, reason = READY, ""
    if selection["status"].startswith("BLOCKED"):
        status, reason = selection["status"], "No actual source-qualified detection/confidence measurement"
    elif len(census["original_choice_alphabet"]) != 2:
        status, reason = "BLOCKED_UNQUALIFIED_SOURCE_CHOICE_ALPHABET", "Pinned binary brighter-disc task requires exactly two source-observed original choice labels"
    elif census["duplicate_eligible_session_identities"]:
        status, reason = "BLOCKED_DUPLICATE_SOURCE_SESSION", "Resolve original source duplicate session identities before defining a split"
    elif census["eligible_unexposed_people"] < sum(cfg["training"]["people"].values()):
        status, reason = "BLOCKED_INSUFFICIENT_ELIGIBLE_PEOPLE", "Fewer independent source-qualified unexposed human participants than fixed requested split"
    target = Path(output_dir).resolve()
    if target.exists():
        raise core.StudyError("Preserve existing prepared artifact; use a new output directory")
    target.mkdir(parents=True)
    core.write_json(target/"source_census.json", census)
    core.write_json(target/"construct_selection.json", selection)
    training_cfg = {**cfg["training"], "study": STUDY, "source_id": SOURCE,
                    "response_alphabet": census["original_choice_alphabet"],
                    "response_alphabet_evidence": "Complete pinned source census of original choice-role markers; no assumed L/R mapping",
                    "assay_id": ASSAY, "response_role": "source You press choice marker only",
                    "source_generator_sha256": generator["sha256"], "source_role_config_sha256": provenance["config"]["sha256"]}
    core.write_json(target/"training.config.json", training_cfg)
    split = None
    if status == READY:
        with role_context(census=census):
            split = native.make_split(source_root, manifest_path, target/"training.config.json", target/"split")
        qualify_split(target/"split", target/"training.config.json", open_test=True)
    for item in [provenance["config"], provenance["manifest"], generator]+provenance["code"]:
        if core.sha256_file(item["path"]) != item["sha256"]:
            raise core.StudyError("Preparation input/code changed during complete source scan")
    for item in census["source_resources"]:
        if core.sha256_file(item["path"]) != item["sha256"]:
            raise core.StudyError("Actual behavioural source changed during preparation")
    proposal_discovery.verify_outputs(discovery_dir)
    outputs = {name: core.sha256_file(target/name) for name in ("source_census.json", "construct_selection.json", "training.config.json")}
    if split:
        outputs.update({"split/"+p.name: core.sha256_file(p) for p in sorted((target/"split").iterdir()) if p.is_file()})
    receipt = {"schema_version": 1, "status": status, "reason": reason, "created_utc": datetime.now(timezone.utc).isoformat(),
               **provenance, "discovery_receipt": selection["discovery_receipt"], "source_resources": census["source_resources"],
               "outputs": outputs, "requested_independent_people": cfg["training"]["people"],
               "actual_split_people": split["independent_people"] if split else {},
               "eligible_unexposed_people": census["eligible_unexposed_people"], "assay_id": ASSAY,
               "native_training_executed": False, "cross_system_comparison_permitted": False,
               "neural_to_behavioural_mapping_qualified": False, "result_direction_gate": False,
               "runtime": {"python": sys.version, "platform": platform.platform()}, "scientific_scope": cfg["scientific_scope"]}
    core.write_json(target/"prepare_receipt.json", receipt)
    return receipt


def qualify_split(split_dir, config_path, *, open_test):
    split, data = native.verify_split(split_dir, config_path, open_test=open_test)
    cfg = core.read_json(config_path)
    for role, rows in data.items():
        for row in rows:
            if row.get("study") != STUDY or row.get("source_id") != SOURCE or source_identity({"experiment": row.get("experiment"), "participant": row.get("source_participant"), "study": row.get("study")}) is None:
                raise core.StudyError("Frozen split contains a different source/task")
            flags = row.get("exposure_flags", {})
            if flags.get("is_psych101") is not False or flags.get("is_psych101_test") is not False:
                raise core.StudyError("Frozen split is not explicitly provider-unexposed")
            events = choice_responses(row["text"])
            if len(events) != row["original_response_count"] or any(e["response"].strip() not in cfg["response_alphabet"] for e in events):
                raise core.StudyError("Frozen original source choice role/count/alphabet changed")
    return split, data


def verify_prepared(directory, *, open_test=False):
    root = Path(directory).resolve()
    receipt = core.read_json(root/"prepare_receipt.json")
    if receipt.get("schema_version") != 1 or receipt.get("status") != READY:
        raise core.StudyError("Native training requires an actual qualified brightness participant split")
    for name, expected in receipt["outputs"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or core.sha256_file(path) != expected:
            raise core.StudyError("Frozen prepared output changed: "+name)
    for item in [receipt["config"], receipt["manifest"], receipt["generator"], receipt["discovery_receipt"]]+receipt["code"]+receipt["source_resources"]:
        if core.sha256_file(item["path"]) != item["sha256"]:
            raise core.StudyError("Prepared input/code/source changed: "+item["path"])
    cfg = load_config(receipt["config"]["path"])
    proposal_discovery.verify_outputs(receipt["discovery_dir"])
    selection = construct_selection(receipt["discovery_dir"])
    if selection != core.read_json(root/"construct_selection.json") or selection["status"] != "prespecified_construct_candidate_selected":
        raise core.StudyError("Discovery-led construct selection differs from actual qualified source")
    training_cfg = core.read_json(root/"training.config.json")
    census = core.read_json(root/"source_census.json")
    if training_cfg["response_alphabet"] != census["original_choice_alphabet"] or len(training_cfg["response_alphabet"]) != 2:
        raise core.StudyError("Source-observed original choice alphabet differs")
    expected_training = {**cfg["training"], "study": STUDY, "source_id": SOURCE,
        "response_alphabet": census["original_choice_alphabet"],
        "response_alphabet_evidence": "Complete pinned source census of original choice-role markers; no assumed L/R mapping",
        "assay_id": ASSAY, "response_role": "source You press choice marker only",
        "source_generator_sha256": receipt["generator"]["sha256"], "source_role_config_sha256": receipt["config"]["sha256"]}
    if training_cfg != expected_training:
        raise core.StudyError("Native training configuration differs from prepared role/budget contract")
    split, data = qualify_split(root/"split", root/"training.config.json", open_test=open_test)
    if split["independent_people"] != cfg["training"]["people"]:
        raise core.StudyError("Actual independent person split differs from declared budget")
    eligible = {(e["source_resource_id"], e["experiment"], e["participant"], e["text_sha256"]) for e in census["sessions"] if e.get("eligible")}
    for rows in data.values():
        for row in rows:
            if (row["source_resource_id"], row["experiment"], row["source_participant"], row["text_sha256"]) not in eligible:
                raise core.StudyError("Split text not in actual source-qualified eligible census")
    return receipt, training_cfg, split, data


def _verify_breadth_training(directory, prepared, cfg, model):
    record = native._verify_training(directory, Path(prepared)/"split", Path(prepared)/"training.config.json", model)
    if record.get("assay_id") != ASSAY or record.get("human_assay") != human_assay(cfg) or record.get("proposal_breadth_prepared") != _record(Path(prepared)/"prepare_receipt.json") or record.get("proposal_breadth_code_sha256") != core.sha256_file(__file__):
        raise core.StudyError("Native trained artifact lacks original brightness-role provenance")
    _, data = qualify_split(Path(prepared)/"split", Path(prepared)/"training.config.json", open_test=False)
    captured = record.get("source_role_training_events", {})
    if set(captured) != {"train", "validation"}:
        raise core.StudyError("Native original training/validation role observations missing")
    with role_context():
        for role in ("train", "validation"):
            observations = captured[role]["observations"]
            if not observations or native.human_observation_fingerprint(data[role], observations) != captured[role]["original_human_observations_sha256"] or captured[role]["native_token_events_sha256"] != record[role+"_events_sha256"]:
                raise core.StudyError("Actual native training source observation fingerprint differs")
            original = {(row["session_id"], event["index"]): ordinal for row in data[role] for ordinal, event in enumerate(choice_responses(row["text"]))}
            if any(original.get((event["session_id"], int(event["response_index"])), cfg["max_responses_per_session"]) >= cfg["max_responses_per_session"] for event in observations):
                raise core.StudyError("Native training selected-role ordinal budget exceeded")
            if role == "validation" and any(n > cfg["validation_events_per_person"] for n in Counter(event["person_id"] for event in observations).values()):
                raise core.StudyError("Native validation person event budget exceeded")
    order, updates = optimizer_observations(captured["train"]["observations"], cfg)
    if canonical_hash(order) != record["actual_training_order_sha256"] or canonical_hash(record.get("actual_optimizer_observation_order")) != canonical_hash(order) or updates != record["actual_optimizer_steps"]:
        raise core.StudyError("Native optimizer budget/order differs from actual original source observations")
    return record


def train(prepared, model, output):
    receipt, cfg, _, _ = verify_prepared(prepared, open_test=False)
    root = Path(prepared).resolve()
    with role_context(training_cfg=cfg, prepared_record=_record(root/"prepare_receipt.json")):
        native.train(receipt["source_root"], receipt["manifest"]["path"], root/"split", root/"training.config.json", model, output)
        result = _verify_breadth_training(output, root, cfg, model)
    return result


def _verify_matched_training(prepared, cfg, centaur_trained, base_trained):
    records = {name: _verify_breadth_training(directory, prepared, cfg, name)
               for name, directory in (("centaur", centaur_trained), ("base", base_trained))}
    keys = ("train_events_sha256", "validation_events_sha256", "actual_training_order_sha256", "optimizer_steps",
            "actual_optimizer_steps", "epochs", "trainable_parameters", "split_files_sha256", "assay_id")
    if any(records["centaur"][key] != records["base"][key] for key in keys):
        raise core.StudyError("Test remains sealed: native human adaptation rows/budgets are not matched")
    return records


def evaluate(prepared, model, centaur_trained, base_trained, output):
    # Rehash test files without reading outcomes until both real training receipts qualify.
    receipt, cfg, _, _ = verify_prepared(prepared, open_test=False)
    root = Path(prepared).resolve()
    _verify_matched_training(root, cfg, centaur_trained, base_trained)
    qualify_split(root/"split", root/"training.config.json", open_test=True)
    trained = centaur_trained if model == "centaur" else base_trained
    with role_context():
        return native.evaluate(receipt["source_root"], receipt["manifest"]["path"], root/"split", root/"training.config.json", trained, model, output,
                               centaur_trained=centaur_trained, base_trained=base_trained)


def verify_choice_budget(data, rows, cfg):
    """Enforce selected-role ordinal without rewriting source marker indices."""
    original = {(row["session_id"], event["index"]): (row["person_id"], event["response"].strip(), ordinal)
                for row in data["test"] for ordinal, event in enumerate(choice_responses(row["text"]))}
    seen, people = set(), Counter()
    for row in rows:
        key = row["session_id"], int(row["response_index"])
        value = original.get(key)
        if key in seen or value is None or value[:2] != (row["person_id"], row["response"]) or value[2] >= cfg["max_responses_per_session"]:
            raise core.StudyError("Heldout original choice role/identity/ordinal budget incompatible")
        seen.add(key)
        people[row["person_id"]] += 1
    if not rows or any(n > cfg["test_events_per_person"] for n in people.values()):
        raise core.StudyError("Heldout original person/event budget incompatible")


def compare(prepared, centaur_train, base_train, centaur_eval, base_eval, output):
    _, cfg, _, _ = verify_prepared(prepared, open_test=False)
    root = Path(prepared).resolve()
    _verify_matched_training(root, cfg, centaur_train, base_train)
    _, data = qualify_split(root/"split", root/"training.config.json", open_test=True)
    for directory in (centaur_eval, base_eval):
        verify_choice_budget(data, native._csv_rows(Path(directory)/"heldout_likelihoods.csv"), cfg)
    with role_context(comparison_config=root/"training.config.json"):
        result = native.compare(root/"split", root/"training.config.json", centaur_train, base_train, centaur_eval, base_eval, output)
        native.verify_comparison(output)
    record = {"schema_version": 1, "status": "proposal_breadth_native_comparison_verified",
              "prepared": _record(root/"prepare_receipt.json"), "comparison": _record(Path(output)/"comparison_receipt.json"),
              "code": _code_records(), "assay_id": ASSAY, "original_marker_indices_preserved": True,
              "heldout_choice_ordinal_budget_verified": True, "both_native_models_human_finetuned": True,
              "neural_to_behavioural_mapping_qualified": False, "result_direction_gate": False}
    core.write_json(Path(output)/"breadth_receipt.json", record)
    return {**result, "proposal_breadth_status": record["status"], "neural_to_behavioural_mapping_qualified": False}


def verify_comparison(directory):
    root = Path(directory).resolve()
    record = core.read_json(root/"breadth_receipt.json")
    if record.get("status") != "proposal_breadth_native_comparison_verified":
        raise core.StudyError("Actual role-qualified behavioural comparison missing")
    for item in [record["prepared"], record["comparison"]]+record["code"]:
        if core.sha256_file(item["path"]) != item["sha256"]:
            raise core.StudyError("Behavioural breadth comparison source/code changed")
    prepared = Path(record["prepared"]["path"]).parent
    _, cfg, _, _ = verify_prepared(prepared, open_test=False)
    receipt = core.read_json(root/"comparison_receipt.json")
    _verify_matched_training(prepared, cfg, receipt["input_paths"]["centaur_train"], receipt["input_paths"]["base_train"])
    _, data = qualify_split(prepared/"split", prepared/"training.config.json", open_test=True)
    for model in ("centaur", "base"):
        verify_choice_budget(data, native._csv_rows(Path(receipt["input_paths"][model+"_eval"])/"heldout_likelihoods.csv"), cfg)
    with role_context(comparison_config=prepared/"training.config.json"):
        return native.verify_comparison(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("discovery", "source-root", "manifest", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    for name in ("train", "evaluate", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--prepared", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name in ("train", "evaluate"):
            p.add_argument("--model", choices=("centaur", "base"), required=True)
        if name == "evaluate":
            p.add_argument("--centaur-trained", type=Path, required=True)
            p.add_argument("--base-trained", type=Path, required=True)
        if name == "compare":
            for role in ("centaur-train", "base-train", "centaur-eval", "base-eval"):
                p.add_argument("--"+role, type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.discovery, args.source_root, args.manifest, args.config, args.generator, args.output)
    elif args.command == "train":
        result = train(args.prepared, args.model, args.output)
    elif args.command == "evaluate":
        result = evaluate(args.prepared, args.model, args.centaur_trained, args.base_trained, args.output)
    elif args.command == "compare":
        result = compare(args.prepared, args.centaur_train, args.base_train, args.centaur_eval, args.base_eval, args.output)
    else:
        result = verify_comparison(args.comparison)
    print(json.dumps({k: result[k] for k in ("status", "reason", "assay_id", "eligible_unexposed_people", "native_training_executed", "proposal_breadth_status") if k in result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
