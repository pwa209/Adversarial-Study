"""Audit human adaptation before any comparison across neural and AI systems.

This does not compare unrelated EEG and language-model representations. An
adapted checkpoint is necessary; a matched human assay and protected evaluation
are also necessary. Missing evidence is emitted as an explicit development
record rather than replaced with an analogy between tasks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import sha256_file, write_json


def _receipt(directory, filename):
    path = Path(directory).resolve() / filename
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected an actual model execution receipt")
    return path, value


def audit(clinical, biological, behaviour, output, *, manifold=None, representations=None):
    """Retain actual training evidence and identify missing cross-assay evidence."""
    clinical_path, clinical_record = _receipt(clinical, "training_receipt.json")
    behavioural_path, behavioural_record = _receipt(behaviour, "comparison_receipt.json")
    biological_path, biological_record = _receipt(biological, "training_receipt.json")
    from native_tools.human_finetune import verify_adaptation as verify_clinical
    from .extension_finetune import verify_comparison as verify_behaviour
    from .human_generative_adaptation import verify_adaptation as verify_generative
    verify_clinical(Path(clinical).resolve())
    verify_behaviour(Path(behaviour).resolve())
    verify_generative(Path(biological).resolve())
    # Native trainers prove optimizer updates and changed checkpoint identity;
    # the generative fit proves train-only parameter selection and disjoint tests.
    neural_training = clinical_record.get("native_training_executed") is True
    behavioural_training = behavioural_record.get("all_models_human_finetuned") is True
    generative_fitting = biological_record.get("human_parameter_fitting_executed") is True
    prerequisites = {
        "clinical_human_task_copy": neural_training,
        "centaur_and_same_base_comparator": behavioural_training,
        "generative_human_parameter_fitting": generative_fitting,
    }
    geometry_inputs = {}
    if (manifold is None) != (representations is None):
        raise ValueError("The manifold and adapted-representation evidence must be supplied together")
    if manifold is not None:
        from .manifold_mechanisms import verify_bridge
        from .representation_bridge import verify_representation
        verify_bridge(Path(manifold).resolve())
        verify_representation(Path(representations).resolve())
        map_path, map_training = _receipt(manifold, "training_receipt.json")
        representation_path, _ = _receipt(representations, "receipt.json")
        if json.loads(representation_path.read_text(encoding="utf-8")).get("clinical_training_receipt_sha256") != sha256_file(clinical_path):
            raise ValueError("Manifold comparison used a different adapted clinical checkpoint lineage")
        prerequisites["manifold_human_parameter_map"] = map_training.get("human_parameter_fitting_executed") is True
        geometry_inputs = {
            "manifold": {"path": str(map_path), "sha256": sha256_file(map_path)},
            "adapted_representation": {"path": str(representation_path), "sha256": sha256_file(representation_path)},
        }
    assays = {"clinical": clinical_record.get("assay_id"), "behavioural": behavioural_record.get("assay_id"),
              "biological": biological_record.get("assay_id")}
    same_assay = all(assays.values()) and len(set(assays.values())) == 1
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "human_adaptation_audited_cross_assay_mapping_required" if all(prerequisites.values()) else "cross_system_human_training_prerequisites_missing",
        "all_systems_human_adapted": all(prerequisites.values()),
        "training_prerequisites": prerequisites,
        "input_receipts": {
            "clinical": {"path": str(clinical_path), "sha256": sha256_file(clinical_path)},
            "behavioural": {"path": str(behavioural_path), "sha256": sha256_file(behavioural_path)},
            "biological": {"path": str(biological_path), "sha256": sha256_file(biological_path)},
            **geometry_inputs,
        },
        "clinical_assay": clinical_record.get("assay_id"),
        "behavioural_assay": behavioural_record.get("assay_id"),
        "biological_assay": biological_record.get("assay_id"),
        "declared_assay_ids_agree": bool(same_assay),
        "assay_mapping_status": "identical_declared_assay_still_requires_mapping_qualification" if same_assay else "unmatched_human_assays",
        "assay_mismatch_reason": None if same_assay else "Current neural detection, behavioural lottery choice and temporal-proxy dynamics measure different human quantities",
        "clinical_released_instrument": "fixed discovery reference; adapted human-task copy is separate",
        "biological_adaptation_kind": "Human-constrained full state-trajectory effective observation map" if manifold is not None else "human-constrained parameter fitting; effective temporal observation map",
        "geometry_mechanism_bridge_verified": manifold is not None,
        "cross_modal_comparison_executed": False,
        "shared_human_assay_qualified": False,
        "required_next_evidence": [
            "Identical prospectively specified human construct, conditions and outcomes across systems",
            "Qualified mapping from neural and biological observations to that human outcome",
            "Separate training, validation and protected test people; all previous source exposure retained",
            "Adaptation completed for every system before test outcomes or cross-system contrasts are opened",
            "Frozen diagnostic input and matched test case identities with actual model predictions",
        ],
        "no_result_direction_or_significance_gate": True,
        "interpretation": "Training qualification supports a future comparison; it does not equate existing unrelated task assays",
    }
    write_json(output / "cross_system_readiness.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", required=True)
    parser.add_argument("--biological", required=True)
    parser.add_argument("--behaviour", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifold")
    parser.add_argument("--representations")
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.clinical, args.biological, args.behaviour, args.output,
        manifold=args.manifold, representations=args.representations), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
