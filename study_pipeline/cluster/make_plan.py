"""Source-pinned original clinical-model-guided mechanistic programme on Narval.

Numerical/source integrity gates protect executable analyses. A phenotype,
model failure, preferred mechanism, or significant effect never admits a stage.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

# Pristine published cache, verified afresh in every release. Source bytes never
# become execution state; builds, candidates, normalized EEG and results are new.
NATIVE_CACHE_TAG = "20261004T165416Z-b6b22b33a4ba"


def make_plan(source, tag, manifest, extension_manifest=None):
    project = "/home/pwa209/adversarial-study"
    scratch = "/scratch/pwa209/adversarial-study"
    run = project + "/runs/" + tag
    data_root = scratch + "/data/public-20261004"
    extension_root = scratch + "/data/extensions-20261004"
    normalized = scratch + "/normalized/" + tag
    native = scratch + "/native/" + NATIVE_CACHE_TAG
    results = scratch + "/results/" + tag
    qualification = run + "/qualification/native_detector.json"
    plan = {"schema_version": 2, "study_id": "adversarial-consciousness",
        "scientific_programme": "clinical_model_guided_human_mechanistic_adversarial_discovery",
        "source": str(source), "source_manifest_sha256": hashlib.sha256((Path(source) / "SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
        "account": "def-ptewarie_cpu", "gpu_account": "def-ptewarie_gpu",
        "project": project, "scratch": scratch, "run_root": run,
        "environment": scratch + "/environments/" + tag, "results": results,
        "controller_cpus": 1, "max_allocated_cpus": 16, "max_allocated_gpus": 2,
        "storage_allowance_bytes": 1_000_000_000_000, "entry_allowance": 150000,
        "cross_system_policy": {"human_finetuning_before_comparison": True,
            "released_clinical_checkpoint_role": "fixed discovery instrument; separate human-task adapted copy",
            "protected_human_assay_mapping_required": True},
        "result_direction_gates": False, "independent_evaluation_opened": False, "stages": []}
    groups = []

    def add(name, command, parents=(), cpus=1, memory="8G", walltime="02:00:00", role="runtime_qualification", gpus=0):
        item = {"id": name, "command": command, "depends_on": list(parents),
                "cpus": cpus, "memory": memory, "walltime": walltime, "evidence_role": role}
        if gpus: item.update(gpus=gpus, gpu_type="a100", account=plan["gpu_account"])
        plan["stages"].append(item)

    def transfer(name, *, metadata=None, manifest_name="sources.public.json", resources=(), sources=(), parents=("environment",), initial=False):
        group = {"id": name, "destination": native if metadata else (extension_root if manifest_name == "sources.extensions.json" else data_root)}
        if metadata:
            group.update(kind="figshare_v2", metadata=metadata, include_detector=True, include_config_bank=True)
        else:
            group.update(kind="public_manifest", manifest="{pipeline}/configs/" + manifest_name,
                         resource_ids=list(resources), source_ids=list(sources))
        if not initial: group["requires_stage"] = "qualify_detector"
        groups.append(group)
        add(name, ["{python}", "{pipeline}/cluster/transfer.py", "verify", "--plan", "{run}/execution_plan.json", "--group", name],
            parents, cpus=1, walltime="04:00:00", role="scheduled_source_cache_verification")
        plan["stages"][-1]["transfer_group"] = name

    runtime_cache = "{pipeline}/configs/dependency_runtime_cache.json"
    add("environment", ["bash", "{pipeline}/cluster/environment.sh", plan["environment"], "{source}", "{run}", runtime_cache], cpus=4, memory="16G", walltime="01:00:00")
    transfer("native_sources", metadata="{source}/source_records/toker_v2/figshare_v2_metadata.json", initial=True)
    pilot_resources = [r["resource_id"] for r in manifest["resources"] if r["source_id"] == "monash_wanderim" and r.get("participant_id") == "monash_wanderim_306"]
    if len(pilot_resources) != 4: raise ValueError("Expected pinned Monash pilot EEG triplet and behaviour")
    transfer("acquire_monash_pilot", resources=pilot_resources, initial=True)
    add("nftsim_runtime", ["{python}", "-m", "native_tools.nftsim_runtime", "--release-dir", native,
        "--output-dir", results + "/nftsim_runtime", "--compiler", "g++"], ["native_sources"], cpus=2)
    add("effective_recovery", ["{python}", "-m", "adversarial_study.effective_dynamics", "--synthetic-recovery", "--output-dir", results + "/effective_recovery"], ["environment"], cpus=2, role="assumptions_only_synthetic")

    def convert(name, source_name, output, parents, subject=None):
        command = ["{python}", "-m", "adversarial_study.source_adapters", source_name,
            "--source-root", data_root, "--manifest", "{pipeline}/configs/sources.public.json", "--output", output,
            "--release-dir", native, "--qualification", qualification]
        if subject: command += ["--subject", subject]
        if source_name == "tactile": command += ["--task", "adapt", "--capacity", "detection"]
        add(name, command, parents, cpus=4, memory="32G", walltime="08:00:00", role="empirical_development_source_qualification")

    def prepared(prefix, output, parents, scoring_parents=(), pilot=False):
        config = output + "/study.config.json"
        pointer = run + "/" + prefix + ".prepared.json"
        base = ["{python}", "{pipeline}/cluster/pilot.py"]
        args = ["--config", config, "--pointer", pointer]
        scores = results + "/" + prefix + ".scores.csv"
        add(prefix + "_prepare", base + ["prepare"] + args, parents, cpus=4, memory="32G", walltime="06:00:00", role="empirical_development")
        if pilot:
            add("qualify_detector", base + ["qualify"] + args, [prefix + "_prepare"], cpus=4, memory="16G", walltime="02:00:00")
            scoring_parents = ["qualify_detector"]
        add(prefix + "_score", base + ["batch"] + args + ["--scores", scores], [prefix + "_prepare", *scoring_parents], cpus=4, memory="16G", walltime="12:00:00", role="empirical_development")
        add(prefix + "_discovery", ["{python}", "-m", "adversarial_study.model_guided", "discover", "--pointer", pointer, "--scores", scores,
            "--qualification", qualification, "--output", results + "/" + prefix + "_discovery"], [prefix + "_score"], cpus=2, memory="16G", walltime="04:00:00", role="clinical_model_guided_empirical_discovery")
        add(prefix + "_trajectories", ["{python}", "-m", "adversarial_study.model_guided", "trajectories", "--pointer", pointer, "--scores", scores,
            "--qualification", qualification, "--output", results + "/" + prefix + "_trajectories"], [prefix + "_score"], cpus=4, memory="32G", walltime="06:00:00", role="multichannel_human_dynamics")
        add(prefix + "_dynamics", base + ["dynamics"] + args + ["--output-dir", results + "/" + prefix + "_dynamics"],
            [prefix + "_prepare"], cpus=4, memory="16G", walltime="06:00:00", role="empirical_effective_model_comparison")
        add(prefix + "_geometry", ["{python}", "-m", "adversarial_study.activation_geometry", "--pointer", pointer, "--scores", scores,
            "--qualification", qualification, "--output-dir", results + "/" + prefix + "_geometry"],
            [prefix + "_score"], cpus=2, memory="16G", walltime="06:00:00", role="clinical_model_representation_measurement")
        return {"pointer": pointer, "scores": scores, "trajectories": results + "/" + prefix + "_trajectories"}

    pilot = normalized + "/monash_pilot"
    convert("convert_monash_pilot", "monash", pilot, ["acquire_monash_pilot", "native_sources"], "306")
    prepared("pilot", pilot, ["convert_monash_pilot"], pilot=True)

    # Actual competing biological models and paired diagnostic inputs, executed
    # independently of whether natural model judgments favour any hypothesis.
    smoke = results + "/biological_smoke"
    bank = results + "/biological_bank"
    add("biological_smoke", ["{python}", "-m", "native_tools.mechanistic_native", "--release-dir", native,
        "--output-dir", smoke, "--workers", "4", "--timeout-s", "180", "--config-limit", "1", "--search-budget", "1"],
        ["nftsim_runtime"], cpus=4, memory="16G", walltime="01:00:00", role="native_biological_runtime_qualification")
    add("biological_bank", ["{python}", "-m", "native_tools.mechanistic_native", "--release-dir", native,
        "--output-dir", bank, "--workers", "4", "--timeout-s", "180", "--config-limit", "281", "--search-budget", "48"],
        ["biological_smoke"], cpus=4, memory="16G", walltime="12:00:00", role="competing_biological_generative_bank")
    bank_scores = results + "/simulation_scores.csv"
    add("simulation_score", ["{python}", "-m", "native_tools.detector_worker", "--mode", "batch", "--release-dir", native,
        "--jobs-csv", bank + "/detector_jobs.csv", "--output-csv", bank_scores,
        "--qualification-json", qualification, "--timeout", "42600"],
        ["biological_bank", "qualify_detector"], cpus=4, memory="16G", walltime="12:00:00", role="frozen_clinical_model_simulation_interrogation")

    neural = []
    transfer("acquire_monash_cohort", sources=["monash_wanderim"], parents=["qualify_detector"])
    convert("convert_monash_cohort", "monash", normalized + "/monash_cohort", ["acquire_monash_cohort"])
    neural.append(("cohort", prepared("cohort", normalized + "/monash_cohort", ["convert_monash_cohort"])))
    transfer("acquire_paris_archive", sources=["mindblanking_zenodo17649277"], parents=["qualify_detector"])
    convert("convert_paris_cohort", "paris", normalized + "/paris_cohort", ["acquire_paris_archive"])
    neural.append(("paris", prepared("paris", normalized + "/paris_cohort", ["convert_paris_cohort"])))
    tactile_resources = [r["resource_id"] for r in manifest["resources"] if r["source_id"] == "openneuro_ds001785" and
        ("task-adapt" in r["relative_path"] or r["relative_path"].split("/")[-1] in {"README", "dataset_description.json", "participants.tsv"})]
    transfer("acquire_tactile_cohort", resources=tactile_resources, parents=["qualify_detector"])
    convert("convert_tactile_cohort", "tactile", normalized + "/tactile_cohort", ["acquire_tactile_cohort"])
    neural.append(("tactile", prepared("tactile", normalized + "/tactile_cohort", ["convert_tactile_cohort"])))

    if extension_manifest is None:
        extension_path = Path(source) / "study_pipeline/configs/sources.extensions.json"
        if not extension_path.exists(): raise ValueError("Original programme requires pinned sleep/behaviour resources")
        extension_manifest = json.loads(extension_path.read_text(encoding="utf-8"))
    transfer("acquire_psych", manifest_name="sources.extensions.json", sources=["psych101", "psych201"], initial=True)
    add("behavioural_breadth", ["{python}", "-m", "adversarial_study.extension_sources", "psych", "--source-root", extension_root,
        "--manifest", "{pipeline}/configs/sources.extensions.json", "--output", results + "/behavioural_breadth"],
        ["acquire_psych"], cpus=4, memory="32G", walltime="06:00:00", role="behavioural_task_exposure_and_sequential_comparators")
    pilot_sleep_resources = ["figshare:59168237", "figshare:41090870"]
    if set(pilot_sleep_resources) - {r["resource_id"] for r in extension_manifest["resources"]}:
        raise ValueError("Pinned dense sleep pilot identifiers missing")
    transfer("acquire_dream_pilot", manifest_name="sources.extensions.json", sources=["dream_registry"], resources=pilot_sleep_resources, initial=True)
    def dream(name, dataset, output, parents, subject=None):
        cmd = ["{python}", "-m", "adversarial_study.extension_sources", "dream", "--dataset", dataset,
            "--source-root", extension_root, "--manifest", "{pipeline}/configs/sources.extensions.json", "--output", output,
            "--release-dir", native, "--qualification", qualification]
        if subject: cmd += ["--subject", subject]
        add(name, cmd, parents, cpus=4, memory="32G", walltime="06:00:00", role="report_linked_sleep_source_qualification")
    dream("convert_dream_pilot", "tononi", normalized + "/dream_pilot", ["acquire_dream_pilot", "native_sources"], "1")
    prepared("dream_pilot", normalized + "/dream_pilot", ["convert_dream_pilot"], ["qualify_detector"])
    transfer("acquire_dream_cohort", manifest_name="sources.extensions.json", sources=["dream_tononi_serial", "dream_registry"], parents=["qualify_detector"])
    dream("convert_dream_cohort", "tononi", normalized + "/dream_cohort", ["acquire_dream_cohort", "convert_dream_pilot"])
    neural.append(("dream_cohort", prepared("dream_cohort", normalized + "/dream_cohort", ["convert_dream_cohort"])))
    transfer("acquire_dream_sparse", manifest_name="sources.extensions.json", sources=["dream_oudiette_n1", "dream_registry"], parents=["qualify_detector"])
    dream("qualify_dream_sparse", "oudiette", normalized + "/dream_sparse", ["acquire_dream_sparse"])
    # Sparse scoring is deliberately absent until source physical units qualify.

    def biological_loop(prefix, observations, parents):
        fit = results + "/" + prefix + "_mechanistic_fit"
        command = ["{python}", "-m", "adversarial_study.model_guided", "fit"]
        for observation in observations: command += ["--observations", observation]
        command += ["--bank", bank + "/simulation_inventory.csv", "--bank-jobs", bank + "/detector_jobs.csv",
            "--bank-scores", bank_scores, "--qualification", qualification, "--bank-receipt", bank + "/bank_receipt.json", "--output", fit]
        add(prefix + "_mechanistic_fit", command, ["simulation_score", *parents], cpus=4, memory="32G", walltime="06:00:00", role="empirically_constrained_competing_generative_accounts")
        search = results + "/" + prefix + "_adversarial_search"
        add(prefix + "_adversarial_search", ["{python}", "-m", "adversarial_study.model_guided", "search", "--fit", fit, "--output", search],
            [prefix + "_mechanistic_fit"], cpus=2, memory="16G", walltime="04:00:00", role="constrained_adversarial_mechanism_discrimination")
        add(prefix + "_diagnostic_freeze", ["{python}", "-m", "adversarial_study.model_guided", "diagnostic", "--search", search,
            "--output", results + "/" + prefix + "_diagnostic"], [prefix + "_adversarial_search"], cpus=2, memory="16G", role="diagnostic_prediction_and_human_protocol_preparation")
    biological_loop("pilot", [results + "/pilot_trajectories"], ["pilot_trajectories", "pilot_discovery", "pilot_geometry"])
    for prefix, fields in neural:
        biological_loop(prefix, [fields["trajectories"]], [prefix + "_trajectories", prefix + "_discovery", prefix + "_geometry"])
    # Joint development explicitly separates sources and participant groups.
    biological_loop("joint", [fields["trajectories"] for _, fields in neural],
        [prefix + suffix for prefix, _ in neural for suffix in ("_trajectories", "_discovery", "_geometry")])
    add("behavioural_relevance", ["{python}", "-m", "adversarial_study.extension_relevance", "--human-fit", results + "/joint_mechanistic_fit",
        "--behaviour", results + "/behavioural_breadth", "--output", results + "/behavioural_relevance"],
        ["joint_mechanistic_fit", "behavioural_breadth"], cpus=2, memory="16G", role="emerged_human_capacity_behavioural_mapping_audit")

    # Native behavioural model uses its documented public 4bit base and adapter;
    # no substituted language model or synthetic activation stands in for it.
    transfer("acquire_centaur", manifest_name="sources.extensions.json", sources=["centaur_adapter", "centaur_base_4bit"], initial=True)
    add("centaur_resource_audit", ["{python}", "-m", "adversarial_study.extension_centaur", "audit", "--source-root", extension_root,
        "--manifest", "{pipeline}/configs/sources.extensions.json", "--output", "{run}/qualification/centaur_access.json"],
        ["acquire_centaur"], cpus=1, memory="8G", walltime="02:00:00", role="exact_native_behavioural_resource_admission")
    add("centaur_dependencies", ["bash", "{pipeline}/cluster/centaur_environment.sh", plan["environment"] + "-centaur", "{run}", runtime_cache],
        ["environment"], cpus=4, memory="32G", walltime="02:00:00", role="native_behavioural_model_dependency_qualification", gpus=2)
    # Human training precedes every between-model behavioural comparison. The
    # released Centaur/base inference pilot is not submitted as a comparison.
    ft_config = "{pipeline}/configs/extension_finetune.json"
    ft_split = results + "/behavioural_human_split"
    ft_base = ["{python}", "-m", "adversarial_study.extension_finetune"]
    ft_sources = ["--source-root", extension_root, "--manifest", "{pipeline}/configs/sources.extensions.json", "--config", ft_config]
    add("behavioural_human_split", ft_base + ["split"] + ft_sources + ["--output", ft_split],
        ["behavioural_breadth"], cpus=2, memory="16G", walltime="04:00:00", role="participant_disjoint_human_training_allocation")
    ct = results + "/centaur_human_finetune"
    bt = results + "/base_human_finetune"
    for model, destination, parents in (
        ("centaur", ct, ["behavioural_human_split", "centaur_resource_audit", "centaur_dependencies"]),
        ("base", bt, ["behavioural_human_split", "centaur_resource_audit", "centaur_dependencies", "centaur_human_finetune"]),
    ):
        add(model + "_human_finetune", ft_base + ["train"] + ft_sources + ["--split", ft_split, "--model", model, "--output", destination],
            parents, cpus=4, memory="128G", walltime="12:00:00", role="native_human_data_weight_finetuning", gpus=2)
    ce = results + "/centaur_human_evaluation"
    be = results + "/base_human_evaluation"
    for model, trained, destination in (("centaur", ct, ce), ("base", bt, be)):
        add(model + "_human_evaluation", ft_base + ["evaluate"] + ft_sources + ["--split", ft_split,
            "--trained", trained, "--model", model, "--centaur-trained", ct, "--base-trained", bt, "--output", destination],
            ["centaur_human_finetune", "base_human_finetune"], cpus=4, memory="128G", walltime="08:00:00",
            role="post_finetuning_protected_human_prediction", gpus=2)
        plan["stages"][-1]["required_human_training_stages"] = ["centaur_human_finetune", "base_human_finetune"]
    cmp = results + "/human_finetuned_behavioural_comparison"
    add("human_finetuned_behavioural_comparison", ft_base + ["compare", "--config", ft_config, "--split", ft_split, "--centaur-train", ct,
        "--base-train", bt, "--centaur-eval", ce, "--base-eval", be, "--output", cmp],
        ["centaur_human_evaluation", "base_human_evaluation"], cpus=2, memory="16G",
        role="human_finetuned_cross_system_comparison")
    plan["stages"][-1]["required_human_training_stages"] = ["centaur_human_finetune", "base_human_finetune"]
    clinical_export = results + "/clinical_human_training_export"
    clinical_train = results + "/clinical_human_finetune"
    tactile_fields = next(fields for prefix, fields in neural if prefix == "tactile")
    add("clinical_human_training_export", ["{python}", "-m", "native_tools.human_finetune", "export",
        "--pointer", tactile_fields["pointer"], "--scores", tactile_fields["scores"], "--qualification", qualification,
        "--assay", "detection", "--maximum-examples", "50000", "--output-dir", clinical_export],
        ["tactile_score"], cpus=4, memory="32G", walltime="06:00:00", role="participant_disjoint_human_neural_adaptation_export")
    add("clinical_human_finetune", ["{python}", "-m", "native_tools.human_finetune", "train", "--export-dir", clinical_export,
        "--release-dir", native, "--output-dir", clinical_train],
        ["clinical_human_training_export"], cpus=4, memory="32G", walltime="12:00:00", role="native_human_data_weight_finetuning")
    generative_train = results + "/generative_human_adaptation"
    add("generative_human_adaptation", ["{python}", "-m", "adversarial_study.human_generative_adaptation",
        "--fit", results + "/joint_mechanistic_fit", "--output", generative_train],
        ["joint_mechanistic_fit"], cpus=4, memory="32G", walltime="06:00:00", role="human_data_generative_parameter_adaptation")
    # The neural state-space and representation results enter the generative
    # fit and its diagnostic freeze. They are no longer descriptive sidecars.
    manifold_config = "{pipeline}/configs/neural_manifold.json"
    states = results + "/human_neural_state_space"
    state_command = ["{python}", "-m", "adversarial_study.neural_state_space", "--config", manifold_config,
        "--clinical-trained", clinical_train, "--output", states]
    for _, fields in neural:
        state_command += ["--trajectory", fields["trajectories"]]
    add("human_neural_state_space", state_command,
        ["clinical_human_finetune", *[prefix + "_trajectories" for prefix, _ in neural]],
        cpus=4, memory="64G", walltime="12:00:00", role="participant_validated_neural_state_space")
    adapted_geometry = results + "/adapted_clinical_geometry"
    add("adapted_clinical_geometry", ["{python}", "-m", "native_tools.adapted_geometry", "--states", states,
        "--trained", clinical_train, "--release-dir", native, "--output-dir", adapted_geometry],
        ["human_neural_state_space", "clinical_human_finetune"],
        cpus=4, memory="32G", walltime="12:00:00", role="post_finetuning_protected_human_prediction")
    plan["stages"][-1]["required_human_training_stages"] = ["clinical_human_finetune"]
    representation = results + "/human_representation_bridge"
    representation_command = ["{python}", "-m", "adversarial_study.representation_bridge", "--states", states,
        "--clinical-trained", clinical_train, "--adapted-geometry", adapted_geometry,
        "--config", manifold_config, "--output", representation]
    for prefix, _ in neural:
        representation_command += ["--geometry", results + "/" + prefix + "_geometry"]
    add("human_representation_bridge", representation_command,
        ["human_neural_state_space", "adapted_clinical_geometry", *[prefix + "_geometry" for prefix, _ in neural]],
        cpus=4, memory="32G", walltime="08:00:00", role="human_finetuned_cross_system_comparison")
    plan["stages"][-1]["required_human_training_stages"] = ["clinical_human_finetune"]
    bridge = results + "/manifold_mechanistic_bridge"
    add("manifold_mechanistic_bridge", ["{python}", "-m", "adversarial_study.manifold_mechanisms", "--states", states,
        "--fit", results + "/joint_mechanistic_fit", "--representations", representation,
        "--config", manifold_config, "--output", bridge],
        ["human_neural_state_space", "human_representation_bridge", "joint_mechanistic_fit"],
        cpus=4, memory="64G", walltime="12:00:00", role="human_data_manifold_generative_parameter_adaptation")
    manifold_freeze = results + "/manifold_diagnostic_freeze"
    add("manifold_diagnostic_freeze", ["{python}", "-m", "adversarial_study.manifold_diagnostic", "freeze",
        "--states", states, "--representations", representation, "--bridge", bridge,
        "--config", manifold_config, "--output", manifold_freeze],
        ["manifold_mechanistic_bridge"], cpus=2, memory="16G", walltime="02:00:00",
        role="geometry_mechanism_diagnostic_prediction_freeze")
    add("cross_system_training_audit", ["{python}", "-m", "adversarial_study.cross_system", "--clinical", clinical_train,
        "--biological", generative_train, "--behaviour", cmp, "--manifold", bridge,
        "--representations", representation, "--output", results + "/cross_system_training_audit"],
        ["clinical_human_finetune", "generative_human_adaptation", "human_finetuned_behavioural_comparison", "manifold_mechanistic_bridge"],
        cpus=2, memory="16G", role="post_human_adaptation_matched_assay_readiness")
    for name, reason, parents in (
        ("independent_neural_replication", "New untouched people/sources require secured access or collection after the frozen manifold-mechanism diagnostic and precision plan; target100 informative is provisional.", ["manifold_diagnostic_freeze"]),
        ("new_randomized_human_experiment", "Recruit and ethically collect the frozen manifold-mechanism diagnostic contrast in independent humans; target100-120 informative,2sessions,2sites is provisional. Use manifold_diagnostic evaluate with prospective registration and complete randomized denominators. Any cross-system test requires completed human fine-tuning for all models, an identical qualified human assay, protected test participants and frozen predictions. Human data are not collected.", ["manifold_diagnostic_freeze", "cross_system_training_audit"])):
        plan["stages"].append({"id": name, "external_input": reason, "depends_on": parents})
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("transfer_plan", Path(__file__).with_name("transfer.py"))
    module = module_from_spec(spec); spec.loader.exec_module(module)
    plan["transfer"] = {"groups": sorted(groups, key=lambda g: module.GROUP_ORDER.index(g["id"])),
        "continue_independent_groups": True, "max_rate_bytes_per_second": 8388608,
        "max_bytes": plan["storage_allowance_bytes"], "max_files": plan["entry_allowance"]}
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError("Plan exists; preserve it")
    manifest = json.loads((args.source / "study_pipeline/configs/sources.public.json").read_text())
    extensions = json.loads((args.source / "study_pipeline/configs/sources.extensions.json").read_text())
    plan = make_plan(args.source.resolve(), args.tag, manifest, extensions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
