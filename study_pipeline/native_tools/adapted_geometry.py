"""Actual hidden activations from the human-finetuned clinical network copy.

Original native-preprocessed human signals enter the saved adaptedNet, never a
Python surrogate. The released instrument stays fixed. Geometry remains a
within-person, same-channel, same-layer measurement; it is not a brain manifold.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from adversarial_study import core,native
from adversarial_study.activation_geometry import _load_mat,_vector,_cosine,matlab_layer_field_name
from .human_finetune import verify_adaptation,validate_split
from .detector_worker import matlab_string,run_matlab
from .nftsim_runtime import require_scheduled_job


def _own(root,name):
    root=Path(root).resolve();part=Path(name)
    if part.is_absolute() or ".." in part.parts:raise core.StudyError("Adapted activation artifact escapes its role")
    path=(root/part).resolve()
    if not path.is_relative_to(root) or not path.is_file():raise core.StudyError("Adapted activation artifact missing/outside its role")
    return path


def source_jobs(states,trained):
    """Recover the exact receipt-bound original inputs, without selecting outcomes."""
    from adversarial_study.neural_state_space import verify_state_space
    adaptation=verify_adaptation(Path(trained).resolve())
    state_root=Path(states).resolve();state=verify_state_space(state_root)
    rows=core.read_csv(state_root/"rows.csv")
    original={(r["participant_id"],r["source_id"],r["episode_id"]):r for r in rows if r.get("status")=="included"};wanted=set(original)
    contexts=state.get("native_contexts",[])
    if not isinstance(contexts,list) or not contexts:raise core.StudyError("State artifact lacks exact original native scoring contexts")
    split=validate_split(adaptation["split"]);jobs={};inputs=[]
    for context in contexts:
        for key in ["jobs","scores","qualification"]:
            path=Path(context[key+"_path"]).resolve()
            if core.sha256_file(path)!=context[key+"_sha256"]:raise core.StudyError("Original state native scoring context changed")
            inputs.append({"path":str(path),"sha256":core.sha256_file(path)})
        actual=native.import_scores(Path(context["scores_path"]),Path(context["jobs_path"]),Path(context["qualification_path"]))
        for row in actual:
            identity=row["participant_id"],row["source_id"],row["episode_id"]
            if identity not in wanted:continue
            copied={k:row[k] for k in ["job_id","episode_id","participant_id","source_id","channel_name","input_path","sfreq_hz","signal_sha256"]}
            copied["recording_id"]=original[identity]["recording_id"]
            copied["clinical_split"]=split.get(row["participant_id"],"source_held_out")
            if row["job_id"] in jobs and jobs[row["job_id"]]!=copied:raise core.StudyError("Conflicting original native job identity")
            jobs[row["job_id"]]=copied
    if not jobs or { (r["participant_id"],r["source_id"],r["episode_id"]) for r in jobs.values()}!=wanted:
        raise core.StudyError("Included human states and actual native input episodes differ")
    layers=adaptation.get("explicit_trained_backbone_layers",[])
    if not isinstance(layers,list) or not layers or len(set(layers))!=len(layers):raise core.StudyError("Adapted native backbone layer inventory missing")
    fields=[matlab_layer_field_name(layer) for layer in layers]
    if len(set(fields))!=len(fields):raise core.StudyError("Adapted hidden layer field mapping collides")
    return state,adaptation,sorted(jobs.values(),key=lambda r:r["job_id"]),inputs,layers


def _measure(directory,plan,execution,jobs,*,pair_sink=None):
    """Verify every returned native vector and retain every dependent pair."""
    root=Path(directory).resolve();outputs=execution.get("activation_outputs",[])
    if isinstance(outputs,dict):outputs=[outputs]
    if not isinstance(outputs,list):raise core.StudyError("Native adapted activation binding registry invalid")
    bindings={}
    for item in outputs:
        identity=item.get("job_id")
        if identity in bindings or not identity:raise core.StudyError("Duplicate/empty adapted activation identity")
        path=_own(root,item["path"])
        if core.sha256_file(path)!=item["sha256"]:raise core.StudyError("Actual adapted activation bytes changed")
        bindings[identity]=(path,item)
    predictions=core.read_csv(_own(root,"adapted_predictions.csv"))
    statuses={r["job_id"]:r for r in predictions}
    if len(statuses)!=len(predictions) or set(statuses)!={r["job_id"] for r in jobs}:raise core.StudyError("Native adapted job coverage differs")
    if set(bindings)-set(statuses):raise core.StudyError("Unregistered adapted activation outputs")
    layers=plan["layers"];mapping={name:matlab_layer_field_name(name) for name in layers};inventory=[];groups=defaultdict(list);pairs=[]
    for row in jobs:groups[row["participant_id"],row["source_id"],row["channel_name"]].append(row)
    for (person,source,channel),group in sorted(groups.items()):
        # Do not retain a whole cohort's large convolutional vectors in RAM.
        vectors={};local=[]
        for job in sorted(group,key=lambda r:r["job_id"]):
            entry={k:job[k] for k in ["job_id","episode_id","participant_id","source_id","channel_name","clinical_split"]}
            entry["layers"]={};status=statuses[job["job_id"]]
            if status["status"]!="ok":
                entry["status"]="native_activation_unavailable";entry["reason"]=status.get("error","")
                entry["layers"]={name:{"status":"native_activation_unavailable"} for name in layers}
            else:
                if job["job_id"] not in bindings:raise core.StudyError("Successful adapted job lacks actual activation binding")
                path,item=bindings[job["job_id"]];values=_load_mat(path)
                if set(values)!=set(mapping.values()):raise core.StudyError("Returned adapted hidden layer set differs")
                entry.update(status="native_adapted_layer_set_verified",activation_path=str(path),activation_sha256=item["sha256"])
                for name,field in mapping.items():
                    vector,descriptor=_vector(values[field]);entry["layers"][name]=descriptor
                    if vector is not None:vectors[job["job_id"],name]=vector
            inventory.append(entry);local.append(entry)
        by_id={r["job_id"]:r for r in local}
        for a,b in combinations(sorted(group,key=lambda r:(r["episode_id"],r["job_id"])),2):
            for layer in layers:
                ia,ib=by_id[a["job_id"]]["layers"][layer],by_id[b["job_id"]]["layers"][layer]
                value=None;status="activation_unavailable"
                if ia["status"]==ib["status"]=="available":
                    if ia["shape"]!=ib["shape"]:status="shape_mismatch"
                    else:value=_cosine(vectors[a["job_id"],layer],vectors[b["job_id"],layer]);status="computed" if value is not None else "undefined_zero_norm"
                row={"participant_id":person,"source_id":source,"channel_name":channel,"layer_name":layer,
                              "job_id_a":a["job_id"],"job_id_b":b["job_id"],"episode_id_a":a["episode_id"],"episode_id_b":b["episode_id"],
                              "clinical_split":a["clinical_split"],"cosine_distance":value,"metric_status":status}
                if pair_sink is None:pairs.append(row)
                else:pair_sink(row)
    return inventory,pairs


def export(states,trained,release_dir,output_dir,*,matlab="matlab",timeout_s=14400):
    require_scheduled_job();state,adaptation,jobs,inputs,layers=source_jobs(states,trained)
    release=native.verify_release(Path(release_dir).resolve())
    if Path(release_dir).resolve()!=Path(adaptation["original_release_dir"]).resolve():raise core.StudyError("Adapted copy original release differs")
    if len(jobs)>50000 or not 1<=timeout_s<=43200:raise core.StudyError("Adapted export exceeds fixed native budget")
    root=Path(output_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    core.write_csv(root/"adapted_jobs.csv",jobs)
    plan={"status":"verified_adapted_native_export_plan","states_path":str(Path(states).resolve()),"trained_path":str(Path(trained).resolve()),
          "states_receipt_sha256":core.sha256_file(Path(states)/"receipt.json"),"clinical_training_receipt_sha256":core.sha256_file(Path(trained)/"training_receipt.json"),
          "adapted_checkpoint":adaptation["trained_checkpoint"],"original_release":release,"jobs_sha256":core.sha256_file(root/"adapted_jobs.csv"),
          "layers":layers,"original_native_inputs":inputs,"clinical_split":adaptation["split"],"source_code_sha256":core.sha256_file(__file__),
          "helper_sha256":core.sha256_file(Path(__file__).with_suffix(".m"))}
    core.write_json(root/"export_plan.json",plan)
    directory=Path(__file__).resolve().parent
    command="addpath("+matlab_string(directory)+"); addpath("+matlab_string(directory.parent/"matlab")+"); adapted_geometry("+",".join(matlab_string(p) for p in [root,Path(release_dir).resolve()])+ ");"
    worker=run_matlab(command,root/"matlab_activations.log",matlab=matlab,timeout=timeout_s);core.write_json(root/"worker_receipt.json",worker)
    if worker["returncode"]!=0:raise core.StudyError("Actual adapted native activation export failed; inspect retained log")
    execution=core.read_json(root/"native_receipt.json")
    _verify_execution(root,plan,execution)
    import csv
    metric_counts=Counter();count=0
    fields=sorted(["participant_id","source_id","channel_name","layer_name","job_id_a","job_id_b","episode_id_a","episode_id_b","clinical_split","cosine_distance","metric_status"])
    with (root/"layer_distances.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        def emit(row):
            nonlocal count
            writer.writerow({k:"" if v is None else v for k,v in row.items()});count+=1;metric_counts[row["metric_status"]]+=1
        inventory,_=_measure(root,plan,execution,jobs,pair_sink=emit)
    core.write_json(root/"activation_inventory.json",inventory)
    receipt={"status":"actual_human_adapted_activation_geometry_completed","states_path":plan["states_path"],"trained_path":plan["trained_path"],
             "states_receipt_sha256":plan["states_receipt_sha256"],"clinical_training_receipt_sha256":plan["clinical_training_receipt_sha256"],
             "adapted_checkpoint":plan["adapted_checkpoint"],"export_plan_sha256":core.sha256_file(root/"export_plan.json"),
             "native_receipt_sha256":core.sha256_file(root/"native_receipt.json"),"distance_table_sha256":core.sha256_file(root/"layer_distances.csv"),
             "activation_inventory_sha256":core.sha256_file(root/"activation_inventory.json"),"layers":layers,"clinical_split":adaptation["split"],
             "jobs":len(jobs),"pairs":count,"participants":len({r["participant_id"] for r in jobs}),"metric_counts":dict(metric_counts),
             "all_original_same_person_same_channel_pairs_retained":True,"actual_adapted_native_activations":True,
             "interpretation":"Actual post-human-finetuning detector representations; dependent within-person pairs, not human manifold coordinates or independent participants"}
    core.write_json(root/"receipt.json",receipt);verify_adapted_geometry(root);native.verify_release(Path(release_dir));return receipt


def _verify_execution(root,plan,execution):
    if execution.get("status")!="adapted_native_activation_export_completed" or execution.get("forward_correspondence_verified") is not True or execution.get("activation_repeats_verified") is not True:
        raise core.StudyError("Actual adapted forward correspondence/activation repeats not qualified")
    expected={"adapted_checkpoint_sha256":plan["adapted_checkpoint"]["sha256"],"jobs_sha256":plan["jobs_sha256"],
              "export_plan_sha256":core.sha256_file(root/"export_plan.json"),"helper_sha256":plan["helper_sha256"],
              "clinical_training_receipt_sha256":plan["clinical_training_receipt_sha256"],"states_receipt_sha256":plan["states_receipt_sha256"]}
    if any(execution.get(k)!=v for k,v in expected.items()):raise core.StudyError("Adapted native export checkpoint/input/helper binding differs")
    if execution.get("predictions_sha256")!=core.sha256_file(_own(root,"adapted_predictions.csv")):raise core.StudyError("Actual adapted predictions changed")


def verify_adapted_geometry(directory):
    root=Path(directory).resolve();receipt=core.read_json(_own(root,"receipt.json"))
    if receipt.get("status")!="actual_human_adapted_activation_geometry_completed":raise core.StudyError("Actual human-adapted activation geometry absent")
    state,adaptation,jobs,inputs,layers=source_jobs(receipt["states_path"],receipt["trained_path"])
    for key,path in [("states_receipt_sha256",Path(receipt["states_path"])/"receipt.json"),("clinical_training_receipt_sha256",Path(receipt["trained_path"])/"training_receipt.json")]:
        if core.sha256_file(path)!=receipt[key]:raise core.StudyError("Adapted activation parent receipt changed")
    for key,name in [("export_plan_sha256","export_plan.json"),("native_receipt_sha256","native_receipt.json"),("distance_table_sha256","layer_distances.csv"),("activation_inventory_sha256","activation_inventory.json")]:
        if core.sha256_file(_own(root,name))!=receipt[key]:raise core.StudyError("Adapted geometry artifact changed")
    plan=core.read_json(root/"export_plan.json");execution=core.read_json(root/"native_receipt.json")
    if plan["adapted_checkpoint"]!=adaptation["trained_checkpoint"] or receipt["adapted_checkpoint"]!=adaptation["trained_checkpoint"] or plan["layers"]!=layers or receipt["layers"]!=layers or receipt["clinical_split"]!=adaptation["split"]:
        raise core.StudyError("Adapted geometry uses a different actual checkpoint/layer/clinical split")
    written=core.read_csv(_own(root,"adapted_jobs.csv"));expected=[{k:str(v) for k,v in row.items()} for row in jobs]
    if written!=expected or core.sha256_file(root/"adapted_jobs.csv")!=plan["jobs_sha256"]:raise core.StudyError("Actual adapted input jobs changed")
    if plan["source_code_sha256"]!=core.sha256_file(__file__) or plan["helper_sha256"]!=core.sha256_file(Path(__file__).with_suffix(".m")):
        raise core.StudyError("Adapted geometry implementation differs from the frozen exporter")
    _verify_execution(root,plan,execution)
    import csv
    with (root/"layer_distances.csv").open(encoding="utf-8",newline="") as stream:
        reader=iter(csv.DictReader(stream))
        def verify_pair(row):
            actual=next(reader,None);expected={k:"" if v is None else str(v) for k,v in row.items()}
            if actual is None or {k:v.strip() for k,v in actual.items()}!=expected:
                raise core.StudyError("Rehashed adapted pair geometry differs from actual native vectors")
        inventory,_=_measure(root,plan,execution,jobs,pair_sink=verify_pair)
        if next(reader,None) is not None:raise core.StudyError("Rehashed adapted pair geometry adds uncomputed native pairs")
    if inventory!=json.loads((root/"activation_inventory.json").read_text(encoding="utf-8")):raise core.StudyError("Adapted vector inventory differs from actual native outputs")
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["states","trained","release-dir","output-dir"]:parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--matlab",default="matlab");parser.add_argument("--timeout-s",type=int,default=14400)
    args=vars(parser.parse_args(argv));print(json.dumps(export(**args),indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
