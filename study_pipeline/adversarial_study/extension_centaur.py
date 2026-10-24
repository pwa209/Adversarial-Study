"""Native source-pinned Centaur branch: offline likelihood and representations.

This operates on real behavioural transcripts and real released model weights.
The base Llama comparator uses the identical model with the released adapter
disabled. A zero-block ablation diagnoses model computation, not a brain or a
human consciousness mechanism. Resource/cache and dependency audits stay
distinct from executed GPU measurements.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np

from .acquisition import canonical_hash, verify_acquired
from .core import sha256_file, write_csv, write_json
from .extension_sources import marked_responses

EXPECTED_MODELS = {
    "centaur_base_4bit": ("unsloth/Meta-Llama-3.1-70B-bnb-4bit", "a009b8db2439814febe725486a5ed388f12a8744"),
    "centaur_adapter": ("marcelbinz/Llama-3.1-Centaur-70B-adapter", "159600db8be99dc183c289923148dfd96cbd8e07"),
}
REQUIRED_FILES = {
    "centaur_base_4bit": {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"} |
                         {f"model-{i:05d}-of-00006.safetensors" for i in range(1,7)},
    "centaur_adapter": {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json"},
}


def _json(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resource_audit(source_root,manifest_path):
    manifest=_json(manifest_path); root=Path(source_root).resolve()
    resources=[r for r in manifest["resources"] if r["source_id"] in ["centaur_base_4bit","centaur_adapter"]]
    if not resources: raise ValueError("source-pinned Centaur model manifest missing")
    sources={s["source_id"]:s for s in manifest["sources"]}
    for source_id,(repository,revision) in EXPECTED_MODELS.items():
        source=sources.get(source_id,{})
        if (source.get("repository"),source.get("revision")) != (repository,revision):
            raise ValueError("exact documented Centaur base/adapter source identity missing or changed")
        filenames={Path(r["relative_path"]).name for r in resources if r["source_id"]==source_id}
        if REQUIRED_FILES[source_id]-filenames:
            raise ValueError("model manifest omits required original architecture/tokenizer/weight files")
    receipts=[]; failures=[]
    for resource in resources:
        try: receipts.append(verify_acquired(root,resource))
        except (ValueError,OSError) as error:
            failures.append({"resource_id":resource["resource_id"],"status":"missing_or_unqualified","reason":str(error)})
    provider_gated={s["source_id"]:s.get("provider_gated") for s in manifest["sources"] if s["source_id"] in ["centaur_base_4bit","centaur_adapter"]}
    return {"status":"READY_CACHED_MODELS" if not failures else "ACCESS_REQUIRED" if any(v is True for v in provider_gated.values()) else "SOURCE_TRANSFER_PENDING",
            "manifest_sha256":canonical_hash(manifest),"resource_count":len(resources),"expected_bytes":sum(r["expected_bytes"] for r in resources),
            "resource_receipts":receipts,"failures":failures,"provider_gated":provider_gated,
            "gpu_request":{"gpu_count":2,"gpu_type":"A10040GB","aggregate_memory_gib_min":76,"cpu_cores":4},
            "license":"sourceLlama3.1communitylicense; retainedoriginalcards",
            "native_model_executed":False,"synthetic":False}


def dependency_audit():
    versions={}; missing=[]
    for package in ["torch","transformers","peft","accelerate","bitsandbytes","safetensors","sentencepiece"]:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: missing.append(package)
    return {"python":platform.python_version(),"executable":sys.executable,"versions":versions,"missing":missing,
            "status":"packages_available_runtime_unqualified" if not missing else "GPU_RUNTIME_DEPENDENCIES_REQUIRED"}


def audit(source_root,manifest_path,output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("complete model-cache hashing is scheduled CPU work")
    result=resource_audit(source_root,manifest_path)
    result["dependency_audit"]=dependency_audit(); result["slurm_job_id"]=os.environ["SLURM_JOB_ID"]
    write_json(output,result); return result


@contextmanager
def zero_block(model,index):
    """Explicit entire decoder-block output intervention, preserving structure."""
    layers=model.get_base_model().model.layers
    if index < 0 or index >= len(layers): raise ValueError("ablation block outside real native architecture")
    def zero(_module,_inputs,output):
        if isinstance(output,tuple): return (output[0]*0,)+output[1:]
        return output*0
    hook=layers[index].register_forward_hook(zero)
    try: yield
    finally: hook.remove()


def _encode_candidates(tokenizer,prefix,choices,max_context):
    prefix_ids=tokenizer.encode(prefix,add_special_tokens=True)
    if len(prefix_ids)>max_context: raise ValueError("sourceprefixexceedsfixedcontextbound; no silenttruncation")
    suffixes={}
    for choice in choices:
        whole=tokenizer.encode(prefix+choice,add_special_tokens=True)
        if whole[:len(prefix_ids)]!=prefix_ids:
            raise ValueError("native tokenizer boundary changes responseprefix; sourceevent needs qualification")
        suffix=whole[len(prefix_ids):]
        if not suffix or len(whole)>max_context: raise ValueError("response exceeds native contextbound")
        suffixes[choice]=suffix
    return prefix_ids,suffixes


def _candidate_likelihoods(model,torch,prefix_ids,suffixes):
    device=model.get_input_embeddings().weight.device
    values={}
    for choice,suffix in suffixes.items():
        ids=torch.tensor([prefix_ids+suffix],dtype=torch.long,device=device)
        with torch.inference_mode(): output=model(input_ids=ids,use_cache=False,return_dict=True)
        log_prob=torch.log_softmax(output.logits[0].float(),dim=-1)
        score=0.0
        for i,token in enumerate(suffix): score+=float(log_prob[len(prefix_ids)+i-1,token].cpu())
        if not math.isfinite(score): raise ValueError("native nonfinite response loglikelihood")
        values[choice]=score
        del output,log_prob,ids
    maximum=max(values.values()); normalizer=maximum+math.log(sum(math.exp(v-maximum) for v in values.values()))
    return {choice:{"raw_log_likelihood":value,"conditional_choice_probability":math.exp(value-normalizer),
                   "conditional_choice_log_likelihood":value-normalizer} for choice,value in values.items()}


def _representations(model,torch,prefix_ids,selected_layers):
    layers=model.get_base_model().model.layers; captures={}; hooks=[]
    for index in selected_layers:
        if index < 0 or index >= len(layers): raise ValueError("selected native block outside real architecture")
        def capture(_module,_inputs,output,index=index):
            value=output[0] if isinstance(output,tuple) else output
            captures[str(index)]=value[0,-1,:].detach().float().cpu().numpy()
        hooks.append(layers[index].register_forward_hook(capture))
    try:
        device=model.get_input_embeddings().weight.device
        ids=torch.tensor([prefix_ids],dtype=torch.long,device=device)
        with torch.inference_mode(): result=model(input_ids=ids,use_cache=False,return_dict=True)
        del result,ids
    finally:
        for hook in hooks: hook.remove()
    if set(captures)!={str(v) for v in selected_layers} or any(not np.isfinite(v).all() for v in captures.values()):
        raise ValueError("native selectedlayer activation missing/nonfinite")
    return captures


def run_native(source_root,manifest_path,sessions_path,config_path,output_dir):
    if not os.environ.get("SLURM_JOB_ID"): raise ValueError("Centaurnativeinference requires allocated SlurmGPUworker")
    output=Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_DATASETS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    pins=resource_audit(source_root,manifest_path)
    write_json(output/"model_source_audit.json",pins)
    if pins["status"]!="READY_CACHED_MODELS": raise ValueError("originalCentaurresources unavailable; inspect pinnedcacheaudit")
    deps=dependency_audit(); write_json(output/"dependency_audit.json",deps)
    if deps["missing"]: raise ValueError("nativeGPUdependenciesmissing; no simulatedmodelreplacement")
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:
        raise ValueError("native branch requires exactlytwo visible allocated GPUs")
    devices=[{"index":i,"name":torch.cuda.get_device_name(i),"memory_bytes":torch.cuda.get_device_properties(i).total_memory} for i in range(2)]
    if sum(d["memory_bytes"] for d in devices) < 76*1024**3:
        raise ValueError("allocatedGPUmemory below officialnative80GBqualificationtarget")
    config=_json(config_path); settings=config["behaviour"]; root=Path(source_root).resolve()
    base_path=root/"centaur_base_4bit"; adapter_path=root/"centaur_adapter"
    # Same documented4bit base, not a substitute; prequantization config is original.
    model=AutoModelForCausalLM.from_pretrained(base_path,local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.bfloat16,device_map="balanced",max_memory={0:"36GiB",1:"36GiB","cpu":"0GiB"},low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,adapter_path,local_files_only=True,is_trainable=False)
    model.eval(); tokenizer=AutoTokenizer.from_pretrained(adapter_path,local_files_only=True,trust_remote_code=False,use_fast=True)
    sessions=[]
    with Path(sessions_path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip(): sessions.append(json.loads(line))
    if not sessions or len(sessions)>settings["max_centaur_sessions"]:
        raise ValueError("native source-orderpilot sessioncount differsfromfrozenbound")
    choices=settings["response_alphabet"]; rows=[]; excluded=[]; activations=[]; baseline_repeat=None
    for session in sessions:
        text=session["text"]
        if hashlib.sha256(text.encode()).hexdigest()!=session["text_sha256"] or session.get("synthetic") is not False:
            raise ValueError("actualbehavioural source text pin mismatch")
        events=marked_responses(text)[:settings["max_centaur_responses"]]
        for event in events:
            identity=session["session_id"]+f"_{event['index']:04d}"
            try: prefix,suffixes=_encode_candidates(tokenizer,event["prefix"],choices,settings["max_context_tokens"])
            except ValueError as error:
                excluded.append({"session_id":session["session_id"],"response_index":event["index"],"reason":str(error)});continue
            response=event["response"].strip()
            if response not in choices: raise ValueError("nativepilotresponseoutsideoriginaltaskalphabet")
            results={"centaur_adapter":_candidate_likelihoods(model,torch,prefix,suffixes)}
            with model.disable_adapter():
                results["llama_same_base"]=_candidate_likelihoods(model,torch,prefix,suffixes)
                base_vectors=_representations(model,torch,prefix,settings["selected_native_layers"])
            with zero_block(model,settings["ablation_layer"]):
                results["centaur_zero_block"]=_candidate_likelihoods(model,torch,prefix,suffixes)
            native_vectors=_representations(model,torch,prefix,settings["selected_native_layers"])
            if baseline_repeat is None:
                repeated=_candidate_likelihoods(model,torch,prefix,suffixes)
                differences=[abs(results["centaur_adapter"][choice]["raw_log_likelihood"]-repeated[choice]["raw_log_likelihood"]) for choice in choices]
                baseline_repeat={"max_absolute_loglikelihood_difference":max(differences),"tolerance":1e-3,
                                 "source_event":identity,"within_tolerance":max(differences)<=1e-3}
                if not baseline_repeat["within_tolerance"]: raise ValueError("native repeatedlikelihood forwardpassfailed")
            for model_name,result in results.items():
                value=result[response]
                rows.append({"session_id":session["session_id"],"source_id":session["source_id"],"study":session["study"],
                             "experiment":session["experiment"],"source_participant":session["source_participant"],
                             "response_index":event["index"],"response":response,"model":model_name,
                             "context_tokens":len(prefix),"response_tokens":len(suffixes[response]),**value,
                             "conditional_alphabet":"L,R", "role":"development",**session["exposure_flags"]})
            path=output/(identity+".activations.npz")
            np.savez(path,**{f"centaur_block_{k}":v for k,v in native_vectors.items()},**{f"base_block_{k}":v for k,v in base_vectors.items()})
            activations.append({"session_id":session["session_id"],"response_index":event["index"],"path":path.name,
                                "sha256":sha256_file(path),"position":"lastgenuinecontexttokenbeforehumanresponse",
                                "selected_blocks":settings["selected_native_layers"],"architecture":"actual80decoderblocksLlama3.1-70B",
                                "model_source_manifest_sha256":pins["manifest_sha256"]})
    names=sorted({k for row in rows for k in row}) or ["session_id","response_index","model","conditional_choice_log_likelihood"]
    write_csv(output/"native_likelihoods.csv",rows,names)
    write_csv(output/"excluded_events.csv",excluded,["session_id","response_index","reason"])
    report={"status":"native_centaur_executed" if rows else "BLOCKED_NO_QUALIFIED_EVENTS","slurm_job_id":os.environ["SLURM_JOB_ID"],
            "devices":devices,"source_manifest_sha256":pins["manifest_sha256"],"sessions_sha256":sha256_file(sessions_path),
            "configuration_sha256":sha256_file(config_path),"native_rows":len(rows),"native_events":len(activations),
            "excluded_events":len(excluded),"activation_receipts":activations,"repeat_forward_pass":baseline_repeat,
            "dependency_versions":deps["versions"],"synthetic":False,"role":"development",
            "comparisons":"actualCentauradapter, exactsamebaseadapterdisabled, fixedblock60zerointervention",
            "interpretation":"conditionalactualL/Rlikelihoodsandcomputationalrepresentations; nohumanEEG/consciousnessclaims; fixedpilotnotdiscoveryselectedmechanism",
            "behavioural_generalisation_established":False,
            "selection":"actualsource-order taskpilot; noparticipant/choiceadmissionbyfindingdirection",
            "result_table_sha256":sha256_file(output/"native_likelihoods.csv")}
    write_json(output/"native_receipt.json",report)
    if not rows: raise ValueError("no qualifiedactualnativeevents; exclusions preserved")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    audit_parser=sub.add_parser("audit"); run_parser=sub.add_parser("run")
    for p in [audit_parser,run_parser]:
        p.add_argument("--source-root",required=True);p.add_argument("--manifest",required=True);p.add_argument("--output",required=True)
    run_parser.add_argument("--sessions",required=True);run_parser.add_argument("--config",required=True)
    args=parser.parse_args(argv)
    if args.command=="audit": result=audit(args.source_root,args.manifest,args.output)
    else: result=run_native(args.source_root,args.manifest,args.sessions,args.config,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ["resource_receipts","failures","activation_receipts"]},indent=2));return 0


if __name__=="__main__": raise SystemExit(main())
