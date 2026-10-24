"""Matched human fine-tuning before behavioural cross-system comparisons.

Both original Centaur and its exact base comparator learn from the same human
training participants. Test participants never supply optimizer updates or
model selection. This is bounded development training, not a claim of adequate
foundation-model retraining or an independently unexposed clinical cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random

import numpy as np

from .acquisition import canonical_hash
from .core import sha256_file,write_csv,write_json
from .extension_sources import _behaviour_rows,marked_responses
from .extension_centaur import resource_audit,dependency_audit,_encode_candidates,_candidate_likelihoods,_representations


def _json(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:return [json.loads(line) for line in stream if line.strip()]


def _csv_rows(path):
    import csv
    with Path(path).open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))


def _owned_file(directory,relative,description):
    """Only read the named role's own artifact, including after symlink resolution."""
    root=Path(directory).resolve();name=Path(relative)
    if name.is_absolute() or ".." in name.parts or not name.parts:
        raise ValueError(description+" escapes its artifact role")
    path=(root/name).resolve()
    if not path.is_relative_to(root) or path==root or not path.is_file():
        raise ValueError(description+" escapes its artifact role or is missing")
    return path


def human_observation_fingerprint(data,events):
    """Reconstruct source observation identity without a tokenizer or model import."""
    original={}
    for row in data:
        for event in marked_responses(row["text"]):
            key=(row["session_id"],event["index"])
            if key in original:raise ValueError("duplicate original human observation identity")
            original[key]={"session_id":row["session_id"],"person_id":row["person_id"],"response_index":event["index"],
                           "response":event["response"].strip(),"text_sha256":row["text_sha256"],
                           "prefix_sha256":hashlib.sha256(event["prefix"].encode()).hexdigest(),
                           **{k:row.get(k) for k in ["source_id","source_resource_id","resource_sha256","exposure_flags"]}}
    selected=[];seen=set()
    for event in events:
        key=(event["session_id"],int(event["response_index"]))
        if key in seen:raise ValueError("duplicate heldout human observation")
        seen.add(key);row=original.get(key)
        if row is None or (row["person_id"],row["response"])!=(event["person_id"],event["response"]):
            raise ValueError("heldout row differs from original sealed human observation")
        selected.append(row)
    if not selected:raise ValueError("empty heldout observations cannot certify a comparison")
    return canonical_hash(selected)


def assign_people(people,counts,seed):
    if any(not isinstance(n,int) or n<1 for n in counts.values()):raise ValueError("positive train/validation/test people counts required")
    unique=set(people)
    ordered=sorted(unique,key=lambda p:hashlib.sha256((str(seed)+"\0"+p).encode()).hexdigest())
    if len(ordered)<sum(counts.values()):raise ValueError("insufficient genuine eligible independent participant identities")
    assignments={};offset=0
    for role in ["train","validation","test"]:
        for person in ordered[offset:offset+counts[role]]:assignments[person]=role
        offset+=counts[role]
    return assignments


def make_split(source_root,manifest_path,config_path,output_dir):
    root=Path(source_root).resolve();manifest=_json(manifest_path);cfg=_json(config_path)
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    resources=[r for r in manifest["resources"] if r["source_id"]==cfg["source_id"]]
    if not resources:raise ValueError("actual configured human behavioural sources absent")
    candidates=[];exclusions=[];receipts={};people=set();required=sum(cfg["people"].values())
    # Complete source scan ensures later sessions of an admitted person stay together.
    for resource,receipt,row in _behaviour_rows(resources,root):
        receipts[resource["resource_id"]]=receipt
        study=str(row.get("study",row.get("experiment",""))).split("/")[0]
        experiment=str(row.get("experiment",study));participant=str(row.get("participant",""))
        if cfg["study"].lower() not in (study+" "+experiment).lower():continue
        flags={k:row.get(k) for k in ["is_psych101","is_psych101_test","in_eval"]}
        person=study+"\0"+participant
        try:
            if not participant:raise ValueError("original participant identity missing")
            if cfg["require_psych101_unexposed"] and (flags["is_psych101"] is not False or flags["is_psych101_test"] is not False):
                raise ValueError("provider prior Psych101 training/test exposure absent or affirmative")
            events=marked_responses(row["text"])
            if not events or any(e["response"].strip() not in cfg["response_alphabet"] for e in events):
                raise ValueError("original response alphabet/event schema incompatible")
            if person not in people and len(people)>=required:continue
            people.add(person)
            identity=hashlib.sha256((study+"\0"+experiment+"\0"+participant).encode()).hexdigest()
            candidates.append({"session_id":identity,"person_id":hashlib.sha256(person.encode()).hexdigest(),
                               "person_group_key":person,"source_id":resource["source_id"],"study":study,"experiment":experiment,
                               "source_participant":participant,"text":row["text"],"text_sha256":hashlib.sha256(row["text"].encode()).hexdigest(),
                               "exposure_flags":flags,"source_resource_id":resource["resource_id"],"resource_sha256":receipt["sha256"],
                               "synthetic":False,"original_response_count":len(events)})
        except (ValueError,KeyError) as error:
            exclusions.append({"study":study,"experiment":experiment,"participant":participant,"reason":str(error)})
    try:assignments=assign_people(people,cfg["people"],cfg["seed"])
    except ValueError as error:
        write_json(output/"split_receipt.json",{"status":"BLOCKED_INSUFFICIENT_ELIGIBLE_PEOPLE","reason":str(error),"eligible_people":len(people),"exclusions":exclusions,
                   "manifest_sha256":canonical_hash(manifest),"human_training_executed":False});raise
    session_ids=[r["session_id"] for r in candidates]
    if len(set(session_ids))!=len(session_ids):raise ValueError("duplicate original human session identity")
    counts=Counter();files={};person_roles={}
    for role in ["train","validation","test"]:
        selected=sorted((r for r in candidates if assignments[r["person_group_key"]]==role),key=lambda r:r["session_id"])
        path=output/(role+".jsonl")
        with path.open("w",encoding="utf-8",newline="\n") as stream:
            for row in selected:
                exported={key:value for key,value in row.items() if key!="person_group_key"};exported["fine_tune_role"]=role
                person_roles[row["person_id"]]=role;stream.write(json.dumps(exported,allow_nan=False)+"\n");counts[role]+=1
        files[path.name]=sha256_file(path)
    write_csv(output/"excluded_sessions.csv",exclusions,["study","experiment","participant","reason"])
    receipt={"status":"participant_disjoint_human_split_frozen","source_manifest_sha256":canonical_hash(manifest),
             "configuration_sha256":sha256_file(config_path),"files_sha256":files,"person_roles":person_roles,
             "independent_people":dict(Counter(person_roles.values())),"sessions":dict(counts),"source_receipts":list(receipts.values()),
             "selection":"first48 source-order compatible people; deterministic sourceidentity hash assigns roles; sameperson allsessions share role",
             "prior_exposure":"Psych101flags explicitly false; other historical LLM exposure unknown; testprotectedfromthisnewtraining only",
             "human_training_executed":False,"synthetic":False}
    write_json(output/"split_receipt.json",receipt);return receipt


def verify_split(split_dir,config_path,*,open_test=True):
    root=Path(split_dir).resolve();receipt=_json(_owned_file(root,"split_receipt.json","human split receipt"))
    if receipt.get("status")!="participant_disjoint_human_split_frozen" or receipt["configuration_sha256"]!=sha256_file(config_path):
        raise ValueError("original human split/config receipt not qualified")
    roles={};data={}
    for role in ["train","validation","test"]:
        path=_owned_file(root,role+".jsonl","human split table")
        if sha256_file(path)!=receipt["files_sha256"][path.name]:raise ValueError("immutable participant split changed")
        data[role]=_jsonl(path) if role!="test" or open_test else []
        for row in data[role]:
            if row.get("synthetic") is not False or row["text_sha256"]!=hashlib.sha256(row["text"].encode()).hexdigest():
                raise ValueError("real human text identity differs from split")
            previous=roles.setdefault(row["person_id"],role)
            if previous!=role:raise ValueError("person crosses fine-tune/evaluation boundary")
    expected={person:role for person,role in receipt["person_roles"].items() if role!="test" or open_test}
    if roles!=expected:raise ValueError("participant role inventory mismatch")
    return receipt,data


def _runtime(source_root,manifest_path,output):
    if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("WORLD_SIZE","1"))!=1:
        raise ValueError("native fine-tuning requires one scheduled model-parallel process")
    pins=resource_audit(source_root,manifest_path);write_json(output/"model_source_audit.json",pins)
    deps=dependency_audit();write_json(output/"dependency_audit.json",deps)
    if pins["status"]!="READY_CACHED_MODELS" or deps["missing"]:raise ValueError("source weights or native dependencies not qualified")
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:raise ValueError("native70B fine-tuning requires exactlytwo allocated GPUs")
    devices=[{"index":i,"name":torch.cuda.get_device_name(i),"memory_bytes":torch.cuda.get_device_properties(i).total_memory} for i in range(2)]
    if sum(d["memory_bytes"] for d in devices)<76*1024**3:raise ValueError("native training GPU memory below qualification target")
    return torch,pins,deps,devices


def explicit_device_map():
    result={"model.embed_tokens":0,"model.rotary_emb":0,"model.norm":1,"lm_head":1}
    result.update({f"model.layers.{i}":0 if i<40 else 1 for i in range(80)})
    return result


def _load_model(source_root,model_name,torch,training,trained_adapter=None):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel,LoraConfig,get_peft_model,prepare_model_for_kbit_training
    root=Path(source_root).resolve();tokenizer=AutoTokenizer.from_pretrained(root/"centaur_adapter",local_files_only=True,trust_remote_code=False,use_fast=True)
    base=AutoModelForCausalLM.from_pretrained(root/"centaur_base_4bit",local_files_only=True,trust_remote_code=False,
             torch_dtype=torch.bfloat16,device_map=explicit_device_map(),low_cpu_mem_usage=True,attn_implementation="sdpa")
    if base.config.num_hidden_layers!=80:raise ValueError("native source architecture differs from original80block70B model")
    if training:
        base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,
                 gradient_checkpointing_kwargs={"use_reentrant":False})
    if trained_adapter is not None:
        model=PeftModel.from_pretrained(base,trained_adapter,local_files_only=True,is_trainable=False)
    elif model_name=="centaur":
        model=PeftModel.from_pretrained(base,root/"centaur_adapter",local_files_only=True,is_trainable=training)
    else:
        config=_json(root/"centaur_adapter/adapter_config.json")
        # Fresh adapter matches original parameter budget exactly, including rsLoRA.
        model=get_peft_model(base,LoraConfig(r=config["r"],lora_alpha=config["lora_alpha"],lora_dropout=config["lora_dropout"],
                  target_modules=config["target_modules"],bias=config["bias"],task_type="CAUSAL_LM",use_rslora=config.get("use_rslora",False)))
    model.config.use_cache=False
    if training:model.train()
    else:model.eval()
    return model,tokenizer


def _token_events(data,tokenizer,cfg):
    events=[];exclusions=[]
    for row in data:
        for event in marked_responses(row["text"])[:cfg["max_responses_per_session"]]:
            try:prefix,suffixes=_encode_candidates(tokenizer,event["prefix"],cfg["response_alphabet"],cfg["max_context_tokens"])
            except ValueError as error:
                exclusions.append({"session_id":row["session_id"],"person_id":row["person_id"],"response_index":event["index"],"reason":str(error)});continue
            response=event["response"].strip()
            if response not in suffixes:raise ValueError("human event differs from source response alphabet")
            events.append({"session_id":row["session_id"],"person_id":row["person_id"],"response_index":event["index"],
                           "response":response,"prefix_ids":prefix,"suffixes":suffixes,"exposure_flags":row["exposure_flags"]})
    if not events:raise ValueError("no genuine compatible human events; no padding, synthetic substitution or history truncation")
    return events,exclusions


def _person_event_limit(events,limit):
    counts=Counter();selected=[]
    for event in events:
        if counts[event["person_id"]]<limit:
            selected.append(event);counts[event["person_id"]]+=1
    return selected


def event_fingerprint(events):
    return canonical_hash([{k:r[k] for k in ["session_id","person_id","response_index","response","prefix_ids","suffixes"]} for r in events])


def _response_loss(model,torch,event):
    suffix=event["suffixes"][event["response"]];device=model.get_input_embeddings().weight.device
    ids=torch.tensor([event["prefix_ids"]+suffix],dtype=torch.long,device=device)
    kwargs={"input_ids":ids,"use_cache":False,"return_dict":True}
    parameters=inspect.signature(model.get_base_model().forward).parameters
    key="logits_to_keep" if "logits_to_keep" in parameters else "num_logits_to_keep" if "num_logits_to_keep" in parameters else None
    if key:kwargs[key]=len(suffix)+1
    result=model(**kwargs)
    logits=result.logits[0,:len(suffix)] if key else result.logits[0,len(event["prefix_ids"])-1:-1]
    labels=torch.tensor(suffix,dtype=torch.long,device=logits.device)
    return torch.nn.functional.cross_entropy(logits.float(),labels,reduction="mean")


def _parameter_hash(model):
    digest=hashlib.sha256()
    for name,parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode());digest.update(str(tuple(parameter.shape)).encode());digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def _memory(torch):
    return [{"device":i,"allocated_bytes":torch.cuda.memory_allocated(i),"reserved_bytes":torch.cuda.memory_reserved(i),
             "peak_allocated_bytes":torch.cuda.max_memory_allocated(i)} for i in range(2)]


def train(source_root,manifest_path,split_dir,config_path,model_name,output_dir):
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    cfg=_json(config_path);split,data=verify_split(split_dir,config_path,open_test=False)
    torch,pins,deps,devices=_runtime(source_root,manifest_path,output)
    torch.manual_seed(cfg["seed"]);random.seed(cfg["seed"]);np.random.seed(cfg["seed"])
    try:
        model,tokenizer=_load_model(source_root,model_name,torch,True)
        events,excluded=_token_events(data["train"],tokenizer,cfg);validation,val_excluded=_token_events(data["validation"],tokenizer,cfg)
        validation=_person_event_limit(validation,cfg["validation_events_per_person"])
        # Test outcomes are hash-verified but never opened as tokenized events during training.
        initial_hash=_parameter_hash(model)
        parameters=[p for name,p in model.named_parameters() if p.requires_grad]
        names=[name for name,p in model.named_parameters() if p.requires_grad]
        if not parameters or any("lora_" not in n for n in names):raise ValueError("native parameter qualification found trainable non-LoRA weights")
        count=sum(p.numel() for p in parameters)
        probe=_response_loss(model,torch,events[0]);probe.backward()
        finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
        gradient_devices={str(p.device) for p in parameters if p.grad is not None and float(p.grad.detach().abs().sum())>0}
        if not finite or gradient_devices!={"cuda:0","cuda:1"}:raise ValueError("native two-device gradient flow not qualified")
        qualification={"status":"native_two_gpu_qlora_gradient_qualified","probe_loss":float(probe.detach().cpu()),
                       "gradient_devices":sorted(gradient_devices),"trainable_parameters":count,"memory":_memory(torch),
                       "device_map":explicit_device_map(),"no_optimizer_update_during_probe":True}
        write_json(output/"resource_qualification.json",qualification)
        model.zero_grad(set_to_none=True);del probe
        optimizer=torch.optim.AdamW(parameters,lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
        log=[];updates=0;seen=[];generator=random.Random(cfg["seed"])
        for epoch in range(cfg["epochs"]):
            order=list(range(len(events)));generator.shuffle(order)
            for first in range(0,len(order),cfg["gradient_accumulation"]):
                if updates>=cfg["max_optimizer_steps"]:break
                indices=order[first:first+cfg["gradient_accumulation"]];optimizer.zero_grad(set_to_none=True)
                losses=[]
                for index in indices:
                    loss=_response_loss(model,torch,events[index]);(loss/len(indices)).backward()
                    value=float(loss.detach().cpu())
                    if not math.isfinite(value):raise ValueError("nonfinite actual human training loss")
                    losses.append(value);seen.append((epoch,events[index]["session_id"],events[index]["response_index"]))
                    del loss
                gradient_norm=torch.nn.utils.clip_grad_norm_(parameters,cfg["max_gradient_norm"])
                if not math.isfinite(float(gradient_norm.cpu())):raise ValueError("nonfinite optimizer gradient")
                optimizer.step();updates+=1
                log.append({"epoch":epoch,"optimizer_step":updates,"human_response_events":len(indices),
                            "mean_response_loss":sum(losses)/len(losses),"gradient_norm":float(gradient_norm.cpu())})
                # Persist progress independent of loss direction; no automatic ambiguous resubmission.
                write_json(output/"training_progress.json",{"status":"training_in_progress","model":model_name,"completed_optimizer_steps":updates,"last":log[-1]})
        model.zero_grad(set_to_none=True);final_hash=_parameter_hash(model)
        if final_hash==initial_hash:raise ValueError("optimizer completed without changing trainable human-adapted weights")
        model.eval();val_rows=[]
        for event in validation:
            result=_candidate_likelihoods(model,torch,event["prefix_ids"],event["suffixes"])[event["response"]]
            val_rows.append({"session_id":event["session_id"],"person_id":event["person_id"],"response_index":event["response_index"],
                             "response":event["response"],**result})
        adapter=output/"adapter";model.save_pretrained(adapter,safe_serialization=True)
        files={p.name:sha256_file(p) for p in adapter.iterdir() if p.is_file()}
        write_csv(output/"training_losses.csv",log)
        write_csv(output/"validation_likelihoods.csv",val_rows)
        write_csv(output/"excluded_events.csv",excluded+val_excluded,["session_id","person_id","response_index","reason"])
        receipt={"status":"human_finetuning_completed","model":model_name,"slurm_job_id":os.environ["SLURM_JOB_ID"],
                 "source_manifest_sha256":pins["manifest_sha256"],"split_receipt_sha256":sha256_file(Path(split_dir)/"split_receipt.json"),
                 "split_files_sha256":split["files_sha256"],"configuration_sha256":sha256_file(config_path),
                 "train_events_sha256":event_fingerprint(events),"validation_events_sha256":event_fingerprint(validation),
                 "actual_training_order_sha256":canonical_hash(seen),"optimizer_steps":updates,"epochs":cfg["epochs"],
                 "training_events":len(events),"validation_events":len(validation),"test_events_consumed":0,
                 "trainable_parameters":count,"initial_trainable_weights_sha256":initial_hash,"final_trainable_weights_sha256":final_hash,
                 "weights_changed":True,"adapter_files_sha256":files,"devices":devices,"memory":_memory(torch),
                 "native_training_executed":True,"human_data":True,"adaptation_completed":True,"actual_optimizer_steps":updates,
                 "data_split_sha256":sha256_file(Path(split_dir)/"split_receipt.json"),
                 "assay_id":"psych201:nasioulas2024feedback:original_LR_choice",
                 "human_assay":{"assay_id":"psych201:nasioulas2024feedback:original_LR_choice","target_field":"human_LR_response",
                                "source_ids":["psych201"],"input_modality":"behavioural_task_transcript",
                                "label_semantics":"original prompted Left/Right lottery choice; not EEG awareness/detection"},
                 "split":{"train_participant_ids":sorted(p for p,r in split["person_roles"].items() if r=="train"),
                          "validation_participant_ids":sorted(p for p,r in split["person_roles"].items() if r=="validation"),
                          "test_participant_ids":sorted(p for p,r in split["person_roles"].items() if r=="test"),
                          "split_sha256":sha256_file(Path(split_dir)/"split_receipt.json")},
                 "trained_checkpoint":{"path":str(adapter/"adapter_model.safetensors"),"sha256":files.get("adapter_model.safetensors")},
                 "train_examples_sha256":event_fingerprint(events),"pretrained_exposure":{"status":"unknown","provider_Psych101_flags":False},
                 "dependency_versions":deps["versions"],"selection":"fixed equal update budget; validation no model-selection gate",
                 "cross_system_comparison_performed":False,"synthetic":False}
        write_json(output/"training_receipt.json",receipt);return receipt
    except Exception as error:
        write_json(output/"training_failure.json",{"status":"NATIVE_TRAINING_NOT_COMPLETED","model":model_name,
                   "error_type":type(error).__name__,"reason":str(error),"resource_qualification":"inspect actual GPU memory/gradient receipt",
                   "cross_system_comparison_permitted":False})
        raise


def _verify_training(directory,split_dir,config_path,model_name):
    directory=Path(directory).resolve();receipt=_json(_owned_file(directory,"training_receipt.json","human training receipt"))
    if receipt.get("status")!="human_finetuning_completed" or receipt.get("weights_changed") is not True or receipt.get("model")!=model_name:
        raise ValueError("actual human fine-tuning receipt absent; released-only comparison prohibited")
    if receipt.get("native_training_executed") is not True or receipt.get("actual_optimizer_steps",0)<1:
        raise ValueError("no actual native human optimizer updates recorded")
    if receipt["split_receipt_sha256"]!=sha256_file(Path(split_dir)/"split_receipt.json") or receipt["configuration_sha256"]!=sha256_file(config_path):
        raise ValueError("fine-tuning source split/config mismatch")
    if receipt["initial_trainable_weights_sha256"]==receipt["final_trainable_weights_sha256"]:raise ValueError("unchanged adapter cannot certify human fine-tuning")
    files=receipt.get("adapter_files_sha256",{})
    if not files.get("adapter_model.safetensors"):raise ValueError("actual trained native weight artifact missing")
    for name,value in files.items():
        if len(Path(name).parts)!=1:raise ValueError("trained native artifact escapes its adapter role")
        path=_owned_file(directory,Path("adapter")/name,"trained native weight artifact")
        if path.stat().st_size<1 or sha256_file(path)!=value:raise ValueError("trained native weight artifact changed")
    checkpoint=receipt.get("trained_checkpoint",{})
    weight=_owned_file(directory,"adapter/adapter_model.safetensors","trained checkpoint")
    if Path(checkpoint.get("path","")).resolve()!=weight or checkpoint.get("sha256")!=files["adapter_model.safetensors"]:
        raise ValueError("trained checkpoint differs from its own native adapter artifact")
    return receipt


def evaluate(source_root,manifest_path,split_dir,config_path,trained_dir,model_name,output_dir,*,centaur_trained,base_trained):
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False);cfg=_json(config_path)
    # Unseal the test only after both real adapters have changed through human training.
    both={name:_verify_training(path,split_dir,config_path,name) for name,path in [("centaur",centaur_trained),("base",base_trained)]}
    matched=["train_events_sha256","validation_events_sha256","actual_training_order_sha256","optimizer_steps","trainable_parameters"]
    if any(both["centaur"][k]!=both["base"][k] for k in matched):raise ValueError("test stays sealed: unmatched human adaptation budget")
    training=_verify_training(trained_dir,split_dir,config_path,model_name)
    own_role_path=centaur_trained if model_name=="centaur" else base_trained
    if sha256_file(Path(trained_dir)/"training_receipt.json")!=sha256_file(Path(own_role_path)/"training_receipt.json"):
        raise ValueError("evaluation model differs from jointly unsealed human-trained pair")
    split,data=verify_split(split_dir,config_path,open_test=True)
    torch,pins,deps,devices=_runtime(source_root,manifest_path,output)
    model,tokenizer=_load_model(source_root,model_name,torch,False,Path(trained_dir)/"adapter")
    events,excluded=_token_events(data["test"],tokenizer,cfg);rows=[];activations=[]
    events=_person_event_limit(events,cfg["test_events_per_person"])
    for event in events:
        value=_candidate_likelihoods(model,torch,event["prefix_ids"],event["suffixes"])[event["response"]]
        rows.append({"model":model_name,"session_id":event["session_id"],"person_id":event["person_id"],
                     "response_index":event["response_index"],"response":event["response"],**value,**event["exposure_flags"]})
        vectors=_representations(model,torch,event["prefix_ids"],[20,60])
        filename=event["session_id"]+f"_{event['response_index']:04d}.activations.npz"
        np.savez(output/filename,**{f"trained_block_{k}":v for k,v in vectors.items()})
        activations.append({"session_id":event["session_id"],"response_index":event["response_index"],"path":filename,"sha256":sha256_file(output/filename)})
    write_csv(output/"heldout_likelihoods.csv",rows);write_csv(output/"excluded_events.csv",excluded,["session_id","person_id","response_index","reason"])
    receipt={"status":"human_finetuned_heldout_evaluation_completed","model":model_name,"slurm_job_id":os.environ["SLURM_JOB_ID"],
             "training_receipt_sha256":sha256_file(Path(trained_dir)/"training_receipt.json"),"split_receipt_sha256":sha256_file(Path(split_dir)/"split_receipt.json"),
             "test_events_sha256":event_fingerprint(events),"test_people":len({r["person_id"] for r in rows}),
             "test_events":len(events),"test_training_updates":0,"likelihoods_sha256":sha256_file(output/"heldout_likelihoods.csv"),
             "heldout_rows_sha256":canonical_hash(_csv_rows(output/"heldout_likelihoods.csv")),
             "human_observations_sha256":human_observation_fingerprint(data["test"],events),
             "both_models_human_finetuned_before_test":True,"assay_id":training["assay_id"],
             "both_training_receipts_sha256":{"centaur":sha256_file(Path(centaur_trained)/"training_receipt.json"),"base":sha256_file(Path(base_trained)/"training_receipt.json")},
             "human_assay":training["human_assay"],"adaptation_completed":True,"trained_checkpoint":training["trained_checkpoint"],
             "test_observation_ids":[r["session_id"]+":"+str(r["response_index"]) for r in events],
             "test_predictions":{"path":str(output/"heldout_likelihoods.csv"),"sha256":sha256_file(output/"heldout_likelihoods.csv")},
             "activation_receipts":activations,"source_manifest_sha256":pins["manifest_sha256"],"synthetic":False,
             "cross_system_comparison_performed":False,"prior_exposure":"protectedfromnewfinetuning; olderLLMpretrainingexposureunknown"}
    write_json(output/"evaluation_receipt.json",receipt);return receipt


def _comparison_layout(paths,output):
    roles={name:Path(path).resolve() for name,path in paths.items() if name!="config"}
    roles["comparison"]=Path(output).resolve()
    values=list(roles.values())
    if len(set(values))!=len(values) or any(a in b.parents for a in values for b in values if a!=b):
        raise ValueError("comparison artifact roles must be distinct non-overlapping directories")
    root=Path(os.path.commonpath([str(path) for path in values])).resolve()
    if root==Path(root.anchor):raise ValueError("comparison artifact root cannot be a filesystem root")
    config=Path(paths["config"]).resolve()
    if not config.is_file() or config.parent==Path(config.anchor):raise ValueError("bounded comparison configuration missing")
    return root,config.parent


def compare(split_dir,config_path,centaur_train,base_train,centaur_eval,base_eval,output_dir,*,_verify_only=False):
    paths={name:str(Path(path).resolve()) for name,path in {"split":split_dir,"config":config_path,
           "centaur_train":centaur_train,"base_train":base_train,"centaur_eval":centaur_eval,"base_eval":base_eval}.items()}
    output=Path(output_dir).resolve();artifact_root,configuration_root=_comparison_layout(paths,output)
    cfg=_json(config_path)
    split,data=verify_split(split_dir,config_path)
    trains={name:_verify_training(path,split_dir,config_path,name) for name,path in [("centaur",centaur_train),("base",base_train)]}
    keys=["train_events_sha256","validation_events_sha256","actual_training_order_sha256","optimizer_steps","actual_optimizer_steps","epochs","trainable_parameters","split_files_sha256","assay_id"]
    if any(trains["centaur"][k]!=trains["base"][k] for k in keys):raise ValueError("systems did not receive identical human training rows and parameter/update budget")
    if trains["centaur"]["split_files_sha256"]!=split["files_sha256"]:raise ValueError("training split file inventory differs from frozen human inputs")
    training_hashes={name:sha256_file(_owned_file(path,"training_receipt.json","human training receipt"))
                     for name,path in [("centaur",centaur_train),("base",base_train)]}
    evals={};tables={}
    for name,path in [("centaur",centaur_eval),("base",base_eval)]:
        path=Path(path).resolve();receipt=_json(_owned_file(path,"evaluation_receipt.json","heldout evaluation receipt"))
        if receipt.get("status")!="human_finetuned_heldout_evaluation_completed" or receipt.get("model")!=name:
            raise ValueError("completed trained-model heldout evaluation missing")
        if receipt.get("both_models_human_finetuned_before_test") is not True or receipt.get("test_training_updates")!=0:
            raise ValueError("test was not held out until both human adaptations completed")
        if receipt["split_receipt_sha256"]!=sha256_file(Path(split_dir)/"split_receipt.json"):
            raise ValueError("heldout evaluation used a different frozen human split")
        if receipt["training_receipt_sha256"]!=training_hashes[name] or receipt.get("both_training_receipts_sha256")!=training_hashes:
            raise ValueError("heldout evaluation tied to wrong trained model")
        likelihoods=_owned_file(path,"heldout_likelihoods.csv","heldout result table")
        if receipt["likelihoods_sha256"]!=sha256_file(likelihoods):raise ValueError("actual heldout result table changed")
        predictions=receipt.get("test_predictions",{})
        if Path(predictions.get("path","")).resolve()!=likelihoods or predictions.get("sha256")!=receipt["likelihoods_sha256"]:
            raise ValueError("heldout prediction artifact differs from its own evaluation role")
        tables[name]=_csv_rows(likelihoods)
        if canonical_hash(tables[name])!=receipt.get("heldout_rows_sha256"):raise ValueError("heldout result row fingerprint changed")
        if human_observation_fingerprint(data["test"],tables[name])!=receipt.get("human_observations_sha256"):
            raise ValueError("heldout source human observation fingerprint changed")
        if receipt.get("test_events")!=len(tables[name]) or receipt.get("test_people")!=len({r["person_id"] for r in tables[name]}):
            raise ValueError("heldout participant/event census differs from actual table")
        observations=[r["session_id"]+":"+str(int(r["response_index"])) for r in tables[name]]
        if receipt.get("test_observation_ids")!=observations or receipt.get("assay_id")!=trains[name]["assay_id"]:
            raise ValueError("heldout assay/observation inventory mismatch")
        if "test_events_per_person" in cfg and any(n>cfg["test_events_per_person"] for n in Counter(r["person_id"] for r in tables[name]).values()):
            raise ValueError("heldout person budget exceeds frozen configuration")
        if "max_responses_per_session" in cfg and any(int(r["response_index"])>=cfg["max_responses_per_session"] for r in tables[name]):
            raise ValueError("heldout response budget exceeds frozen configuration")
        evals[name]=receipt
    if evals["centaur"]["test_events_sha256"]!=evals["base"]["test_events_sha256"]:raise ValueError("systems tested on different actual human events")
    key=lambda r:(r["session_id"],int(r["response_index"]))
    indexed={n:{key(r):r for r in rows} for n,rows in tables.items()}
    if set(indexed["centaur"])!=set(indexed["base"]):raise ValueError("heldout episode pairing mismatch")
    groups=defaultdict(list);paired=[]
    actual_test={}
    for row in data["test"]:
        for event in marked_responses(row["text"]):
            actual_test[(row["session_id"],event["index"])]=(row["person_id"],event["response"].strip())
    for event,a in indexed["centaur"].items():
        b=indexed["base"][event]
        if a["person_id"]!=b["person_id"] or a["response"]!=b["response"]:raise ValueError("heldout human identity/outcome mismatch")
        if actual_test.get(event)!=(a["person_id"],a["response"]) or split["person_roles"].get(a["person_id"])!="test":
            raise ValueError("claimed cross-system test outcome differs from original sealed human observations")
        delta=float(a["conditional_choice_log_likelihood"])-float(b["conditional_choice_log_likelihood"])
        if not math.isfinite(delta):raise ValueError("nonfinite matched human comparison")
        groups[a["person_id"]].append(delta);paired.append({"session_id":event[0],"response_index":event[1],"person_id":a["person_id"],"centaur_minus_base_loglikelihood":delta})
    report={"status":"cross_system_comparison_after_matched_human_finetuning","independent_test_people":len(groups),
            "heldout_events":len(paired),"equal_person_mean_loglikelihood_difference":float(np.mean([np.mean(v) for v in groups.values()])),
            "training_receipts_sha256":training_hashes,
            "schema_version":2,"input_paths":paths,"artifact_root":str(artifact_root),"configuration_root":str(configuration_root),
            "input_receipts_sha256":{name:sha256_file(_owned_file(path,filename,"comparison input receipt"))
                                    for name,path,filename in [("split",split_dir,"split_receipt.json"),("centaur_train",centaur_train,"training_receipt.json"),
                                                              ("base_train",base_train,"training_receipt.json"),("centaur_eval",centaur_eval,"evaluation_receipt.json"),
                                                              ("base_eval",base_eval,"evaluation_receipt.json")]},
            "split_files_sha256":split["files_sha256"],"human_observations_sha256":evals["centaur"]["human_observations_sha256"],
            "identical_training_and_evaluation_budget_verified":True,"synthetic":False,
            "all_models_human_finetuned":True,"assay_id":trains["centaur"]["assay_id"],
            "interpretation":"bounded task-specific model comparison after real matched human fine-tuning; no neural/consciousness/generalmechanismclaim and no predeterminedwinner",
            "same_training_people_prior_centaur_exposure":"providerPsych101flagsfalse; broaderpretrainingunknown"}
    report["input_receipts_sha256"]["config"]=sha256_file(config_path)
    if _verify_only:return report,paired
    output.mkdir(parents=True,exist_ok=False);write_csv(output/"paired_heldout_comparison.csv",paired)
    report["paired_table"]={"path":str(output/"paired_heldout_comparison.csv"),"sha256":sha256_file(output/"paired_heldout_comparison.csv"),
                            "rows_sha256":canonical_hash(_csv_rows(output/"paired_heldout_comparison.csv"))}
    write_json(output/"comparison_receipt.json",report);return report


def verify_comparison(directory):
    """Reverify actual paired human adaptation artifacts; never load models or write."""
    directory=Path(directory).resolve();receipt_path=_owned_file(directory,"comparison_receipt.json","comparison receipt")
    receipt=_json(receipt_path)
    if receipt.get("status")!="cross_system_comparison_after_matched_human_finetuning" or receipt.get("schema_version")!=2:
        raise ValueError("artifact-verifiable human-finetuned comparison receipt missing")
    names={"split","config","centaur_train","base_train","centaur_eval","base_eval"}
    paths=receipt.get("input_paths",{})
    if set(paths)!=names or any(not Path(path).is_absolute() for path in paths.values()):
        raise ValueError("comparison input role paths missing or not absolute")
    root,config_root=_comparison_layout(paths,directory)
    if str(root)!=receipt.get("artifact_root") or str(config_root)!=receipt.get("configuration_root"):
        raise ValueError("comparison input artifact roots changed")
    for name,path in paths.items():
        if name!="config" and not Path(path).resolve().is_relative_to(root):raise ValueError("comparison input escapes its bounded role root")
    expected,paired=compare(paths["split"],paths["config"],paths["centaur_train"],paths["base_train"],
                            paths["centaur_eval"],paths["base_eval"],directory,_verify_only=True)
    for name,value in expected.items():
        if receipt.get(name)!=value:raise ValueError("comparison receipt differs from actual verified inputs: "+name)
    table=_owned_file(directory,"paired_heldout_comparison.csv","paired comparison result table");record=receipt.get("paired_table",{})
    rows=_csv_rows(table)
    if Path(record.get("path","")).resolve()!=table or record.get("sha256")!=sha256_file(table) or record.get("rows_sha256")!=canonical_hash(rows):
        raise ValueError("paired comparison result table changed")
    reconstructed=[{k:str(v) for k,v in row.items()} for row in paired]
    if rows!=reconstructed:raise ValueError("paired comparison results differ from actual heldout native evaluations")
    return {"status":"human_finetuned_comparison_artifacts_verified","comparison_receipt_sha256":sha256_file(receipt_path),
            "all_models_human_finetuned":True,"assay_id":receipt["assay_id"],"heldout_events":len(rows),
            "artifact_root":str(root),"verification_only":True,"gpu_execution":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    for name in ["split","train","evaluate","compare"]:
        p=sub.add_parser(name);p.add_argument("--config",required=True);p.add_argument("--output",required=True)
        if name!="compare":p.add_argument("--source-root",required=True);p.add_argument("--manifest",required=True)
        if name!="split":p.add_argument("--split",required=True)
        if name in ["train","evaluate"]:p.add_argument("--model",choices=["centaur","base"],required=True)
        if name=="evaluate":
            p.add_argument("--trained",required=True);p.add_argument("--centaur-trained",required=True);p.add_argument("--base-trained",required=True)
        if name=="compare":
            for key in ["centaur-train","base-train","centaur-eval","base-eval"]:p.add_argument("--"+key,required=True)
    a=parser.parse_args(argv)
    if a.command=="split":r=make_split(a.source_root,a.manifest,a.config,a.output)
    elif a.command=="train":r=train(a.source_root,a.manifest,a.split,a.config,a.model,a.output)
    elif a.command=="evaluate":r=evaluate(a.source_root,a.manifest,a.split,a.config,a.trained,a.model,a.output,centaur_trained=a.centaur_trained,base_trained=a.base_trained)
    else:r=compare(a.split,a.config,a.centaur_train,a.base_train,a.centaur_eval,a.base_eval,a.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ["person_roles","source_receipts","activation_receipts"]},indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
