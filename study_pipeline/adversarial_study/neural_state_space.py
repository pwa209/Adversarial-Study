"""Human sensor state spaces with participant-disjoint, history-conditioned tests.

PCA and PCA of linear time-delay coordinates are candidate observation models,
not nonlinear manifolds, source anatomy or consciousness states. Channels and
learned transforms use training people; validation selects rank/ridge/family.
Each episode retains its own clock and boundaries. Test early history can
condition a recorded nuisance baseline; test futures never fit that baseline,
source scaling, axes, transition models or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import core, data, discovery, model_guided as guided, signals
from .core import StudyError
from .human_generative_adaptation import participant_split

DEFINITION = "human_sensor_state_space_v1"
DEFAULTS = {"maximum_states": 400, "maximum_channels": 32, "minimum_channels": 4,
            "rank_grid": [2,4,8], "ridge_grid": [.001,.01,.1], "delay_lags": [0,1,2],
            "train_fraction": .65, "forecast_horizons_s": [.1,.5], "bootstrap_samples": 500,
            "minimum_people_per_source": 6, "training_channel_support": .8,
            "episode_centering": "early_history_mean"}
DESCRIPTORS = ("state_radius_mean", "state_radius_std", "state_displacement_100ms", "state_displacement_500ms",
               "state_covariance_effective_rank", "state_covariance_participation_rank", "state_recurrence_fraction",
               "state_dwell_median_s", "state_dwell_max_s", "state_transition_rate_hz",
               "state_occupancy_q0", "state_occupancy_q1", "state_occupancy_q2", "state_occupancy_q3")


def options(path):
    config = core.read_json(path)
    if config.get("schema_version") != 1:
        raise StudyError("State-space configuration must have schema_version 1")
    settings = {**DEFAULTS, **config.get("state_space", {})}
    if set(settings)-set(DEFAULTS):
        raise StudyError("Unknown state-space options: "+str(sorted(set(settings)-set(DEFAULTS))))
    settings["seed"] = config.get("seed",20261004)
    for key, low, high in (("maximum_states",32,400),("maximum_channels",2,64),("minimum_channels",2,64),
                           ("bootstrap_samples",100,2000),("minimum_people_per_source",6,1000)):
        if isinstance(settings[key],bool) or not isinstance(settings[key],int) or not low<=settings[key]<=high:
            raise StudyError("Invalid bounded state-space option "+key)
    if settings["minimum_channels"]>settings["maximum_channels"] or not .4<=settings["train_fraction"]<=.8 or not .8<=settings["training_channel_support"]<=1.:
        raise StudyError("Invalid training history or channel support threshold")
    if settings["episode_centering"] not in {"early_history_mean","none"}:
        raise StudyError("Unknown fixed episode nuisance operator")
    if isinstance(settings["seed"],bool) or not isinstance(settings["seed"],int) or settings["seed"]<0:
        raise StudyError("Invalid random seed")
    for key in ("rank_grid","delay_lags"):
        values=settings[key]
        if not isinstance(values,list) or not values or len(values)!=len(set(values)) or any(isinstance(v,bool) or not isinstance(v,int) for v in values):
            raise StudyError("Invalid discrete state-space grid "+key)
    if any(not 1<=r<=16 for r in settings["rank_grid"]) or 0 not in settings["delay_lags"] or any(not 0<=r<=8 for r in settings["delay_lags"]) or len(settings["delay_lags"])>5:
        raise StudyError("Unbounded ranks or delay coordinates")
    for key, low, high in (("ridge_grid",1e-6,10.),("forecast_horizons_s",.025,1.)):
        values=settings[key]
        if not isinstance(values,list) or not values or len(values)>8 or len(values)!=len(set(values)) or any(not math.isfinite(float(v)) or not low<=float(v)<=high for v in values):
            raise StudyError("Invalid bounded numerical grid "+key)
    return settings


def _token(value):
    return "e"+hashlib.sha256(value.encode()).hexdigest()[:32]


def _put(archive, name, array):
    values=np.asarray(array)
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise StudyError("Only actual finite numerical state arrays may be exported")
    with archive.open(name+".npy","w") as stream:
        np.lib.format.write_array(stream,values,allow_pickle=False)


def _contexts(trajectory_paths):
    contexts, rows, inputs, seen = [], [], [], set()
    for directory in trajectory_paths:
        root=Path(directory).resolve(); receipt=guided._verify_output(root)
        if receipt.get("status")!="model_guided_trajectories_completed":
            raise StudyError("State-space input must be completed genuine human trajectories")
        source_rows=core.read_csv(root/"observations.csv")
        if receipt.get("outputs",{}).get("observations.csv")!=core.sha256_file(root/"observations.csv"):
            raise StudyError("Trajectory observation table lacks a pinned source receipt")
        parents=[Path(i["path"]).resolve() for i in receipt["inputs"]]
        pointers=[]
        for path in parents:
            if path.suffix==".json":
                value=core.read_json(path)
                if isinstance(value.get("prepared"),str) and isinstance(value.get("config"),str):
                    pointers.append((path,value))
        if len(pointers)!=1:
            raise StudyError("Trajectory must bind exactly one original preparation pointer")
        pointer_path,pointer=pointers[0]; prepared=Path(pointer["prepared"]).resolve()
        cfg=core.load_config(pointer["config"])
        qualification=core.resolve_path(cfg,"qualification",True)
        score_receipts=[]
        jobs=prepared/"native/native_jobs.csv"
        for path in parents:
            if path.suffix==".json":
                value=core.read_json(path)
                if value.get("status")=="native_batch_completed" and value.get("jobs_sha256")==core.sha256_file(jobs):
                    score_receipts.append(path)
        if len(score_receipts)!=1 or not str(score_receipts[0]).endswith(".receipt.json"):
            raise StudyError("Trajectory lacks an unambiguous actual native scoring context")
        scores=Path(str(score_receipts[0])[:-len(".receipt.json")])
        _,checked,actual,chain=guided._human(pointer_path,scores,qualification)
        if checked.resolve()!=prepared:
            raise StudyError("Prepared human source pointer differs")
        actual_by_id={(r["participant_id"],r["episode_id"]):r for r in actual}
        recordings=core.read_json(prepared/"recordings.json")["recordings"]
        recording_map={r["recording_id"]:r for r in recordings}
        if len(recording_map)!=len(recordings):
            raise StudyError("Duplicate source recording provenance")
        context={"trajectory_path":str(root),"source_receipt_sha256":core.sha256_file(root/"receipt.json"),
                 "pointer_path":str(pointer_path),"config_path":str(Path(pointer["config"]).resolve()),
                 "prepared_path":str(prepared),"jobs_path":str(jobs),"scores_path":str(scores),
                 "qualification_path":str(qualification),"jobs_sha256":core.sha256_file(jobs),"scores_sha256":core.sha256_file(scores),
                 "qualification_sha256":core.sha256_file(qualification),"pointer_sha256":core.sha256_file(pointer_path),
                 "config_sha256":core.sha256_file(pointer["config"]),"prepared_recordings_sha256":core.sha256_file(prepared/"recordings.json"),
                 "source_ids":sorted({r["source_id"] for r in source_rows}),
                 "recordings":recording_map,"cfg":cfg}
        context_index=len(contexts);contexts.append(context)
        inputs.extend([root/"receipt.json",root/"observations.csv",Path(pointer["config"]),prepared/"recordings.json",*chain])
        for row in source_rows:
            ident=(row["participant_id"],row["episode_id"])
            if ident in seen or ident not in actual_by_id:
                raise StudyError("Duplicated or unbound source episode")
            seen.add(ident); original=actual_by_id[ident]
            if row.get("measurement_status")!="actual_multichannel_eeg" or row.get("score_origin")!="native_frozen_v2" or core.flag(row.get("synthetic")):
                raise StudyError("State inference cannot use synthetic or unqualified human observations")
            for key in ("source_id","recording_id","start_s","stop_s"):
                if row.get(key)!=str(original.get(key,"")):
                    raise StudyError("Trajectory identity/time differs from original preparation")
            if float(row["score"])!=float(original["score"]):
                raise StudyError("Trajectory clinical score differs from unchanged native inference")
            recording=recording_map[row["recording_id"]]
            rows.append({**row,"_context":context_index,"signal_path":recording["path"],"signal_sha256":recording["sha256"],
                         "source_trajectory":str(root),"source_receipt_sha256":context["source_receipt_sha256"],
                         "state_token":_token(row["source_id"]+"\0"+row["participant_id"]+"\0"+row["episode_id"])})
    return contexts,rows,list(dict.fromkeys(Path(p).resolve() for p in inputs))


def fixed_split(rows,seed,clinical_trained=None):
    clinical=None; inherited={}
    if clinical_trained is not None:
        from native_tools.human_finetune import verify_adaptation
        result=verify_adaptation(Path(clinical_trained).resolve())
        for split in ("train","validation","test"):
            for person in result["split"][split+"_participant_ids"]:
                if person in inherited:raise StudyError("Clinical adaptation partitions overlap")
                inherited[person]=split
        clinical={"directory":str(Path(clinical_trained).resolve()),"training_receipt_sha256":core.sha256_file(Path(clinical_trained)/"training_receipt.json"),
                  "trained_checkpoint_sha256":result["trained_checkpoint"]["sha256"],"participant_roles":inherited}
    sources=defaultdict(set)
    for row in rows:sources[row["source_id"]].add(row["participant_id"])
    assignments={};source_splits={}
    for source,people in sorted(sources.items()):
        base=participant_split(sorted(people),seed)
        for split in ("train","validation","test"):
            for person in base[split+"_participant_ids"]:
                role=inherited.get(person,split)
                if person in assignments and assignments[person]!=role:
                    raise StudyError("A globally deidentified person has conflicting source partitions")
                assignments[person]=role
        source_splits[source]={split+"_participant_ids":sorted(p for p in people if assignments[p]==split) for split in ("train","validation","test")}
    return {"schema_version":1,"seed":seed,"rule":"Per-source outcome-blind SHA256 participant allocation; actual clinical-adaptation roles inherited for every overlapping person",
            **{split+"_participant_ids":sorted(p for p in assignments if assignments[p]==split) for split in ("train","validation","test")},
            "sources":source_splits,"clinical_adaptation":clinical,
            "interpretation":"Existing development people; completed human-task fine-tuning exposure retained, no independent prospective validation"},assignments


def fixed_channels(rows,settings):
    training=[r for r in rows if r["split"]=="train"]
    if not training:return []
    sets=[set(r.get("trajectory_channel_names",r.get("scalp_qualified_channel_names","")).split(";"))-{ "" } for r in training]
    counts=Counter(c for names in sets for c in names)
    ordered=sorted(counts,key=lambda c:(-counts[c],c))
    chosen=[]
    for name in ordered:
        candidate=chosen+[name]
        support=sum(set(candidate)<=names for names in sets)/len(sets)
        if support>=settings["training_channel_support"]:
            chosen=candidate
        if len(chosen)>=settings["maximum_channels"]:break
    return chosen if len(chosen)>=settings["minimum_channels"] else []


def bin_window(window,fs,start_s,settings):
    width=max(1,math.ceil(window.shape[1]/settings["maximum_states"]))
    n=window.shape[1]//width
    values=window[:,:n*width].reshape(window.shape[0],n,width).mean(axis=2).T
    times=start_s+(np.arange(n)*width+(width-1)/2)/fs
    split=int(math.floor(n*settings["train_fraction"]))
    if split<20 or n-split<10:raise StudyError("Insufficient actual early/late state samples")
    baseline=values[:split].mean(axis=0) if settings["episode_centering"]=="early_history_mean" else np.zeros(window.shape[0])
    return values,times,baseline,width/fs,split


def delay_coordinates(sensor,lags):
    lags=sorted(lags); first=max(lags)
    if len(sensor)<=first+3:raise StudyError("Episode too short for fixed delay coordinates")
    return np.concatenate([sensor[first-lag:len(sensor)-lag] for lag in lags],axis=1),first


def _counts(rows):
    return Counter(r["participant_id"] for r in rows)


def _training_scaling(cache,rows):
    counts=_counts(rows); people=len(counts);mean=None;second=None
    for row in rows:
        token=row["state_token"];x=cache[token+"__bins"][:int(row["state_temporal_split"])]-cache[token+"__baseline"]
        weight=1/(people*counts[row["participant_id"]])
        m=x.mean(axis=0);s=(x*x).mean(axis=0)
        mean=m*weight if mean is None else mean+m*weight
        second=s*weight if second is None else second+s*weight
    std=np.sqrt(np.maximum(second-mean*mean,0.))
    if not np.isfinite(std).all() or np.any(std<=1e-9):raise StudyError("Training common montage has flat/nonfinite observed channels")
    return mean,std


def _sensor(cache,row,mean,std):
    token=row["state_token"]
    return (cache[token+"__bins"]-cache[token+"__baseline"]-mean)/std


def _basis(cache,rows,mean,std,lags):
    counts=_counts(rows);npeople=len(counts);fm=None;second=None
    for row in rows:
        x,first=delay_coordinates(_sensor(cache,row,mean,std),lags)
        x=x[:int(row["state_temporal_split"])-first]
        w=1/(npeople*counts[row["participant_id"]]);m=x.mean(axis=0);s=x.T@x/len(x)
        fm=w*m if fm is None else fm+w*m;second=w*s if second is None else second+w*s
    covariance=(second-np.outer(fm,fm));covariance=(covariance+covariance.T)/2
    eigen,axes=np.linalg.eigh(covariance);order=np.argsort(eigen)[::-1];axes=axes[:,order]
    for j in range(axes.shape[1]):
        if axes[np.argmax(np.abs(axes[:,j])),j]<0:axes[:,j]*=-1
    return fm,axes,np.maximum(eigen[order],0.)


def _transition(cache,rows,model):
    rank=model["rank"];counts=_counts(rows);npeople=len(counts)
    xx=np.zeros((rank+1,rank+1));xy=np.zeros((rank+1,rank))
    radii=[]
    for row in rows:
        x,first=delay_coordinates(_sensor(cache,row,model["mean"],model["std"]),model["lags"])
        z=(x-model["feature_mean"])@model["basis"]
        z=z[:int(row["state_temporal_split"])-first]
        a=np.c_[np.ones(len(z)-1),z[:-1]];b=z[1:]
        weight=1/(npeople*counts[row["participant_id"]]*(len(z)-1));xx+=weight*a.T@a;xy+=weight*a.T@b
        radii.extend(np.linalg.norm(z,axis=1).tolist())
    penalty=np.eye(rank+1)*model["ridge"];penalty[0,0]=0.
    coefficient=np.linalg.solve(xx+penalty,xy)
    model["intercept"]=coefficient[0];model["transition"]=coefficient[1:]
    model["occupancy_edges"]=np.quantile(radii,[.25,.5,.75])
    return model


def transform(sensor,model):
    features,first=delay_coordinates(sensor,model["lags"])
    return (features-model["feature_mean"])@model["basis"],first


def evaluate(sensor,row,model,settings):
    z,first=transform(sensor,model);d=sensor.shape[1]
    cut=int(row["state_temporal_split"])-first
    reconstruction=z@model["basis"][:d].T+model["feature_mean"][:d]
    result={"reconstruction_mse":float(np.mean((reconstruction[cut:]-sensor[first+cut:])**2))}
    futures=[];persistence=[];static=[]
    for seconds in settings["forecast_horizons_s"]:
        steps=max(1,round(seconds/float(row["state_bin_s"])))
        origins=np.arange(max(cut+1,1),len(z)-steps+1)
        if not len(origins):raise StudyError("No genuine late forecast targets at selected horizon")
        predicted=z[origins-1].copy();err=0.;pe=0.;se=0.
        for step in range(steps):
            predicted=model["intercept"]+predicted@model["transition"]
            current=predicted@model["basis"][:d].T+model["feature_mean"][:d]
            actual=sensor[first+origins+step]
            if not np.isfinite(current).all():
                return {**result,"recursive_forecast_mse":None,"status":"nonfinite_unconstrained_future"}
            err+=float(np.mean((current-actual)**2))/steps
            pe+=float(np.mean((sensor[first+origins-1]-actual)**2))/steps
            se+=float(np.mean((model["feature_mean"][:d]-actual)**2))/steps
        result["forecast_"+str(seconds)+"s_mse"]=err;futures.append(err);persistence.append(pe);static.append(se)
    result.update(recursive_forecast_mse=float(np.mean(futures)),persistence_forecast_mse=float(np.mean(persistence)),
                  covariance_static_forecast_mse=float(np.mean(static)),selection_loss=result["reconstruction_mse"]+float(np.mean(futures)),status="finite")
    return result


def _average(records,key):
    people=defaultdict(list)
    for r in records:
        value=guided._finite(r.get(key))
        if value is not None:people[r["participant_id"]].append(value)
    return float(np.mean([np.mean(v) for v in people.values()])) if people else None


def uncertainty(records,key,settings):
    people=defaultdict(list)
    for row in records:
        value=guided._finite(row.get(key))
        if value is not None:people[row["participant_id"]].append(value)
    values=np.asarray([np.mean(v) for _,v in sorted(people.items())])
    if not len(values):return {"participants":0,"mean":None,"interval95":None}
    rng=np.random.default_rng(settings["seed"])
    means=[float(np.mean(values[rng.integers(0,len(values),len(values))])) for _ in range(settings["bootstrap_samples"])]
    return {"participants":len(values),"mean":float(values.mean()),"interval95":np.quantile(means,[.025,.975]).tolist() if len(values)>=2 else None,
            "uncertainty":"Participant resampling conditional on the frozen model; one person supplies no uncertainty estimate"}


def state_descriptors(z,dt,edges):
    centered=z-z.mean(axis=0);eigen=np.maximum(np.linalg.eigvalsh(centered.T@centered/max(1,len(z)-1)),0.)
    total=float(eigen.sum());prob=eigen[eigen>0]/total if total>0 else np.asarray([])
    radius=np.linalg.norm(z,axis=1)
    distance=np.sqrt(np.maximum(0,np.sum(z*z,axis=1)[:,None]+np.sum(z*z,axis=1)[None,:]-2*z@z.T)/z.shape[1])
    eligible=np.abs(np.arange(len(z))[:,None]-np.arange(len(z))[None,:])*dt>=.1-1e-9
    changes=np.linalg.norm(np.diff(z,axis=0),axis=1)/math.sqrt(z.shape[1]);boundaries=np.flatnonzero(changes>.75)+1
    dwell=np.diff(np.r_[0,boundaries,len(z)])*dt
    result={"state_radius_mean":float(radius.mean()),"state_radius_std":float(radius.std()),
            "state_covariance_effective_rank":float(np.exp(-prob@np.log(prob))) if total else 0.,
            "state_covariance_participation_rank":float(total*total/(eigen@eigen)) if total else 0.,
            "state_recurrence_fraction":float(((distance<=.75)&eligible).sum()/eligible.sum()) if eligible.any() else None,
            "state_dwell_median_s":float(np.median(dwell)),"state_dwell_max_s":float(dwell.max()),
            "state_transition_rate_hz":float(len(boundaries)/(len(z)*dt))}
    for label,seconds in (("100ms",.1),("500ms",.5)):
        lag=max(1,round(seconds/dt))
        result["state_displacement_"+label]=float(np.mean(np.linalg.norm(z[lag:]-z[:-lag],axis=1))/math.sqrt(z.shape[1])) if lag<len(z) else None
    occupancy=np.bincount(np.searchsorted(edges,radius,side="right"),minlength=4)/len(radius)
    result.update({"state_occupancy_q"+str(i):float(v) for i,v in enumerate(occupancy)})
    return result,eigen


def _model_arrays(archive,key,model):
    for name in ("mean","std","feature_mean","basis","transition","intercept","occupancy_edges","eigenvalues"):
        _put(archive,key+"__"+name,model[name])
    _put(archive,key+"__lags",model["lags"])


def _select_model(cache,group,settings):
    """Reproducible train-only fit and validation-only model selection."""
    train=[r for r in group if r["split"]=="train"]
    validation=[r for r in group if r["split"]=="validation"]
    if len({r["participant_id"] for r in train})<2 or not validation:
        raise StudyError("insufficient_source_training_or_validation_people")
    mean,std=_training_scaling(cache,train);alternatives=[]
    for family,lags in (("pca",[0]),("delay_linear",sorted(settings["delay_lags"]))):
        feature_mean,basis,eigen=_basis(cache,train,mean,std,lags)
        for rank in settings["rank_grid"]:
            if rank>basis.shape[1]:continue
            for ridge in settings["ridge_grid"]:
                model=_transition(cache,train,{"mean":mean,"std":std,"feature_mean":feature_mean,"basis":basis[:,:rank],
                                  "rank":rank,"ridge":ridge,"lags":lags,"family":family,"eigenvalues":eigen})
                measured=[{**evaluate(_sensor(cache,r,mean,std),r,model,settings),"participant_id":r["participant_id"]} for r in validation]
                loss=_average(measured,"selection_loss") if all(r["status"]=="finite" for r in measured) else None
                alternatives.append((math.inf if loss is None else loss,rank,ridge,family,model))
    if not alternatives:raise StudyError("no_candidate_state_model")
    best=min(alternatives,key=lambda a:a[:4])
    if not math.isfinite(best[0]):raise StudyError("no_numerically_finite_validation_model")
    record={"source_model_key":group[0]["source_model_key"],"source_id":group[0]["source_id"],"channel_names":group[0]["channel_names"].split(";"),
            "family":best[3],"rank":best[1],"ridge":best[2],"delay_lags":best[4]["lags"],
            "training_participant_ids":sorted({r["participant_id"] for r in train}),
            "validation_participant_ids":sorted({r["participant_id"] for r in validation}),
            "validation_candidates":[{"family":a[3],"rank":a[1],"ridge":a[2],"loss":a[0] if math.isfinite(a[0]) else None} for a in alternatives],
            "selection_rule":"Equal-participant reconstruction MSE plus recursive late-future sensor MSE; rank/ridge/family lexical ties fixed",
            "source_bin_s":group[0]["state_bin_s"],"episode_centering":settings["episode_centering"],"baseline_history_fraction":settings["train_fraction"]}
    return best[4],record


def _measure_episode(recording,row,channels,cfg,settings):
    """Measure one fixed montage directly from unchanged source samples."""
    window=data.extract_window(recording,float(row["start_s"]),float(row["stop_s"]),duration_s=10.)
    indices,_=signals.channel_quality_selection(window,recording["sfreq_hz"],recording["channel_names"],cfg.get("channel_quality",{}))
    actual={recording["channel_names"][i] for i in indices}
    if not set(channels)<=actual:raise StudyError("missing_or_unqualified_frozen_training_montage_channel")
    if not set(channels)<=set(row.get("trajectory_channel_names","").split(";")):
        raise StudyError("original_trajectory_mask_does_not_qualify_frozen_montage")
    selected=[recording["channel_names"].index(c) for c in channels]
    raw=window[selected];fs=recording["sfreq_hz"]
    bins,times,baseline,dt,cut=bin_window(raw,fs,float(row["start_s"]),settings)
    key="m"+hashlib.sha256((row["source_id"]+"\0"+str(dt)+"\0"+";".join(channels)).encode()).hexdigest()[:24]
    fields={"channel_names":";".join(channels),"channel_count":len(channels),"source_model_key":key,
            "state_bin_s":dt,"state_sfreq_hz":1/dt,"original_sfreq_hz":fs,"state_temporal_split":cut,
            "baseline_support_stop_s":float(row["start_s"])+cut*dt,
            "state_sensor_input_sha256":hashlib.sha256(np.ascontiguousarray(bins).tobytes()).hexdigest(),
            **{"fixed_"+k:v for k,v in signals.compute_features(raw,fs,cfg.get("features",{})).items()},
            **{"fixed_"+k:v for k,v in guided.multichannel_trajectory(raw,fs).items()},
            "response_marker_status":"missing_no_qualified_source_input"}
    arrays={"bins":bins,"times":times,"baseline":baseline}
    if row.get("source_trial_type")=="stim-adapt":
        event=float(row["stop_s"])
        try:
            response=data.extract_window(recording,event-1,event+3,duration_s=4.)[selected]
            quality=signals.quality_metrics(response,fs,**cfg.get("quality",{}))
            if not quality["qualified"]:raise StudyError("response_signal_quality:"+";".join(quality["reasons"]))
            width=round(dt*fs);n=response.shape[1]//width
            arrays["response"]=response[:,:n*width].reshape(len(channels),n,width).mean(axis=2).T
            arrays["response_times"]=event-1+(np.arange(n)*width+(width-1)/2)/fs
            fields.update(response_marker_status="source_annotated_tactile_event_physical_trigger_alignment_unqualified",response_event_onset_s=event)
        except (ValueError,StudyError) as error:fields["response_marker_status"]="missing_actual_response:"+str(error)
    return fields,arrays


def _weighted_ridge(train,features,target,test_features,ridge):
    counts=_counts(train);weights=np.asarray([1/(len(counts)*counts[r["participant_id"]]) for r in train])
    mean=weights@features;std=np.sqrt(weights@((features-mean)**2));std=np.where(std>1e-9,std,1.)
    x=np.c_[np.ones(len(features)),(features-mean)/std];y=np.asarray(target)
    if y.ndim==1:y=y[:,None]
    penalty=np.eye(x.shape[1])*ridge;penalty[0,0]=0.
    coefficient=np.linalg.solve(x.T@(weights[:,None]*x)+penalty,x.T@(weights[:,None]*y))
    return np.c_[np.ones(len(test_features)),(test_features-mean)/std]@coefficient


def outcome_predictions(rows,settings):
    """Fit source-specific readouts on training labels; choose ridge on validation."""
    results=[]
    numeric=("capacity_value","awareness_rating_raw","confidence")
    for target in (*numeric,"detection","report_category"):
        strata=defaultdict(list)
        for row in rows:
            value=row.get(target)
            if (target=="report_category" and value) or (target!="report_category" and guided._finite(value) is not None):
                key=(row["source_id"],row.get("capacity_construct",target) if target=="capacity_value" else target,
                     row.get("capacity_units","") if target=="capacity_value" else "source_original_code")
                strata[key].append(row)
        for key,group in sorted(strata.items()):
            train=[r for r in group if r["split"]=="train"];val=[r for r in group if r["split"]=="validation"];test=[r for r in group if r["split"]=="test"]
            if len({r["participant_id"] for r in train})<2 or not val or not test:
                results.append({"target":target,"source_id":key[0],"construct":key[1],"status":"insufficient_source_partition_labels"});continue
            categorical=target in {"report_category","detection"}
            classes=sorted({str(r[target]) for r in train}) if categorical else []
            if categorical and len(classes)<2:
                results.append({"target":target,"source_id":key[0],"status":"insufficient_training_category_variation","original_training_categories":classes});continue
            ytrain=np.eye(len(classes))[[classes.index(str(r[target])) for r in train]] if categorical else np.asarray([float(r[target]) for r in train])
            conventional=[k for k in ("fixed_relative_delta","fixed_relative_theta","fixed_relative_alpha","fixed_relative_beta","fixed_relative_gamma",
                                      "fixed_effective_rank","fixed_covariance_participation_rank","feature_spectral_centroid_hz","feature_spectral_entropy")
                          if all(guided._finite(r.get(k)) is not None for r in group)]
            state=[k for k in DESCRIPTORS if all(guided._finite(r.get(k)) is not None for r in group)]
            feature_sets={"clinical_score":["score"],"spectra_covariance":["score",*conventional],
                          "state_space":["score",*state],"state_space_plus_spectra_covariance":["score",*conventional,*state]}
            counts=_counts(train);weights=np.asarray([1/(len(counts)*counts[r["participant_id"]]) for r in train])
            baseline_label=classes[int(np.argmax(weights@ytrain))] if categorical else float(weights@ytrain)
            for label,fields in feature_sets.items():
                tx=np.asarray([[float(r[k]) for k in fields] for r in train]);vx=np.asarray([[float(r[k]) for k in fields] for r in val])
                alternatives=[]
                for ridge in settings["ridge_grid"]:
                    prediction=_weighted_ridge(train,tx,ytrain,vx,ridge)
                    error=[float(classes[int(np.argmax(p))]!=str(r[target])) if categorical else (float(p[0])-float(r[target]))**2 for r,p in zip(val,prediction)]
                    loss=_average([{"participant_id":r["participant_id"],"error":e} for r,e in zip(val,error)],"error")
                    alternatives.append((loss,ridge))
                _,ridge=min(alternatives)
                test_x=np.asarray([[float(r[k]) for k in fields] for r in test]);prediction=_weighted_ridge(train,tx,ytrain,test_x,ridge)
                observed=[]
                for row,pred in zip(test,prediction):
                    value=classes[int(np.argmax(pred))] if categorical else float(pred[0])
                    error=float(value!=str(row[target])) if categorical else (value-float(row[target]))**2
                    null=float(baseline_label!=str(row[target])) if categorical else (baseline_label-float(row[target]))**2
                    observed.append({"participant_id":row["participant_id"],"episode_id":row["episode_id"],"prediction":value,"observed":str(row[target]) if categorical else float(row[target]),"error":error,"training_baseline_error":null})
                results.append({"target":target,"source_id":key[0],"construct":key[1],"units":key[2],"predictor":label,
                                "status":"development_held_participant_source_outcome_prediction","selected_ridge":ridge,"features":fields,
                                "original_training_categories":classes,"unseen_test_categories":sorted({str(r[target]) for r in test}-set(classes)) if categorical else [],
                                "metric":"misclassification" if categorical else "squared_error_original_source_units",
                                "test_error":uncertainty(observed,"error",settings),"training_mean_or_majority_error":uncertainty(observed,"training_baseline_error",settings),
                                "heldout_predictions":observed,
                                "interpretation":"Named source labels remain distinct; fixed-split development readout, not causal neural-to-function identification, independent replication or an awareness probability"})
    return results


def montage_sensitivity(rows):
    groups=defaultdict(list);output=[]
    for row in rows:groups[(row["participant_id"],row["source_id"],row["recording_id"])].append(row)
    for identity,group in sorted(groups.items()):
        group=sorted(group,key=lambda r:(float(r["start_s"]),r["episode_id"]))
        # Consecutive disjoint windows are an outcome-blind bounded pair rule.
        previous=None
        for row in group:
            if previous is not None and float(previous["stop_s"])<=float(row["start_s"]) and previous["channel_names"]==row["channel_names"]:
                a,b=previous,row
                output.append({"participant_id":identity[0],"source_id":identity[1],"recording_id":identity[2],"episode_a":a["episode_id"],"episode_b":b["episode_id"],
                               "fixed_channel_names":a["channel_names"],"full_channel_count_a":a.get("trajectory_channels"),"full_channel_count_b":b.get("trajectory_channels"),
                               "fixed_rank_change":float(b["fixed_trajectory_effective_rank"])-float(a["fixed_trajectory_effective_rank"]),
                               "full_rank_change":float(b["trajectory_effective_rank"])-float(a["trajectory_effective_rank"]) if all(guided._finite(r.get("trajectory_effective_rank")) is not None for r in (a,b)) else None,
                               "fixed_recurrence_change":float(b["fixed_trajectory_recurrence_fraction"])-float(a["fixed_trajectory_recurrence_fraction"]),
                               "full_recurrence_change":float(b["trajectory_recurrence_fraction"])-float(a["trajectory_recurrence_fraction"]) if all(guided._finite(r.get("trajectory_recurrence_fraction")) is not None for r in (a,b)) else None,
                               "interpretation":"Same fixed sensor montage in disjoint within-recording windows; full-scalp variable-mask descriptors are a sensitivity comparison, not an intrinsic neural dimension"})
            previous=row
    return output


def analyze(trajectory,config,output,clinical_trained=None):
    settings=options(config);contexts,rows,inputs=_contexts(trajectory)
    split,assignments=fixed_split(rows,settings["seed"],clinical_trained)
    out=guided._new_output(output);inputs.append(Path(config).resolve())
    if clinical_trained:inputs.append(Path(clinical_trained).resolve()/"training_receipt.json")
    core.write_json(out/"split.json",split)
    sources=defaultdict(list)
    for row in rows:row["split"]=assignments[row["participant_id"]];sources[row["source_id"]].append(row)
    montages={source:fixed_channels(group,settings) for source,group in sources.items()}
    work=out/"sensor_inputs.npz";valid=[];excluded=[];raw_inputs=set()
    with zipfile.ZipFile(work,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as cache:
        by_recording=defaultdict(list)
        for row in rows:by_recording[(row["_context"],row["recording_id"])].append(row)
        for (context_index,rid),group in sorted(by_recording.items()):
            context=contexts[context_index];receipt=context["recordings"][rid]
            if core.sha256_file(receipt["path"])!=receipt["sha256"]:raise StudyError("Actual EEG recording changed")
            recording=data.read_recording(receipt["path"],"npz");raw_inputs.add(Path(receipt["path"]).resolve())
            for row in group:
                row={**row};channels=montages[row["source_id"]]
                try:
                    if not channels:raise StudyError("insufficient_training_supported_common_montage")
                    if len({r["participant_id"] for r in sources[row["source_id"]]})<settings["minimum_people_per_source"]:
                        raise StudyError("insufficient_people_per_source")
                    fields,arrays=_measure_episode(recording,row,channels,context["cfg"],settings)
                    for name,array in arrays.items():_put(cache,row["state_token"]+"__"+name,array)
                    row.update(status="included",**fields)
                    valid.append(row)
                except (ValueError,StudyError) as error:
                    row.update(status="excluded",state_exclusion_reason=str(error),channel_names=";".join(channels));excluded.append(row)
    inputs.extend(sorted(raw_inputs));models={};selection=[];evaluation_rows=[]
    with np.load(work,allow_pickle=False) as cache:
        cache_files=set(cache.files)
        groups=defaultdict(list)
        for row in valid:groups[row["source_model_key"]].append(row)
        for key,group in sorted(groups.items()):
            try:
                models[key],record=_select_model(cache,group,settings);selection.append(record)
            except (StudyError,np.linalg.LinAlgError,FloatingPointError) as error:
                reason=str(error) if str(error)=="insufficient_source_training_or_validation_people" else "source_model_insufficient:"+str(error)
                for row in group:row.update(status="excluded",state_exclusion_reason=reason)
                excluded.extend(group)
        model_path=out/"models.npz"
        with zipfile.ZipFile(model_path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for key,model in sorted(models.items()):_model_arrays(archive,key,model)
        frozen={"schema_version":1,"definition":DEFINITION,"settings":settings,"source_models":selection,"models_sha256":core.sha256_file(model_path),
                "split_sha256":core.sha256_file(out/"split.json"),"frozen_before_test_transformation_utc":datetime.now(timezone.utc).isoformat(),
                "geometry_scope":"Measured sensor PCA/linear-delay state space, not source anatomy or identified nonlinear manifold",
                "conditioning":"Test-person early history supplies an explicit nuisance mean; source scaling, axes and dynamics are learned exclusively from training people"}
        core.write_json(out/"model_selection.json",frozen)
        included=[r for r in valid if r["source_model_key"] in models and r["status"]=="included"]
        with zipfile.ZipFile(out/"states.npz","w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for row in included:
                model=models[row["source_model_key"]];sensor=_sensor(cache,row,model["mean"],model["std"]);z,first=transform(sensor,model)
                token=row["state_token"];times=cache[token+"__times"][first:]
                descriptors,eigen=state_descriptors(z,float(row["state_bin_s"]),model["occupancy_edges"])
                _put(archive,token+"__state",z);_put(archive,token+"__sensor",sensor[first:]);_put(archive,token+"__times_s",times)
                _put(archive,token+"__baseline_uv",cache[token+"__baseline"]);_put(archive,token+"__state_cov_eigenvalues",eigen)
                row.update(**descriptors,state_count=len(z),state_rank=z.shape[1],state_delay_bins=max(model["lags"]),
                           state_start_s=float(times[0]),state_stop_s=float(times[-1]),state_model_sha256=frozen["models_sha256"],
                           state_coordinates="pca" if model["family"]=="pca" else "linear_delay_pca_not_nonlinear_manifold")
                if token+"__response" in cache_files:
                    response=(cache[token+"__response"]-cache[token+"__baseline"]-model["mean"])/model["std"]
                    response_z,rf=transform(response,model);rt=cache[token+"__response_times"][rf:]
                    _put(archive,token+"__response_trace",response[rf:]);_put(archive,token+"__response_state",response_z);_put(archive,token+"__response_times_s",rt)
                result=evaluate(sensor,row,model,settings);row.update({"state_"+k:v for k,v in result.items() if k!="status"})
                row["state_forecast_numerical_status"]=result["status"]
                cut=int(row["state_temporal_split"])-first
                early,_=state_descriptors(z[:cut],float(row["state_bin_s"]),model["occupancy_edges"])
                late,_=state_descriptors(z[cut:],float(row["state_bin_s"]),model["occupancy_edges"])
                row.update({"early_"+k:v for k,v in early.items()});row.update({"late_"+k:v for k,v in late.items()})
                evaluation_rows.append({"participant_id":row["participant_id"],"source_id":row["source_id"],"episode_id":row["episode_id"],"split":row["split"],"control":"actual",**result})
                cut=int(row["state_temporal_split"])
                for control in ("shared_time_shuffle","shared_phase","independent_phase"):
                    rng=np.random.default_rng(int(hashlib.sha256((token+control).encode()).hexdigest()[:8],16));pieces=[]
                    for block in (sensor[:cut],sensor[cut:]):
                        if control=="shared_time_shuffle":pieces.append(block[rng.permutation(len(block))])
                        else:pieces.append(discovery.phase_surrogate(block.T,int(rng.integers(0,2**32-1)),shared_phases=control=="shared_phase").T)
                    surrogate=np.concatenate(pieces,axis=0)
                    evaluation_rows.append({"participant_id":row["participant_id"],"source_id":row["source_id"],"episode_id":row["episode_id"],"split":row["split"],"control":control,**evaluate(surrogate,row,model,settings)})
    output_rows=[{k:v for k,v in r.items() if not k.startswith("_")} for r in [*included,*excluded]]
    core.write_csv(out/"rows.csv",output_rows,sorted({k for r in output_rows for k in r}|{"participant_id","source_id","episode_id","split","status","state_token"}))
    core.write_csv(out/"prediction_metrics.csv",evaluation_rows,sorted({k for r in evaluation_rows for k in r}|{"participant_id","source_id","episode_id","split","control"}))
    sensitivity=montage_sensitivity(included)
    core.write_csv(out/"montage_sensitivity.csv",sensitivity,sorted({k for r in sensitivity for k in r}|{"participant_id","source_id","episode_a","episode_b","fixed_channel_names"}))
    reliability={"schema_version":1,"status":"heldout_state_space_evaluated" if any(r["split"]=="test" for r in included) else "insufficient_heldout_state_space",
                 "prediction_uncertainty":[{"source_id":source,"split":part,"control":control,**uncertainty([r for r in evaluation_rows if r["source_id"]==source and r["split"]==part and r["control"]==control],"recursive_forecast_mse",settings)}
                                           for source in sorted(sources) for part in ("validation","test") for control in ("actual","shared_time_shuffle","shared_phase","independent_phase")],
                 "baseline_comparators":[{"source_id":source,"split":part,"baseline":baseline,**uncertainty([r for r in evaluation_rows if r["source_id"]==source and r["split"]==part and r["control"]=="actual"],baseline,settings)}
                                         for source in sorted(sources) for part in ("validation","test") for baseline in ("persistence_forecast_mse","covariance_static_forecast_mse")],
                 "test_conditioning":"Initial measured state/history is observed; later recursive targets are never inserted. Nuisance baseline uses only early65% history. This is not an unconditional out-of-person forecast.",
                 "controls":"Shared-time shuffle and phase controls are transformed separately inside early/late blocks using frozen actual-data transforms; they are measurement nulls, not human causal interventions",
                 "independent_human_validation":False,"arousal_mapping":"Only genuinely recorded source fields retained; no proxy arousal state is invented"}
    reliability["source_outcome_predictions"]=outcome_predictions(included,settings)
    reliability["descriptor_stability"]=[{"source_id":source,"split":part,"descriptor":feature,
                                          **uncertainty([{"participant_id":r["participant_id"],"early_late_absolute_difference":abs(float(r["early_"+feature])-float(r["late_"+feature]))}
                                                         for r in included if r["source_id"]==source and r["split"]==part and guided._finite(r.get("early_"+feature)) is not None and guided._finite(r.get("late_"+feature)) is not None],"early_late_absolute_difference",settings),
                                          "interpretation":"Within-window stability across early/late intervals, potentially affected by genuine nonstationarity; not test-retest reliability of an invariant state"}
                                         for source in sorted(sources) for part in ("validation","test") for feature in DESCRIPTORS]
    reliability["matched_montage_sensitivity"]={"pair_rows":len(sensitivity),"rule":"Consecutive disjoint same-person/source/recording pairs with the identical training-fixed montage, selected independently of scores/outcomes",
                                               "artifact":"montage_sensitivity.csv","test_people_do_not_choose_montage":True}
    core.write_json(out/"reliability.json",reliability)
    return guided._finish(out,inputs,"neural_state_space_completed",definition=DEFINITION,configuration_sha256=core.sha256_file(config),
                          configuration_path=str(Path(config).resolve()),source_code_sha256=core.sha256_file(__file__),
                          native_contexts=[{k:v for k,v in c.items() if k not in {"cfg","recordings"}} for c in contexts],
                          episode_count=len(included),excluded_episodes=len(excluded),source_model_count=len(models),
                          scientific_status=reliability["status"],split_sha256=core.sha256_file(out/"split.json"),
                          source_anatomy_mapping_qualified=False,nonlinear_manifold_identification_established=False)


def _same_value(actual,expected):
    if isinstance(expected,dict):
        return isinstance(actual,dict) and set(actual)==set(expected) and all(_same_value(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):
        return isinstance(actual,list) and len(actual)==len(expected) and all(_same_value(a,b) for a,b in zip(actual,expected))
    if isinstance(expected,(int,float)) and not isinstance(expected,bool):
        return isinstance(actual,(int,float)) and not isinstance(actual,bool) and math.isfinite(float(actual)) and math.isclose(float(actual),float(expected),rel_tol=1e-8,abs_tol=1e-11)
    return actual==expected


def _array_proof(actual,expected,name,exact=False):
    a=np.asarray(actual);b=np.asarray(expected)
    valid=a.shape==b.shape and np.isfinite(a).all() and (np.array_equal(a,b) if exact else np.allclose(a,b,rtol=1e-8,atol=1e-11))
    if not valid:raise StudyError("State-space numerical reproduction differs: "+name)


def _row_proof(row,fields):
    for key,value in fields.items():
        observed=row.get(key,"")
        if isinstance(value,(float,int)) and not isinstance(value,bool):
            number=guided._finite(observed)
            if number is None or not math.isclose(number,float(value),rel_tol=1e-8,abs_tol=1e-11):
                raise StudyError("State-space measured row differs: "+key)
        elif observed!=("" if value is None else str(value)):
            raise StudyError("State-space original/measured row differs: "+key)


def verify_state_space(directory):
    """Reproduce source bins, dispositions, training fits and validation choices.

    Raw recordings are streamed once per original recording. A rehashed cache
    cannot substitute its own measurements, axes or participant allocation.
    Numerical eigenspace comparisons allow floating point eigensolver variation;
    the actual stored axes must still fit the training covariance and transition.
    """
    root=Path(directory).resolve();receipt=guided._verify_output(root)
    if receipt.get("status")!="neural_state_space_completed" or receipt.get("definition")!=DEFINITION:
        raise StudyError("State-space receipt is absent or has a different measurement definition")
    for name in ("rows.csv","split.json","models.npz","model_selection.json","states.npz","sensor_inputs.npz","reliability.json"):
        if receipt.get("outputs",{}).get(name)!=core.sha256_file(root/name):raise StudyError("Required state-space artifact is not bound")
    config=Path(receipt["configuration_path"]).resolve();settings=options(config)
    if receipt["configuration_sha256"]!=core.sha256_file(config) or receipt["source_code_sha256"]!=core.sha256_file(__file__):
        raise StudyError("State-space source/configuration definition changed")
    selection=core.read_json(root/"model_selection.json");split=core.read_json(root/"split.json")
    if selection["settings"]!=settings or selection["models_sha256"]!=core.sha256_file(root/"models.npz") or selection["split_sha256"]!=core.sha256_file(root/"split.json"):
        raise StudyError("State-space models, settings or partition lineage differs")
    contexts,original,_=_contexts([c["trajectory_path"] for c in receipt["native_contexts"]])
    if [{k:v for k,v in c.items() if k not in {"cfg","recordings"}} for c in contexts]!=receipt["native_contexts"]:
        raise StudyError("Original source/native context lineage differs")
    clinical=split.get("clinical_adaptation");expected,assignments=fixed_split(original,settings["seed"],clinical["directory"] if clinical else None)
    if expected!=split:raise StudyError("State-space human partitions differ from fixed source/clinical allocation")
    rows=core.read_csv(root/"rows.csv");saved={(r["source_id"],r["participant_id"],r["episode_id"]):r for r in rows}
    if len(rows)!=len(original) or len(saved)!=len(rows) or set(saved)!={(r["source_id"],r["participant_id"],r["episode_id"]) for r in original}:
        raise StudyError("State-space silently omitted or duplicated original human episodes")
    sources=defaultdict(list);recordings=defaultdict(list)
    for row in original:
        row["split"]=assignments[row["participant_id"]];sources[row["source_id"]].append(row)
        recordings[(row["_context"],row["recording_id"])].append(row)
        _row_proof(saved[(row["source_id"],row["participant_id"],row["episode_id"])],{k:v for k,v in row.items() if not k.startswith("_")})
    montages={source:fixed_channels(group,settings) for source,group in sources.items()}
    records={m["source_model_key"]:m for m in selection["source_models"]}
    if len(records)!=len(selection["source_models"]):raise StudyError("Duplicated state-space model records")
    groups=defaultdict(list);expected_sensor=set();expected_states=set();proved_models={};excluded={}
    with np.load(root/"states.npz",allow_pickle=False) as states,np.load(root/"models.npz",allow_pickle=False) as models,np.load(root/"sensor_inputs.npz",allow_pickle=False) as sensor:
        for (index,rid),group in sorted(recordings.items()):
            context=contexts[index];source=context["recordings"][rid]
            if core.sha256_file(source["path"])!=source["sha256"]:raise StudyError("Actual source recording changed")
            recording=data.read_recording(source["path"],"npz")
            for original_row in group:
                ident=(original_row["source_id"],original_row["participant_id"],original_row["episode_id"])
                row=saved[ident];channels=montages[original_row["source_id"]];token=original_row["state_token"]
                try:
                    if not channels:raise StudyError("insufficient_training_supported_common_montage")
                    if len({r["participant_id"] for r in sources[original_row["source_id"]]})<settings["minimum_people_per_source"]:
                        raise StudyError("insufficient_people_per_source")
                    fields,arrays=_measure_episode(recording,original_row,channels,context["cfg"],settings)
                except (ValueError,StudyError) as error:
                    excluded[ident]=str(error);continue
                _row_proof(row,fields)
                for name,array in arrays.items():
                    member=token+"__"+name;expected_sensor.add(member)
                    if member not in sensor.files:raise StudyError("Missing actual EEG measurement member")
                    _array_proof(sensor[member],array,"original raw EEG "+member,exact=True)
                groups[fields["source_model_key"]].append({**original_row,**fields})
        if set(sensor.files)!=expected_sensor:raise StudyError("EEG measurement cache contains unsupported/missing source members")
        for key,group in sorted(groups.items()):
            try:
                fitted,expected_record=_select_model(sensor,group,settings)
            except (StudyError,np.linalg.LinAlgError,FloatingPointError) as error:
                reason=str(error) if str(error)=="insufficient_source_training_or_validation_people" else "source_model_insufficient:"+str(error)
                for row in group:excluded[(row["source_id"],row["participant_id"],row["episode_id"])]=reason
                continue
            if key not in records or not _same_value(records[key],expected_record):
                raise StudyError("State-space rank/ridge/family was not selected on genuine validation people")
            model={name:models[key+"__"+name] for name in ("mean","std","feature_mean","basis","transition","intercept","occupancy_edges","eigenvalues")}
            model.update(lags=models[key+"__lags"].tolist(),rank=expected_record["rank"],ridge=expected_record["ridge"],family=expected_record["family"])
            if model["lags"]!=expected_record["delay_lags"]:raise StudyError("State-space delay operator differs from validation choice")
            for name in ("mean","std","feature_mean","eigenvalues"):_array_proof(model[name],fitted[name],"training-only "+name)
            _array_proof(model["basis"]@model["basis"].T,fitted["basis"]@fitted["basis"].T,"training-only PCA eigenspace")
            _array_proof(model["basis"].T@model["basis"],np.eye(model["rank"]),"orthonormal state axes")
            training=[r for r in group if r["split"]=="train"]
            actual_dynamics=_transition(sensor,training,{**model})
            for name in ("transition","intercept","occupancy_edges"):_array_proof(model[name],actual_dynamics[name],"training-only "+name)
            proved_models[key]=model
            for original_row in group:
                ident=(original_row["source_id"],original_row["participant_id"],original_row["episode_id"]);row=saved[ident];token=row["state_token"]
                if row["status"]!="included" or row["state_model_sha256"]!=selection["models_sha256"]:raise StudyError("Fitted source episode was excluded or its basis lineage changed")
                raw=_sensor(sensor,original_row,model["mean"],model["std"]);z,first=transform(raw,model)
                times=sensor[token+"__times"][first:];descriptors,eigen=state_descriptors(z,float(row["state_bin_s"]),model["occupancy_edges"])
                arrays={"state":z,"sensor":raw[first:],"times_s":times,"baseline_uv":sensor[token+"__baseline"],"state_cov_eigenvalues":eigen}
                if token+"__response" in sensor.files:
                    response=(sensor[token+"__response"]-sensor[token+"__baseline"]-model["mean"])/model["std"]
                    response_z,rf=transform(response,model)
                    arrays.update(response_trace=response[rf:],response_state=response_z,response_times_s=sensor[token+"__response_times"][rf:])
                for name,array in arrays.items():
                    member=token+"__"+name;expected_states.add(member)
                    if member not in states.files:raise StudyError("Missing actual state trajectory member")
                    _array_proof(states[member],array,"frozen human state "+member)
                if len(z)>settings["maximum_states"]:raise StudyError("Unbounded source state trajectory")
                _row_proof(row,{**descriptors,"state_count":len(z),"state_rank":z.shape[1],"state_delay_bins":max(model["lags"]),
                                "state_start_s":float(times[0]),"state_stop_s":float(times[-1])})
                cut=int(row["state_temporal_split"])-first
                for prefix,part in (("early_",z[:cut]),("late_",z[cut:])):
                    values,_=state_descriptors(part,float(row["state_bin_s"]),model["occupancy_edges"])
                    _row_proof(row,{prefix+k:v for k,v in values.items()})
        if set(records)!=set(proved_models):raise StudyError("State-space contains a model without an actual train/validation fit")
        expected_model_members={key+"__"+name for key in proved_models for name in ("mean","std","feature_mean","basis","transition","intercept","occupancy_edges","eigenvalues","lags")}
        if set(models.files)!=expected_model_members or set(states.files)!=expected_states:raise StudyError("State-space model/state census contains unsupported members")
        for ident,reason in excluded.items():
            row=saved[ident]
            if row["status"]!="excluded" or row.get("state_exclusion_reason")!=reason:raise StudyError("Original source exclusion disposition differs")
    included_count=len(rows)-len(excluded)
    if receipt["episode_count"]!=included_count or receipt["excluded_episodes"]!=len(excluded) or receipt["source_model_count"]!=len(proved_models):
        raise StudyError("State-space denominator or fitted model census differs")
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory",action="append",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--clinical-trained",type=Path)
    args=parser.parse_args(argv)
    try:
        result=analyze(**vars(args));print(json.dumps({"status":result["status"],"scientific_status":result["scientific_status"],"episodes":result["episode_count"]}))
    except (StudyError,ValueError,OSError,KeyError,np.linalg.LinAlgError) as error:
        parser.exit(2,"Human state-space analysis failed: "+str(error)+"\n")


if __name__=="__main__":main()
