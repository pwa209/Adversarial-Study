"""Human neural-state trajectories -> effective native mechanisms -> challenges.

Only actual Pop.1.V trajectories and paired native input/control simulations
are used. Delay kernels are effective observation maps, not extra populations,
an anatomical EEG forward model or identified physiology. Entire people remain
in their parent frozen split. Validation selects maps/candidates before later
test trajectories are opened. Full multivariate traces and temporal Gram
geometry replace the earlier five scalar summary fit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from . import core, model_guided as guided, human_generative_adaptation as generative

ASSAY_ID = "human_neural_state_dynamics_v1"
ENDPOINT = {"name": "latent_response_path_rms_v1", "formula": "Interpolate actual frozen latent response coordinates onto96points linspace(-0.9,2.9s), without extrapolation; sum_t sqrt(sum_j((z[t+1,j]-z[t,j])^2)/rank) over consecutive grid points with0<=relative_event_time<=2.9s", "units": "source-specific training-scaled latent coordinate units", "time_window_s": [0., 2.9], "interpretation": "Response trajectory path at the declared grid resolution, not a conscious-experience duration or an anatomical spread measure"}


def rotation_geometry(state, points=16):
    """Entire normalized temporal Gram and increment-Gram matrices; rotation invariant."""
    y = np.asarray(state, dtype=float)
    if y.ndim != 2 or min(y.shape) < 1 or len(y) < 4 or not np.isfinite(y).all():
        raise core.StudyError("Geometry requires an actual finite multivariate state trace")
    positions = np.linspace(0, len(y)-1, points)
    sampled = np.column_stack([np.interp(positions, np.arange(len(y)), y[:,j]) for j in range(y.shape[1])])
    centered = sampled-sampled.mean(axis=0)
    energy = float(np.mean(np.sum(centered**2, axis=1)))
    if energy <= 1e-15:
        return np.zeros(points*(points+1)//2+(points-1)*points//2)
    gram = centered@centered.T/energy
    increments = np.diff(centered, axis=0)
    increment_gram = increments@increments.T/energy
    return np.r_[gram[np.triu_indices(points)], increment_gram[np.triu_indices(points-1)]]


def path_endpoint(state, relative_times):
    y, t = np.asarray(state, float), np.asarray(relative_times, float)
    selected = (t >= 0) & (t <= 3)
    if y.ndim != 2 or len(y) != len(t) or selected.sum() < 2 or not np.isfinite(y).all():
        raise core.StudyError("Endpoint requires a genuine frozen latent response trace")
    return float(np.sqrt(np.mean(np.diff(y[selected], axis=0)**2, axis=1)).sum())


def _rank_limit(weights, rank):
    u, s, vt = np.linalg.svd(weights, full_matrices=False)
    k = min(int(rank), len(s))
    return (u[:,:k]*s[:k])@vt[:k]


def _interpolate(values, source_times, target_times):
    y, t, q = np.asarray(values, float), np.asarray(source_times, float), np.asarray(target_times, float)
    if y.ndim != 2 or len(y) != len(t) or len(t) < 4 or not np.isfinite(y).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or q.min() < t[0]-1e-8 or q.max() > t[-1]+1e-8:
        raise core.StudyError("Observation alignment cannot pad, extrapolate or join episode boundaries")
    return np.column_stack([np.interp(q, t, y[:,j]) for j in range(y.shape[1])])


def _config(path):
    cfg = core.read_json(path); section = cfg.get("mechanisms", {})
    result = {"seed": int(cfg.get("seed", 20261004)), "lags": section.get("delay_lags", [0,1,2,4,8]), "ridges": section.get("ridge_grid", [.001,.01,.1,1.]),
              "ranks": section.get("observation_rank_grid", [2,4]), "strengths": section.get("observation_strength_grid", [0.,1.]),
              "maximum_states": int(section.get("maximum_bank_states", 400)), "maximum_candidates": int(section.get("maximum_candidates", 100)),
              "bootstrap": int(section.get("bootstrap_samples", 500)), "support_quantile": float(section.get("joint_support_quantile", .95)),
              "minimum_train": int(section.get("minimum_training_people", 3)), "minimum_validation": int(section.get("minimum_validation_people", 1)), "minimum_test": int(section.get("minimum_test_people", 2)),
              "early_fraction": float(cfg.get("state_space", {}).get("train_fraction", .65))}
    if not 16 <= result["maximum_states"] <= 400 or not 1 <= result["maximum_candidates"] <= 200 or not 20 <= result["bootstrap"] <= 2000 or not .5 <= result["support_quantile"] < 1 or not .5 <= result["early_fraction"] <= .8:
        raise core.StudyError("Mechanistic bridge exceeds declared bounded design")
    if not isinstance(result["lags"], list) or not result["lags"] or result["lags"][0] != 0 or len(result["lags"]) > 12 or any(isinstance(v,bool) or not isinstance(v,int) or not 0 <= v <= 20 for v in result["lags"]) or sorted(set(result["lags"])) != result["lags"]:
        raise core.StudyError("Native observation lags must be unique bounded nonnegative indices")
    if any(not math.isfinite(float(v)) or not 0 < float(v) <= 10 for v in result["ridges"]) or len(result["ridges"]) > 8 or any(not 1 <= int(v) <= 8 for v in result["ranks"]) or len(result["ranks"]) > 4 or any(float(v) not in (0.,.5,1.) for v in result["strengths"]):
        raise core.StudyError("Invalid bounded observation-map grid")
    result["delay_seconds_per_index"] = 10/result["maximum_states"]
    return result


def _load_parents(states, fit, representations):
    from .neural_state_space import verify_state_space
    from .representation_bridge import verify_representation
    state_receipt = verify_state_space(states)
    representation_receipt = verify_representation(representations)
    if Path(representation_receipt["input_paths"]["states"]).resolve()!=Path(states).resolve() or representation_receipt["input_receipts_sha256"]["states"]!=core.sha256_file(Path(states)/"receipt.json"):
        raise core.StudyError("Adapted representation map does not use these actual human state coordinates")
    fit_receipt, bank, provenance = generative.verify_fit(fit)
    rows = core.read_csv(Path(states)/"rows.csv")
    included = [r for r in rows if r.get("status") == "included"]
    if len({r["episode_id"] for r in included}) != len(included):
        raise core.StudyError("Repeated actual state episode")
    people = defaultdict(set)
    for row in included:
        if row.get("split") not in ("train", "validation", "test"):
            raise core.StudyError("State artifact lacks a fixed human partition")
        people[row["participant_id"]].add(row["split"])
    if any(len(values) != 1 for values in people.values()):
        raise core.StudyError("Human people cross mechanism partitions")
    return state_receipt, representation_receipt, fit_receipt, bank, provenance, rows


def _native_bank(rows):
    from native_tools.nftsim_runtime import parse_output
    from native_tools.mechanistic_native import output_signal, pulse_response
    traces, pairs, rejected = {}, {}, []
    for row in sorted(rows, key=lambda r:r["simulation_id"]):
        parsed = parse_output(Path(row["source_output_path"]))
        signal = output_signal(parsed)
        t = parsed["values"][:,0]
        if t[0] < -1e-8 or t[-1] < 9.9 or t[-1] > 10.1:
            raise core.StudyError("Native bank does not have its documented ten-second relative output clock")
        scale = float(np.std(signal, ddof=1))
        if not math.isfinite(scale) or scale <= 1e-15:
            rejected.append({"simulation_id": row["simulation_id"], "reason": "native_scalar_cannot_be_standardized"}); continue
        traces[row["simulation_id"]] = {"row": row, "signal": signal, "times": t, "mean": float(signal.mean()), "scale": scale, "parsed": parsed}
    grouped = defaultdict(list)
    for trace in traces.values():
        if trace["row"].get("pair_id"): grouped[(trace["row"].get("variant"),trace["row"]["pair_id"])].append(trace)
    for key, values in sorted(grouped.items()):
        controls = [v for v in values if float(v["row"].get("input_amplitude",0)) == 0]
        pulses = [v for v in values if float(v["row"].get("input_amplitude",0)) > 0]
        if len(controls) != 1 or len(pulses) != 1:
            rejected.append({"simulation_id": str(key), "reason": "missing_actual_paired_input_or_control"}); continue
        control, pulse = controls[0], pulses[0]
        try:
            metrics = pulse_response(control["parsed"], pulse["parsed"])
        except ValueError as exc:
            rejected.append({"simulation_id": pulse["row"]["simulation_id"], "reason": "unqualified_pair: "+str(exc)}); continue
        pairs[control["row"]["simulation_id"]] = {"control": control, "pulse": pulse, "metrics": metrics, "onset_s": 2.}
    for trace in traces.values(): del trace["parsed"]
    return traces, pairs, rejected


def _native_kernel(trace, target_times, lag_seconds, reference=None):
    reference = trace if reference is None else reference
    x = (trace["signal"]-reference["mean"])/reference["scale"]
    queries = np.asarray(target_times)[:,None]-np.asarray(lag_seconds)[None,:]
    if queries.min() < trace["times"][0]-1e-8 or queries.max() > trace["times"][-1]+1e-8:
        raise core.StudyError("Native delay kernel cannot extrapolate unobserved history")
    return np.column_stack([np.interp(queries[:,j],trace["times"],x) for j in range(queries.shape[1])])


def _residuals(directory):
    """Numeric train/validation errors only; test errors cannot select this bridge."""
    path = Path(directory)/"episode_representation_residuals.csv"
    values, evidence = defaultdict(list), defaultdict(set)
    if not path.is_file(): return {}, {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("split") not in ("train", "validation"): continue
            predictor = row.get("predictor", "").lower()
            if predictor != "activation" or row.get("target") != "state_distribution_mmd": continue
            value = guided._finite(row.get("mean_pair_loss"))
            if value is None or value < 0: continue
            key = row["source_id"], row["episode_id"], row["split"]
            values[key].append(value); evidence[key].add(row["participant_id"])
    return {key:float(np.mean(v)) for key,v in values.items()}, {key:sorted(v) for key,v in evidence.items()}


def _source_examples(rows, archive, source_key, cfg, partition, grid=None):
    selected = [r for r in rows if r.get("status") == "included" and r["source_model_key"] == source_key and r["split"] == partition]
    examples, rejected = [], []
    if grid is None:
        counts, starts = [], []
        for row in selected:
            times = np.asarray(archive[row["state_token"]+"__times_s"],float)
            counts.append(int(np.sum(times >= times[-1]-10)))
            starts.append(max(0.,9.998-(times[-1]-times[0])))
        if not counts: return [], [], None
        n = min(cfg["maximum_states"], max(16,int(np.median(counts))))
        earliest = max(max(cfg["lags"])*cfg["delay_seconds_per_index"], max(starts))
        if earliest >= 5: return [], [{"source_model_key":source_key,"reason":"less_than_five_seconds_common_native_human_support"}], None
        grid = np.linspace(earliest,9.998,n)
    for row in selected:
        token = row["state_token"]
        try:
            times = np.asarray(archive[token+"__times_s"],float)
            y = _interpolate(archive[token+"__state"], times, times[-1]-9.998+grid)
            early_count = max(4, int(math.floor(len(y)*cfg["early_fraction"])))
            if early_count >= len(y)-2: raise core.StudyError("No later trajectory target")
            baseline_stop = guided._finite(row.get("baseline_support_stop_s"))
            if baseline_stop is not None and baseline_stop > times[-1]-9.998+grid[early_count-1]+1e-8:
                raise core.StudyError("Parent nuisance baseline reaches a later forecast target")
            example = {"row":row,"y":y,"early":early_count,"times":grid,"response":None,"response_times":None}
            response_key = token+"__response_state"
            if response_key in archive.files:
                onset = guided._finite(row.get("response_event_onset_s"))
                if onset is not None:
                    rt = np.asarray(archive[token+"__response_times_s"],float)-onset
                    response_grid = np.linspace(-.9,2.9,96)
                    try:
                        response = _interpolate(archive[response_key],rt,response_grid)
                        if response.shape[1] != y.shape[1]: raise core.StudyError("Response coordinates differ from the trained source state basis")
                        example.update(response=response,response_times=response_grid)
                    except core.StudyError as exc: example["response_unavailable_reason"] = str(exc)
            examples.append(example)
        except (core.StudyError,KeyError) as exc:
            rejected.append({"episode_id":row["episode_id"],"participant_id":row["participant_id"],"source_model_key":source_key,"split":partition,"reason":str(exc)})
    return examples,rejected,grid


def _person_weights(examples, residuals, strength):
    people = defaultdict(int)
    for e in examples: people[e["row"]["participant_id"]] += 1
    recorded = [residuals[(e["row"]["source_id"],e["row"]["episode_id"],"train")] for e in examples if (e["row"]["source_id"],e["row"]["episode_id"],"train") in residuals]
    center, scale = (float(np.mean(recorded)),max(float(np.std(recorded)),1e-9)) if recorded else (0.,1.)
    weights = []
    for e in examples:
        row = e["row"]; value = residuals.get((row["source_id"],row["episode_id"],"train"),center)
        weights.append((1+strength*max(0.,min(2.,(value-center)/scale)))/people[row["participant_id"]])
    return np.asarray(weights)/sum(weights), {"mean":center,"std":scale,"observations":len(recorded),"strength":strength}


def _nearest_trace(examples, predictions, strength):
    # Constant per-episode alignment is inferred exclusively from early states.
    cut=examples[0]["early"]
    if any(e["early"]!=cut for e in examples): raise core.StudyError("Conditioning clocks differ")
    y=np.asarray([e["y"][:cut] for e in examples]);p=predictions[:,:cut]
    a,b=y.reshape(len(y),-1),p.reshape(len(p),-1)
    distance=np.mean(a*a,axis=1)[:,None]+np.mean(b*b,axis=1)[None,:]-2*a@b.T/a.shape[1]
    differences=y.mean(axis=1)[:,None,:]-p.mean(axis=1)[None,:,:]
    distance-=(2*strength-strength**2)*np.mean(differences**2,axis=2)
    indices=np.argmin(distance,axis=1)
    return indices,differences[np.arange(len(y)),indices]*strength


def _fit_weights(X,Y,weights,ridge,rank):
    w=np.concatenate([np.full(len(x),v/len(x)) for x,v in zip(X,weights)])
    x,y=np.concatenate(X),np.concatenate(Y)
    mx=np.sum(x*w[:,None],axis=0); my=np.sum(y*w[:,None],axis=0)
    centered=x-mx; target=y-my
    XX=(centered*w[:,None]).T@centered
    W=np.linalg.solve(XX+ridge*np.eye(x.shape[1]),(centered*w[:,None]).T@target)
    W=_rank_limit(W,rank)
    return W,my-mx@W


def _fit_one(train,validation,natives,pairs,cfg,options,residuals):
    lagset,ridge,rank,strength=options
    lags=np.asarray(lagset)*cfg["delay_seconds_per_index"]
    kernels=np.asarray([_native_kernel(trace,train[0]["times"],lags) for trace in natives])
    weights,representation_scale=_person_weights(train,residuals,1.)
    # Initialization uses all early temporal Gram entries, not scalar summaries.
    early=train[0]["early"]
    native_geometry=np.asarray([rotation_geometry(z[:early]) for z in kernels])
    human_geometry=np.asarray([rotation_geometry(e["y"][:e["early"]]) for e in train])
    d=np.sum(human_geometry**2,axis=1)[:,None]+np.sum(native_geometry**2,axis=1)[None,:]-2*human_geometry@native_geometry.T
    assigned=np.argmin(d,axis=1)
    for _ in range(2):
        W,b=_fit_weights([kernels[i,:e["early"]] for i,e in zip(assigned,train)],[e["y"][:e["early"]] for e in train],weights,ridge,rank)
        prediction=kernels@W+b
        assigned,_=_nearest_trace(train,prediction,strength)
    W,b=_fit_weights([kernels[i,:e["early"]] for i,e in zip(assigned,train)],[e["y"][:e["early"]] for e in train],weights,ridge,rank)
    support=sorted(set(assigned.tolist()),key=lambda i:natives[i]["row"]["simulation_id"])
    supported_natives=[natives[i] for i in support]
    supported_kernels=kernels[support]
    responseX,responseY,response_weights=[],[],[]
    for index,e,weight in zip(assigned,train,weights):
        pair=pairs.get(natives[index]["row"]["simulation_id"])
        if e["response"] is None or pair is None: continue
        q=e["response_times"]+pair["onset_s"]
        delta=_native_kernel(pair["pulse"],q,lags,pair["control"])-_native_kernel(pair["control"],q,lags)
        pre=e["response_times"]<0
        responseX.append(delta); responseY.append(e["response"]-e["response"][pre].mean(axis=0));response_weights.append(weight)
    Wr=None
    if responseX:
        w=np.concatenate([np.full(len(x),v/len(x)) for x,v in zip(responseX,response_weights)])
        x,y=np.concatenate(responseX),np.concatenate(responseY)
        Wr=_rank_limit(np.linalg.solve((x*w[:,None]).T@x+ridge*np.eye(x.shape[1]),(x*w[:,None]).T@y+ridge*W),rank)
    fitted={"W":W,"bias":b,"response_W":Wr,"lags":list(lagset),"ridge":ridge,"rank":rank,"strength":strength,
            "support_ids":[v["row"]["simulation_id"] for v in supported_natives],"train_assignments":[natives[i]["row"]["simulation_id"] for i in assigned],
            "training_episode_weights":weights.tolist(),
            "representation_scale":representation_scale,"response_training_episodes":len(responseX),"response_training_people":len({e["row"]["participant_id"] for i,e in zip(assigned,train) if e["response"] is not None and natives[i]["row"]["simulation_id"] in pairs})}
    A,ab=_fit_weights([e["y"][:e["early"]-1] for e in train],[e["y"][1:e["early"]] for e in train],weights,.01,train[0]["y"].shape[1])
    fitted["baseline_A"],fitted["baseline_bias"]=A,ab
    rows,arrays,metrics=_predict(validation,supported_natives,pairs,cfg,fitted,"validation")
    fitted["validation_metrics"]=metrics
    return fitted,rows,arrays


def _predict(examples,natives,pairs,cfg,fitted,partition):
    if not examples: return [],{}, {"status":"no_actual_partition_trajectories","objective":None,"people":0,"episodes":0}
    lags=np.asarray(fitted["lags"])*cfg["delay_seconds_per_index"]
    kernels=np.asarray([_native_kernel(v,examples[0]["times"],lags) for v in natives])
    predictions=kernels@fitted["W"]+fitted["bias"]
    indices,offsets=_nearest_trace(examples,predictions,fitted["strength"])
    rows,arrays,person_loss= [],{},defaultdict(list)
    for i,(e,index,offset) in enumerate(zip(examples,indices,offsets)):
        row=e["row"]; actual=e["y"]; cut=e["early"]; predicted=predictions[index]+offset
        scale=max(float(np.mean(actual[:cut]**2)),1e-9)
        mse=float(np.mean((predicted[cut:]-actual[cut:])**2)/scale)
        persistence=float(np.mean((actual[cut-1]-actual[cut:])**2)/scale)
        autoregressive=[];last=actual[cut-1]
        for _ in range(len(actual)-cut):
            last=last@fitted["baseline_A"]+fitted["baseline_bias"];autoregressive.append(last)
        ar_mse=float(np.mean((np.asarray(autoregressive)-actual[cut:])**2)/scale)
        geometry=float(np.mean((rotation_geometry(predicted[cut:])-rotation_geometry(actual[cut:]))**2))
        response_loss=None;response_endpoint=None;prediction_endpoint=None
        if e["response"] is not None and fitted["response_W"] is not None and natives[index]["row"]["simulation_id"] in pairs:
            pair=pairs[natives[index]["row"]["simulation_id"]]; q=e["response_times"]+pair["onset_s"]
            delta=_native_kernel(pair["pulse"],q,lags,pair["control"])-_native_kernel(pair["control"],q,lags)
            baseline=e["response"][e["response_times"]<0].mean(axis=0)
            rhat=delta@fitted["response_W"]+baseline
            post=e["response_times"]>=0
            response_loss=float(np.mean((rhat[post]-e["response"][post])**2)/scale)
            response_endpoint=path_endpoint(e["response"],e["response_times"])
            prediction_endpoint=path_endpoint(rhat,e["response_times"])
            arrays[row["state_token"]+"__response_prediction"]=rhat
        objective=mse+.25*geometry+(.25*response_loss if response_loss is not None else 0.)
        person_loss[row["participant_id"]].append((objective,mse,persistence,geometry,ar_mse))
        arrays[row["state_token"]+"__prediction"]=predicted
        rows.append({"episode_id":row["episode_id"],"participant_id":row["participant_id"],"source_id":row["source_id"],"split":partition,"simulation_id":natives[index]["row"]["simulation_id"],"later_state_standardized_mse":mse,"persistence_standardized_mse":persistence,"autoregressive_standardized_mse":ar_mse,"temporal_gram_loss":geometry,"response_standardized_mse":response_loss,"observed_latent_response_path_rms_v1":response_endpoint,"predicted_latent_response_path_rms_v1":prediction_endpoint,"objective":objective,"latent_rank":actual.shape[1],"grid_states":len(actual),"conditioning_states":cut,"actual_response_available":e["response"] is not None,"response_mapping_status":"annotated_event_association_physical_units_unqualified" if response_loss is not None else "no_supported_actual_response_prediction"})
    values=np.asarray([np.mean(v,axis=0) for v in person_loss.values()])
    metrics={"status":"effective_trajectory_prediction_only","objective":float(values[:,0].mean()),"later_state_standardized_mse":float(values[:,1].mean()),"persistence_standardized_mse":float(values[:,2].mean()),"temporal_gram_loss":float(values[:,3].mean()),"autoregressive_standardized_mse":float(values[:,4].mean()),"people":len(person_loss),"episodes":len(examples),"response_predictions":sum(r["response_standardized_mse"] is not None for r in rows)}
    return rows,arrays,metrics


def joint_support(examples, quantile=.95, minimum_people=3):
    """Learn a joint temporal-geometry support metric with other-person calibration."""
    descriptors=np.asarray([rotation_geometry(e["y"]) for e in examples])
    mean=descriptors.mean(axis=0);scale=descriptors.std(axis=0);scale=np.where(scale>1e-6,scale,1.)
    x=(descriptors-mean)/scale
    _,singular,vt=np.linalg.svd(x,full_matrices=False)
    variation_rank=int(np.sum(singular>1e-8))
    k=min(8,max(1,len(examples)-1),max(1,variation_rank))
    basis=vt[:k];scores=x@basis.T
    score_scale=np.maximum(scores.std(axis=0),.1)
    residual=x-scores@basis
    residual_scale=max(float(np.sqrt(np.mean(residual**2))),.1)
    model={"mean":mean,"scale":scale,"basis":basis,"score_scale":score_scale,"residual_scale":np.asarray([residual_scale]),"train_descriptor":descriptors,"people":[e["row"]["participant_id"] for e in examples],"episode_ids":[e["row"]["episode_id"] for e in examples],"variation_rank":variation_rank}
    radii=[];count=min(minimum_people,max(1,len(set(model["people"]))-1))
    for descriptor,person in zip(descriptors,model["people"]):
        distance=_support_distances(descriptor,model)
        other={p:min(d for p0,d in zip(model["people"],distance) if p0==p) for p in set(model["people"])-{person}}
        if len(other)>=count: radii.append(sorted(other.values())[count-1])
    model["radius"]=float(np.quantile(radii,quantile)) if radii else 0.
    model["required_people"]=count
    return model


def _support_distances(descriptor,model):
    allx=(model["train_descriptor"]-model["mean"])/model["scale"]
    query=(np.asarray(descriptor)-model["mean"])/model["scale"]
    delta=allx-query
    projected=delta@model["basis"].T
    orthogonal=delta-projected@model["basis"]
    return np.sqrt(np.mean((projected/model["score_scale"])**2,axis=1)+np.mean(orthogonal**2,axis=1)/float(model["residual_scale"][0])**2)


def check_support(state,model):
    distances=_support_distances(rotation_geometry(state),model)
    per_person={p:min(float(d) for p0,d in zip(model["people"],distances) if p0==p) for p in set(model["people"])}
    ordered=sorted(per_person.values());k=model["required_people"]
    distance=ordered[k-1] if len(ordered)>=k else math.inf
    status="uninformative_joint_state_geometry" if model["variation_rank"]==0 else ("within_joint_training_manifold_support" if distance<=model["radius"]+1e-8 else "outside_joint_training_manifold_support")
    return {"status":status,"distance":distance,"radius":model["radius"],"nearby_training_people":sum(v<=model["radius"]+1e-8 for v in per_person.values()),"required_people":k}


def _bootstrap_maps(train,natives,pairs,cfg,fitted,seed):
    """Person bootstrap conditional on frozen assignments; no test observations."""
    lags=np.asarray(fitted["lags"])*cfg["delay_seconds_per_index"]
    by_id={v["row"]["simulation_id"]:v for v in natives}
    people=sorted({e["row"]["participant_id"] for e in train});index={p:i for i,p in enumerate(people)}
    D=len(lags);R=train[0]["y"].shape[1]
    xx=np.zeros((len(people),D,D));xy=np.zeros((len(people),D,R));sx=np.zeros((len(people),D));sy=np.zeros((len(people),R));total=np.zeros(len(people))
    rxx=np.zeros_like(xx);rxy=np.zeros_like(xy)
    weights=np.asarray(fitted["training_episode_weights"],float)
    # The fixed representation weights are re-used rather than bootstrapping their already fitted readout.
    for e,identity,weight in zip(train,fitted["train_assignments"],weights):
        p=index[e["row"]["participant_id"]];v=by_id[identity]
        x=_native_kernel(v,e["times"],lags)[:e["early"]];y=e["y"][:e["early"]];w=weight/len(x)
        xx[p]+=w*x.T@x;xy[p]+=w*x.T@y;sx[p]+=weight*x.mean(axis=0);sy[p]+=weight*y.mean(axis=0);total[p]+=weight
        if e["response"] is not None and identity in pairs and fitted["response_W"] is not None:
            pair=pairs[identity];q=e["response_times"]+pair["onset_s"]
            delta=_native_kernel(pair["pulse"],q,lags,pair["control"])-_native_kernel(pair["control"],q,lags)
            target=e["response"]-e["response"][e["response_times"]<0].mean(axis=0)
            rxx[p]+=weight/len(delta)*delta.T@delta;rxy[p]+=weight/len(delta)*delta.T@target
    rng=np.random.default_rng(seed);Ws=[];Wr=[]
    for _ in range(cfg["bootstrap"]):
        count=np.bincount(rng.integers(0,len(people),len(people)),minlength=len(people))
        norm=float(count@total);mx=count@sx/norm;my=count@sy/norm
        X=np.einsum('i,ijk->jk',count,xx)/norm-np.outer(mx,mx)
        Y=np.einsum('i,ijk->jk',count,xy)/norm-np.outer(mx,my)
        w=_rank_limit(np.linalg.solve(X+fitted["ridge"]*np.eye(D),Y),fitted["rank"]);Ws.append(w)
        if fitted["response_W"] is not None:
            a=np.einsum('i,ijk->jk',count,rxx)/norm;b=np.einsum('i,ijk->jk',count,rxy)/norm
            Wr.append(_rank_limit(np.linalg.solve(a+fitted["ridge"]*np.eye(D),b+fitted["ridge"]*w),fitted["rank"]))
    return np.asarray(Ws),np.asarray(Wr) if Wr else None


def _map_key(source,account,variant):
    return "map_"+hashlib.sha256((source+"\0"+account+"\0"+variant).encode()).hexdigest()[:20]


def _before_test(rows,archive,traces,pairs,cfg,residuals):
    accounts=defaultdict(list)
    for v in traces.values():
        if float(v["row"].get("input_amplitude",0))==0:
            accounts[(v["row"]["account_class"],v["row"].get("variant","unspecified"))].append(v)
    maps,metadata,validation_rows,validation_arrays,excluded=[],[],[],{},[]
    arrays={};sources={}
    lagsets=sorted({tuple(cfg["lags"]),tuple(cfg["lags"][:min(3,len(cfg["lags"]))])},key=lambda v:(len(v),v))
    for source in sorted({r["source_model_key"] for r in rows if r.get("status")=="included"}):
        train,rejected,grid=_source_examples(rows,archive,source,cfg,"train")
        excluded.extend(rejected)
        if grid is None or len({e["row"]["participant_id"] for e in train})<cfg["minimum_train"]:
            excluded.append({"source_model_key":source,"reason":"insufficient_actual_training_people"});continue
        validation,rejected,_=_source_examples(rows,archive,source,cfg,"validation",grid)
        excluded.extend(rejected)
        if len({e["row"]["participant_id"] for e in validation})<cfg["minimum_validation"]:
            excluded.append({"source_model_key":source,"reason":"insufficient_actual_validation_people"});continue
        support=joint_support(train,cfg["support_quantile"],cfg["minimum_train"])
        ready_people={p:set() for p in ("train","validation")}
        for e in train+validation:
            r=e["row"]
            if (r["source_id"],r["episode_id"],r["split"]) in residuals:ready_people[r["split"]].add(r["participant_id"])
        ready=len(ready_people["train"])>=cfg["minimum_train"] and len(ready_people["validation"])>=cfg["minimum_validation"]
        sources[source]={"train":train,"validation":validation,"grid":grid,"support":support,"representation_ready":ready,"representation_people":{k:sorted(v) for k,v in ready_people.items()}}
        for (account,variant),native in sorted(accounts.items()):
            alternatives=[]
            for options in itertools.product(lagsets,cfg["ridges"],cfg["ranks"],cfg["strengths"]):
                fitted,predictions,predicted_arrays=_fit_one(train,validation,native,pairs,cfg,options,residuals)
                objective=fitted["validation_metrics"]["objective"]
                alternatives.append((objective if objective is not None else math.inf,options,fitted,predictions,predicted_arrays))
            selected=min(alternatives,key=lambda v:(v[0],v[1]))
            _,options,fitted,predictions,predicted_arrays=selected
            key=_map_key(source,account,variant)
            fitted.update(map_key=key,source_model_key=source,account_class=account,variant=variant,representation_ready=ready)
            fitted["bootstrap_W"],fitted["bootstrap_response_W"]=_bootstrap_maps(train,native,pairs,cfg,fitted,cfg["seed"]+int(key[-6:],16))
            for name in ("W","bias","baseline_A","baseline_bias","response_W","bootstrap_W","bootstrap_response_W"):
                if fitted[name] is not None:arrays[key+"__"+name]=fitted[name]
            arrays[key+"__times_s"]=grid
            for name in ("mean","scale","basis","score_scale","residual_scale","train_descriptor"):
                arrays[key+"__support_"+name]=support[name]
            description={k:v for k,v in fitted.items() if not isinstance(v,np.ndarray)}
            description["response_W_present"]=fitted["response_W"] is not None
            for k in ("response_W","bootstrap_response_W"):description.pop(k,None)
            description["native_parameters"]=[{"simulation_id":v["row"]["simulation_id"],"config_sha256":v["row"]["config_sha256"],"parameters":{k:value for k,value in v["row"].items() if k.startswith("param_")}} for v in native if v["row"]["simulation_id"] in fitted["support_ids"]]
            description["validation_alternatives"]=[{"delay_lags":list(v[1][0]),"ridge":v[1][1],"rank":v[1][2],"strength":v[1][3],"metrics":v[2]["validation_metrics"]} for v in alternatives]
            description["joint_support"]={"radius":support["radius"],"required_people":support["required_people"],"training_episode_ids":support["episode_ids"],"training_people":support["people"],"definition":"Training-whitened full temporal Gram geometry plus orthogonal residual distance; threshold from nearest other-person training support"}
            metadata.append(description);maps.append(fitted)
            validation_rows.extend(dict(row,map_key=key,source_model_key=source,account_class=account,variant=variant) for row in predictions)
            validation_arrays.update({key+"__"+name:value for name,value in predicted_arrays.items()})
    return maps,metadata,arrays,validation_rows,validation_arrays,excluded,sources


def _candidate_predictions(maps,sources,traces,pairs,cfg):
    proposals=[(m,control_id,pair) for m in maps for control_id,pair in sorted(pairs.items()) if (pair["control"]["row"]["account_class"],pair["control"]["row"].get("variant","unspecified"))==(m["account_class"],m["variant"])]
    rows,arrays=[],{}
    for m,identity,pair in proposals[:cfg["maximum_candidates"]]:
        source=sources[m["source_model_key"]];lags=np.asarray(m["lags"])*cfg["delay_seconds_per_index"]
        control=pair["control"];pulse=pair["pulse"]
        control_state=_native_kernel(control,source["grid"],lags)@m["W"]+m["bias"]
        support=check_support(control_state,source["support"])
        response_times=np.linspace(-.9,2.9,96);q=response_times+pair["onset_s"]
        zc=_native_kernel(control,q,lags);zp=_native_kernel(pulse,q,lags,control);delta=zp-zc
        W=m["response_W"] if m["response_W"] is not None else m["W"]
        predicted_delta=delta@W
        endpoint=path_endpoint(predicted_delta,response_times)
        bootstrap=m["bootstrap_response_W"] if m["bootstrap_response_W"] is not None else m["bootstrap_W"]
        values=[path_endpoint(delta@w,response_times) for w in bootstrap]
        low,high=np.quantile(values,[.025,.975])
        candidate_id=m["map_key"]+"__"+pair["control"]["row"]["pair_id"]
        arrays[candidate_id+"__response_delta"]=predicted_delta;arrays[candidate_id+"__response_times_s"]=response_times
        rows.append({"candidate_id":candidate_id,"map_key":m["map_key"],"source_model_key":m["source_model_key"],"account_class":m["account_class"],"variant":m["variant"],"control_simulation_id":identity,"control_config_sha256":control["row"]["config_sha256"],"pulse_simulation_id":pulse["row"]["simulation_id"],"pulse_config_sha256":pulse["row"]["config_sha256"],"input_amplitude_model_units":pulse["row"]["input_amplitude"],"input_width_s":pulse["row"]["input_width_s"],"native_output_onset_s":pair["onset_s"],"clinical_control_score":control["row"]["score"],"clinical_pulse_score":pulse["row"]["score"],"clinical_score_change":float(pulse["row"]["score"])-float(control["row"]["score"]),"native_control_output_sha256":control["row"]["source_output_sha256"],"native_pulse_output_sha256":pulse["row"]["source_output_sha256"],"latent_response_path_rms_v1":endpoint,"conditional_person_bootstrap_low":float(low),"conditional_person_bootstrap_high":float(high),"joint_support_status":support["status"],"joint_support_distance":support["distance"],"joint_support_radius":support["radius"],"joint_support_people":support["nearby_training_people"],"representation_link_ready":m["representation_ready"],"observed_response_calibration_episodes":m["response_training_episodes"],"response_mapping_status":"effective_annotated_response_calibration_physical_units_unqualified" if m["response_W"] is not None else "temporal_observation_map_without_human_response_calibration","human_intervention_qualified":False,"status":"retained_supported_computational_challenge" if m["representation_ready"] and support["status"].startswith("within") else "retained_unsupported_challenge"})
    contrasts=[]
    for a,b in itertools.combinations(rows,2):
        if a["source_model_key"]!=b["source_model_key"] or a["account_class"]==b["account_class"] or a["variant"]!=b["variant"] or float(a["input_amplitude_model_units"])!=float(b["input_amplitude_model_units"]):continue
        if a["status"]!=b["status"] or a["status"]!="retained_supported_computational_challenge":continue
        contrasts.append({"account_A":a["candidate_id"],"account_B":b["candidate_id"],"endpoint_A":a["latent_response_path_rms_v1"],"endpoint_B":b["latent_response_path_rms_v1"],"endpoint_difference":a["latent_response_path_rms_v1"]-b["latent_response_path_rms_v1"],"control_clinical_score_difference":abs(float(a["clinical_control_score"])-float(b["clinical_control_score"])),"description":"Distinct actual parameter families under a matched native input; response difference is conditional on effective observation maps"})
    contrasts.sort(key=lambda r:(-abs(r["endpoint_difference"]),r["account_A"],r["account_B"]))
    diagnostic={"status":"no_qualified_human_diagnostic","chosen_candidate":None,"leading_computational_contrast":contrasts[0] if contrasts else None,"computational_contrasts":contrasts,"primary_neural_endpoint":None,"candidate_neural_endpoint":ENDPOINT,"human_intervention_qualified":False,"human_function_mapping_qualified":False,"anatomical_mapping_qualified":False,"selection_inputs":"Actual training/validation trajectories, adapted-representation residuals and native input/control outputs only; no test trajectory or response outcome used","candidate_selection_rule":"Retain bounded deterministic actual pairs, require joint other-person training support and actual adapted-representation overlap; rank supported cross-account matched-input contrasts by predicted response-path difference","uncertainty":"Participant bootstrap conditional on frozen native assignments and observation-map hyperparameters; excludes full model-selection and human intervention uncertainty","mapping_limitations":["Scalar native cortical voltage supports an effective delay/kernel map only","Source annotation does not qualify physical stimulus timing/units or a simulator-to-human intervention calibration","No randomized matched human control-response contrast or human functional observation bridge is supplied"],"prediction_scope":"Computational candidates frozen before development held-person evaluation; a feasible independent human diagnostic requires new qualified measurements"}
    return rows,arrays,diagnostic,{"actual_native_pair_proposals":len(proposals),"evaluated_pairs":len(rows),"not_evaluated_design_budget":max(0,len(proposals)-len(rows))}


def _sensitivity(maps,sources,traces,cfg):
    rows,equivalence=[],[]
    for m in maps:
        native=[traces[s] for s in m["support_ids"]];lags=np.asarray(m["lags"])*cfg["delay_seconds_per_index"]
        predicted=[_native_kernel(v,sources[m["source_model_key"]]["grid"],lags)@m["W"]+m["bias"] for v in native]
        for i,a in enumerate(native):
            if len(native)<2:
                rows.append({"map_key":m["map_key"],"simulation_id":a["row"]["simulation_id"],"status":"only_one_training_supported_configuration_no_parameter_recovery"});continue
            distances=[math.inf if j==i else float(np.mean((predicted[i]-p)**2)) for j,p in enumerate(predicted)]
            j=int(np.argmin(distances));b=native[j]
            parameter_distance=float(np.linalg.norm([math.log(float(a["row"].get(k,1)))-math.log(float(b["row"].get(k,1))) for k in ("param_gain_factor","param_coupling_factor","param_history_factor")]))
            scale=max(float(np.mean(predicted[i]**2)),1e-9);relative=math.sqrt(distances[j]/scale)
            identity={"map_key":m["map_key"],"simulation_id":a["row"]["simulation_id"],"nearest_prediction_equivalent_id":b["row"]["simulation_id"],"relative_latent_rms_difference":relative,"log_relative_parameter_distance":parameter_distance,"status":"descriptive_unidentifiable_effective_parameters" if relative<=.01 else "distinct_predictions_under_frozen_effective_map","physiological_recovery_established":False}
            rows.append(identity)
            if relative<=.01:equivalence.append(identity)
    return rows,{"descriptive_equivalence_tolerance_relative_rms":.01,"statistical_equivalence_established":False,"physiological_parameter_identification_established":False,"classes":equivalence,"interpretation":"Training-supported native configurations may have observationally similar mapped trajectories; this preserves ambiguity, not recovery of human synapses"}


def _after_freeze(rows,archive,maps,sources,traces,pairs,cfg):
    predictions,arrays,metrics,excluded=[],{},[],[]
    for source,context in sorted(sources.items()):
        test,rejected,_=_source_examples(rows,archive,source,cfg,"test",context["grid"])
        excluded.extend(rejected)
        for m in [m for m in maps if m["source_model_key"]==source]:
            native=[traces[s] for s in m["support_ids"]]
            result,predicted,summary=_predict(test,native,pairs,cfg,m,"test")
            predictions.extend(dict(r,map_key=m["map_key"],source_model_key=source,account_class=m["account_class"],variant=m["variant"]) for r in result)
            arrays.update({m["map_key"]+"__"+k:v for k,v in predicted.items()})
            metrics.append({"map_key":m["map_key"],"source_model_key":source,"account_class":m["account_class"],"variant":m["variant"],"metrics":summary,"adequate_test_people":summary["people"]>=cfg["minimum_test"],"representation_link_ready":context["representation_ready"],"no_result_direction_gate":True})
    return predictions,arrays,metrics,excluded


def _csv(path,rows,required):
    fields=sorted(set(required)|{k for r in rows for k in r})
    core.write_csv(path,rows,fields)


def _same_numeric(actual,expected,context):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise core.StudyError(context+" schema differs")
        for key,value in expected.items():_same_numeric(actual[key],value,context+"."+key)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):raise core.StudyError(context+" list differs")
        for a,b in zip(actual,expected):_same_numeric(a,b,context)
    elif isinstance(expected,(float,np.floating)):
        if not isinstance(actual,(float,int)) or not math.isfinite(float(actual)) or not math.isclose(float(actual),float(expected),rel_tol=1e-8,abs_tol=1e-10):raise core.StudyError(context+" numerical evidence was not produced by the frozen fit")
    elif actual!=expected:raise core.StudyError(context+" evidence differs")


def _check_npz(path,expected):
    with np.load(path,allow_pickle=False) as saved:
        if set(saved.files)!=set(expected):raise core.StudyError("Learned map/prediction array keys differ")
        for key,value in expected.items():
            if saved[key].shape!=np.asarray(value).shape or not np.isfinite(saved[key]).all() or not np.allclose(saved[key],value,rtol=1e-8,atol=1e-10):
                raise core.StudyError("Learned map/prediction was not generated from actual declared training and native trajectories: "+key)


def _check_csv(path,expected):
    actual=core.read_csv(path)
    if len(actual)!=len(expected):raise core.StudyError("Actual account/candidate predictions silently omit or add observations")
    for a,b in zip(actual,expected):
        for key,value in b.items():
            written="" if value is None else str(value)
            if isinstance(value,(float,np.floating)):
                number=guided._finite(a.get(key))
                if number is None or not math.isclose(number,float(value),rel_tol=1e-8,abs_tol=1e-10):raise core.StudyError("Rehashed prediction does not follow the frozen observation map")
            elif a.get(key)!=written:raise core.StudyError("Actual prediction identity or scientific status differs")


def bridge(states,fit,config,output,representations):
    states,fit,representations=map(lambda p:Path(p).resolve(),(states,fit,representations))
    config=Path(config).resolve();cfg=_config(config)
    state_receipt,rep_receipt,fit_receipt,bank,provenance,rows=_load_parents(states,fit,representations)
    traces,pairs,native_rejected=_native_bank(bank)
    residuals,_=_residuals(representations)
    out=guided._new_output(output)
    with np.load(states/"states.npz",allow_pickle=False) as archive:
        maps,descriptions,map_arrays,validation_rows,validation_arrays,excluded,sources=_before_test(rows,archive,traces,pairs,cfg,residuals)
        candidates,candidate_arrays,diagnostic,candidate_census=_candidate_predictions(maps,sources,traces,pairs,cfg)
        sensitivity,equivalence=_sensitivity(maps,sources,traces,cfg)
        np.savez_compressed(out/"observation_map.npz",**map_arrays)
        np.savez_compressed(out/"candidate_trajectories.npz",**candidate_arrays)
        selected={"schema_version":1,"maps":descriptions,"config":cfg,"source_representation_overlap":{k:{"ready":v["representation_ready"],"people":v["representation_people"]} for k,v in sources.items()},"selection_rule":"Minimum equal-person validation later multivariate state MSE plus0.25temporal-Gram loss plus0.25actual annotated-response MSE when available; no test targets; fixed grids break ties","representation_objective":"Training weights emphasize actual adapted-activation-to-state-distribution map residuals; source-matched train/validation overlap is separately required","observation_scope":"Final genuine10s only, conditioned on early65% by default. Interpolation aligns actual sampled trajectories, never adds independent observations or generated120s evidence","parameter_kind":"Existing native configuration selection plus effective scalar delay/kernel observation weights; no anatomical populations invented"}
        core.write_json(out/"model_selection.json",selected)
        core.write_json(out/"parameter_equivalence.json",equivalence)
        _csv(out/"parameter_sensitivity.csv",sensitivity,["map_key","simulation_id","status"])
        _csv(out/"candidate_predictions.csv",candidates,["candidate_id","map_key","status","control_simulation_id","pulse_simulation_id"])
        diagnostic.update(observation_map_sha256=core.sha256_file(out/"observation_map.npz"),model_selection_sha256=core.sha256_file(out/"model_selection.json"),candidate_predictions_sha256=core.sha256_file(out/"candidate_predictions.csv"),parameter_equivalence_sha256=core.sha256_file(out/"parameter_equivalence.json"),candidate_census=candidate_census)
        core.write_json(out/"diagnostic.json",diagnostic)
        split={p:sorted({r["participant_id"] for r in rows if r.get("status")=="included" and r["split"]==p}) for p in ("train","validation","test")}
        training={"schema_version":1,"status":"effective_human_state_mechanisms_fitted" if maps else "insufficient_actual_state_mechanism_training","assay_id":ASSAY_ID,"human_data":True,"human_parameter_fitting_executed":bool(maps),"native_training_executed":False,"adaptation_kind":"actual_native_configuration_and_effective_kernel_fit","training_person_ids":split["train"],"train_person_ids":split["train"],"validation_person_ids":split["validation"],"test_person_ids":split["test"],"split":split,"data_split_sha256":state_receipt.get("split_sha256",core.sha256_file(states/"split.json")),"observation_map":{"path":str(out/"observation_map.npz"),"sha256":core.sha256_file(out/"observation_map.npz")},"source_states_receipt_sha256":core.sha256_file(states/"receipt.json"),"source_fit_receipt_sha256":core.sha256_file(fit/"receipt.json"),"representation_receipt_sha256":core.sha256_file(representations/"receipt.json"),"adapted_clinical_checkpoint_sha256":rep_receipt.get("adapted_checkpoint_sha256"),"native_provenance":provenance,"maps":len(maps),"representation_qualified_source_maps":sum(m["representation_ready"] for m in maps),"native_parameter_kind":"Source-pinned existing parameter configurations; numerical variant retained; physiological identification unestablished","actual_response_fitted_episodes":sum(m["response_training_episodes"] for m in maps),"source_response_episode_denominator_note":"Episodes reused across competing accounts; this sum is not independent human sample size","test_outcomes_used_for_selection":False,"frozen_before_test_utc":datetime.now(timezone.utc).isoformat(),"candidate_freeze_sha256":core.sha256_file(out/"diagnostic.json"),"source_code_sha256":core.sha256_file(__file__),"historical_exposure":"Public development states and parent analyses already exposed; this stage does not establish independent protected human validation"}
        training["scientific_status"]="actual_state_representation_effective_mechanism_bridge_fitted" if any(m["representation_ready"] for m in maps) else "insufficient_adapted_representation_overlap_for_full_mechanistic_bridge"
        training["representation_geometry_mechanism_bridge_ready"]=any(m["representation_ready"] for m in maps)
        core.write_json(out/"training_receipt.json",training)
        # Selection artifacts have been written and hashed before loading test state arrays.
        test_rows,test_arrays,metrics,test_excluded=_after_freeze(rows,archive,maps,sources,traces,pairs,cfg)
    all_predictions=validation_rows+test_rows
    np.savez_compressed(out/"state_predictions.npz",**(validation_arrays|test_arrays))
    _csv(out/"account_predictions.csv",all_predictions,["map_key","episode_id","participant_id","source_id","split","simulation_id","later_state_standardized_mse"])
    _csv(out/"excluded_observations.csv",excluded+test_excluded,["episode_id","participant_id","source_model_key","split","reason"])
    _csv(out/"native_pair_failures.csv",native_rejected,["simulation_id","reason"])
    evaluation={"schema_version":1,"status":"development_held_person_multivariate_trajectories_evaluated" if test_rows else "insufficient_actual_heldout_state_trajectories","assay_id":ASSAY_ID,"training_receipt_sha256":core.sha256_file(out/"training_receipt.json"),"observation_map_sha256":training["observation_map"]["sha256"],"data_split_sha256":training["data_split_sha256"],"test_person_ids":sorted({r["participant_id"] for r in test_rows}),"test_observation_ids":sorted({r["episode_id"] for r in test_rows}),"metrics":metrics,"account_predictions_sha256":core.sha256_file(out/"account_predictions.csv"),"actual_test_response_predictions":sum(r["response_standardized_mse"] is not None for r in test_rows),"independent_human_validation":False,"human_function_mapping_qualified":False,"human_intervention_qualified":False,"anatomical_mapping_qualified":False,"evaluated_utc":datetime.now(timezone.utc).isoformat(),"failed_or_weak_predictions_retained":True,"uncertainty":"Conditional participant bootstrap of effective observation maps; source-specific coordinate systems and response annotation limitations retained"}
    core.write_json(out/"evaluation_receipt.json",evaluation)
    inputs=[states/"receipt.json",states/"rows.csv",states/"states.npz",states/"models.npz",fit/"receipt.json",fit/"scored_bank.csv",representations/"receipt.json",representations/"episode_representation_residuals.csv",config]
    return guided._finish(out,inputs,"manifold_mechanistic_bridge_completed",input_paths={"states":str(states),"fit":str(fit),"representations":str(representations),"config":str(config)},human_parameter_fitting_executed=bool(maps),scientific_status=training["scientific_status"],assay_id=ASSAY_ID,human_function_mapping_qualified=False,human_intervention_qualified=False,actual_native_trajectories=len(traces),actual_native_pairs=len(pairs),source_code_sha256=core.sha256_file(__file__),denominator_source_episodes=len(rows),mechanistically_fitted_source_keys=sorted(sources),test_prediction_episodes=len({r["episode_id"] for r in test_rows}))


def verify_bridge(directory):
    """Re-fit training/validation maps and regenerate predictions; hashes alone do not suffice."""
    root=Path(directory).resolve();receipt=guided._verify_output(root)
    if receipt.get("status")!="manifold_mechanistic_bridge_completed" or receipt.get("source_code_sha256")!=core.sha256_file(__file__):raise core.StudyError("Missing frozen mechanistic bridge/code provenance")
    paths=receipt["input_paths"];states,fit,representations=(Path(paths[k]).resolve() for k in ("states","fit","representations"));config=Path(paths["config"]).resolve()
    state_receipt,rep_receipt,fit_receipt,bank,provenance,rows=_load_parents(states,fit,representations)
    cfg=_config(config);traces,pairs,native_rejected=_native_bank(bank);residuals,_=_residuals(representations)
    training=core.read_json(root/"training_receipt.json");evaluation=core.read_json(root/"evaluation_receipt.json")
    if training.get("test_outcomes_used_for_selection") is not False or training.get("native_training_executed") is not False or training.get("assay_id")!=ASSAY_ID or training.get("native_provenance")!=provenance:
        raise core.StudyError("Mechanistic training kind/native evidence differs")
    if Path(training["observation_map"]["path"]).resolve()!=root/"observation_map.npz" or training["observation_map"]["sha256"]!=core.sha256_file(root/"observation_map.npz") or training["candidate_freeze_sha256"]!=core.sha256_file(root/"diagnostic.json"):
        raise core.StudyError("Mechanistic map/candidate freeze not bound to actual artifacts")
    with np.load(states/"states.npz",allow_pickle=False) as archive:
        maps,descriptions,map_arrays,validation_rows,validation_arrays,excluded,sources=_before_test(rows,archive,traces,pairs,cfg,residuals)
        _check_npz(root/"observation_map.npz",map_arrays)
        selected=core.read_json(root/"model_selection.json")
        _same_numeric(selected["maps"],descriptions,"Learned configuration/kernel selection")
        if selected.get("config")!=cfg:raise core.StudyError("Mechanistic map grid differs from declared source config")
        candidates,candidate_arrays,diagnostic,census=_candidate_predictions(maps,sources,traces,pairs,cfg)
        _check_npz(root/"candidate_trajectories.npz",candidate_arrays);_check_csv(root/"candidate_predictions.csv",candidates)
        sensitivity,equivalence=_sensitivity(maps,sources,traces,cfg)
        _same_numeric(core.read_json(root/"parameter_equivalence.json"),equivalence,"Parameter ambiguity")
        _check_csv(root/"parameter_sensitivity.csv",sensitivity)
        saved_diagnostic=core.read_json(root/"diagnostic.json")
        for key,value in diagnostic.items():_same_numeric(saved_diagnostic.get(key),value,"Pretest diagnostic "+key)
        if saved_diagnostic.get("candidate_census")!=census:raise core.StudyError("Bounded actual candidate census differs")
        for name,key in (("observation_map.npz","observation_map_sha256"),("model_selection.json","model_selection_sha256"),("candidate_predictions.csv","candidate_predictions_sha256"),("parameter_equivalence.json","parameter_equivalence_sha256")):
            if saved_diagnostic.get(key)!=core.sha256_file(root/name):raise core.StudyError("Diagnostic freeze references different learned artifacts")
        test_rows,test_arrays,metrics,test_excluded=_after_freeze(rows,archive,maps,sources,traces,pairs,cfg)
    _check_npz(root/"state_predictions.npz",validation_arrays|test_arrays)
    _check_csv(root/"account_predictions.csv",validation_rows+test_rows)
    _check_csv(root/"excluded_observations.csv",excluded+test_excluded)
    _check_csv(root/"native_pair_failures.csv",native_rejected)
    _same_numeric(evaluation.get("metrics"),metrics,"Actual held-person evaluation")
    split={p:sorted({r["participant_id"] for r in rows if r.get("status")=="included" and r["split"]==p}) for p in ("train","validation","test")}
    if training.get("split")!=split or training.get("training_person_ids")!=split["train"] or any(training.get(p+"_person_ids")!=split[p] for p in split) or training.get("maps")!=len(maps) or training.get("human_parameter_fitting_executed")!=bool(maps) or training.get("representation_qualified_source_maps")!=sum(m["representation_ready"] for m in maps) or training.get("data_split_sha256")!=state_receipt.get("split_sha256"):
        raise core.StudyError("Actual human fitting/split completion differs")
    expected_status="actual_state_representation_effective_mechanism_bridge_fitted" if any(m["representation_ready"] for m in maps) else "insufficient_adapted_representation_overlap_for_full_mechanistic_bridge"
    if training.get("scientific_status")!=expected_status or receipt.get("scientific_status")!=expected_status or training.get("representation_geometry_mechanism_bridge_ready")!=any(m["representation_ready"] for m in maps):
        raise core.StudyError("Missing adapted-representation overlap cannot qualify the full mechanism bridge")
    if training.get("source_states_receipt_sha256")!=core.sha256_file(states/"receipt.json") or training.get("source_fit_receipt_sha256")!=core.sha256_file(fit/"receipt.json") or training.get("representation_receipt_sha256")!=core.sha256_file(representations/"receipt.json") or training.get("adapted_clinical_checkpoint_sha256")!=rep_receipt.get("adapted_checkpoint_sha256"):
        raise core.StudyError("Actual neural/native/adapted representation parent differs")
    if evaluation.get("training_receipt_sha256")!=core.sha256_file(root/"training_receipt.json") or evaluation.get("observation_map_sha256")!=training["observation_map"]["sha256"] or evaluation.get("data_split_sha256")!=training["data_split_sha256"] or evaluation.get("account_predictions_sha256")!=core.sha256_file(root/"account_predictions.csv") or evaluation.get("test_observation_ids")!=sorted({r["episode_id"] for r in test_rows}) or evaluation.get("test_person_ids")!=sorted({r["participant_id"] for r in test_rows}):
        raise core.StudyError("Actual frozen-map test checkpoint/observations differ")
    if datetime.fromisoformat(evaluation["evaluated_utc"])<datetime.fromisoformat(training["frozen_before_test_utc"]):raise core.StudyError("Evaluation predates the recorded candidate freeze")
    if any(evaluation.get(key) is not False for key in ("independent_human_validation","human_function_mapping_qualified","human_intervention_qualified","anatomical_mapping_qualified")):
        raise core.StudyError("Effective development map cannot acquire unmeasured human/anatomical qualification")
    return training


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("states","fit","config","output","representations"):parser.add_argument("--"+key,type=Path,required=True)
    try:
        result=bridge(**vars(parser.parse_args(argv)))
        print(json.dumps({"status":result["status"],"human_parameter_fitting_executed":result["human_parameter_fitting_executed"]}))
    except (core.StudyError,ValueError,OSError,KeyError,OverflowError,FloatingPointError) as exc:parser.exit(2,"Neural-state mechanism bridge failed: "+str(exc)+"\n")


if __name__=="__main__":main()
