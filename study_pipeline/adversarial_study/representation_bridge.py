"""Held-person mappings from actual human-adapted detector representations.

Same-person, same-channel hidden activation distances predict differences in
measured human neural state distributions, temporal descriptors and original
human outcomes. Detector activations are not brain coordinates. Human labels
never define the neural manifold. The released clinical instrument is only an
optional descriptive reference; numerical representation maps use the actually
human-finetuned copy and exclude its training/validation people from map tests.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import core
from .acquisition import canonical_hash


SPECTRAL=("feature_spectral_centroid_hz","feature_spectral_entropy","state_covariance_effective_rank","state_covariance_participation_rank")
DYNAMICS=("state_radius_mean","state_radius_std","state_displacement_100ms","state_displacement_500ms",
          "state_recurrence_fraction","state_dwell_median_s","state_dwell_max_s","state_transition_rate_hz",
          "state_occupancy_q0","state_occupancy_q1","state_occupancy_q2","state_occupancy_q3")
NEURAL_TARGETS=("state_displacement_100ms","state_displacement_500ms","state_dwell_median_s","state_transition_rate_hz")
PREDICTORS=("score_only","spectral_covariance","geometry_dynamics","activation","combined")


def _finite(value):
    try:result=float(value)
    except (TypeError,ValueError):return None
    return result if math.isfinite(result) else None


def _own(directory,name):
    root=Path(directory).resolve();part=Path(name)
    if part.is_absolute() or ".." in part.parts:raise core.StudyError("Representation artifact escapes its output role")
    path=(root/part).resolve()
    if not path.is_relative_to(root) or not path.is_file():raise core.StudyError("Representation artifact missing/outside its role")
    return path


def _geometry_rows(path):
    import csv
    with Path(path).open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream):yield {k:v.strip() for k,v in row.items()}


def effective_config(config):
    section=config.get("representations",config.get("representation_bridge",{}))
    aliases={"ridge_grid":"ridge_alphas","minimum_training_people":"minimum_train_people","minimum_validation_people":"minimum_validation_people",
             "minimum_test_people":"minimum_test_people","bootstrap_samples":"bootstrap_replicates"}
    defaults={"ridge_grid":[.001,.01,.1,1.],"minimum_training_people":3,"minimum_validation_people":1,"minimum_test_people":2,
              "bootstrap_samples":500,"maximum_pairs_per_participant":2000,"maximum_pair_observations":200000,
              "trajectory_samples":32,"seed":config.get("seed",20261004),
              "numeric_outcomes":["capacity_value","awareness_rating_raw","detection","confidence","performance"],
              "nominal_outcomes":["report_category","experience_report"]}
    result={}
    for key,default in defaults.items():
        alias=aliases.get(key,key)
        if key!=alias and key in section and alias in section and section[key]!=section[alias]:raise core.StudyError("Conflicting representation configuration aliases")
        result[key]=section.get(key,section.get(alias,default))
    if not result["ridge_grid"] or any(_finite(v) is None or float(v)<=0 for v in result["ridge_grid"]):raise core.StudyError("Positive finite ridge grid required")
    for key,upper in [("minimum_training_people",10000),("minimum_validation_people",10000),("minimum_test_people",10000),
                      ("bootstrap_samples",10000),("maximum_pairs_per_participant",10000),("maximum_pair_observations",500000),("trajectory_samples",64)]:
        value=result[key]
        if isinstance(value,bool) or not isinstance(value,int) or not 1<=value<=upper:raise core.StudyError("Unbounded representation configuration: "+key)
    result["ridge_grid"]=sorted(set(float(v) for v in result["ridge_grid"]))
    return result


def _verified_inputs(states,clinical_trained,adapted_geometry,config_path,geometry):
    from native_tools.human_finetune import verify_adaptation,validate_split
    from native_tools.adapted_geometry import verify_adapted_geometry
    from .neural_state_space import verify_state_space
    clinical=verify_adaptation(Path(clinical_trained).resolve())
    state=verify_state_space(Path(states).resolve());adapted=verify_adapted_geometry(Path(adapted_geometry).resolve())
    if Path(adapted["states_path"]).resolve()!=Path(states).resolve() or Path(adapted["trained_path"]).resolve()!=Path(clinical_trained).resolve():
        raise core.StudyError("Actual adapted geometry belongs to different states/trained human model")
    if adapted["adapted_checkpoint"]!=clinical["trained_checkpoint"]:raise core.StudyError("Numerical representation comparison is not using the actual human-adapted checkpoint")
    inputs={"states":str(Path(states).resolve()),"clinical_trained":str(Path(clinical_trained).resolve()),
            "adapted_geometry":str(Path(adapted_geometry).resolve()),"config":str(Path(config_path).resolve()),
            "geometry":[str(Path(path).resolve()) for path in geometry]}
    pins={"states":core.sha256_file(Path(states)/"receipt.json"),"clinical_training":core.sha256_file(Path(clinical_trained)/"training_receipt.json"),
          "adapted_geometry":core.sha256_file(Path(adapted_geometry)/"receipt.json"),"config":core.sha256_file(config_path)}
    # Frozen references cannot enter numeric candidates, select a layer or supply
    # substitute vectors. Their source receipts remain visible for discovery.
    references=[]
    for path in geometry:
        root=Path(path).resolve();receipt=core.read_json(_own(root,"receipt.json"))
        references.append({"path":str(root),"receipt_sha256":core.sha256_file(root/"receipt.json"),"status":receipt.get("status"),
                           "role":"descriptive_released_instrument_reference_only; excludedfromnumericadaptedcomparison"})
    rows=core.read_csv(Path(states)/"rows.csv");split=core.read_json(Path(states)/"split.json");roles=validate_split(split)
    clinical_roles=validate_split(clinical["split"])
    for row in rows:
        if row.get("status")=="included" and roles.get(row["participant_id"])!=row.get("split"):
            raise core.StudyError("Human state row differs from its frozen participant split")
    return inputs,pins,references,rows,roles,clinical_roles,clinical


def _weights(rows):
    counts=Counter(r["participant_id"] for r in rows);w=np.asarray([1./counts[r["participant_id"]] for r in rows])
    return w/w.sum()


def fit_ridge(x,y,people,alpha):
    """Each training person supplies equal total weight; test never sets scales."""
    rows=[{"participant_id":p} for p in people];w=_weights(rows)
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    means=w@x;scale=np.sqrt(w@((x-means)**2));scale[scale<=1e-12]=1.
    design=np.c_[np.ones(len(x)),(x-means)/scale];penalty=np.eye(design.shape[1]);penalty[0,0]=0.
    beta=np.linalg.solve(design.T@(w[:,None]*design)+float(alpha)*penalty,design.T@(w[:,None]*y if y.ndim==2 else w*y))
    return {"means":means,"scale":scale,"beta":beta}


def predict_ridge(model,x):
    x=np.asarray(x,dtype=float);return np.c_[np.ones(len(x)),(x-model["means"])/model["scale"]]@model["beta"]


def _softmax(values):
    values=np.asarray(values,dtype=float);values-=values.max(axis=1,keepdims=True);p=np.exp(values);return p/p.sum(axis=1,keepdims=True)


def _metric(rows,truth,predicted,kind,classes):
    if kind=="numeric":loss=(np.asarray(predicted)-np.asarray(truth))**2
    else:
        probabilities=_softmax(predicted);indices={label:i for i,label in enumerate(classes)}
        loss=np.asarray([-math.log(max(float(probabilities[i,indices[label]]) if label in indices else 0.,1e-12)) for i,label in enumerate(truth)])
    return float(_weights(rows)@loss),loss


def _ci(values,seed,count):
    values=np.asarray(values,dtype=float)
    if len(values)<2:return [None,None]
    rng=np.random.default_rng(seed);samples=values[rng.integers(0,len(values),size=(count,len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(samples,[.025,.975])]


def _neural_distribution_distance(a,b,bandwidth,count):
    # Actual uniformly spaced observed states; no interpolation or cross-montage
    # coordinates. Biased empirical RBF MMD measures state distributions, not
    # temporal correspondence. Dynamic targets separately retain ordering.
    a=a[np.linspace(0,len(a)-1,min(count,len(a)),dtype=int)];b=b[np.linspace(0,len(b)-1,min(count,len(b)),dtype=int)]
    def kernel(x,y):
        distance=np.maximum((x*x).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*x@y.T,0.)
        return np.exp(-distance/(2*bandwidth**2)).mean()
    return float(np.sqrt(max(0.,kernel(a,a)+kernel(b,b)-2*kernel(a,b))))


def _pair_data(states,adapted_geometry,rows,roles,clinical_roles,cfg):
    included={(r["participant_id"],r["source_id"],r["episode_id"]):r for r in rows if r.get("status")=="included"}
    if len(included)!=sum(r.get("status")=="included" for r in rows):raise core.StudyError("Duplicate actual human state identity")
    layers=core.read_json(Path(adapted_geometry)/"receipt.json")["layers"]
    grouped=defaultdict(lambda:defaultdict(list));census=[];byperson=defaultdict(set)
    for row in _geometry_rows(Path(adapted_geometry)/"layer_distances.csv"):
        person=row["participant_id"];source=row["source_id"];a,b=sorted([row["episode_id_a"],row["episode_id_b"]])
        if a==b:continue
        key=(person,source,a,b)
        if (person,source,a) not in included or (person,source,b) not in included:raise core.StudyError("Adapted activation pair lacks exact measured human state endpoints")
        if row["layer_name"] not in layers:raise core.StudyError("Unknown actual adapted layer")
        byperson[person].add(key)
    admitted=set()
    for person,keys in byperson.items():
        order=sorted(keys,key=lambda key:hashlib.sha256((str(cfg["seed"])+"\0"+json.dumps(key)).encode()).hexdigest())
        admitted.update(order[:cfg["maximum_pairs_per_participant"]])
    if len(admitted)>cfg["maximum_pair_observations"]:raise core.StudyError("Representation pair budget exceeded; no outcome-dependent thinning")
    # Two source-table passes retain pair identities for the full census and
    # only bounded admitted scalar distances, never every channel-level row.
    for row in _geometry_rows(Path(adapted_geometry)/"layer_distances.csv"):
        a,b=sorted([row["episode_id_a"],row["episode_id_b"]]);key=row["participant_id"],row["source_id"],a,b
        if key in admitted and row.get("metric_status")=="computed":
            value=_finite(row.get("cosine_distance"))
            if value is not None:grouped[key][row["layer_name"]].append(value)
    predictions=core.read_csv(Path(adapted_geometry)/"adapted_predictions.csv");jobs={r["job_id"]:r for r in core.read_csv(Path(adapted_geometry)/"adapted_jobs.csv")}
    head=defaultdict(list)
    for row in predictions:
        if row["status"]=="ok":
            job=jobs[row["job_id"]];p=np.asarray(json.loads(row["human_class_probabilities"]),dtype=float).reshape(-1)
            if not np.isfinite(p).all() or np.any(p<0) or abs(p.sum()-1)>1e-5:raise core.StudyError("Actual adapted head probability invalid")
            head[job["participant_id"],job["source_id"],job["episode_id"]].append(p)
    with np.load(Path(states)/"states.npz",allow_pickle=False) as archive:
        trajectories={key:np.asarray(archive[row["state_token"]+"__state"],dtype=float) for key,row in included.items()}
    bandwidths={};domains=defaultdict(list)
    for key,row in included.items():
        if roles[key[0]]=="train":domains[row["source_id"],row["source_model_key"]].append(key)
    for domain,keys in domains.items():
        persons=defaultdict(list)
        for key in keys:
            x=trajectories[key]
            if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():raise core.StudyError("Actual measured human state trajectory invalid")
            persons[key[0]].append(float(np.sqrt(2*np.var(x,axis=0,ddof=1).sum())))
        bandwidth=float(np.median([np.mean(v) for v in persons.values()]))
        if math.isfinite(bandwidth) and bandwidth>1e-12:bandwidths[domain]=bandwidth
    output=[]
    all_keys=sorted(key for keys in byperson.values() for key in keys)
    for key in all_keys:
        bylayer=grouped.get(key,{})
        person,source,ea,eb=key;a=included[person,source,ea];b=included[person,source,eb];role=roles[person]
        reason=""
        if key not in admitted:reason="fixed_metadata_only_pair_budget"
        elif role=="test" and clinical_roles.get(person) in ("train","validation"):reason="clinical_adaptation_train_or_validation_person_excluded_from_comparative_test"
        elif a["source_model_key"]!=b["source_model_key"]:reason="different_actual_neural_coordinate_domains"
        elif (source,a["source_model_key"]) not in bandwidths:reason="no_training_person_neural_scale"
        distances={}
        for layer in layers:
            actual=bylayer.get(layer,[])
            if actual:distances[layer]=float(np.mean(actual))
        if not reason and set(distances)!=set(layers):reason="missing_actual_adapted_activation_layer"
        if not reason and ((person,source,ea) not in head or (person,source,eb) not in head):reason="missing_actual_adapted_task_readout"
        record={"participant_id":person,"source_id":source,"episode_a":ea,"episode_b":eb,"split":role,
                "source_model_key":a["source_model_key"],"status":"excluded" if reason else "included","reason":reason,
                "original_report_a":a.get("report_category",a.get("experience_report","")),"original_report_b":b.get("report_category",b.get("experience_report","")),
                "candidate_layers":len(layers),"available_layers":len(distances) if key in admitted else None,"clinical_split":clinical_roles.get(person,"source_held_out")}
        census.append(record)
        if reason:continue
        score_a=np.mean(head[person,source,ea],axis=0);score_b=np.mean(head[person,source,eb],axis=0)
        if score_a.shape!=score_b.shape:raise core.StudyError("Actual adapted readout class geometry differs")
        features={"adapted_probability_l2_difference":float(np.linalg.norm(score_a-score_b))}
        for field in SPECTRAL+DYNAMICS:
            va,vb=_finite(a.get(field)),_finite(b.get(field))
            if va is not None and vb is not None:features[field]=abs(vb-va)
        targets={"state_distribution_mmd":("numeric",_neural_distribution_distance(trajectories[person,source,ea],trajectories[person,source,eb],bandwidths[source,a["source_model_key"]],cfg["trajectory_samples"]),"measured_neural_state_distribution",None,None)}
        for field in NEURAL_TARGETS+tuple(cfg["numeric_outcomes"]):
            va,vb=_finite(a.get(field)),_finite(b.get(field))
            targets[field]=("numeric",abs(vb-va) if va is not None and vb is not None else None,
                            "measured_neural_dynamic_difference" if field in NEURAL_TARGETS else "original_human_numeric_outcome_difference",a.get(field),b.get(field))
        for field in cfg["nominal_outcomes"]:
            va,vb=a.get(field,""),b.get(field,"")
            targets[field]=("nominal",json.dumps([va,vb],ensure_ascii=False) if va and vb else None,"original_nominal_report_pair",va,vb)
        output.append({**record,"features":features,"activations":distances,"targets":targets})
    return output,census,bandwidths,layers


def _analyse(states,clinical_trained,adapted_geometry,config_path,geometry):
    inputs,pins,references,rows,roles,clinical_roles,clinical=_verified_inputs(states,clinical_trained,adapted_geometry,config_path,geometry)
    cfg=effective_config(core.read_json(config_path));pairs,census,bandwidths,layers=_pair_data(states,adapted_geometry,rows,roles,clinical_roles,cfg)
    maps={};selections=[];predictions=[];metrics=[];missing=[];domains=defaultdict(list)
    for row in pairs:domains[row["source_id"],row["source_model_key"]].append(row)
    for (source,domain),observations in sorted(domains.items()):
        target_names=sorted({field for row in observations for field in row["targets"]})
        for target in target_names:
            valid=[r for r in observations if r["targets"][target][1] is not None]
            for role in ["train","validation","test"]:
                part=[r for r in observations if r["split"]==role];present=[r for r in valid if r["split"]==role]
                missing.append({"source_id":source,"source_model_key":domain,"target":target,"split":role,"pair_observations":len(part),
                                "observed_outcomes":len(present),"missing_outcomes":len(part)-len(present),"independent_people":len({r["participant_id"] for r in present})})
            parts={role:[r for r in valid if r["split"]==role] for role in ["train","validation","test"]}
            needed={"train":cfg["minimum_training_people"],"validation":cfg["minimum_validation_people"],"test":cfg["minimum_test_people"]}
            if any(len({r["participant_id"] for r in parts[role]})<needed[role] for role in needed):
                metrics.append({"source_id":source,"source_model_key":domain,"target":target,"status":"insufficient_independent_people","people":{r:len({v["participant_id"] for v in parts[r]}) for r in parts}});continue
            kind=valid[0]["targets"][target][0];semantics=valid[0]["targets"][target][2]
            train,validation,test=(parts[r] for r in ["train","validation","test"])
            classes=sorted({r["targets"][target][1] for r in train}) if kind=="nominal" else []
            if kind=="nominal" and len(classes)<2:
                metrics.append({"source_id":source,"source_model_key":domain,"target":target,"status":"insufficient_training_original_categories","classes":classes});continue
            # All predictors face identical target-eligible pairs. Only fields
            # actually complete across these pairs enter a comparator; missing
            # outcome counts and unavailable fields remain explicit metadata.
            available=set.intersection(*(set(r["features"]) for r in train));available.discard(target)
            complete=[r for r in valid if available.issubset(r["features"])]
            if len(complete)!=len(valid):
                for role in ["train","validation","test"]:
                    metrics.append({"source_id":source,"source_model_key":domain,"target":target,"split":role,
                                    "status":"missing_training_defined_comparator_measurements",
                                    "excluded_pairs":sum(r["split"]==role and not available.issubset(r["features"]) for r in valid)})
                parts={role:[r for r in complete if r["split"]==role] for role in ["train","validation","test"]}
                if any(len({r["participant_id"] for r in parts[role]})<needed[role] for role in needed):continue
                train,validation,test=(parts[r] for r in ["train","validation","test"])
            fields={"score_only":["adapted_probability_l2_difference"],"spectral_covariance":[f for f in SPECTRAL if f in available],
                    "geometry_dynamics":[f for f in DYNAMICS if f in available],"activation":[],
                    "combined":["adapted_probability_l2_difference"]+[f for f in SPECTRAL+DYNAMICS if f in available]}
            actual=lambda records:[r["targets"][target][1] for r in records]
            y=np.asarray(actual(train),dtype=float) if kind=="numeric" else np.asarray([[float(label==r["targets"][target][1]) for label in classes] for r in train])
            for predictor in PREDICTORS:
                if not fields[predictor] and predictor!="activation":
                    metrics.append({"source_id":source,"source_model_key":domain,"target":target,"predictor":predictor,"status":"missing_measured_comparator_features"});continue
                candidates=[]
                for layer in layers if predictor in ("activation","combined") else [None]:
                    def matrix(records):return np.asarray([[r["features"][f] for f in fields[predictor]]+([r["activations"][layer]] if layer else []) for r in records],dtype=float)
                    x=matrix(train)
                    for alpha in cfg["ridge_grid"]:
                        model=fit_ridge(x,y,[r["participant_id"] for r in train],alpha)
                        value,_=_metric(validation,actual(validation),predict_ridge(model,matrix(validation)),kind,classes)
                        candidates.append((value,"" if layer is None else layer,alpha,model))
                value,layer,alpha,model=min(candidates,key=lambda item:(item[0],item[1],item[2]));selected_layer=layer or None
                name="map_"+hashlib.sha256(json.dumps([source,domain,target,predictor]).encode()).hexdigest()[:20]
                maps.update({name+"__"+key:value for key,value in model.items()})
                selection={"map_id":name,"source_id":source,"source_model_key":domain,"target":target,"target_kind":kind,"semantics":semantics,
                           "predictor":predictor,"features":fields[predictor],"selected_layer":selected_layer,"ridge_alpha":alpha,"classes":classes,
                           "training_people":sorted({r["participant_id"] for r in train}),"validation_people":sorted({r["participant_id"] for r in validation}),
                           "validation_equal_participant_loss":value,"test_used_for_selection":False,"no_result_direction_gate":True}
                selections.append(selection)
                for role,records in [("train",train),("validation",validation),("test",test)]:
                    x=np.asarray([[r["features"][f] for f in fields[predictor]]+([r["activations"][selected_layer]] if selected_layer else []) for r in records])
                    estimates=predict_ridge(model,x);mean_loss,losses=_metric(records,actual(records),estimates,kind,classes)
                    probabilities=_softmax(estimates) if kind=="nominal" else None;by_person=defaultdict(list);class_errors=defaultdict(list)
                    for i,row in enumerate(records):
                        truth=row["targets"][target][1];prediction=float(estimates[i]) if kind=="numeric" else classes[int(np.argmax(probabilities[i]))]
                        by_person[row["participant_id"]].append(float(losses[i]))
                        if kind=="nominal":class_errors[row["participant_id"]].append(float(prediction!=truth))
                        predictions.append({k:row[k] for k in ["participant_id","source_id","source_model_key","episode_a","episode_b"]}|{
                            "split":role,"target":target,"target_kind":kind,"semantics":semantics,"predictor":predictor,"selected_layer":selected_layer,
                            "actual":truth,"predicted":prediction,"loss":float(losses[i]),"original_outcome_a":row["targets"][target][3],
                            "original_outcome_b":row["targets"][target][4],"probabilities":json.dumps(probabilities[i].tolist()) if probabilities is not None else ""})
                    persons=[np.mean(v) for _,v in sorted(by_person.items())];seed=cfg["seed"]+int(name[-6:],16)
                    metrics.append({"source_id":source,"source_model_key":domain,"target":target,"predictor":predictor,"split":role,
                                    "status":"development_held_person_map_evaluated" if role=="test" else "training_or_selection_descriptive",
                                    "target_kind":kind,"semantics":semantics,"independent_people":len(persons),"dependent_pairs":len(records),
                                    "equal_participant_loss":mean_loss,"participant_bootstrap_95ci":_ci(persons,seed,cfg["bootstrap_samples"]),
                                    "loss_kind":"original_numeric_difference_MSE" if kind=="numeric" else "original_report_pair_multiclass_log_loss",
                                    "equal_participant_misclassification":float(np.mean([np.mean(v) for v in class_errors.values()])) if class_errors else None,
                                    "original_categories":classes,"unseen_original_categories":sorted(set(actual(records))-set(classes)) if kind=="nominal" else []})
    residuals=defaultdict(list)
    for row in predictions:
        for endpoint in ["episode_a","episode_b"]:
            key=tuple(row[k] for k in ["participant_id","source_id","source_model_key","split","target","target_kind","predictor"])+(row[endpoint],)
            residuals[key].append(row["loss"])
    episodes=[dict(zip(["participant_id","source_id","source_model_key","split","target","target_kind","predictor","episode_id"],key),
                   paired_observations=len(values),mean_pair_loss=float(np.mean(values)),semantics="endpointattributionofrealpairwisepredictionerror; notabsolutecoordinate reconstruction")
              for key,values in sorted(residuals.items())]
    summary={"effective_configuration":cfg,"sources":sorted({r["source_id"] for r in rows}),"included_pairs":len(pairs),"pair_census":dict(Counter(r["reason"] or "included" for r in census)),
             "clinical_training_people":clinical["split"]["train_participant_ids"],"clinical_validation_people":clinical["split"]["validation_participant_ids"],
             "comparative_test_people":sorted({r["participant_id"] for r in pairs if r["split"]=="test"}),
             "neural_distribution_bandwidths":[{"source_id":s,"source_model_key":d,"bandwidth":v,"fit_partition":"train"} for (s,d),v in sorted(bandwidths.items())],
             "original_outcome_missingness":missing,"metrics":metrics,"maps_fitted":len(selections),"descriptive_frozen_references":references,
             "semantics":"Pairwise predictive alignment of actual adapted detector hidden representations with independently measured same-domain human states/outcomes; notidentity of representations or consciousnessdiagnosis",
             "future_cross_system_comparison":"Requires corresponding human assay and protected paired tests for every other system; this bridge alone does not establish that mapping"}
    return inputs,pins,maps,selections,predictions,episodes,census,summary


def run(states,clinical_trained,adapted_geometry,config,output,*,geometry=()):
    root=Path(output).resolve()
    if root.exists():raise core.StudyError("Representation output exists; preserve previous results")
    inputs,pins,maps,selections,predictions,episodes,census,summary=_analyse(states,clinical_trained,adapted_geometry,config,geometry)
    root.mkdir(parents=True);np.savez(root/"map_weights.npz",**maps)
    core.write_json(root/"map_selection.json",{"maps":selections});core.write_json(root/"representation_summary.json",summary)
    core.write_csv(root/"pair_predictions.csv",predictions,None if predictions else ["participant_id","source_id","episode_a","episode_b","split","target","predictor","actual","predicted","loss"])
    core.write_csv(root/"episode_representation_residuals.csv",episodes,None if episodes else ["participant_id","source_id","episode_id","split","target","predictor","mean_pair_loss"])
    core.write_csv(root/"pair_eligibility.csv",census,None if census else ["participant_id","source_id","episode_a","episode_b","split","status","reason"])
    trained={"status":"actual_adapted_representation_maps_fitted" if selections else "insufficient_qualified_representation_maps","human_data":True,
             "actual_human_adapted_checkpoint":True,"clinical_training_receipt_sha256":pins["clinical_training"],"maps":len(selections),
             "map_weights":{"path":str(root/"map_weights.npz"),"sha256":core.sha256_file(root/"map_weights.npz")},
             "map_selection_sha256":core.sha256_file(root/"map_selection.json"),"partition_rule":"Train-only map/scales; validation-only ridge/layer selection; no refit on validation/test people",
             "test_outcomes_used_for_selection":False,"scope":"Actual human adapted clinical representation mapping; not additional backbone fine-tuning or global consciousness classification"}
    core.write_json(root/"training_receipt.json",trained)
    evaluation={"status":"actual_adapted_representation_held_person_evaluation" if selections else "insufficient_qualified_representation_evaluation",
                "test_people":summary["comparative_test_people"],"clinical_train_or_validation_people_in_test":False,
                "predictions":{"path":str(root/"pair_predictions.csv"),"sha256":core.sha256_file(root/"pair_predictions.csv")},
                "training_receipt_sha256":core.sha256_file(root/"training_receipt.json"),"metrics":summary["metrics"],"synthetic":False}
    core.write_json(root/"evaluation_receipt.json",evaluation)
    receipt={"schema_version":1,"status":"human_adapted_representation_bridge_completed","input_paths":inputs,"input_receipts_sha256":pins,
             "clinical_training_receipt_sha256":pins["clinical_training"],
             "adapted_checkpoint_sha256":core.read_json(Path(clinical_trained)/"training_receipt.json")["trained_checkpoint"]["sha256"],
             "source_code_sha256":core.sha256_file(__file__),"actual_adapted_checkpoint":True,"test_overlap_audited":True,"maps_fitted":len(selections),
             "outputs":{p.name:core.sha256_file(p) for p in sorted(root.iterdir()) if p.is_file()},"scope":summary["semantics"]}
    core.write_json(root/"receipt.json",receipt);return receipt


def verify_representation(directory):
    """Reconstruct maps from original train/validation data; no MATLAB or writes."""
    root=Path(directory).resolve();receipt=core.read_json(_own(root,"receipt.json"))
    if receipt.get("status")!="human_adapted_representation_bridge_completed" or receipt.get("source_code_sha256")!=core.sha256_file(__file__):
        raise core.StudyError("Frozen human-adapted representation bridge missing/changed")
    for name,value in receipt["outputs"].items():
        if core.sha256_file(_own(root,name))!=value:raise core.StudyError("Representation map/data artifact changed")
    paths=receipt["input_paths"];inputs,pins,maps,selections,predictions,episodes,census,summary=_analyse(paths["states"],paths["clinical_trained"],paths["adapted_geometry"],paths["config"],paths["geometry"])
    if pins!=receipt["input_receipts_sha256"] or inputs!=paths:raise core.StudyError("Representation parent source receipt chain changed")
    checkpoint=core.read_json(Path(paths["clinical_trained"])/"training_receipt.json")["trained_checkpoint"]
    if receipt.get("clinical_training_receipt_sha256")!=pins["clinical_training"] or receipt.get("adapted_checkpoint_sha256")!=checkpoint["sha256"]:
        raise core.StudyError("Representation uses a different completed human-finetuned clinical copy")
    with np.load(root/"map_weights.npz",allow_pickle=False) as archive:
        if set(archive.files)!=set(maps) or any(not np.array_equal(archive[k],v) for k,v in maps.items()):raise core.StudyError("Rehashed representation map differs from original training-only fit")
    if core.read_json(root/"map_selection.json")!={"maps":selections} or core.read_json(root/"representation_summary.json")!=summary:
        raise core.StudyError("Representation selection/metrics differ from actual held-person analysis")
    for name,expected in [("pair_predictions.csv",predictions),("episode_representation_residuals.csv",episodes),("pair_eligibility.csv",census)]:
        written=[{k:"" if v is None else str(v) for k,v in row.items()} for row in expected]
        if core.read_csv(root/name)!=written:raise core.StudyError("Rehashed representation predictions/data differ from actual human observations")
    training=core.read_json(root/"training_receipt.json");evaluation=core.read_json(root/"evaluation_receipt.json")
    if training["maps"]!=len(selections) or training["clinical_training_receipt_sha256"]!=pins["clinical_training"] or training["test_outcomes_used_for_selection"] is not False:
        raise core.StudyError("Representation training receipt differs from actual adaptation/map evidence")
    weights=training.get("map_weights",{});predictions_record=evaluation.get("predictions",{})
    if Path(weights.get("path","")).resolve()!=_own(root,"map_weights.npz") or weights.get("sha256")!=core.sha256_file(root/"map_weights.npz") or training.get("map_selection_sha256")!=core.sha256_file(root/"map_selection.json"):
        raise core.StudyError("Representation fitted weights/selection are not bound to their output role")
    if Path(predictions_record.get("path","")).resolve()!=_own(root,"pair_predictions.csv") or predictions_record.get("sha256")!=core.sha256_file(root/"pair_predictions.csv"):
        raise core.StudyError("Representation prediction receipt is not bound to its output role")
    if evaluation["training_receipt_sha256"]!=core.sha256_file(root/"training_receipt.json") or evaluation["test_people"]!=summary["comparative_test_people"] or evaluation["metrics"]!=summary["metrics"]:
        raise core.StudyError("Representation evaluation receipt differs from original held-person analysis")
    return receipt|{"verified_actual_source_chain":True,"verification_only":True}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["states","clinical-trained","adapted-geometry","config","output"]:parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--geometry",type=Path,action="append",default=[])
    args=vars(parser.parse_args(argv));print(json.dumps(run(**args),indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
