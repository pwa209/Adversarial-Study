"""Original DREAM and Psych branches with source-specific provenance.

Sleep reports are retrospective categories, not ten-second consciousness
labels. Behavioural sessions do not enlarge the EEG participant denominator.
No source or model network requests occur here; bounded acquisition is separate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any
import zipfile

import numpy as np

from .acquisition import canonical_hash, verify_acquired
from .core import RECORDING_COLUMNS, sha256_file, write_csv, write_json

DREAM_CATEGORIES = {2: "experience", 1: "experience_without_recall", 0: "no_experience",
                    -1: "experience_with_or_without_recall", -2: "no_experience_or_experience_without_recall",
                    -3: "experience_with_recall_or_no_experience", -4: "unknown"}
DREAM_STAGES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 5: "REM"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _csv_bytes(data: bytes):
    # Source exports contain CRCRLF; universal newline handling preserves fields.
    stream = io.StringIO(data.decode("utf-8-sig"), newline=None)
    return list(csv.DictReader(stream))


def _member(archive: zipfile.ZipFile, basename: str):
    candidates = [i for i in archive.infolist() if PurePosixPath(i.filename).name.lower() == basename.lower()]
    if len(candidates) != 1:
        raise ValueError(f"no unique original archive member {basename}")
    return candidates[0]


def _extract(archive, member, destination, max_bytes=100_000_000):
    parts = PurePosixPath(member.filename)
    if parts.is_absolute() or ".." in parts.parts or "\\" in member.filename:
        raise ValueError("unsafe original archive member")
    if member.file_size > max_bytes or member.file_size < 1:
        raise ValueError("original member exceeds declared conversion bound")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("do not overwrite original source extraction")
    digest = hashlib.sha256()
    with archive.open(member) as source, destination.open("xb") as target:
        total = 0
        while block := source.read(1024 * 1024):
            total += len(block)
            if total > max_bytes:
                raise ValueError("original source extraction exceeded bound")
            target.write(block)
            digest.update(block)
    if total != member.file_size:
        raise ValueError("truncated original archive member")
    return {"member": member.filename, "bytes": total, "zip_crc32": f"{member.CRC:08x}", "sha256": digest.hexdigest()}


def parse_edf_header(header: bytes):
    """Qualify EDF16 storage without silently assigning missing physical units."""
    if len(header) < 256 or header[:8].strip() != b"0":
        raise ValueError("only standard EDF16 is supported")
    n = int(header[252:256]); size = int(header[184:192])
    if n < 1 or n > 1024 or size != 256 * (n+1) or len(header) < size:
        raise ValueError("EDF signal/header dimensions invalid")
    reserved = header[192:236].decode("ascii").strip()
    if "EDF+D" in reserved:
        raise ValueError("discontinuous EDF is not a genuine continuous predictor input")
    fields = {}; offset = 256
    for key, width in [("names",16),("transducer",80),("units",8),("physical_min",8),
                       ("physical_max",8),("digital_min",8),("digital_max",8),
                       ("prefilter",80),("samples_per_record",8),("signal_reserved",32)]:
        fields[key] = [header[offset+i*width:offset+(i+1)*width].decode("latin1").strip() for i in range(n)]
        offset += width*n
    for key in ["physical_min","physical_max","digital_min","digital_max"]:
        fields[key] = [float(v) for v in fields[key]]
    fields["samples_per_record"] = [int(v) for v in fields["samples_per_record"]]
    if any(not math.isfinite(v) for k in ["physical_min","physical_max","digital_min","digital_max"] for v in fields[k]):
        raise ValueError("nonfinite EDF calibration")
    if any(a >= b for a,b in zip(fields["digital_min"],fields["digital_max"])) or any(
            a >= b for a,b in zip(fields["physical_min"],fields["physical_max"])):
        raise ValueError("EDF calibration ranges invalid")
    duration = float(header[244:252]); records = int(header[236:244])
    if duration <= 0 or records < 1 or any(v < 1 for v in fields["samples_per_record"]):
        raise ValueError("EDF record dimensions invalid")
    fields.update(header_bytes=size, record_duration_s=duration, data_records=records,
                  duration_s=duration*records, reserved=reserved,
                  patient_raw=header[8:88].decode("latin1").strip(),
                  recording_raw=header[88:168].decode("latin1").strip())
    return fields


def read_edf_original(path, *, authoritative_duration_s, eeg_names):
    """Decode declared source duration, preserving source calibration/reference.

    DREAM Records.csv explicitly supersedes EDF metadata. Tononi export contains
    an additional final EDF record (121 versus120 seconds); use source duration
    as the pre-awakening boundary, never the padded file endpoint.
    """
    path = Path(path)
    with path.open("rb") as stream:
        first = stream.read(256)
        size = int(first[184:192]); header = parse_edf_header(first+stream.read(size-256))
    names = header["names"]
    if set(eeg_names)-set(names):
        raise ValueError("specified source EEG labels absent")
    eeg_indices = [names.index(n) for n in eeg_names]
    rates = [header["samples_per_record"][i]/header["record_duration_s"] for i in eeg_indices]
    if len(set(rates)) != 1 or rates[0] <= 90:
        raise ValueError("source EEG sampling rates incompatible with native clinical predictor")
    sfreq = rates[0]
    expected_n = authoritative_duration_s*sfreq
    if not np.isclose(expected_n, round(expected_n), atol=1e-6, rtol=0) or expected_n < 10*sfreq:
        raise ValueError("DREAM duration does not support exact genuine10s interval")
    if authoritative_duration_s > header["duration_s"] or header["duration_s"]-authoritative_duration_s > header["record_duration_s"]+1e-6:
        raise ValueError("DREAM authoritative duration disagrees beyond one export record")
    samples = header["samples_per_record"]; width = sum(samples)
    expected_size = header["header_bytes"]+2*width*header["data_records"]
    if path.stat().st_size != expected_size:
        raise ValueError("EDF byte length does not match declared source dimensions")
    digital = np.memmap(path, dtype="<i2", mode="r", offset=header["header_bytes"],
                        shape=(header["data_records"],width))
    offsets = np.cumsum([0]+samples)
    raw_values = []; auxiliary = {}; missing_units = []
    for i,name in enumerate(names):
        raw = np.asarray(digital[:,offsets[i]:offsets[i+1]]).reshape(-1)
        values = ((raw.astype(np.float32)-header["digital_min"][i]) *
                  ((header["physical_max"][i]-header["physical_min"][i])/
                   (header["digital_max"][i]-header["digital_min"][i]))+header["physical_min"][i])
        unit = header["units"][i].replace("µ","u").replace("μ","u")
        rate = samples[i]/header["record_duration_s"]
        extent = int(round(authoritative_duration_s*rate))
        values = values[:extent]
        if i in eeg_indices:
            factor = {"uV":1.0,"mV":1000.0,"V":1e6}.get(unit)
            if factor is None:
                missing_units.append(name); raw_values.append(values)
            else:
                raw_values.append(values*factor)
        else:
            auxiliary[name] = {"values":values,"units":header["units"][i],"sfreq_hz":rate}
    del digital
    return {"data":np.stack(raw_values),"sfreq_hz":sfreq,"channel_names":eeg_names,
            "units":"uV" if not missing_units else "UNKNOWN", "missing_unit_channels":missing_units,
            "auxiliary":auxiliary,"header":header,
            "duration_s":authoritative_duration_s,
            "discarded_export_duration_s":header["duration_s"]-authoritative_duration_s}


def convert_dream(dataset, source_root, manifest_path, output_dir, *, subjects=None,
                  release_dir=None, qualification=None, config_path=None):
    manifest = _json(manifest_path)
    if config_path is None: config_path = Path(__file__).parents[1]/"configs/extension_study.json"
    settings = _json(config_path)["sleep"][dataset]
    source_id = settings["source_id"]; root=Path(source_root).resolve(); output=Path(output_dir).resolve()
    resources = [r for r in manifest["resources"] if r["source_id"]==source_id]
    output.mkdir(parents=True,exist_ok=False); (output/"source_material").mkdir()
    available=[]; archive_receipts=[]
    for r in resources:
        path=root/r["relative_path"]
        if path.is_file():
            archive_receipts.append(verify_acquired(root,r)); available.append(path)
    if not available: raise ValueError("no checksum-qualified original DREAM archives")
    if dataset=="tononi":
        metadata_resource=next(r for r in resources if r["resource_id"]==settings["metadata_resource_id"])
        metadata=root/metadata_resource["relative_path"]
        verify_acquired(root,metadata_resource)
    else: metadata=available[0]
    with zipfile.ZipFile(metadata) as archive:
        info=_member(archive,"Records.csv"); records=_csv_bytes(archive.read(info))
        extra_infos=[i for i in archive.infolist() if PurePosixPath(i.filename).name=="AdditionalData.csv"]
        extra=_csv_bytes(archive.read(extra_infos[0])) if extra_infos else []
        if dataset=="tononi":
            coordinate=_member(archive,"HydroCelGSN256v10.sfp")
            _extract(archive,coordinate,output/"source_material"/"HydroCelGSN256v10.sfp",max_bytes=100_000)
    additional={str(r["Case ID"]):r for r in extra}
    selected=[r for r in records if not subjects or str(r["Subject ID"]) in set(subjects)]
    if len({r["Case ID"] for r in selected})!=len(selected): raise ValueError("duplicate source DREAM case IDs")
    members={}
    for path in available:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.filename.lower().endswith(".edf"):
                    key=PurePosixPath(info.filename).name
                    if key in members: raise ValueError("duplicate EDF across source ZIPs")
                    members[key]=(path,info.filename)
    rows=[]; episodes=[]; eligibility=[]; audits=[]
    for source in selected:
        filename=PurePosixPath(source["Filename"].lstrip("/")).name
        if not filename.lower().endswith(".edf"): filename+=".edf"
        state={"source_case_id":source["Case ID"],"source_subject_id":source["Subject ID"],
               "source_filename":filename,"status":"pending","reason":"","outcome_selection_applied":False}
        if filename not in members:
            state.update(status="not_acquired",reason="originalEDFnotpresentinselectedarchives")
            eligibility.append(state); continue
        try:
            archive_path,name=members[filename]
            original_path=output/"source_material"/filename
            with zipfile.ZipFile(archive_path) as archive:
                source_receipt=_extract(archive,archive.getinfo(name),original_path,max_bytes=100_000_000)
            with original_path.open("rb") as stream:
                first=stream.read(256); header=parse_edf_header(first+stream.read(int(first[184:192])-256))
            if dataset=="tononi":
                eeg_names=[f"Chan {i}" for i in range(1,258)]
                if header["names"]!=eeg_names: raise ValueError("originalTononiEEGschema differsfrominspectedrelease")
            else: eeg_names=settings["instrument_channels"]
            result=read_edf_original(original_path,authoritative_duration_s=float(source["Duration"]),eeg_names=eeg_names)
            rid=source_id+"_"+str(source["Case ID"]); pid=source_id+"_"+str(source["Subject ID"])
            signal_path=output/(rid+".npz")
            if result["units"]!="uV":
                unknown_path=output/(rid+".unqualified_units.npz")
                np.savez(unknown_path,source_physical_values=result["data"],sfreq_hz=result["sfreq_hz"],
                         channel_names=np.asarray(eeg_names,dtype="U"),physical_units=np.asarray(
                             [header["units"][header["names"].index(n)] for n in eeg_names],dtype="U"))
                state.update(status="BLOCKED_UNKNOWN_PHYSICAL_UNITS",reason="producerunits evidence required before clinical scoring",
                             preserved_unqualified_values=str(unknown_path),source_sha256=source_receipt["sha256"])
                eligibility.append(state); continue
            np.savez(signal_path,data=result["data"],sfreq_hz=result["sfreq_hz"],channel_names=np.asarray(eeg_names,dtype="U"),
                     units="uV",reference=settings["reference_status"],gap_intervals_s=np.empty((0,2)))
            row={"recording_id":rid,"participant_id":pid,"source_id":source_id,"cohort_id":source_id,
                 "site_id":"Madison_Wisconsin" if dataset=="tononi" else "Paris_Brain_Institute",
                 "session_id":"source_case_"+source["Case ID"],"role":"development","signal_path":signal_path.name,
                 "signal_format":"npz","modality":"EEG","units":"uV","duration_s":result["duration_s"],
                 "sfreq_hz":result["sfreq_hz"],"channel_count":len(eeg_names),"detector_exposure":"unknown",
                 "analyst_exposure":"unknown","synthetic":False,"reference":settings["reference_status"],
                 "source_conversion":"auditedEDF16physicalgain; nofilter/resample/rereference",
                 "source_signal_sha256":source_receipt["sha256"]}
            code=int(source["Experience"]); stage=int(source["Last sleep stage"])
            if code not in DREAM_CATEGORIES or stage not in DREAM_STAGES: raise ValueError("unrecognised source DREAM category/stage")
            duration=result["duration_s"]; ad=additional.get(source["Case ID"],{})
            episode={"episode_id":rid+"_preawakening","recording_id":rid,"start_s":duration-10,
                     "stop_s":duration,"report_start_s":"","report_stop_s":"","report_category":DREAM_CATEGORIES[code],
                     "capacity_value":"","source_report_code":code,"source_sleep_stage_code":stage,
                     "sleep_stage":DREAM_STAGES[stage],"report_point_s":duration,"report_interval_status":"duration_unspecified_retrospective_report",
                     "report_instruction":"mental activity immediately prior to awakening",
                     "timing_basis":"authoritativeRecordsdurationendsatawakening; metadata10sEEGnotexperience-durationlabel",
                     "time_of_awakening_raw":source["Time of awakening"],"source_case_id":source["Case ID"],
                     "source_original_report_category":{"experience":"DE","experience_without_recall":"DEWR","no_experience":"NE"}.get(DREAM_CATEGORIES[code],"source_combined_category"),
                     "perception_report_raw":ad.get("Perception (yes=1, no=0)",""),
                     "experience_duration_category_raw":ad.get("Duration of continuous experiences before alarm 0-5s=1, >5-10s=2, >10s-1min=3,>1-10min=4,>10min-1h=5, >1h=6","")}
            rows.append(row); episodes.append(episode)
            state.update(status="converted",recording_id=rid)
            audits.append({"recording_id":rid,"source_member":source_receipt,"normalised_sha256":sha256_file(signal_path),
                           "edf_header":header,"authoritative_preawakening_duration_s":duration,
                           "discarded_export_duration_s":result["discarded_export_duration_s"],
                           "reference_change_applied":False,"source_report":source,"additional_data":ad})
        except (ValueError,OSError,KeyError) as error:
            state.update(status="not_qualified",reason=f"{type(error).__name__}: {error}")
        eligibility.append(state)
    names=list(RECORDING_COLUMNS)+["reference","source_conversion","source_signal_sha256"]
    write_csv(output/"recordings.csv",rows,names)
    episode_columns=sorted({k for r in episodes for k in r}) or ["episode_id","recording_id","start_s","stop_s","report_category","capacity_value"]
    write_csv(output/"episodes_development.csv",episodes,episode_columns)
    write_csv(output/"source_eligibility.csv",eligibility,sorted({k for r in eligibility for k in r}) or ["source_case_id","status","reason"])
    cfg=_json(Path(__file__).parents[1]/"configs/study.template.json")
    cfg.update(study_id="adversarial-consciousness-"+source_id,mode="empirical",path_base=".")
    cfg["paths"].update(recordings="recordings.csv",episodes_development="episodes_development.csv",data_root=".",output_dir="runs")
    if release_dir: cfg["paths"]["release_dir"]=str(Path(release_dir).resolve())
    if qualification: cfg["paths"]["qualification"]=str(Path(qualification).resolve())
    cfg["instrument"].update(channel_names=settings["instrument_channels"],channel_aggregation="median")
    cfg["quality"].update(amplitude_reference="channel_median")
    cfg["channel_quality"]={"enabled":True,"minimum_channels":16 if dataset=="tononi" else 3,
                            "max_abs_uv":250.0,"flat_std_uv":0.01,"amplitude_reference":"channel_median"}
    cfg["discovery"]["relationship_columns"]=["perception_report_raw","experience_duration_category_raw"]
    cfg["analysis"].update(construct="DREAM retrospective experience development",timing_definition="source10s_preawakening",enable_prediction=False)
    cfg["source_conversion"]={"adapter":"dream_"+dataset,"manifest_sha256":canonical_hash(manifest),"synthetic":False,
                              "native_channel_selection":settings["instrument_channels"],"channel_selection_basis":settings.get("selection_basis","original three EEG derivations"),
                              "prior_source_selection":settings.get("source_selection","originalstudyselection"),"protected_evaluation_qualification":False}
    write_json(output/"study.config.json",cfg)
    report={"status":"compatible_development_inputs" if rows else "BLOCKED_NO_QUALIFIED_EEG",
            "schema_version":1,"source_id":source_id,"manifest_sha256":canonical_hash(manifest),"source_archives":archive_receipts,
            "candidate_episodes":len(selected),"converted_recordings":len(rows),"converted_participants":len({r["participant_id"] for r in rows}),
            "selected_episodes":len(episodes),"records":audits,"eligibility":eligibility,
            "role":"development","synthetic":False,"detector_exposure":"unknown",
            "no_favourable_result_gate":True,"scientific_interpretation":"sleep comparison, not independent checkpointvalidation; reportcontentandno-recall keptseparate"}
    write_json(output/"source_qualification.json",report)
    return report


def marked_responses(text: str):
    """Return genuine response events; reject malformed or nested delimiters."""
    if not isinstance(text,str): raise ValueError("behaviour text must be source string")
    events=[]; position=0
    while True:
        start=text.find("<<",position)
        stray=text.find(">>",position)
        if start<0:
            if stray>=0: raise ValueError("unmatched closing response delimiter")
            break
        if 0<=stray<start: raise ValueError("unmatched closing response delimiter")
        end=text.find(">>",start+2)
        if end<0 or "<<" in text[start+2:end]: raise ValueError("malformed/nested markedresponse")
        response=text[start+2:end]
        if not response.strip(): raise ValueError("empty markedresponse")
        events.append({"index":len(events),"prefix":text[:start+2],"response":response,"start":start+2,"stop":end})
        position=end+2
    return events


def binary_prequential(events):
    """Proper source-instruction L/R likelihoods, fitted only on past choices."""
    counts={"L":0,"R":0}; transitions=defaultdict(lambda:{"L":0,"R":0}); previous=None
    rows=[]
    for event in events:
        response=event["response"].strip()
        if response not in counts: raise ValueError("response not in original fixed taskinstruction L/R alphabet")
        probabilities={"uniform":0.5,"beta_bernoulli":(counts[response]+0.5)/(sum(counts.values())+1),
                       "response_persistence":0.5 if previous is None else (0.8 if response==previous else 0.2),
                       "markov1":0.5 if previous is None else (transitions[previous][response]+0.5)/(sum(transitions[previous].values())+1)}
        rows.extend({"response_index":event["index"],"response":response,"model":model,
                     "observed_response_probability":probability,"log_likelihood":math.log(probability),
                     "training_choices":sum(counts.values()),"conditioning":"sourceinstructionL/Ralphabet; sameparticipantpastresponsesonly"}
                    for model,probability in probabilities.items())
        counts[response]+=1
        if previous is not None: transitions[previous][response]+=1
        previous=response
    return rows


def _behaviour_rows(resources,root):
    for resource in resources:
        receipt=verify_acquired(root,resource); path=root/resource["relative_path"]
        if path.suffix==".parquet":
            try: import pyarrow.parquet as parquet
            except ImportError as error: raise RuntimeError("Psych201 source requires pyarrow; no CSV/mock replacement") from error
            for batch in parquet.ParquetFile(path).iter_batches(batch_size=128):
                for i,row in enumerate(batch.to_pylist()): yield resource,receipt,row
        else:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip(): yield resource,receipt,json.loads(line)


def audit_psych(source_root,manifest_path,output_dir,*,source_ids=None,config_path=None):
    root=Path(source_root).resolve(); manifest=_json(manifest_path); output=Path(output_dir).resolve()
    output.mkdir(parents=True,exist_ok=False)
    if config_path is None: config_path=Path(__file__).parents[1]/"configs/extension_study.json"
    config=_json(config_path); settings=config["behaviour"]
    resources=[r for r in manifest["resources"] if r["source_id"] in (source_ids or ["psych101","psych201"])]
    if not resources: raise ValueError("no selected actual behavioural source resources")
    census=Counter(); candidate_errors=[]; receipts={}; comparator=[]; sessions=[]; pilot_people=set()
    response_count=0; flagged=Counter(); study_counts=Counter(); pilot_candidates=0
    for resource,receipt,row in _behaviour_rows(resources,root):
        receipts[resource["resource_id"]]=receipt
        sid=resource["source_id"]; study=str(row.get("study",row.get("experiment","")))
        experiment=str(row.get("experiment",study)); participant=str(row.get("participant",""))
        census[sid]+=1; study_counts[(sid,study,experiment)]+=1
        flags={name:row.get(name) for name in ["is_psych101","is_psych101_test","in_eval"]}
        if sid=="psych101": flags["is_psych101"]=True
        for k,v in flags.items(): flagged[(sid,k,str(v))]+=1
        try:
            events=marked_responses(row["text"]); response_count+=len(events)
            if not participant or not experiment: raise ValueError("sourceparticipant/experimentidentitymissing")
            if settings["pilot_study"].lower() not in (study+" "+experiment).lower(): continue
            pilot_candidates+=1
            comparisons=binary_prequential(events)
            identity=hashlib.sha256((sid+"\0"+study+"\0"+experiment+"\0"+participant).encode()).hexdigest()
            comparator.extend({"session_id":identity,"source_id":sid,"study":study,"experiment":experiment,
                               "source_participant":participant,**r,**flags} for r in comparisons)
            # Pilot admission is source-order/source identity only, independent of likelihood.
            person=(sid,study,participant)
            if person not in pilot_people and len(pilot_people)<settings["max_centaur_sessions"]:
                pilot_people.add(person)
                sessions.append({"session_id":identity,"source_id":sid,"study":study,"experiment":experiment,
                                 "source_participant":participant,"text":row["text"],"source_resource_id":resource["resource_id"],
                                 "resource_sha256":receipt["sha256"],"text_sha256":hashlib.sha256(row["text"].encode()).hexdigest(),
                                 "exposure_flags":flags,"role":"development","synthetic":False,
                                 "source_row_sha256":canonical_hash(row),"native_responses_requested":settings["max_centaur_responses"]})
        except (ValueError,KeyError) as error:
            candidate_errors.append({"source_id":sid,"study":study,"experiment":experiment,"participant":participant,
                                     "status":"not_qualified","reason":str(error)})
    with (output/"centaur_sessions.jsonl").open("w",encoding="utf-8",newline="\n") as stream:
        for row in sessions: stream.write(json.dumps(row,allow_nan=False)+"\n")
    names=["session_id","source_id","study","experiment","source_participant","response_index","response","model",
           "observed_response_probability","log_likelihood","training_choices","conditioning","is_psych101","is_psych101_test","in_eval"]
    write_csv(output/"conventional_likelihoods.csv",comparator,names)
    write_csv(output/"task_census.csv",[{"source_id":s,"study":study,"experiment":exp,"sessions":n} for (s,study,exp),n in sorted(study_counts.items())],
              ["source_id","study","experiment","sessions"])
    write_csv(output/"source_eligibility.csv",candidate_errors,["source_id","study","experiment","participant","status","reason"])
    report={"status":"behavioural_development_sources_audited","manifest_sha256":canonical_hash(manifest),
            "sources_sessions":dict(census),"marked_human_responses":response_count,
            "task_experiment_counts":len(study_counts),"source_overlap_flags":[{"source_id":s,"flag":k,"value":v,"sessions":n} for (s,k,v),n in sorted(flagged.items())],
            "pilot_study":settings["pilot_study"],"pilot_candidate_sessions":pilot_candidates,"pilot_selected_sessions":len(sessions),
            "pilot_selection":"first12 source-order distinct study-participant identities, independentofresponses/modeloutcomes",
            "source_receipts":list(receipts.values()),"sessions_sha256":sha256_file(output/"centaur_sessions.jsonl"),
            "conventional_likelihoods_sha256":sha256_file(output/"conventional_likelihoods.csv"),
            "scientific_role":"behaviouralbreadthandcomputationalprocesscomparison; no EEG and nohumanconsciousnessscore",
            "interpretation":"fixed simple prequential responsehistory comparators; not validatedbiologicalmechanism; full Psych201countmayoverlapPsych101",
            "protected_human_evaluation":False,"centaur_pretraining_exposure":"unknown beyond explicitPsych101trainingflags",
            "synthetic":False}
    write_json(output/"source_qualification.json",report)
    if not sessions: raise ValueError("requested original task unavailable; full source audit retained; no alternate task substituted")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    dream=sub.add_parser("dream"); dream.add_argument("--dataset",choices=["tononi","oudiette"],required=True)
    psych=sub.add_parser("psych"); psych.add_argument("--source-id",action="append")
    for p in [dream,psych]:
        p.add_argument("--source-root",required=True);p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);p.add_argument("--config")
    dream.add_argument("--subject",action="append");dream.add_argument("--release-dir");dream.add_argument("--qualification")
    args=parser.parse_args(argv)
    if args.command=="dream":
        report=convert_dream(args.dataset,args.source_root,args.manifest,args.output,subjects=args.subject,release_dir=args.release_dir,qualification=args.qualification,config_path=args.config)
    else: report=audit_psych(args.source_root,args.manifest,args.output,source_ids=args.source_id,config_path=args.config)
    print(json.dumps({k:v for k,v in report.items() if k not in ["records","eligibility","source_receipts","source_archives"]},indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
