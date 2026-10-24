"""Link observed human capacities to documented behavioural task candidates.

Task similarity is an explicit development hypothesis. This audit never turns
an EEG association into a validated behavioural or biological mechanism. Every
grouped prediction and available task remains visible, including unavailable
capacities, insufficient data, exposure and unqualified cross-assay mappings.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from .core import sha256_file,write_csv,write_json

COMMIT="c63dea18897393b706227b50024b27fa8d8907d8"
ROOT_URL="https://github.com/marcelbinz/Psych-201/blob/"+COMMIT+"/"
TASKS=[
    {"study":"hellmann_unpublished_brightness","family":"perceptual_discrimination_and_confidence",
     "candidate_targets":["detection","confidence"],"assays":"left/right brighterdisc judgment; confidencecontinuous-1to1; sourceRTs",
     "evidence_url":ROOT_URL+"hellmann_unpublished_brightness/README.md",
     "source_generator_url":ROOT_URL+"hellmann_unpublished_brightness/generate_prompts.py",
     "required_mapping":"Specify how tactile detection and sourceconfidence correspond to visualdiscrimination andcontinuousconfidence; shared computationalprediction; crossassaycalibration"},
    {"study":"pirrone_2018_dots","family":"magnitude_and_perceptual_discrimination",
     "candidate_targets":["detection"],"assays":"accuracyinmore-dotsconditions; separatevaluecollectioncondition; sourceRTs",
     "evidence_url":ROOT_URL+"pirrone_2018_dots/README.md",
     "source_generator_url":ROOT_URL+"pirrone_2018_dots/generate_prompts.py",
     "required_mapping":"Retain taskconditions; valueselectionisnotperception; map fixedsensoryevidence/decisionprediction across modalities before claimedgeneralisation"},
    {"study":"haridi2024memory","family":"associative_and_sequence_memory",
     "candidate_targets":["memory"],"assays":"word-paircuedrecall; lettersequencememory; no-recallresponse0",
     "evidence_url":ROOT_URL+"haridi2024memory/README.md",
     "source_generator_url":ROOT_URL+"haridi2024memory/generate_prompts.py",
     "required_mapping":"Independent paired human memory assay required; mindblanking/no-recallreport is not objective memory performance"},
    {"study":"sandbrink2024metacontrol","family":"information_sampling_and_metacontrol",
     "candidate_targets":["metacontrol"],"assays":"observeversusbet, choicefeedback, manipulationofcontrol",
     "evidence_url":ROOT_URL+"sandbrink2024metacontrol/README.md",
     "source_generator_url":ROOT_URL+"sandbrink2024metacontrol/generate_prompts.py",
     "required_mapping":"Independent information-sampling/controlassay and shared mechanism prediction required; thought-probecategoryisnotmetacontrolperformance"},
    {"study":"nasioulas2024feedback","family":"value_choice_and_feedback",
     "candidate_targets":["value_choice"],"assays":"specifiedbinarylotteries; L/Rchoice; partial/completefeedbackconditions",
     "evidence_url":ROOT_URL+"nasioulas2024feedback/README.md",
     "source_generator_url":ROOT_URL+"nasioulas2024feedback/generate_prompts.py",
     "required_mapping":"Value-choiceperformanceassay andaccount prediction required; first12Centaurruntimepilot doesnotestablishfunctionalbreadth"},
]


def relevance_audit(human_fit,behaviour,output_dir):
    human=Path(human_fit).resolve(); behaviour=Path(behaviour).resolve();output=Path(output_dir).resolve()
    prediction_file=human/"human_outcome_predictions.json"
    prediction=json.loads(prediction_file.read_text(encoding="utf-8"))
    source_audit=json.loads((behaviour/"source_qualification.json").read_text(encoding="utf-8"))
    with (behaviour/"task_census.csv").open(encoding="utf-8",newline="") as stream: census=list(csv.DictReader(stream))
    output.mkdir(parents=True,exist_ok=False)
    results=prediction.get("results")
    if not isinstance(results,list):raise ValueError("actual human grouped-prediction results missing")
    attempts=[]
    for result in results:
        target=result.get("target","");construct=result.get("construct","")
        for task in TASKS:
            matching=target in task["candidate_targets"] or (target=="capacity_value" and construct in task["candidate_targets"])
            if not matching:continue
            available=[r for r in census if task["study"] in (r["study"]+" "+r["experiment"])]
            attempts.append({"human_target":target,"human_construct":construct,"human_source_id":result.get("source_id",""),
                             "human_predictor":result.get("predictor",""),"human_prediction_status":result.get("status",""),
                             "grouping":result.get("grouping",""),"independent_participants":result.get("independent_participants",result.get("groups",0)),
                             "equal_participant_mse":result.get("equal_participant_mse"),"candidate_study":task["study"],
                             "candidate_family":task["family"],"available_sessions":sum(int(r["sessions"]) for r in available),
                             "task_evidence_url":task["evidence_url"],"required_mapping":task["required_mapping"],
                             "mapping_status":"candidate_construct_analogy_requires_qualification",
                             "neural_to_behavioural_mechanism_validated":False})
    columns=["human_target","human_construct","human_source_id","human_predictor","human_prediction_status","grouping",
             "independent_participants","equal_participant_mse","candidate_study","candidate_family","available_sessions",
             "task_evidence_url","required_mapping","mapping_status","neural_to_behavioural_mechanism_validated"]
    write_csv(output/"candidate_functional_mappings.csv",attempts,columns)
    write_json(output/"all_grouped_human_predictions.json",prediction)
    write_json(output/"documented_task_catalogue.json",{"source_commit":COMMIT,"tasks":TASKS,"all_actual_task_census":census})
    # No automatic selection from effect size/significance, proxy fit or task title.
    report={"status":"behavioural_relevance_development_audit_completed","human_predictions_sha256":sha256_file(prediction_file),
            "behaviour_audit_sha256":sha256_file(behaviour/"source_qualification.json"),"task_census_sha256":sha256_file(behaviour/"task_census.csv"),
            "all_grouped_predictions_retained":len(results),"candidate_mapping_count":len(attempts),
            "qualified_neural_to_behavioural_mappings":0,"native_discovery_driven_sessions_selected":0,
            "next_requirement":"Qualify shared assay/construct, source-specific responseoptionsandconditions, trainingexposure, measuredhumanfunctionalprediction and prospectiveaccount comparison before claiming behaviouralgeneralisation",
            "centaur_runtime_pilot_study":source_audit.get("pilot_study"),
            "centaur_runtime_pilot_role":"native dependency/likelihood/representation qualification; separatefromdiscoveryselectedfunctionaltest",
            "experience_report_and_capacity_kept_distinct":True,"no_result_direction_or_significance_gate":True,
            "no_claim_of_shared_neural_or_biological_mechanism":True,"source_overlap_flags":source_audit.get("source_overlap_flags",[]),
            "scientific_role":"open originalbehaviouralbranch; identifiescodeablecandidatefamiliesandmissingdiagnostic mappings instead of assertingunsupportedbreadth",
            "source_previous_native_training_exposure":"retainedexplicitPsych101flags; otherLLMpretrainingexposureunknown"}
    write_json(output/"relevance_receipt.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-fit",required=True);parser.add_argument("--behaviour",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args(argv);print(json.dumps(relevance_audit(args.human_fit,args.behaviour,args.output),indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
