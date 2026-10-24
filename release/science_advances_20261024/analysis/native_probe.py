"""Scheduled actual-input technical checks without preferred result gating."""
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path

import driver
import numpy as np


def numpy_scalar(value):
    if isinstance(value,np.generic): return value.item()
    raise TypeError("Unsupported technical proof type: "+type(value).__name__)


def run():
    driver.require(os.environ.get('SLURM_JOB_ID','').isdigit(), 'Actual-input parsing requires scheduled compute')
    value, training = driver.validated_inputs()
    checks = {}
    for name in driver.STAGES:
        module = importlib.import_module(name)
        checks[name] = json.loads(json.dumps(module.qualification_probe(driver.stage_inputs(name,value)), default=numpy_scalar, allow_nan=False))
        driver.require(isinstance(checks[name],dict) and isinstance(checks[name].get('status'),str),
                       'Actual technical probe did not produce a receipt: '+name)
    return {'status':'cross_mechanistic_actual_inputs_qualified','utc':datetime.now(timezone.utc).isoformat(),
            'job_id':os.environ['SLURM_JOB_ID'],'source_sha256':driver.sha(__file__),
            'input_manifest_sha256':driver.sha(Path(__file__).parent/'inputs.json'),
            'training_gate':training,'actual_input_probes':checks,'raw_labels_preserved':True,
            'scientific_arrays_opened_on_scheduled_compute':True,'full_scientific_analysis_executed':False,
            'results_or_hypotheses_used_as_qualification_gate':False,'completed_work_repeated':False}


if __name__ == '__main__':
    print(json.dumps(run(),separators=(',',':'),allow_nan=False))
