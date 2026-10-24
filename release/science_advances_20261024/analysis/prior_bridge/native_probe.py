"""Scheduled qualification against existing native files, before full-bank work."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

import driver
import native_population
import population_bridge

IDS = ('pristine__published_000',
       'scaled_dendrite_v1_explicit_patch__candidate_000__control',
       'scaled_dendrite_v1_explicit_patch__candidate_000__pulse')


def run():
    driver.require(os.environ.get('SLURM_JOB_ID', '').isdigit(), 'Actual native parsing requires a scheduled worker')
    value, training = driver.validated_inputs()
    with Path(value['files']['bank_inventory']['path']).open(encoding='utf8', newline='') as stream:
        rows = {r['simulation_id']: r for r in csv.DictReader(stream)}
    traces, records = {}, []
    for name in IDS:
        trace = native_population.load_native(rows[name], value['bank_root'])
        traces[name] = trace
        records.append({'simulation_id': name, 'source_output_sha256': rows[name]['source_output_sha256'],
                        'config_sha256': rows[name]['config_sha256'], 'rows': len(trace['times']),
                        'first_observed_time_s': float(trace['times'][0]), 'last_observed_time_s': float(trace['times'][-1]),
                        'columns': trace['columns'], 'voltage_rate_shape': list(trace['values'].shape),
                        'recorded_noise_observed': True, 'recorded_pulse_observed': trace['stimulus_observed']})
    pair = native_population.qualify_pair(traces[IDS[1]], traces[IDS[2]])
    selection = json.loads(Path(value['files']['mechanism_selection']['path']).read_text())
    requested = max(selection['config']['lags'])*selection['config']['delay_seconds_per_index']
    with Path(value['files']['account_predictions']['path']).open(encoding='utf8', newline='') as stream:
        cuts = {r['map_key']: int(r['conditioning_states']) for r in csv.DictReader(stream)}
    clocks = {}
    with np.load(value['files']['observation_map']['path'], allow_pickle=False) as archive:
        for description in selection['maps']:
            key = description['map_key']; grid = np.asarray(archive[key+'__times_s'], float)
            support = population_bridge.available_history(grid, list(traces.values()), requested)
            features = {view: list(population_bridge.native_features(traces[IDS[0]], grid, cuts[key], view,
                        support['used_max_delay_s']).shape) for view in population_bridge.VIEWS}
            clocks[key] = {'frozen_grid_first_s': float(grid[0]), 'frozen_grid_last_s': float(grid[-1]),
                           'conditioning_states': cuts[key], 'support': support, 'view_shapes': features}
    return {'status': 'actual_native_clock_and_pair_schema_qualified', 'utc': datetime.now(timezone.utc).isoformat(),
            'job_id': os.environ['SLURM_JOB_ID'], 'source_sha256': driver.sha(__file__),
            'input_manifest_sha256': driver.sha(Path(__file__).parent/'inputs.json'),
            'native_cache_schema': native_population.SCHEMA, 'samples': records, 'pair': pair,
            'frozen_clock_and_native_feature_proofs': clocks,
            'training_gate': training, 'human_state_target_arrays_decoded': False,
            'frozen_clock_arrays_read': True, 'new_simulations': False,
            'raw_labels_preserved': True, 'pulse_onset_reference': 'configured onset minus Output.Start, unchanged',
            'selected_scientific_direction': False, 'full_bank_executed': False}


if __name__ == '__main__':
    print(json.dumps(run(), separators=(',', ':'), allow_nan=False))
