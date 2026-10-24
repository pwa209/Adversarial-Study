"""Lossless CSV export of completed results; no fits, new tests or resampling."""
import csv
import hashlib
import json
import shutil
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT = ROOT / 'statistics'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(name, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(',', ':')) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    return len(rows)


def leaves(value, pointer=''):
    if isinstance(value, dict):
        for key, child in value.items():
            escaped = str(key).replace('~', '~0').replace('/', '~1')
            yield from leaves(child, pointer + '/' + escaped)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from leaves(child, pointer + '/' + str(idx))
    else:
        yield pointer, value


def flatten_record(row):
    return {pointer.lstrip('/').replace('/', '__'): value for pointer, value in leaves(row)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load(ROOT / 'provenance/figure_data_manifest.json')
    for name, expected in manifest['data_file_sha256'].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == expected, name
    inventory = []
    for path in sorted(DATA.rglob('*.csv')):
        relative = path.relative_to(DATA)
        destination = OUT / 'source_tables' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            n = sum(1 for _ in reader)
        inventory.append({'file': str(destination.relative_to(ROOT)).replace('\\', '/'), 'source': 'data/' + relative.as_posix(), 'rows': n, 'columns': columns, 'operation': 'byte-identical source CSV copy'})

    all_values = []
    for path in sorted(DATA.rglob('*.json')):
        for pointer, value in leaves(load(path)):
            all_values.append({'source_file': 'data/' + path.relative_to(DATA).as_posix(), 'json_pointer': pointer, 'value': value,
                               'value_type': 'null' if value is None else type(value).__name__})
    write('all_saved_result_values.csv', all_values)
    numerical = [row for row in all_values if row['value_type'] in ('int', 'float')]
    write('all_saved_numerical_results.csv', numerical)

    boot = load(DATA / 'mechanism_compensation/configuration_bootstrap.json')
    recovery = []
    response = []
    for task, destination in (('parameter_recovery', recovery), ('finite_response_prediction', response)):
        for row in boot[task]['results']:
            destination.append(flatten_record(row) | {'bootstrap_repetitions': boot[task]['repetitions'], 'bootstrap_seed': boot[task]['seed'],
                'interval_scope': 'finite configuration bank, conditional on selected fits; not human-population inference', 'p_value': '', 'p_value_status': 'not computed by the source analysis'})
    write('S01_parameter_recovery.csv', recovery)
    write('S02_finite_response_prediction.csv', response)

    human = load(DATA / 'human_spatial/summary.json')
    write('S03_human_spatial_all_tests.csv', (flatten_record({key: value for key, value in row.items() if key not in ('participant_effects', 'leave_one_person_out')}) for row in human['comparisons']))
    write('S04_human_participant_effects_and_leave_one_out.csv', (
        {'comparison_index': idx + 1, 'contrast': row['contrast'], 'panel': row['panel'], 'kind': row['kind'], 'record_type': kind, **flatten_record(record)}
        for idx, row in enumerate(human['comparisons']) for kind in ('participant_effects', 'leave_one_person_out') for record in row.get(kind, [])))
    pathway = load(DATA / 'pathway_balance/configuration_cluster_bootstrap.json')
    write('S05_pathway_configuration_bootstrap.csv', (flatten_record(row) for row in pathway))
    bridge = load(DATA / 'prior/population_bridge/population_bridge.json')
    write('S06_human_forecast_all_account_comparisons.csv', (
        {'analysis_set': key, **flatten_record({k: v for k, v in row.items() if k not in ('person_differences', 'leave_one_person_out', 'person_ids')})}
        for key in ('all_account_comparisons', 'validation_selected_account_comparisons') for row in bridge[key]))
    write('S07_human_forecast_participant_effects.csv', (
        {'analysis_set': key, 'source_model_key': row['source_model_key'], 'predictor': row['predictor'], 'comparator': row['comparator'], 'record_type': kind, **flatten_record(record)}
        for key in ('all_account_comparisons', 'validation_selected_account_comparisons') for row in bridge[key]
        for kind in ('person_differences', 'leave_one_person_out') for record in row.get(kind, [])))
    summary = load(DATA / 'pathway_balance/summary.json')
    write('S08_pathway_matching_dispositions.csv', ({'json_pointer': pointer, 'value': value} for pointer, value in leaves({key: summary[key] for key in ('matched_comparison_counts', 'matched_comparison_dispositions', 'bridge_fit_support', 'strict_supported_pulse_contrasts')})))
    for path in sorted((ROOT / 'plot_data').glob('*.csv')):
        shutil.copyfile(path, OUT / ('display_' + path.name))
    descriptive = []
    for filename, group_columns, metrics in (
        ('pathway_balance/population_pathway_balance.csv', ('mode', 'scope', 'population'), ('cancellation_for_inference',)),
        ('mechanism_compensation/zero_input_features.csv', ('variant',), ('V_raw_participation', 'Q_raw_participation'))):
        with (DATA / filename).open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[column] for column in group_columns)].append(row)
        for group, members in groups.items():
            for metric in metrics:
                values = []
                for row in members:
                    try:
                        value = float(row[metric])
                        if math.isfinite(value): values.append(value)
                    except (TypeError, ValueError): pass
                quartiles = statistics.quantiles(values, n=4, method='inclusive') if len(values) > 1 else values * 3
                descriptive.append({'source': filename, **dict(zip(group_columns, group)), 'metric': metric,
                    'total_rows': len(members), 'finite_rows': len(values), 'unavailable_rows': len(members)-len(values),
                    'mean': statistics.mean(values) if values else '', 'median': statistics.median(values) if values else '',
                    'q1': quartiles[0] if values else '', 'q3': quartiles[2] if values else '',
                    'minimum': min(values) if values else '', 'maximum': max(values) if values else '',
                    'operation': 'new descriptive aggregation of pinned source CSV; no inferential test'})
    write('S09_descriptive_summaries.csv', descriptive)
    write('source_table_inventory.csv', inventory)
    report = {'source_files_verified': len(manifest['data_file_sha256']), 'byte_identical_source_csv_tables': len(inventory),
              'json_leaf_values_exported': len(all_values), 'numerical_json_leaves_exported': len(numerical),
              'new_fits_tests_or_bootstraps': False, 'new_descriptive_summary_rows': len(descriptive),
              'scope': 'All saved aggregate result tables and every JSON leaf in the 37 pinned sources underlying this manuscript and figure set, including nulls, model coefficients, sensitivity tests, and diagnostic results.'}
    (ROOT / 'provenance/statistics_export_receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
