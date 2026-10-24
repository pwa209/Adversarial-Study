"""Validate completed tables and flatten existing summaries; no fitting or new tests."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
DATA=OUT/'data'
PLOT=OUT/'plot_data'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csvrows(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (PLOT/name).open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def main():
    PLOT.mkdir(exist_ok=True)
    pins={}
    for path in sorted((OUT/'provenance').glob('*_retrieval.json')):
        for name,spec in read(path)['files'].items():
            assert sha(DATA/name)==spec['sha256'] and (DATA/name).stat().st_size==spec['bytes']
            pins[name]=spec['sha256']
    assert len(pins)==37, 'The complete pinned source set is required.'
    inventory=DATA/'simulation_inventory.csv'
    assert sha(inventory)=='1c2876091578167d2c93ae41a6f5602619e407f99fe56c07cda4148e579acb4f'
    assert sha(DATA/'electrode_coordinates.sfp')=='59885415704f21152304183670a947547d1651a0862a8ee41ce5e5eb71ca5c29'
    role={r['simulation_id']:r for r in csvrows(DATA/'mechanism_compensation/configuration_roles.csv')}
    factors=[]
    for r in csvrows(inventory):
        if r['pair_id'] and float(r['input_amplitude'])==0:
            factors.append({**{k:r[k] for k in ('simulation_id','account_class','pair_id','param_gain_factor','param_coupling_factor','param_history_factor')},
                            'role':role[r['simulation_id']]['role'],'configuration_group':role[r['simulation_id']]['configuration_group']})
    assert len(factors)==48 and len({r['configuration_group'] for r in factors})==48
    assert {role:sum(r['role']==role for r in factors) for role in ('train','validation','test')}=={'train':28,'validation':10,'test':10}
    write('factor_roles.csv',factors)
    module=DATA/'mechanism_compensation';summary=read(module/'summary.json');boot=read(module/'configuration_bootstrap.json')
    assert summary['zero_input_configurations']==616 and summary['qualified_finite_pairs']==48
    for name,key in (('recovery_summary.csv','parameter_recovery'),('response_summary.csv','finite_response_prediction')):
        write(name,[{k:v for k,v in row.items() if not isinstance(v,(dict,list))}|
                    {'lower':row['bank_stability_mse_percentiles_2p5_97p5'][0], 'upper':row['bank_stability_mse_percentiles_2p5_97p5'][1],
                     'difference_lower':row['bank_stability_difference_percentiles_2p5_97p5'][0],
                     'difference_upper':row['bank_stability_difference_percentiles_2p5_97p5'][1],
                     'replicates':boot[key]['repetitions']} for row in boot[key]['results']])
    validation=[];complexity=[]
    for task in ('parameter_recovery','finite_response_prediction'):
        for variant,result in summary[task].items():
            for model,fit in result['models'].items():
                complexity.append({'task':task,'variant':variant,'model':model,'effective_df':fit['effective_df'],
                                   'input_columns':fit.get('input_columns',0),'ridge':fit.get('ridge','')})
                for r in fit.get('validation_grid',[]):validation.append({'task':task,'variant':variant,'model':model,**r})
    write('validation_grid.csv',validation);write('model_complexity.csv',complexity)
    coverage=read(module/'parameter_coverage.json')['scaled_dendrite_v1_explicit_patch']
    svd=[];load=[]
    for rep,result in coverage['training_feature_parameter_geometry']['representations'].items():
        for idx,s in enumerate(result['singular_values']):svd.append({'representation':rep,'component':idx+1,'singular_value':s})
        for i,row in enumerate(result['standardized_log_factor_directions_columns']):
            for j,value in enumerate(row):load.append({'representation':rep,'factor':('gain','coupling','history')[i],'component':j+1,'loading':value})
    write('sensitivity_singular_values.csv',svd);write('sensitivity_directions.csv',load)
    human=read(DATA/'human_spatial/summary.json');comparisons=[];effects=[]
    assert human['eligible_people']==31 and human['eligible_episodes']==183 and len(human['comparisons'])==14
    for idx,r in enumerate(human['comparisons']):
        key=f'H{idx+1:02d}'
        comparisons.append({k:v for k,v in r.items() if not isinstance(v,(dict,list))}|
                           {'comparison_id':key,'lower':r.get('ci95',['',''])[0],'upper':r.get('ci95',['',''])[1]})
        for effect in r.get('participant_effects',[]):effects.append({'comparison_id':key,'contrast':r['contrast'],'panel':r['panel'],'kind':r['kind'],**effect})
    write('human_comparisons.csv',comparisons);write('human_person_effects.csv',effects)
    bootstrap=read(DATA/'pathway_balance/configuration_cluster_bootstrap.json');assert len(bootstrap)==576
    write('pathway_cluster_bootstrap.csv',bootstrap)
    balance=csvrows(DATA/'pathway_balance/population_pathway_balance.csv');assert len(balance)==7524
    assert all(r['voltage_sum_qualified']=='True' for r in balance)
    assert all(float(r['maximum_absolute_sum_error_V'])<=float(r['fixed_rounding_tolerance_V']) for r in balance)
    bridge=read(DATA/'prior/population_bridge/population_bridge.json')
    comparisons=bridge.get('validation_selected_account_comparisons',[])
    bridge_comp=[]
    for r in comparisons:
        bridge_comp.append({k:v for k,v in r.items() if not isinstance(v,(dict,list))})
    write('human_bridge_comparisons.csv',bridge_comp)
    design={'manuscript_scope':'Observation constraints on mechanistic inference in a single-node neural-field model',
            'palette':['#f5f2ec','#d8d4cd','#756458','#b54a3c'],
            'background':'#ffffff','all_text':'#000000','alignment':'Common inner plot margins within comparable panel grids',
            'main_figures':5,'supplementary_figures':7,
            'render_backend':'R 4.6.1 base graphics; Cairo PDF/SVG; 400 dpi PNG rasterized from the canonical R PDF using Poppler',
            'new_model_fitting':False,'new_significance_tests':False,'new_bootstrap_computations':False,
            'uncertainty':'Previously computed finite-bank configuration bootstrap ranges; human conditional participant bootstrap intervals',
            'matched_capacity_analysis_available':False,'independent_human_mechanism_validated':False,
            'human_bootstrap_replicates':10000,'simulation_bootstrap_replicates':2000,
            'data_file_sha256':pins}
    (OUT/'provenance/figure_data_manifest.json').write_text(json.dumps(design,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'verified_source_files':len(pins),'derived_plot_tables':len(list(PLOT.glob('*.csv'))),'new_fits_or_tests':False}))


if __name__=='__main__':main()
