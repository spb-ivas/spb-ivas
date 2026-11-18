"""Exercise a fit, JSON reload, and three separate one-year scenario calculations."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'release'))
from fitted import fit, predict, validate_model
from prepared import prepare


def verify(data_dir, revision, through_year, scenario_year):
    frame, audit = prepare(data_dir, revision)
    model = fit(frame, through_year, {'revision':revision,'input_sha256':audit['input_sha256']})
    encoded = json.dumps(model, allow_nan=False)
    loaded = json.loads(encoded)
    validate_model(loaded)
    baseline = predict(frame, model, scenario_year).set_index(['FIPS','YEAR'],verify_integrity=True).sort_index()
    def run(**options):
        return predict(frame,loaded,scenario_year,**options).set_index(['FIPS','YEAR'],verify_integrity=True).sort_index()
    pd.testing.assert_frame_equal(run(),baseline,check_exact=True)
    outputs = {}
    for name, options in [('baseline',{}),('tmax_plus2',{'tmax':2.}),('et_plus10',{'et':10.}),('pine_minus10',{'pine':-10.})]:
        scenario = run(**options)
        other = baseline.index.get_level_values('YEAR') != scenario_year
        pd.testing.assert_frame_equal(scenario.loc[other],baseline.loc[other],check_exact=True)
        current = scenario.xs(scenario_year, level='YEAR')
        base = baseline.xs(scenario_year,level='YEAR')
        common = current.SPB_spots.notna() & base.SPB_spots.notna()
        total = current.SPB_spots.sum(min_count=1)
        outputs[name] = {'scored':int(current.SPB_spots.notna().sum()),'unsupported':int(current.SPB_spots.isna().sum()),
                         'sum_modeled_spots':float(total) if pd.notna(total) else None, 'paired_count':int(common.sum()),
                         'paired_change_spots':float((current.SPB_spots[common]-base.SPB_spots[common]).sum()) if common.any() else None,
                         'changed_classes_on_common_rows':int((current.Risk_category[common]!=base.Risk_category[common]).sum()),
                         'unchanged_nontarget_rows':int(other.sum())}
    # Score and compare only where a held-out observation and both estimates exist.
    support = frame.set_index(['FIPS','YEAR'],verify_integrity=True)[['SPOTS','PREV_SPOTS']]
    scores = baseline[['SPB_spots']].join(support).dropna()
    scores = scores.loc[scores.index.get_level_values('YEAR') > through_year]
    byyear={}
    for yr, rows in scores.groupby(level='YEAR'):
        byyear[str(yr)]={'observed_pairs':len(rows),'mae_ols':float((rows.SPB_spots-rows.SPOTS).abs().mean()),
                         'mae_prior_count':float((rows.PREV_SPOTS-rows.SPOTS).abs().mean())}
    return {'revision':revision,'through_year':through_year,'scenario_year':scenario_year,'all_county_years':len(baseline),
            'all_counties':int(frame.FIPS.nunique()),'target_counties':len(baseline.xs(scenario_year,level='YEAR')),
            'training':model['training'],'coefficients':model['coefficients'],'transform':model['transform'],
            'scenario_comparisons':outputs,'temporal_holdout':byyear,
            'prepared_predictor_missing_counts':{c:int(frame[c].isna().sum()) for c in ['TMAX_C','ET_growing','PINE_ACRES','STD_1','PINE_DENSITY','PREV_SPOTS']},
            'model_reloaded_identically':True}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'examples/demo')
    parser.add_argument('--revision',default='20250101')
    parser.add_argument('--through-year',type=int,default=2021)
    parser.add_argument('--year',type=int,default=2025)
    a=parser.parse_args()
    print(json.dumps(verify(a.data_dir,a.revision,a.through_year,a.year),indent=2,allow_nan=False))
