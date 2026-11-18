#!/usr/bin/env python3
"""Execute and record the existing OLS calibration block in isolated scratch.

This audits original forecast.py through fitting, shuffled CV and held-out
prediction. It intentionally stops before VIF, sensitivity, maps and CSV export.
It does not save/deploy a model or change the configured web coefficients.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

RELEASE = Path(__file__).resolve().parent
EXPECTED_FORECAST_SHA256 = '68c3dee918c0ec8379b381c5203b296f3d193db29c8b5ed70e66d7572a74c608'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=RELEASE/'local_data')
    parser.add_argument('--revision', default='20250405')
    parser.add_argument('--mode', choices=['legacy', 'explicit_compatibility'], default='explicit_compatibility')
    parser.add_argument('--output', type=Path, required=True, help='Audit JSON; a sibling .log records existing training output')
    parser.add_argument('--allow-legacy-data', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if not args.revision.isdigit() or len(args.revision) != 8:
        parser.exit(2, 'Revision must contain eight date digits\n')
    output = args.output.resolve()
    logfile = output.with_suffix('.log')
    if not args.force and (output.exists() or logfile.exists()):
        parser.exit(2, 'Audit output/log exists; choose a new filename or use --force\n')
    source = RELEASE/'original/spbom_web'
    if sha256(source/'forecast.py') != EXPECTED_FORECAST_SHA256:
        parser.exit(2, 'Archived forecast.py changed; trace markers require review\n')
    data = args.data_dir.resolve()
    from audit_inputs import audit
    input_audit = audit(data, args.revision)
    if input_audit['blocking_issues'] and not args.allow_legacy_data:
        parser.exit(2, 'Input audit failed: '+ '; '.join(input_audit['blocking_issues'])+
                    '. Use --allow-legacy-data only for labelled existing-data reproduction.\n')
    inputs = {name: data/(name+'_'+args.revision+'.csv')
              for name in ['cpc', 'fia', 'prism_stable', 'prism_provisional', 'spots']}
    wassi = data/('wassi_'+args.revision+'.csv')
    if not wassi.is_file():
        wassi = data/('wassi_'+args.revision+'_p0.csv')
    inputs['wassi'] = wassi
    for path in inputs.values():
        if not path.is_file():
            parser.exit(2, 'Missing prepared input: '+path.name+'\n')
    # Keep Matplotlib caches inside the audit sandbox, then import the same
    # dependencies as the archived calibration/preprocessing source.
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {'mode': args.mode, 'revision': args.revision,
              'purpose': 'Audit execution of existing OLS fitting/CV/hold-out; no ecological skill or coefficient lineage claim',
              'source_hashes': {name:sha256(source/name) for name in ['forecast.py','model_code.py']},
              'inputs': [{'role': role, 'filename': path.name, 'sha256':sha256(path)} for role,path in inputs.items()],
              'input_audit': input_audit, 'allow_legacy_data': args.allow_legacy_data,
              'stages': {}, 'cv_scores': []}
    previous_cwd = Path.cwd()
    previous_trace = sys.gettrace()
    with tempfile.TemporaryDirectory(prefix='spbom-calibration-audit-') as temporary:
        scratch = Path(temporary)
        os.environ['MPLCONFIGDIR'] = str(scratch/'mplconfig')
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        import statsmodels
        import matplotlib
        record['versions'] = {'python': sys.version, 'numpy': np.__version__,
                              'pandas': pd.__version__, 'scipy': scipy.__version__,
                              'sklearn': sklearn.__version__, 'statsmodels': statsmodels.__version__}
        original_use = matplotlib.use
        matplotlib.use = lambda backend,*a,**k: original_use('Agg' if backend=='TkAgg' else backend,*a,**k)
        class AuditStop(Exception):
            pass
        def floats(series):
            return {str(k):float(v) for k,v in series.items()}
        def frame_summary(frame):
            return {'rows':len(frame), 'columns':list(frame.columns),
                    'counties':int(frame.FIPS.nunique()),
                    'years':[int(frame.YEAR.min()),int(frame.YEAR.max())],
                    'duplicate_county_year_excess':int(frame.duplicated(['FIPS','YEAR']).sum())}
        def trace(frame,event,arg):
            if event != 'line' or frame.f_code.co_name != 'forecast' or Path(frame.f_code.co_filename).name != 'forecast.py':
                return trace
            loc=frame.f_locals;line=frame.f_lineno
            if line==103:
                record['stages']['processed']=frame_summary(loc['data'])
                record['boxcox_lambda']=float(loc['maxlog'])
                record['processed_missing_by_column']={str(k):int(v) for k,v in loc['data'].isna().sum().items() if v}
            elif line==126:
                record['stages']['train']=frame_summary(loc['train_data'])
                record['stages']['test']=frame_summary(loc['test_data'])
                record['random_seed']=int(loc['RANDOM_SEED'])
                record['candidate_year_condition']='YEAR < '+str(loc['previous_year'])
                record['scaler_rows']=len(loc['scaler'].df)
                record['scaler']=loc['scaler'].to_dict()
                for name in ['train_data','test_data']:
                    record[name+'_index_sha256']=hashlib.sha256(loc[name].index.to_numpy().astype('<i8').tobytes()).hexdigest()
            elif line in [143,162,173]:
                label={143:'all_features_ols',162:'select_k_best_ols',173:'five_predictor_ols'}[line]
                result=loc['result'];matrix=loc['X']
                record['stages'][label]={'nobs':int(result.nobs),'predictors':list(matrix.columns),
                    'rank':int(result.model.rank),'columns':int(matrix.shape[1]),
                    'training_r_squared_transformed':float(result.rsquared),
                    'coefficients':floats(result.params),'missing_predictor_cells':int(matrix.isna().sum().sum())}
            elif line==217:
                record['cv_scores'].append(float(loc['kmodel'].score(loc['X'].iloc[loc['test']],loc['y'].iloc[loc['test']])))
            elif line==227:
                from sklearn.metrics import r2_score
                record['heldout_r_squared_transformed']=float(r2_score(loc['test_y'],loc['test_prediction']))
                record['heldout_prediction_rows']=len(loc['test_prediction'])
            elif line==237:
                record['intentional_stop']='After OLS fitting, shuffled CV and held-out prediction; before VIF, sensitivity, maps, monthly diagnostics and CSV export'
                raise AuditStop()
            return trace
        (scratch/'data').mkdir()
        for name in ['forecast.py','model_code.py']:
            shutil.copy2(source/name,scratch/name)
        for role,path in inputs.items():
            (scratch/'data'/(role+'_'+args.revision+'.csv')).symlink_to(path)
        (scratch/'data/source').symlink_to(data/'source',target_is_directory=True)
        os.chdir(scratch);sys.path.insert(0,str(scratch))
        with logfile.open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            try:
                import model_code
                if args.mode=='explicit_compatibility':
                    original_init=model_code.Scaler.__init__
                    def explicit_init(self,frame):
                        if isinstance(frame,pd.DataFrame):
                            self.df=pd.DataFrame({'MEAN':frame.mean(),'STD':frame.std()});self.is_scaled=[]
                        else:original_init(self,frame)
                    model_code.Scaler.__init__=explicit_init
                    record['external_interception']='Headless backend and explicit-index scaler; original source bytes unchanged'
                else:
                    record['external_interception']='Headless backend only; original scaler unchanged'
                import forecast
                sys.settrace(trace)
                ordered=['cpc','prism_stable','prism_provisional','wassi','fia','spots']
                forecast.forecast(args.revision,*[str(scratch/'data'/(role+'_'+args.revision+'.csv')) for role in ordered])
                record['core_training_executed']=True
                record['completed_full_function']=True
            except AuditStop:
                record['core_training_executed']=True
                record['completed_full_function']=False
            except Exception as error:
                record['core_training_executed']=False
                record['completed_full_function']=False
                record['failure']={'type':type(error).__name__,'message':str(error)}
                traceback.print_exc()
            finally:
                sys.settrace(previous_trace)
                os.chdir(previous_cwd)
                sys.path.remove(str(scratch))
                matplotlib.use=original_use
    record['source_hashes_unchanged']={name:sha256(source/name)==value for name,value in record['source_hashes'].items()}
    output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'mode':args.mode,'core_training_executed':record['core_training_executed'],
                      'completed_full_function':record['completed_full_function'],'result':str(output)},indent=2))
    if not record['core_training_executed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
