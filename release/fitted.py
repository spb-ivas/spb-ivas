"""Saved-state OLS example for the corrected SPB-IVAS workflow.

``fit`` estimates every learned quantity on eligible observed rows through a
cutoff. ``predict`` never refits those quantities. PREV_SPOTS must already mean
the observed count for the same FIPS at YEAR - 1 (a calendar join, not row shift).
Both response and lag are densities per 1,000 *current-row* pine acres, shifted
by one before Box-Cox. Zero counts are retained, not replaced with pseudo-counts.

This is a statistical example, not evidence of ecological forecast skill. Risk
classes compare predictions with a frozen distribution of observed training
counts; they are relative categories, not outbreak probabilities.
"""
from collections import Counter
import hashlib
import json
import math
import platform
import re

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.special import boxcox, inv_boxcox

SCHEMA_VERSION = 1
FEATURE_ORDER = ['TMAX_C', 'ET_growing', 'STD_1', 'PINE_DENSITY', 'PREV']
SCALED_COLUMNS = ['TMAX_C', 'ET_growing', 'PINE_DENSITY']
REQUIRED_COLUMNS = ['FIPS', 'YEAR', 'TMAX_C', 'ET_growing', 'PINE_DENSITY',
                    'STD_1', 'PINE_ACRES', 'PREV_SPOTS', 'SPOTS', 'PPT_growing']
RISK_MULTIPLIERS = [.5, 1., 1.5, 2., 2.5, 3.]
RISK_LEVELS = [0., .5, 1., 1.5, 2., 2.5, 3.]
LAG_DEFINITION = ('PREV_SPOTS is the observed count at matching FIPS and YEAR-1; '
                  'lag density is 1000*PREV_SPOTS/current-row PINE_ACRES; '
                  'apply the saved Box-Cox lambda to 1+lag density')
MAX_CONDITION_NUMBER = 1e12


def _number(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be a finite number, not a boolean')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + ' must be a finite number') from error
    if not math.isfinite(result):
        raise ValueError(name + ' must be finite')
    return result


def _year(value, name):
    result = _number(value, name)
    if result != int(result) or not 1 <= result <= 9999:
        raise ValueError(name + ' must be an integer year in 1..9999')
    return int(result)


def _schema(frame):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError('frame must be a pandas DataFrame')
    if frame.columns.duplicated().any():
        raise ValueError('frame contains duplicate column names')
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError('Missing input columns: ' + ', '.join(missing))


def _prepare(frame, require_observed=False):
    """Build complete model terms and explicit row exclusions; no filling."""
    _schema(frame)
    values = {name:pd.to_numeric(frame[name], errors='coerce').to_numpy(dtype=float, na_value=np.nan, copy=True)
              for name in REQUIRED_COLUMNS}
    reasons = [[] for _ in range(len(frame))]
    def mark(mask, reason):
        for position in np.flatnonzero(mask):
            reasons[position].append(reason)
    fips = values['FIPS']; years = values['YEAR']
    valid_fips = np.isfinite(fips) & (fips >= 1) & (fips <= 99999)
    valid_fips &= np.where(np.isfinite(fips), fips, 0) == np.floor(np.where(np.isfinite(fips), fips, 0))
    valid_year = np.isfinite(years) & (years >= 1) & (years <= 9999)
    valid_year &= np.where(np.isfinite(years), years, 0) == np.floor(np.where(np.isfinite(years), years, 0))
    mark(~valid_fips, 'invalid_FIPS')
    mark(~valid_year, 'invalid_YEAR')
    normalized_fips = [str(int(x)).zfill(5) if ok else str(original) if pd.notna(original) else None
                       for x,ok,original in zip(fips,valid_fips,frame['FIPS'])]
    keys = pd.DataFrame({'FIPS':normalized_fips, 'YEAR':years})
    mark(keys.duplicated(['FIPS','YEAR'],keep=False).to_numpy() & valid_fips & valid_year,
         'duplicate_county_year')
    predictors = ['TMAX_C','ET_growing','PINE_DENSITY','STD_1','PINE_ACRES','PREV_SPOTS']
    for name in predictors:
        mark(~np.isfinite(values[name]), 'missing_or_nonfinite_' + name)
    for name in ['ET_growing','PREV_SPOTS']:
        mark(np.isfinite(values[name]) & (values[name] < 0), 'negative_' + name)
    mark(np.isfinite(values['PINE_ACRES']) & (values['PINE_ACRES'] <= 0), 'nonpositive_PINE_ACRES')
    for name in ['PINE_DENSITY','STD_1']:
        mark(np.isfinite(values[name]) & ((values[name] < 0) | (values[name] > 1)),
             'out_of_range_' + name)
    if require_observed:
        mark(~np.isfinite(values['SPOTS']), 'unobserved_or_nonfinite_SPOTS')
        mark(np.isfinite(values['SPOTS']) & (values['SPOTS'] < 0), 'negative_SPOTS')
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        lag_density = 1000. * values['PREV_SPOTS'] / values['PINE_ACRES']
        density = 1000. * values['SPOTS'] / values['PINE_ACRES']
    mark(~np.isfinite(lag_density), 'nonfinite_lag_density')
    if require_observed:
        mark(~np.isfinite(density), 'nonfinite_response_density')
    eligible = np.array([not row for row in reasons],dtype=bool)
    return values, normalized_fips, eligible, reasons, density, lag_density


def _features(values, lag_density, lmbda):
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        lag = boxcox(1. + lag_density, lmbda)
    return np.column_stack([values['TMAX_C'], values['ET_growing'], values['STD_1'],
                            values['PINE_DENSITY'], lag])


def _design(features, scaler):
    result = np.array(features, dtype=float, copy=True)
    for name in SCALED_COLUMNS:
        index = FEATURE_ORDER.index(name)
        result[:,index] = (result[:,index]-float(scaler['mean'][name]))/float(scaler['std'][name])
    return np.column_stack([np.ones(len(result)),result])


def fit(frame, through_year, provenance):
    """Fit the five-predictor OLS example and return a JSON-compatible artifact.

    Only nonnegative observed SPOTS with complete, valid predictors/lag and
    YEAR <= through_year are used. Every duplicate county/year is excluded.
    No county filling, response imputation or feature-term skipping is performed.
    ``provenance`` is caller-supplied JSON metadata about source files/processing.
    """
    cutoff = _year(through_year,'through_year')
    _schema(frame)
    if not isinstance(provenance,dict):
        raise ValueError('provenance must be a JSON-serializable dictionary')
    try:
        source_provenance = json.loads(json.dumps(provenance,allow_nan=False))
    except (ValueError,TypeError,OverflowError,RecursionError) as error:
        raise ValueError('provenance must contain only finite JSON values') from error
    years = pd.to_numeric(frame['YEAR'],errors='coerce')
    # Inspect eligibility only within the training cutoff: held-out/future labels
    # and covariates cannot change learned state or exclusion reasons.
    numeric_years=years.to_numpy(dtype=float,na_value=np.nan)
    candidate_mask=np.isfinite(numeric_years)&(numeric_years<=cutoff)
    candidates = frame.loc[candidate_mask].copy()
    values,normalized,eligible,reasons,density,lag_density = _prepare(candidates,True)
    nobs = int(eligible.sum())
    ncolumns = 1 + len(FEATURE_ORDER)
    if nobs <= ncolumns:
        raise ValueError('Need more than six eligible observed training rows for OLS')
    y_density = density[eligible]
    shifted = 1. + y_density
    if not np.isfinite(shifted).all() or np.any(shifted < 1) or np.ptp(shifted) <= 0:
        raise ValueError('Training shifted density must be finite and vary')
    try:
        response,lmbda = stats.boxcox(shifted)
    except (ValueError,FloatingPointError,RuntimeError) as error:
        raise ValueError('Could not fit the training-only shifted Box-Cox transform') from error
    if not np.isfinite(response).all() or not math.isfinite(float(lmbda)):
        raise ValueError('Training Box-Cox transform produced nonfinite values')
    if np.ptp(response)<=0:
        raise ValueError('Training Box-Cox response has no numerical variation')
    selected = {name:arr[eligible] for name,arr in values.items()}
    features = _features(selected,lag_density[eligible],float(lmbda))
    if not np.isfinite(features).all():
        raise ValueError('Training design contains invalid transformed predictors')
    scaler = {'columns':list(SCALED_COLUMNS), 'mean':{}, 'std':{}, 'ddof':1}
    for name in SCALED_COLUMNS:
        index = FEATURE_ORDER.index(name)
        mean = float(features[:,index].mean())
        std = float(features[:,index].std(ddof=1))
        if not math.isfinite(std) or std <= 0:
            raise ValueError('Training predictor has zero or invalid variance: ' + name)
        scaler['mean'][name] = mean;scaler['std'][name] = std
    matrix = _design(features,scaler)
    rank = int(np.linalg.matrix_rank(matrix))
    condition = float(np.linalg.cond(matrix))
    if rank != ncolumns:
        raise ValueError('Training design is rank deficient: rank %d for %d columns' % (rank,ncolumns))
    if not math.isfinite(condition) or condition > MAX_CONDITION_NUMBER:
        raise ValueError('Training design is numerically ill-conditioned')
    coefficients,_,solved_rank,_ = np.linalg.lstsq(matrix,response,rcond=None)
    if solved_rank != ncolumns or not np.isfinite(coefficients).all():
        raise ValueError('OLS failed to produce a finite full-rank solution')
    observed = selected['SPOTS']
    risk_mean = float(observed.mean());risk_std = float(observed.std(ddof=1))
    if not math.isfinite(risk_std) or risk_std <= 0:
        raise ValueError('Observed training spot counts must vary to define risk thresholds')
    thresholds = [risk_mean + multiple*risk_std for multiple in RISK_MULTIPLIERS]
    if not np.isfinite(thresholds).all():
        raise ValueError('Training risk thresholds are not finite')
    residual = response-matrix@coefficients
    rss = float(residual@residual)
    centered = response-response.mean()
    tss = float(centered@centered)
    if not math.isfinite(rss) or not math.isfinite(tss) or tss<=0:
        raise ValueError('Training residual/response variation is numerically invalid')
    data = candidates.loc[eligible,REQUIRED_COLUMNS].copy()
    data['FIPS'] = np.array(normalized,dtype=object)[eligible]
    for name in REQUIRED_COLUMNS[1:]:
        data[name] = selected[name]
    data = data.sort_values(['FIPS','YEAR']).reset_index(drop=True)
    # PPT is display-only; exclude it from the learned-state data fingerprint.
    fingerprint_cols = [x for x in REQUIRED_COLUMNS if x != 'PPT_growing']
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(data[fingerprint_cols],index=False).to_numpy(dtype='<u8').tobytes()).hexdigest()
    exclusions = Counter(reason for row in reasons for reason in row)
    model = {
        'schema_version':SCHEMA_VERSION,
        'model_type':'shifted_boxcox_ols',
        'feature_order':list(FEATURE_ORDER),
        'coefficients':dict(zip(['const']+FEATURE_ORDER,[float(v) for v in coefficients])),
        'transform':{'name':'shifted_boxcox_density','shift':1.,'lambda':float(lmbda),
                     'density_unit':'spots per 1000 current-row pine acres',
                     'lag_definition':LAG_DEFINITION,
                     'inverse_policy':'inverse Box-Cox minus one; clip negative density to zero; invalid domain stays missing'},
        'scaler':scaler,
        'risk':{'reference':'observed eligible training SPOTS', 'mean':risk_mean,'std':risk_std,
                'ddof':1,'multipliers':list(RISK_MULTIPLIERS),'thresholds':thresholds,
                'levels':list(RISK_LEVELS),'interpretation':'relative count class, not outbreak probability'},
        'training':{'through_year':cutoff, 'eligible_rows':nobs,
                    'candidate_rows':int(len(candidates)), 'excluded_rows':int(len(candidates)-nobs),
                    'outside_cutoff_or_invalid_year_rows':int(len(frame)-len(candidates)),
                    'exclusion_reason_counts':dict(sorted(exclusions.items())),
                    'zero_spot_rows':int((observed==0).sum()),
                    'year_min':int(selected['YEAR'].min()),'year_max':int(selected['YEAR'].max()),
                    'county_count':int(data.FIPS.nunique()),
                    'design_rank':rank,'design_columns':ncolumns,
                    'condition_number':condition,'maximum_condition_number':MAX_CONDITION_NUMBER,
                    'residual_degrees_of_freedom':nobs-ncolumns,
                    'residual_sum_squares_transformed':rss,
                    'in_sample_r_squared_transformed':float(1.-rss/tss),
                    'data_sha256':fingerprint,
                    'observation_policy':'SPOTS is an observed nonnegative count; missing is not zero; no count imputation',
                    'provenance':source_provenance,
                    'versions':{'python':platform.python_version(),'numpy':np.__version__,
                                'pandas':pd.__version__,'scipy':scipy.__version__}},
    }
    validate_model(model)
    return model


def validate_model(model):
    """Raise ValueError for malformed, nonfinite or incompatible fitted state."""
    if not isinstance(model,dict):
        raise ValueError('Model artifact must be a dictionary')
    try:
        json.dumps(model,allow_nan=False)
    except (ValueError,TypeError,OverflowError,RecursionError) as error:
        raise ValueError('Model artifact must contain finite JSON-serializable values') from error
    def require(condition,message):
        if not condition:
            raise ValueError('Invalid model artifact: '+message)
    def artifact_number(value,name):
        require(isinstance(value,(int,float)) and not isinstance(value,bool),name+' must be a JSON number')
        return _number(value,name)
    try:
        require(type(model['schema_version']) is int and model['schema_version']==SCHEMA_VERSION,'unsupported schema_version')
        require(model['model_type']=='shifted_boxcox_ols','unsupported model_type')
        require(model['feature_order']==FEATURE_ORDER,'feature_order does not match the supported design')
        coefs=model['coefficients']
        require(isinstance(coefs,dict) and set(coefs)==set(['const']+FEATURE_ORDER),'coefficient names do not match feature_order')
        for name in ['const']+FEATURE_ORDER:
            artifact_number(coefs[name],'coefficient '+name)
        transform=model['transform']
        require(transform['name']=='shifted_boxcox_density','unsupported response transform')
        require(artifact_number(transform['shift'],'transform shift')==1.,'response shift must be one')
        artifact_number(transform['lambda'],'Box-Cox lambda')
        require(transform['density_unit']=='spots per 1000 current-row pine acres','unsupported density unit')
        require(transform['lag_definition']==LAG_DEFINITION,'unsupported lag definition')
        require(transform['inverse_policy']=='inverse Box-Cox minus one; clip negative density to zero; invalid domain stays missing','unsupported inverse policy')
        scaler=model['scaler']
        require(scaler['columns']==SCALED_COLUMNS and type(scaler['ddof']) is int and scaler['ddof']==1,'invalid scaler columns/ddof')
        for field in ['mean','std']:
            require(isinstance(scaler[field],dict) and set(scaler[field])==set(SCALED_COLUMNS),'invalid scaler '+field+' columns')
        for name in SCALED_COLUMNS:
            artifact_number(scaler['mean'][name],'scaler mean '+name)
            require(artifact_number(scaler['std'][name],'scaler std '+name)>0,'scaler standard deviations must be positive')
        risk=model['risk']
        require(risk['reference']=='observed eligible training SPOTS','unsupported risk reference')
        require(type(risk['ddof']) is int and risk['ddof']==1,'risk ddof must be one')
        mean=artifact_number(risk['mean'],'risk mean');std=artifact_number(risk['std'],'risk std')
        require(mean>=0 and std>0,'risk mean/std invalid')
        require(risk['multipliers']==RISK_MULTIPLIERS and risk['levels']==RISK_LEVELS,'invalid risk multipliers/levels')
        for value in risk['multipliers']+risk['levels']:
            artifact_number(value,'risk level/multiplier')
        require(isinstance(risk['thresholds'],list) and len(risk['thresholds'])==6,'need six risk thresholds')
        thresholds=np.array([artifact_number(x,'risk threshold') for x in risk['thresholds']])
        require(np.all(np.diff(thresholds)>0),'risk thresholds must increase strictly')
        require(np.allclose(thresholds,mean+std*np.array(RISK_MULTIPLIERS),rtol=1e-12,atol=1e-12),'risk thresholds do not match frozen training reference')
        training=model['training']
        require(type(training['through_year']) is int,'training cutoff must be an integer')
        cutoff=_year(training['through_year'],'training cutoff')
        require(type(training['year_min']) is int and type(training['year_max']) is int,'training year range must contain integers')
        for name in ['eligible_rows','candidate_rows','excluded_rows','outside_cutoff_or_invalid_year_rows','zero_spot_rows','county_count','design_rank','design_columns','residual_degrees_of_freedom']:
            require(type(training[name]) is int and training[name]>=0,'invalid training count '+name)
        require(training['eligible_rows']>6 and training['design_rank']==training['design_columns']==6,'training design must have rank six and more than six rows')
        require(training['candidate_rows']==training['eligible_rows']+training['excluded_rows'],'training row counts inconsistent')
        require(training['residual_degrees_of_freedom']==training['eligible_rows']-6,'invalid residual degrees of freedom')
        require(training['zero_spot_rows']<=training['eligible_rows'] and 1<=training['county_count']<=training['eligible_rows'],'invalid training county/zero counts')
        require(1<=_year(training['year_min'],'training year_min')<=_year(training['year_max'],'training year_max')<=cutoff,'training years exceed cutoff')
        require(0<artifact_number(training['condition_number'],'condition number')<=MAX_CONDITION_NUMBER,'ill-conditioned fitted design')
        require(artifact_number(training['maximum_condition_number'],'condition limit')==MAX_CONDITION_NUMBER,'unsupported condition limit')
        require(artifact_number(training['residual_sum_squares_transformed'],'training RSS')>=0,'negative training RSS')
        require(-1e-12<=artifact_number(training['in_sample_r_squared_transformed'],'training R-squared')<=1.+1e-12,'invalid training R-squared')
        require(isinstance(training['data_sha256'],str) and re.fullmatch('[0-9a-f]{64}',training['data_sha256']) is not None,'invalid training data fingerprint')
        require(isinstance(training['provenance'],dict),'provenance must be a dictionary')
        require(isinstance(training['exclusion_reason_counts'],dict) and all(isinstance(k,str) and type(v) is int and v>=0 for k,v in training['exclusion_reason_counts'].items()),'invalid exclusion counts')
        require(isinstance(training['versions'],dict) and all(isinstance(training['versions'].get(k),str) and training['versions'][k] for k in ['python','numpy','pandas','scipy']),'invalid dependency versions')
    except (KeyError,TypeError,IndexError,AttributeError) as error:
        raise ValueError('Invalid or incomplete fitted-model artifact') from error


def export_identity(frame, model):
    """Identify the exact calibration and inference inputs in an exported table."""
    canonical = json.dumps(model, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {'model_sha256': hashlib.sha256(canonical).hexdigest(),
            'input_revision': frame.attrs.get('input_revision', ''),
            'input_manifest_sha256': frame.attrs.get('input_manifest_sha256', '')}


def predict(frame, model, year, tmax=0., et=0., pine=0.):
    """Apply frozen fitted state to every row; scenarios affect ``year`` only.

    ``tmax`` is an additive Celsius change; ``et`` and ``pine`` are percent
    changes to ET_growing and pine-area fraction, respectively. Pine acres,
    precipitation and the observed lag are unchanged. ET is already millimetres:
    no legacy 25.4 display multiplier is used. Missing/ineligible predictions and
    their risk classes remain NaN with explicit reasons.
    """
    validate_model(model)
    target=_year(year,'scenario year')
    delta=_number(tmax,'temperature adjustment')
    et_percent=_number(et,'ET percentage')
    pine_percent=_number(pine,'pine-density percentage')
    if et_percent < -100 or pine_percent < -100:
        raise ValueError('ET and pine-density percentages cannot be below -100')
    values,normalized,eligible,reasons,_,lag_density=_prepare(frame,False)
    target_rows=values['YEAR']==target
    if not target_rows.any():
        raise ValueError('Scenario year %d is absent from the supplied frame' % target)
    with np.errstate(over='ignore',invalid='ignore'):
        values['TMAX_C'][target_rows]+=delta
        values['ET_growing'][target_rows]*=1.+et_percent/100.
        values['PINE_DENSITY'][target_rows]*=1.+pine_percent/100.
    for name in ['TMAX_C','ET_growing','PINE_DENSITY']:
        invalid=~np.isfinite(values[name])
        for position in np.flatnonzero(invalid & eligible):
            reasons[position].append('scenario_nonfinite_'+name)
        eligible &= ~invalid
    invalid=(values['PINE_DENSITY']<0)|(values['PINE_DENSITY']>1)
    for position in np.flatnonzero(invalid & eligible):
        reasons[position].append('scenario_out_of_range_PINE_DENSITY')
    eligible &= ~invalid
    rates=np.full(len(frame),np.nan);counts=np.full(len(frame),np.nan);categories=np.full(len(frame),np.nan)
    positions=np.flatnonzero(eligible)
    if len(positions):
        selected={name:arr[positions] for name,arr in values.items()}
        features=_features(selected,lag_density[positions],float(model['transform']['lambda']))
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            matrix=_design(features,model['scaler'])
            coefficients=np.array([model['coefficients'][name] for name in ['const']+FEATURE_ORDER],dtype=float)
            linear=matrix@coefficients
            density=inv_boxcox(linear,float(model['transform']['lambda']))-1.
            density=np.maximum(density,0.)
            spots=density*selected['PINE_ACRES']/1000.
        valid=np.isfinite(matrix).all(axis=1)&np.isfinite(linear)&np.isfinite(density)&np.isfinite(spots)
        for position in positions[~valid]:
            reasons[position].append('inverse_transform_or_prediction_domain')
            eligible[position]=False
        good=positions[valid]
        rates[good]=density[valid];counts[good]=spots[valid]
        categories[good]=np.array(model['risk']['levels'])[np.searchsorted(model['risk']['thresholds'],spots[valid],side='right')]
    actual=values['SPOTS'].copy();actual[~np.isfinite(actual)|(actual<0)]=np.nan
    result=pd.DataFrame(index=frame.index)
    result['FIPS']=normalized
    result['YEAR']=frame['YEAR'].to_numpy(copy=True)
    result['SPB_spots']=counts
    result['Risk_category']=categories
    result['Actual spots']=actual
    result['Max April Temperature']=values['TMAX_C']
    result['ET_mm_per_month']=values['ET_growing']
    result['Evap_Transporation']=result['ET_mm_per_month']  # Historical CSV column spelling.
    result['Pine_density']=values['PINE_DENSITY']
    result['Rain']=values['PPT_growing']
    result['SPB_per_thousand_acre']=rates
    result['Eligible']=eligible
    result['Reason']=[';'.join(row) for row in reasons]
    result['Prediction_status']=np.where(eligible,'predicted','ineligible')
    return result
