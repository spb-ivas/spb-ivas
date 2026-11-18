"""
Forecast pipeline for Southern Pine Beetle outbreak modeling.

This module reads multiple data sources (CPC, PRISM, WASSI, FIA, SPOTS),
merges and processes them, fits an OLS model to transformed SPOTS response,
evaluates/plots diagnostics, generates spatial predictions, and writes an
output CSV with model scores and categorical risk.

Notes
- The code intentionally preserves existing computation to avoid any changes
  in numerical results. Cleanup is limited to documentation, comments, and
  removal of unused imports. Interactive breakpoints are commented out.
- The visualization backend is set to Agg to allow rendering without a GUI.
"""

from model_code import *  # noqa: F401,F403 (intentional star import from project module)
import os
import matplotlib
matplotlib.use('Agg')  # Use the Anti-Grain Geometry (AGG) backend instead of TkAgg

def forecast(REVISION,fcpc,prism_stable,prism_prov,fwassi,ffia,fspot):
    """
    Run the end‑to‑end forecast workflow.

    Parameters
    - REVISION: int or str-like revision identifier (e.g., 20240408). The
      first 4 digits are used as the forecasting YEAR; YEAR-1 used for prior plots.
    - fcpc: path to CPC CSV file (merged with PRISM after pivoting)
    - prism_stable: path to stable PRISM monthly CSV
    - prism_prov: path to provisional PRISM monthly CSV
    - fwassi: path to WASSI monthly CSV
    - ffia: path to FIA annual CSV
    - fspot: path to SPOTS annual CSV

    Returns
    - None. Writes `output_{REVISION}.csv` and displays figures.
    """
    current_dir=os.getcwd()
    print("Current Directory:", current_dir)
    #os.chdir("./spb_2021")

    pd.options.display.max_rows = 6
    pd.options.display.precision = 3

    
    # Fixed random seed for reproducibility of splits / CV scores.
    RANDOM_SEED = 1314897167  # import random; random.randint(0, 2**32 - 1)

    # Debug breakpoint (kept for reference but disabled to avoid halting).
    # import ipdb; ipdb.set_trace()
    

    # CPC data (county-year) to update PRISM monthly aggregates where available.
    cpc = pd.read_csv(fcpc, float_precision='high',engine='c').set_index(['FIPS', 'YEAR'])
    prism = pd.concat([
            pd.read_csv(prism_stable, float_precision='round_trip'),
            pd.read_csv(prism_prov, float_precision='round_trip')
        ]).sort_values(by=['FIPS', 'YEAR', 'MONTH']) \
          .rename(columns={'TMAX_C': 'TMAX', 'PPT_MM': 'PPT', 'TAVG_C': 'TMEAN'}) \
            .pivot_table(index=['FIPS', 'YEAR'], columns='MONTH', values=['PPT', 'TMAX', 'TMEAN']) \
              .reset_index()

    # Flatten multi-indexed columns like ('PPT', 3) -> 'PPT_3'
    prism.columns = prism.columns.map(lambda col: '_'.join([str(idx) for idx in col if idx]))
    prism = prism.set_index(['FIPS', 'YEAR'])
    # Update PRISM features with CPC values on aligned county-year rows.
    prism.update(cpc)
    prism = prism.reset_index()

    # WASSI monthly hydrologic variables pivoted to county-year wide format.
    wassi = (pd.read_csv(fwassi, float_precision='round_trip')
               .pivot_table(index=['FIPS', 'YEAR'], columns='MONTH', values=['RO', 'ET', 'SOLPSAT', 'GEP'])
               .reset_index())
    wassi.columns = wassi.columns.map(lambda col: '_'.join([str(idx) for idx in col if idx]))

    # Annual covariates and response.
    fia = pd.read_csv(ffia, float_precision='round_trip')
    spots = pd.read_csv(fspot, float_precision='round_trip')

    # Merge all sources on county-year.
    data = pd.merge(prism, wassi, on=['FIPS', 'YEAR'], how='left')
    data = data.merge(fia, on=['FIPS', 'YEAR'], how='left')
    # Forward-fill FIA attributes within each county where a year is missing.
    data.loc[:, fia.columns] = data.loc[:, fia.columns].ffill() # (method='ffill')

    data = pd.merge(data, spots, on=['FIPS', 'YEAR'], how='left')
    data = data.sort_values(by=['FIPS', 'YEAR'])
    
    # Explicit intercept column for some analyses/plots (model will add its own const as needed).
    data['const'] = 1
    fig, ax = plt.subplots(1, figsize=(6, 4))
    ax.set_title('Row Counts')
    ax.plot(wassi.groupby('YEAR').FIPS.count(), label='wassi')
    ax.plot(prism.groupby('YEAR').FIPS.count(), label='prism')
    ax.plot(fia.groupby('YEAR').FIPS.count(), label='fia')
    ax.plot(spots.groupby('YEAR').FIPS.count(), label='spots')
    ax.plot(data.groupby('YEAR').FIPS.count(), label='data', linestyle=':')
    ax.legend(loc=1)
    #*************************************************************************************
    # Feature engineering, target transformation (returns transformed Y and lambda for inverse Box-Cox).
    data, maxlog = process_data(data, plot=True)

    normtest(data.lnY)
    normtest(data.Yt)

    # Forecast/plot year derived from REVISION. E.g., 20240408 -> YEAR=2024; previous_year=2023.
    revision_str = str(REVISION)[:4]
    previous_year = int(revision_str)-1 
    
    train_data, test_data = train_test_split(data.loc[data.YEAR < previous_year], 
                                            test_size=.2, random_state=RANDOM_SEED)
    train_data, test_data = train_data.copy(), test_data.copy()

    # Columns to standardize before modeling (hydro/met and density).
    scaled_params = [col for col in list(data.columns) if re.match(r'ET|PPT|RO|TMIN|TMAX|TMEAN|PINE_DENSITY', col)]

    try:
        scaler
    except NameError:
        scaler = Scaler(train_data[scaled_params])  # Created once; reused for test/plot data

    scaler.scale(train_data)
    scaler.scale(test_data)

    # Model matrix (drop identifiers and response/derived columns).
    independent = train_data.drop(['FIPS', 'geometry','YEAR', 'SPOTS', 'const', 'Y', 'lnY', 'lnlnY', 'Yt'],
                                axis=1, errors='ignore').copy()

    # Response: transformed target (from process_data).
    y = train_data[['Yt']].copy()

    # Add intercept and coerce any object columns to numeric (safely).
    X = sm.add_constant(independent)

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fit baseline OLS with all engineered features.
    model = sm.OLS(y, X)
    result=model.fit()

    print(result.summary())

    from sklearn.feature_selection import SelectPercentile, SelectKBest, f_regression

    # Univariate screening to select top predictors (kept for diagnostics/inspection).
    X = independent.drop(['PINE_ACRES', 'TOTAL_ACRES', 'LAT', 'STD_0'], axis=1)

    selector = SelectKBest(score_func=f_regression, k=8) 
    selector.fit(X, y.values.ravel())

    X = X.loc[:, selector.get_support()]
    X = sm.add_constant(X)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Fit reduced model on screened features for comparison.
    model=sm.OLS(y, X)
    result=model.fit()
    print(result.summary())

    # Parsimonious model with key drivers used for additional diagnostics.
    np.corrcoef(data['PREV'], data['Y'])
    X = sm.add_constant(independent[['TMAX_C', 'ET_growing', 'STD_1', 'PINE_DENSITY','PREV']]) 
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')

    model = sm.OLS(y, X)
    result=model.fit()
    print(result.summary())
    
    (X.drop(columns='const', errors='ignore')
    .corr()
    .style.background_gradient())
#    .set_precision(2))

    fig, ax = plt.subplots(1, figsize=(6,4))
    ax.scatter(result.fittedvalues, result.resid, c='blue')
    ax.set_title('Residuals vs. Fitted')
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    _ = plt.xlim(limits(result.fittedvalues))
    _ = plt.ylim(limits(result.resid))
    _ = ax.plot(ax.get_xlim(),(0,0), 'r-')

    fig = sm.qqplot(result.resid, line='45', fit=True)
    fig.axes[0].set_title('Normal QQ')
    plt.show()

    fig, ax = plt.subplots(1, figsize=(6,4))
    ax.scatter(result.predict(X), y, c='blue')

    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    lim = limits(result.predict(X), y)
    _ = plt.xlim(lim)
    _ = plt.ylim(lim)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="-", color='r')

    round(result.rsquared, 3)
    from sklearn.metrics import r2_score
    round(r2_score(y, result.fittedvalues),3)

    from sklearn import linear_model, model_selection

    # Simple K-fold CV to assess generalization (using the same seed for reproducibility).
    k_fold = model_selection.KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    lm = linear_model.LinearRegression(fit_intercept=True)   # fit_intercept: whether to calculate the intercept
                                                            # for this model. If set to False, no intercept will 
                                                            # be used in calculations 
                                                            # (e.g. data is expected to be already centered).
    for train, test in k_fold.split(X):
        kmodel = lm.fit(X.iloc[train], y.iloc[train])
        print(round(kmodel.score(X.iloc[test], y.iloc[test]), 3))

    # Hold-out evaluation with the reduced feature set shape.
    test_y = test_data.loc[:, y.columns]
    test_X = sm.add_constant(test_data).loc[:, X.columns]

    test_prediction = result.predict(test_X)

    round(r2_score(test_y, test_prediction),3)

    fig, ax = plt.subplots(1, figsize=(6,4))
    ax.scatter(test_prediction, test_y, c='blue')

    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    lim = limits(test_prediction, test_y)
    _ = plt.xlim(lim)
    _ = plt.ylim(lim)
    _ = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="-", color='r')

    vif(result)
    sa(result)

    # Prepare full geometry-enabled dataset for mapping the fitted model.
    plot_data, _ = process_data(read_data(add_geometry=True,REVISION=REVISION))

    scaler.scale(plot_data)

    
    def run_model(dataframe, parameters):
        """Apply a linear model defined by params to columns of `dataframe`."""
        return (dataframe.loc[:, parameters.keys()] * parameters).sum(axis=1)


    # Predicted transformed response at each county‑year using fitted params.
    plot_data['MODEL'] = run_model(plot_data, result.params)

    plot_data
    YEAR = int(revision_str)

    states = state_subset(plot_data, YEAR)

    actual = result.model.endog_names  # usually 'Yt'
    vlim = {'vmax': max(plot_data[actual].max(), plot_data['MODEL'].max()), 
            'vmin': min(plot_data[actual].min(), plot_data['MODEL'].min())}

    ax = setup_map(f'SPOTS for {YEAR - 1}', states=states)
    _ = (plot_data
            .loc[plot_data.YEAR == YEAR - 1]
            .plot(ax=ax, column='SPOTS', alpha=0.75, legend=True, cmap='Reds'))

    ax = setup_map(f'SPOTS for {YEAR}', states=states)
    _ = (plot_data
            .loc[plot_data.YEAR == YEAR]
            .plot(ax=ax, column='SPOTS', alpha=0.75, legend=True, cmap='Reds', vmin=0, vmax=100 ))

    ax = setup_map(f'Actual Transformed Spots Per Thousand Acres of Pine for {YEAR}', states=states)
    _ = (plot_data
            .loc[plot_data.YEAR == YEAR]
            .plot(ax=ax, column=actual, alpha=0.75, legend=True, cmap='Reds', **vlim))

    ax = setup_map(f'Predicted Transformed Spots Per Thousand Acres of Pine for {YEAR}', states=states)
    _ = (plot_data
            .loc[plot_data.YEAR == YEAR]
            .plot(ax=ax, column='MODEL', alpha=0.75, legend=True, cmap='Reds', **vlim))

    from scipy.special import inv_boxcox

    # Debug breakpoint (disabled).
    # import ipdb; ipdb.set_trace()

    
    # Inverse transform to get modeled spots per thousand acres of pine.
    plot_data['MSPOTS'] = inv_boxcox(pd.to_numeric(plot_data['MODEL'],errors='coerce'), maxlog)

    # Convert to modeled total spots by scaling with pine acres.
    plot_data['PSPOTS'] = plot_data['MSPOTS'] * plot_data['PINE_ACRES']/1000

    ax = setup_map(f'Modeled Spots Per Thousand Acres for {YEAR}', states=states)
    _ =(plot_data.loc[plot_data.YEAR == YEAR].plot(ax=ax, column='PSPOTS', alpha=0.75, legend=True, cmap='Reds'))


    ax = setup_map(f'Modeled Spots for {YEAR}', states=states)
    _=(plot_data.loc[plot_data.YEAR == YEAR].plot(ax=ax, column='MSPOTS', alpha=0.75, legend=True, cmap='Reds', vmin=0,vmax=100))
    
    plt.show()

    
    # Month-by-month small models to inspect seasonal importance.
    for month in range(3,11):

        X = sm.add_constant(independent[[f'TMAX_{month}', f'ET_{month}', 'STD_1', 'PINE_DENSITY', 'PREV']])
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
        model = sm.OLS(y, X)
        result=model.fit()
        print(round(result.rsquared, 3))
    for col in plot_data.columns:
        if plot_data[col].dtype == 'object':
            plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')

    plot_data.loc[plot_data.YEAR < YEAR, ['FIPS', 'YEAR', 'Yt', 'MODEL']].eval('Yt - MODEL').hist()

    (plot_data.loc[(plot_data.YEAR < YEAR) &
                (plot_data.SPOTS > 1), 
                ['FIPS', 'YEAR', 'Yt', 'MODEL']]
            .eval('Yt - MODEL').hist())

    (plot_data.loc[(plot_data.YEAR < YEAR) &
                (plot_data.SPOTS > 1), 
                ['FIPS', 'YEAR', 'Y', 'MSPOTS']]).eval('Y - MSPOTS').hist()

    plot_data.groupby('YEAR').MSPOTS.sum().max()

    plot_data.loc[(plot_data.FIPS == 1001) & (plot_data.YEAR == 1991)]['PSPOTS']
    # Export core outputs and categorical risk tiers based on standard deviations.
    model=plot_data.loc[:, ['FIPS', 'YEAR', 'Yt', 'MODEL', 'MSPOTS', 'PSPOTS']].rename(columns={'MSPOTS': 'MODEL_SPKA', 'PSPOTS': 'MODEL_SPOTS'})
    mean,std=model.MODEL_SPOTS.mean(),model.MODEL_SPOTS.std()
    model['CATEGORY'] = np.where(model.MODEL_SPOTS >= mean, 0, 0)                             # Low
    model['CATEGORY'] = np.where(model.MODEL_SPOTS >= mean+(std*0.5), 0.5, model['CATEGORY']) # Low
    model['CATEGORY'] = np.where(model.MODEL_SPOTS >= mean+(std*1.0), 1.0, model['CATEGORY']) # Marginal
    model['CATEGORY'] = np.where(model.MODEL_SPOTS >= mean+(std*1.5), 1.5, model['CATEGORY']) # Slight
    model['CATEGORY'] = np.where(model.MODEL_SPOTS >= mean+(std*2.0), 2.0, model['CATEGORY']) # Enhanced
    model['CATEGORY'] = np.where(model.MODEL_SPOTS >= mean+(std*2.5), 2.5, model['CATEGORY']) # Moderate
    model['CATEGORY'] = np.where(model.MODEL_SPOTS >= mean+(std*3.0), 3.0, model['CATEGORY']) # High
    model.to_csv(f'output_{REVISION}.csv', index=False)
    
    #plot_data.to_file(f'output_{REVISION}.shp',driver='ESRI Shapefile')
#    plot_data.loc[:, ['FIPS', 'YEAR', 'Yt', 'MODEL', 'MSPOTS', 'PSPOTS']].rename(columns={'MSPOTS': 'MODEL_SPKA', 'PSPOTS': 'MODEL_SPOTS'}).to_csv(f'output_{REVISION}.csv', index=False)
    
    


if __name__=='__main__':

    REVISION = 20240408
    
    fcpc=f'data/cpc_{REVISION}.csv'
    prism_stable=f'data/prism_stable_{REVISION}.csv'
    prism_prov=f'data/prism_provisional_{REVISION}.csv'
    fwassi=f'data/wassi_{REVISION}.csv'
    ffia=f'data/fia_{REVISION}.csv'
    fspot=f'data/spots_{REVISION}.csv'


    forecast(REVISION,fcpc,prism_stable,prism_prov,fwassi,ffia,fspot)



