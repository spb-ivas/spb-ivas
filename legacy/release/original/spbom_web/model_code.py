import re
from contextlib import contextmanager
from math import floor, ceil
import ipdb

# G. van Rossum, Python tutorial, Technical Report CS-R9526,
# Centrum voor Wiskunde en Informatica (CWI), Amsterdam, May 1995."
from platform import python_version
#print(f'Python {python_version()}')

# Fernando Pérez, Brian E. Granger, IPython: A System for Interactive Scientific Computing,
# Computing in Science and Engineering, vol. 9, no. 3, pp. 21-29, May/June 2007,
# doi:10.1109/MCSE.2007.53. URL: https://ipython.org
import IPython
#print(f'IPython {IPython.__version__}')

# Travis E, Oliphant. A guide to NumPy, USA: Trelgol Publishing, (2006).
import numpy as np
#print(f'Numpy {np.__version__}')

# Wes McKinney. Data Structures for Statistical Computing in Python,
# Proceedings of the 9th Python in Science Conference, 51-56 (2010) (publisher link)
import pandas as pd
#print(f'Pandas {pd.__version__}')

import geopandas as gpd
#print(f'Geopandas {gpd.__version__}')

# Hunter, John D. "Matplotlib: A 2D graphics environment."
# Computing in science & engineering 9.3 (2007): 90-95.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#print(f'Matplotlib {matplotlib.__version__}')

# Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source Scientific Tools for Python,
# 2001-, http://www.scipy.org/ [Online; accessed 2018-12-21].
import scipy
from scipy import stats
#print(f'SciPy {scipy.__version__}')

# Seabold, Skipper, and Josef Perktold. “Statsmodels: Econometric and statistical modeling with python.”
# Proceedings of the 9th Python in Science Conference. 2010.
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
#print(f'Statsmodels {statsmodels.__version__}')

# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
import sklearn
from sklearn.model_selection import train_test_split
#print(f'Scikit-learn {sklearn.__version__}')

# Herman, J. and Usher, W. (2017) SALib: An open-source Python library for sensitivity analysis.
# Journal of Open Source Software, 2(9). doi:10.21105/joss.00097
import SALib
from SALib.sample import saltelli
from SALib.analyze import sobol
#
@contextmanager
def max_rows(row_count):
    with pd.option_context('display.max_rows', row_count):
        yield


@contextmanager
def precision(places):
    with pd.option_context('display.precision', places):
        yield


def read_data(add_geometry=False, plot_rows=False, REVISION=20240408):
    cpc = pd.read_csv(f'data/cpc_{REVISION}.csv', float_precision='high',engine='c').set_index(['FIPS', 'YEAR'])
    prism_1= pd.read_csv(f'data/prism_stable_{REVISION}.csv', float_precision='round_trip')
    prism_2= pd.read_csv(f'data/prism_provisional_{REVISION}.csv', float_precision='round_trip')
    prism = pd.concat([prism_1,prism_2]).sort_values(by=['FIPS', 'YEAR', 'MONTH']).rename(columns={'TMAX_C': 'TMAX', 'PPT_MM': 'PPT', 'TAVG_C': 'TMEAN'}).pivot_table(index=['FIPS', 'YEAR'], columns='MONTH', values=['PPT', 'TMAX', 'TMEAN']).reset_index()
    prism.columns = prism.columns.map(lambda col: '_'.join([str(idx) for idx in col if idx]))
    prism = prism.set_index(['FIPS', 'YEAR'])
    prism.update(cpc)
    prism = prism.reset_index()

    #wassi = (pd.read_csv(f'data/wassi_20190206.csv', float_precision='round_trip')
    wassi = (pd.read_csv(f'data/wassi_{REVISION}.csv', float_precision='round_trip')
               .pivot_table(index=['FIPS', 'YEAR'], columns='MONTH', values=['RO', 'ET', 'SOLPSAT', 'GEP'])
               .reset_index())
    wassi.columns = wassi.columns.map(lambda col: '_'.join([str(idx) for idx in col if idx]))

    #     fia_old = pd.read_csv(f'data/fia_20181002.csv', float_precision='round_trip')
    #     fia_new = pd.read_csv(f'data/fia_{REVISION}.csv', float_precision='round_trip')
    #     fia_new = fia_new.loc[(fia_new.YEAR > fia_old.YEAR.max()) & fia_new.FIPS.isin(fia_old.FIPS)]
    #     fia = fia_old.append(fia_new).sort_values(by=['FIPS', 'YEAR']).reset_index(drop=True)

    #fia = pd.read_csv(f'data/fia_20190318.csv', float_precision='round_trip')
    #spots = pd.read_csv(f'data/spots_20190318.csv', float_precision='round_trip')
    fia = pd.read_csv(f'data/fia_{REVISION}.csv', float_precision='round_trip')
    spots = pd.read_csv(f'data/spots_{REVISION}.csv', float_precision='round_trip')

    data = pd.merge(prism, wassi, on=['FIPS', 'YEAR'], how='left')
    data = data.merge(fia, on=['FIPS', 'YEAR'], how='left')
    data.loc[:, fia.columns] = data.loc[:, fia.columns].ffill() # na(method='ffill')

    data = pd.merge(data, spots, on=['FIPS', 'YEAR'], how='left')
    data = data.sort_values(by=['FIPS', 'YEAR'])

    data['const'] = 1

    if add_geometry:
        geo = gpd.read_file('data/source/county/county.shp').rename(columns={'FIPSasnum': 'FIPS'})
        data = pd.merge(geo[['FIPS', 'geometry']], data, on='FIPS')

    if plot_rows:
        fig, ax = plt.subplots(1, figsize=(6, 4))
        ax.set_title('Row Counts')
        ax.plot(wassi.groupby('YEAR').FIPS.count(), label='wassi')
        ax.plot(prism.groupby('YEAR').FIPS.count(), label='prism')
        ax.plot(fia.groupby('YEAR').FIPS.count(), label='fia')
        ax.plot(spots.groupby('YEAR').FIPS.count(), label='spots')
        ax.plot(data.groupby('YEAR').FIPS.count(), label='data', linestyle=':')
        ax.legend(loc=1)

    return data


def _has_previous_adjacent_spots(data, limit=0):
    data['ADJ'] = 0
    data['PREV'] = data.groupby(['FIPS']).SPOTS.shift(1)
    adj = pd.read_csv('data/source/county_adjacency.csv')
    for row in data.loc[data.PREV > limit].iterrows():
        data.loc[((data.FIPS.isin(adj.loc[adj.target == int(row[1].FIPS)].adjacent.values)) &
                  (data.YEAR == int(row[1].YEAR))), 'ADJ'] = 1
    data = data.drop(columns='PREV')
    return data


def process_data(data, plot=False, Tmax_adjust=0.0, ET_adjust=0.0, Pine_density_adjust=0.0, REVISION=20240408):
    data['SPOTS'].fillna(1, inplace=True)
    data.loc[data.SPOTS == 0, 'SPOTS'] = 1

    data['Y'] = data.SPOTS / (data.PINE_ACRES / 1000)
    data['lnY'] = data.Y.apply(np.log)

    #     data['lnlnY'] = data.lnY + 7
    #     data['lnlnY'] = data.lnlnY.apply(np.log)

    xt, maxlog = _boxcox(data.Y, plot)
    data['Yt'] = xt

    data = _has_previous_adjacent_spots(data, limit=1)

    data['PREV'] = np.nan
    data['PREV'] = data.groupby(['FIPS']).SPOTS.shift(1)
    data['PREV'].fillna(1, inplace=True)
    data['PREV'] = data.PREV / (data.PINE_ACRES / 1000)
    data['PREV'] = stats.boxcox(data['PREV'], lmbda=maxlog)

    data['ET_growing'] = data.loc[:, [f'ET_{m}' for m in range(3, 11)]].mean(axis=1)
    data['PPT_growing'] = data.loc[:, [f'PPT_{m}' for m in range(3, 11)]].sum(axis=1)
    #data['TMAX_C'] = data.loc[:, [f'TMAX_{m}' for m in range(1, 13)]].max(axis=1)
    data['TMAX_C'] = data['TMAX_4'] #data.loc[:, [f'TMAX_{m}' for m in range(1, 13)]].max(axis=1)
    #data['TMAX_C'] = data.loc[:, [f'TMEAN_{m}' for m in range(1, 13)]].mean(axis=1)    
    data['TMEAN_growing'] = data.loc[:, [f'TMEAN_{m}' for m in range(3, 11)]].mean(axis=1)

    this_year=int(str(REVISION)[:4])
    #ipdb.set_trace()
    data['PINE_DENSITY'] = pd.to_numeric(data['PINE_DENSITY'], errors='coerce')
    data.loc[data['YEAR'] == this_year, 'PINE_DENSITY'] *= (1 + Pine_density_adjust/100.0)
    data.loc[data['YEAR'] == this_year, 'TMAX_C']+=Tmax_adjust
    data.loc[data['YEAR'] == this_year, 'ET_growing']*=(1+ET_adjust/100.0)
        
    return data, maxlog


def normtest(x):
    def rjust_n(*numbers, precision=2):
        numbers = [f'{n:.{precision}f}' for n in numbers]
        return tuple(str.rjust(n, len(max(numbers, key=len))) for n in numbers)

    nm_value, nm_p = stats.normaltest(x)
    jb_value, jb_p = stats.jarque_bera(x)

    nm_value, jb_value = rjust_n(nm_value, jb_value)
    nm_p, jb_p = rjust_n(nm_p, jb_p)

    print('D’Agostino-Pearson:  {}  {}'.format(nm_value, nm_p))
    print('       Jarque-Bera:  {}  {}'.format(jb_value, jb_p))


def _boxcox(data, plot=False):
    xt, maxlog, interval = stats.boxcox(data, alpha=0.5)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        prob = stats.boxcox_normplot(data, -19, 19, plot=ax1)
        ax1.plot(prob[0], prob[1], color='b')
        ax1.axvline(maxlog, color='r')
        ax1.axvline(interval[1], color='g', ls='--')
        ax1.axvline(interval[0], color='g', ls='--')

        stats.probplot(xt, dist=stats.norm, plot=ax2)

    return xt, maxlog


def setup_map(title, states=['01', '05', '12', '13', '21', '22', '28',
                             '37', '40', '45', '47', '48', '51']):
    fig, ax = plt.subplots(1, figsize=(12, 6))
    _ = ax.axis('off')
    ax.set_title(title)
    us = gpd.read_file('data/source/county/county.shp')
    (us.plot(ax=ax, color='white', edgecolor='gray', linewidth=.25))
    (us.loc[us.STATE_FIPS.isin(states)]
     .plot(ax=ax, color='gray', edgecolor='gray', linewidth=.25))
    return ax


def state_subset(dataframe, year):
    return (dataframe
            .loc[dataframe.YEAR == year]
            .FIPS
            .astype(str)
            .str.slice(0, -3)
            .str.rjust(2, '0')
            .unique()
            .tolist())


def vif(result):
    params = result.model.exog_names
    if params[0] == 'const':
        params = params[1:]
        variables = result.model.exog[:, 1:]
    else:
        variables = result.model.exog
    vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]

    print('Variance Inflation Factor')
    print('=========================\n')
    print(pd.Series(dict(zip(params, vif))).to_string())


# from sklearn.preprocessing import scale                  # Doesn't provide a means to
# train_data.loc[:, 'STD_1'] = scale(train_data.STD_1)     # scale testing data.

def sa(result):
    params = result.model.exog_names.copy()
    data = result.model.exog.copy()
    const = True if 'const' in params else False

    if const:
        params.remove('const')
        data = data[:, 1:]

    dataframe = pd.DataFrame(data, columns=params)
    problem = {
        'num_vars': len(params),
        'names': params,
        'bounds': [[dataframe[p].min(), dataframe[p].max()] for p in params]
    }

    param_values = saltelli.sample(problem, 5000)

    Y = np.zeros([param_values.shape[0]])
    for i, X in enumerate(param_values):
        if const:
            X = np.append(1, X)
        Y[i] = result.predict(exog=X)

    Si = sobol.analyze(problem, Y, print_to_console=False)

    def tprint(*args):
        print("\t\t".join([str(a) for a in list(args)]))

    print('Sensitivity Analysis')
    print('====================\n')
    print('Parameter\t1st Order\t1st Confidence\tTotal Order')
    for i in range(len(problem['names'])):
        tprint(problem['names'][i][:7], round(Si['S1'][i], 3), round(Si['S1_conf'][i], 3), round(Si['ST'][i], 3))


def limits(*series):
    return (floor(min([s.values.min() for s in series])),
            ceil(max([s.values.max() for s in series])))


class Scaler:

    def __init__(self, data):

        if isinstance(data, pd.core.frame.DataFrame):
            self.df = pd.DataFrame(columns=['MEAN', 'STD'])
            self.df.loc[:, 'MEAN'] = data.mean()
            self.df.loc[:, 'STD'] = data.std()
        elif isinstance(data, dict):
            self.df = pd.DataFrame(data)

        self.is_scaled = []

    def __repr__(self):

        return self.df.__repr__()

    def to_dict(self):

        return self.df.to_dict()

    def scale(self, dataframe):

        if id(dataframe) not in self.is_scaled:
            dataframe[self.df.index] -= self.df.MEAN
            dataframe[self.df.index] /= self.df.STD
            self.is_scaled.append(id(dataframe))

    def unscale(self, dataframe):

        if id(dataframe) in self.is_scaled:
            dataframe[self.df.index] *= self.df.STD
            dataframe[self.df.index] += self.df.MEAN
            self.is_scaled.remove(id(dataframe))
