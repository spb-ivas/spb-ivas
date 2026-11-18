from model_code import *
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import sys  # Import sys module
from scipy.special import inv_boxcox
import ipdb

def forecast_no_training(REVISION,fcpc,prism_stable,prism_prov,fwassi,ffia,fspot, forecast_year, parameters,Tmax_adjust, ET_adjust, Pine_density_adjust):
    current_dir=os.getcwd()
    print("Current Directory:", current_dir)
    pdf_pages = PdfPages('multiple_spbom_plots.pdf')
    
    RANDOM_SEED = 1314897167

    Tmax_adjust=float(Tmax_adjust)
    ET_adjust=float(ET_adjust)
    Pine_density_adjust = float(Pine_density_adjust)

    plot_data, maxlog = process_data(read_data(add_geometry=True,REVISION=REVISION),False,Tmax_adjust, ET_adjust, Pine_density_adjust, REVISION=REVISION)
    
    scaled_params = [col for col in list(plot_data.columns) if re.match(r'ET|PPT|RO|TMIN|TMAX|TMEAN|PINE_DENSITY', col)]
#    ipdb.set_trace()
    try:
        scaler
    except NameError:
        scaler = Scaler(plot_data[scaled_params])

    scaler.scale(plot_data)

    def run_model(dataframe, parameters):
        
        return (dataframe.loc[:, parameters.keys()] * parameters).sum(axis=1)


    plot_data['MODEL'] = run_model(plot_data, parameters)

    plot_data_copy=plot_data.copy()
    
    columns_to_keep = ['FIPS','MODEL']
    columns_to_keep = ['FIPS', 'TMAX_C', 'ET_growing', 'STD_1', 'PINE_DENSITY', 'PREV','MODEL']
    YEAR = forecast_year
    
    states = state_subset(plot_data, YEAR)

    actual = 'Yt'
    plot_data_copy['MSPOTS'] = inv_boxcox(pd.to_numeric(plot_data_copy['MODEL'],errors='coerce'), maxlog)
    plot_data_copy['PSPOTS'] = plot_data_copy['MSPOTS'] * plot_data_copy['PINE_ACRES']/1000

    scaler.unscale(plot_data)
    plot_data['tmax']=plot_data['TMAX_C']
#    scaler.unscale(plot_data['tmax'])

                
    plot_data['rain'] = plot_data['PPT_growing']
    plot_data['pine_density'] = plot_data['PINE_DENSITY']
    plot_data['evap_transp'] = plot_data['ET_growing']*25.4
    plot_data['spots'] = plot_data_copy['SPOTS']

    plot_data['MSPOTS']=plot_data_copy['MSPOTS']
    plot_data['PSPOTS']=plot_data_copy['PSPOTS']
    
#    ipdb.set_trace()
    model=plot_data.loc[:, ['FIPS', 'YEAR', 'Yt', 'MSPOTS', 'PSPOTS','rain','pine_density','evap_transp','tmax','spots']].rename(columns={'MSPOTS': 'MODEL_SPKA', 'PSPOTS': 'MODEL_SPOTS'})
    mean,std=model.MODEL_SPOTS.mean(),model.MODEL_SPOTS.std()
    model['CATEGORY'] = np.where(model.MODEL_SPOTS >= mean, 0, 0)                             # Low
    model['CATEGORY'] = np.where(model.MODEL_SPOTS >= mean+(std*0.5), 0.5, model['CATEGORY']) # Low
    model['CATEGORY'] = np.where(model.MODEL_SPOTS >= mean+(std*1.0), 1.0, model['CATEGORY']) # Marginal
    model['CATEGORY'] = np.where(model.MODEL_SPOTS >= mean+(std*1.5), 1.5, model['CATEGORY']) # Slight
    model['CATEGORY'] = np.where(model.MODEL_SPOTS >= mean+(std*2.0), 2.0, model['CATEGORY']) # Enhanced
    model['CATEGORY'] = np.where(model.MODEL_SPOTS >= mean+(std*2.5), 2.5, model['CATEGORY']) # Moderate
    model['CATEGORY'] = np.where(model.MODEL_SPOTS >= mean+(std*3.0), 3.0, model['CATEGORY']) # High

    model=model.loc[:,['FIPS', 'YEAR', 'Yt', 'MODEL_SPKA', 'MODEL_SPOTS','rain','pine_density','evap_transp','tmax','spots','CATEGORY']].rename(columns={
        'MODEL_SPKA': 'SPB_per_thousand_acre',
        'MODEL_SPOTS': 'SPB_spots',
        'rain': 'Rain',
        'pine_density': 'Pine_density',
        'evap_transp': 'Evap_Transporation',
        'tmax':'Max April Temperature',
        'spots': 'Actual spots',
        'CATEGORY':'Risk_category'
        })

    #model=model.dropna()

    model.to_csv(f'output_{REVISION}.csv', index=False)


if __name__=='__main__':

    if len(sys.argv) != 6:
        print("Usage: python forecast_no_training.py <REVISION> <forecast_year>")
        sys.exit(1)
    
    # Retrieve command line arguments
    REVISION = sys.argv[1]
    forecast_year = sys.argv[2]
    Tmax_adjust=sys.argv[3]
    ET_adjust=sys.argv[4]
    Pine_density_adjust=sys.argv[5]

    # Set model coefficient
    spbom_model = pd.Series({'const': -1.1324, 'TMAX_C': -0.060, 'ET_growing': 0.098, 'STD_1': -0.323, 'PINE_DENSITY': -0.384, 'PREV': 0.729})
    spbom_model = pd.Series({'const': -1.0736, 'TMAX_C': -0.0189, 'ET_growing': 0.0448, 'STD_1': -0.3040, 'PINE_DENSITY': -0.3679, 'PREV': 0.7367})
    spbom_model = pd.Series({'const': -1.0799, 'TMAX_C': -0.0462, 'ET_growing': 0.0194, 'STD_1': -0.2828, 'PINE_DENSITY': -0.3626, 'PREV': 0.7365})
    
    # Set the input data paths
    fcpc=f'data/cpc_{REVISION}.csv'
    prism_stable=f'data/prism_stable_{REVISION}.csv'
    prism_prov=f'data/prism_provisional_{REVISION}.csv'
    fwassi=f'data/wassi_{REVISION}.csv'
    ffia=f'data/fia_{REVISION}.csv'
    fspot=f'data/spots_{REVISION}.csv'
    
    # Run the Model    
    forecast_no_training(REVISION,fcpc,prism_stable,prism_prov,fwassi,ffia,fspot,forecast_year,spbom_model,Tmax_adjust,ET_adjust,Pine_density_adjust)
