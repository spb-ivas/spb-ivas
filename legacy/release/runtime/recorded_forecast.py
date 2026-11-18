"""Instrument metadata only, then execute the preserved regression CLI unchanged."""
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys
import model_code

metadata = {'versions': {p: importlib.metadata.version(p) for p in
                        ['numpy','pandas','scipy','matplotlib','geopandas','statsmodels','scikit-learn','SALib']}}
original_boxcox = model_code._boxcox
def boxcox(data, plot=False):
    result = original_boxcox(data, plot)
    metadata['boxcox_lambda'] = float(result[1])
    return result
model_code._boxcox = boxcox
original_init = model_code.Scaler.__init__
def scaler_init(self, data):
    if os.environ.get('SPBOM_LEGACY_SCALER') == '1':
        self.df = model_code.pd.DataFrame(columns=['MEAN', 'STD'])
        self.df.loc[:, 'MEAN'] = data.mean()
        self.df.loc[:, 'STD'] = data.std()
        self.is_scaled = []
    else:
        original_init(self, data)
    metadata['scaler'] = self.to_dict()
    metadata['scaler_mode'] = 'legacy' if os.environ.get('SPBOM_LEGACY_SCALER') == '1' else 'explicit_compatibility'
model_code.Scaler.__init__ = scaler_init
for node in ast.parse(Path('forecast_no_training.py').read_text()).body:
    if isinstance(node, ast.If):
        for sub in node.body:
            if isinstance(sub,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='spbom_model' for x in sub.targets):
                metadata['coefficients'] = ast.literal_eval(sub.value.args[0])
sys.argv[0] = 'forecast_no_training.py'
runpy.run_path('forecast_no_training.py', run_name='__main__')
Path('numerical_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
