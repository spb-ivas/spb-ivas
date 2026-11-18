"""Local paths and per-run isolation for the preserved SPBOM GUI."""
import base64
import os
from pathlib import Path
import sys
import tempfile
import uuid
from urllib.parse import quote
import dash
from dash import html
from dash.exceptions import PreventUpdate

RELEASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(RELEASE))
from run_spbom import run_forecast,TOKEN
DATA_DIR=Path(os.environ.get('SPBOM_DATA_DIR',str(RELEASE/'local_data'))).resolve()
OUTPUT_DIR=Path(os.environ.get('SPBOM_OUTPUT_DIR',str(RELEASE/'web_outputs'))).resolve()
DEFAULT_OUTPUT=Path(os.environ.get('SPBOM_DEFAULT_OUTPUT',str(RELEASE/'results/baseline.csv'))).resolve()
COUNTIES=DATA_DIR/'counties.geojson'
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

def allowed_output(path):
    path=Path(path).resolve()
    if path.parent!=OUTPUT_DIR or not path.name.startswith('output_') or path.suffix!='.csv':
        raise ValueError('Select a run output within the configured output directory')
    return path

def run_callback(values):
    ctx=dash.callback_context
    if not ctx.triggered:raise PreventUpdate
    official=ctx.triggered[0]['prop_id'].split('.')[0]=='run-official-script-button'
    response=[dash.no_update]*13
    response[6]='auto';response[7]=''
    try:
        if official:
            revision='20250405'
            tag=str(values['official_tag'] or 'run')
            if not TOKEN.fullmatch(tag):raise ValueError('Run label may use letters, digits, _, ., -')
            suffix=tag+'-'+uuid.uuid4().hex[:8]
            output=OUTPUT_DIR/f'output_{revision}_{suffix}.csv'
            run_forecast(DATA_DIR,output,revision,'2025',values['official_tmax_adjust'],
                         values['official_et_adjust'],values['official_pine_density_adjust'],
                         values['official_et_adjust'],allow_legacy_data=os.environ.get('SPBOM_ALLOW_LEGACY_DATA')=='1')
        else:
            revision=str(values['user_revision'] or '')
            if not TOKEN.fullmatch(revision) or not revision[:4].isdigit():raise ValueError('Revision must begin with four year digits')
            suffix='user-'+uuid.uuid4().hex[:8]
            output=OUTPUT_DIR/f'output_{revision}_{suffix}.csv'
            with tempfile.TemporaryDirectory(prefix='spbom-upload-') as temp:
                data=Path(temp);(data/'source').symlink_to(DATA_DIR/'source',target_is_directory=True)
                for prefix in ['cpc','fia','prism_30yr_normal','prism_provisional','prism_stable','spots','wassi']:
                    content=values[prefix+'_content'];name=values[prefix+'_filename']
                    if name!=f'{prefix}_{revision}.csv' or not content:
                        raise ValueError('Upload '+f'{prefix}_{revision}.csv')
                    (data/name).write_bytes(base64.b64decode(content.split(',',1)[1],validate=True))
                run_forecast(data,output,revision,revision[:4],values['user_tmax_adjust'],
                             values['user_et_adjust'],values['user_pine_density_adjust'],None,
                             allow_legacy_data=os.environ.get('SPBOM_ALLOW_LEGACY_DATA')=='1')
        message_index=0 if official else 3
        response[message_index]='SPBOM model completed successfully (legacy reproduction).'
        response[message_index+1]=html.A('Download '+output.name,href='/download/output/'+quote(revision)+'/'+quote(suffix))
        response[message_index+2]={'display':'block'}
        response[8]=str(output)
        response[9 if official else 11]={'background-color':'#1f77b4','color':'white','cursor':'pointer'}
        response[10 if official else 12]=False
    except (ValueError,OSError,RuntimeError) as error:
        response[0 if official else 3]='Forecast failed: '+str(error)
    return response
