import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.run([sys.executable, str(ROOT/'release/run_spbom.py'), *map(str,args)],
                          check=True, capture_output=True, text=True)


def test_demo_cli_calibrate_reload_and_export(tmp_path):
    model = tmp_path/'fit.json'
    command('fit', '--through-year', 2021, '--output', model)
    data = json.loads(model.read_text())
    assert data['training']['through_year'] == 2021
    assert len(data['training']['provenance']['input_sha256']) == 6
    tables=[]
    for change in (0,2):
        out=tmp_path/f'output{change}.csv'
        command('predict','--model',model,'--year',2025,'--tmax',change,'--output',out)
        tables.append(pd.read_csv(out,dtype={'FIPS':str}).set_index(['FIPS','YEAR']))
    base, warm=tables
    before=base.index.get_level_values('YEAR') != 2025
    pd.testing.assert_frame_equal(base.loc[before],warm.loc[before],check_exact=True)
    assert warm.loc[~before,'SPB_spots'].gt(base.loc[~before,'SPB_spots']).all()
    assert warm.loc[~before,'Actual spots'].isna().all()
    assert warm.index.is_unique
    assert warm.model_sha256.str.fullmatch('[0-9a-f]{64}').all()
    assert warm.input_manifest_sha256.str.fullmatch('[0-9a-f]{64}').all()
    assert (warm.loc[~before, 'tmax_adjustment'] == 2).all()


def test_generate_inputs_matches_shipped_demo(tmp_path):
    subprocess.run([sys.executable,str(ROOT/'examples/generate_demo.py'),'--output',str(tmp_path)], check=True)
    for path in sorted((ROOT/'examples/demo').iterdir()):
        assert (tmp_path/path.name).read_bytes()==path.read_bytes()
