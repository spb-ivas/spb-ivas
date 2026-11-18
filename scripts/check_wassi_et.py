"""Replay watershed ET with separately supplied WaSSI Fortran source and inputs.

The county demos start with prepared ET. The optional 2026 replay retains the
empty routing file because WaSSI writes watershed ET before stream routing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd


def sha(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(model, revision, fortran_dir):
    data = model/'data'
    uid = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456'
    forcing = data/'wassi_input_final_copy.csv'
    config = model/'WASSICB_WEB_2_1/USERS/SGCP/Inputs'
    overlap = data/'source/county_proportion.csv'
    reference = data/f'wassi_{revision}_p0.csv'
    with tempfile.TemporaryDirectory(prefix='wassi-et-') as scratch:
        root = Path(scratch)
        inputs = root/'USERS/SGCP/Inputs'
        # Materialize only required input files; outputs and source edits stay inside scratch.
        shutil.copytree(config, inputs, ignore=shutil.ignore_patterns('CLIMATE'))
        climate = inputs/'INPUTS_US/CLIMATE'
        climate.mkdir()
        shutil.copyfile(forcing, climate/'PRISM_CLIMATE_US.TXT')
        output = root/'USERS/SGCP/Outputs/OUTPUTS_US'
        output.mkdir(parents=True)
        shutil.copytree(fortran_dir, root/'codes')
        general = root/'codes/General.for'
        code = general.read_text()
        assert code.count("BASE_PATH='/root/spbom/spbom_web/WASSICB_WEB_2_1'") == 1
        general.write_text(code.replace('CHARACTER*37 BASE_PATH','CHARACTER*1 BASE_PATH')
                           .replace("BASE_PATH='/root/spbom/spbom_web/WASSICB_WEB_2_1'", "BASE_PATH='.'"))
        prog = root/'wassi'
        subprocess.run(['gfortran','-std=legacy','-fallow-argument-mismatch','-ffixed-line-length-none',
                        '-o',str(prog),*sorted((root/'codes').glob('*.for'))],cwd=root,check=True)
        done = subprocess.run([str(prog)], cwd=root, check=True, capture_output=True, text=True)
        if 'SIMULATION COMPLETE' not in done.stdout:
            raise ValueError('Fortran did not reach its completion marker')
        runoff = output/f'MONTHRUNOFF_US_{uid}.TXT'
        monthly = pd.read_csv(runoff,usecols=['CELL','YEAR','MONTH','AET_MM'])
        weights = pd.read_csv(overlap)
        keys = ['YEAR','MONTH','FIPS']
        county = monthly.merge(weights,left_on='CELL',right_on='huc8').rename(columns={'fips':'FIPS'})
        county['et_mm'] = county.AET_MM * county.proportion
        regen = county.groupby(keys).et_mm.sum(min_count=1)
        obs = pd.read_csv(reference,usecols=keys+['ET']).set_index(keys,verify_integrity=True).ET
        regen = regen[regen.index.get_level_values('YEAR') >= 1990].sort_index()
        obs = obs.sort_index()
        if obs.empty or not regen.index.equals(obs.index):
            raise ValueError('County-month keys differ from baseline')
        delta = np.abs(regen-obs)
        return {'revision':revision, 'forcing_sha256':sha(forcing), 'reference_sha256':sha(reference),
                'control_sha256':sha(inputs/f'GENERAL_{uid}.TXT'), 'overlap_sha256':sha(overlap),
                'monthly_runoff_sha256':sha(runoff),'routing_matrix_bytes':(inputs/'INPUTS_US/ROUTEMATRIX_US.TXT').stat().st_size,
                'county_months':len(obs),'years':[int(monthly.YEAR.min()),int(monthly.YEAR.max())],
                'maximum_absolute_ET_difference_mm':float(delta.max()),
                'matching_ET_at_rtol_atol_1e-8':bool(np.allclose(regen, obs, atol=1e-8,rtol=1e-8))}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',required=True,type=Path)
    parser.add_argument('--fortran-dir', required=True, type=Path,
                        help='Original WaSSI codes directory containing General.for and the other Fortran modules')
    parser.add_argument('--revision',default='20260406')
    args=parser.parse_args()
    report = verify(args.model_dir.resolve(),args.revision,args.fortran_dir.resolve())
    print(json.dumps(report,indent=2,allow_nan=False))
    if not report['matching_ET_at_rtol_atol_1e-8']:
        raise SystemExit('County-month ET differs from the supplied baseline')
