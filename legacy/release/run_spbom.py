#!/usr/bin/env python3
"""Run the archived fixed-coefficient SPBOM implementation in an isolated workspace."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

RELEASE = Path(__file__).resolve().parent
MODEL = RELEASE / 'runtime/model'
TOKEN = re.compile(r'^[A-Za-z0-9_.-]+$')
PREFIXES = ('cpc', 'fia', 'prism_stable', 'prism_provisional', 'spots')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def run_forecast(data_dir, output, revision='20250405', forecast_year=None,
                 tmax=0., et=0., pine=0., wassi_scenario=0, force=False,
                 allow_legacy_data=False, legacy_scaler=False):
    if not TOKEN.fullmatch(revision) or not re.match(r'^\d{4}', revision):
        raise ValueError('Revision must start with four year digits and use only letters, digits, _, ., -')
    values = [float(tmax), float(et), float(pine)]
    if not all(__import__('math').isfinite(x) for x in values):
        raise ValueError('Scenario adjustments must be finite')
    output = Path(output).resolve()
    if output.exists() and not force:
        raise FileExistsError('Output exists: {}'.format(output))
    data_dir = Path(data_dir).resolve()
    files = {p + '_' + revision + '.csv': data_dir / (p + '_' + revision + '.csv') for p in PREFIXES}
    if wassi_scenario is None:
        wassi = data_dir / ('wassi_' + revision + '.csv')
    else:
        wassi_scenario = int(wassi_scenario)
        suffix = ('n' if wassi_scenario < 0 else 'p') + str(abs(wassi_scenario))
        wassi = data_dir / ('wassi_' + revision + '_' + suffix + '.csv')
    files['wassi_' + revision + '.csv'] = wassi
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(str(path))
    from audit_inputs import audit
    input_report = audit(data_dir, revision)
    if input_report['blocking_issues'] and not allow_legacy_data:
        raise ValueError('Input audit failed: '+ '; '.join(input_report['blocking_issues'])+
                         '. Use --allow-legacy-data only for explicitly labelled legacy-data review runs.')
    source = data_dir / 'source'
    if not (source / 'county/county.shp').is_file() or not (source / 'county_adjacency.csv').is_file():
        raise FileNotFoundError('Need data/source/county/county.shp (+ sidecars) and county_adjacency.csv')
    inputs = [{'role': name, 'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}
              for name, path in files.items()]
    for path in sorted((source/'county').glob('county.*')):
        inputs.append({'role': 'county_geometry', 'path': str(path), 'sha256': sha256(path)})
    inputs.append({'role': 'adjacency', 'path': str(source/'county_adjacency.csv'),
                   'sha256': sha256(source/'county_adjacency.csv')})
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='spbom-') as temp:
        scratch = Path(temp)
        (scratch / 'data').mkdir()
        for name, path in files.items():
            (scratch / 'data' / name).symlink_to(path)
        (scratch / 'data/source').symlink_to(source, target_is_directory=True)
        for name in ('model_code.py', 'forecast_no_training.py'):
            shutil.copy2(MODEL / name, scratch / name)
        shutil.copy2(RELEASE/'runtime/recorded_forecast.py', scratch/'recorded_forecast.py')
        command = [sys.executable, 'recorded_forecast.py', revision,
                   str(forecast_year or revision[:4]), str(tmax), str(et), str(pine)]
        environment = dict(os.environ, SPBOM_LEGACY_SCALER='1' if legacy_scaler else '0', MPLBACKEND='Agg', MPLCONFIGDIR=str(scratch/'mplconfig'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        process = subprocess.run(command, cwd=str(scratch), env=environment,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = output.with_suffix('.log')
        log.write_text(process.stdout)
        if process.returncode:
            raise RuntimeError('Forecast failed; see {}'.format(log))
        shutil.copy2(scratch / ('output_' + revision + '.csv'), output)
        details = json.loads((scratch/'numerical_metadata.json').read_text())
    record = {'revision': revision, 'forecast_year_argument': str(forecast_year or revision[:4]),
              'scenario_year_used_by_original': int(revision[:4]), 'tmax_adjust_C': float(tmax),
              'et_adjust_percent': float(et), 'pine_adjust_percent': float(pine),
              'wassi_scenario_file_percent': wassi_scenario,
              'python': sys.version, 'elapsed_seconds': time.monotonic()-start,
              'inputs': inputs, 'output_sha256': sha256(output),
              'implementation_sha256': {p:sha256(MODEL/p) for p in ['model_code.py','forecast_no_training.py']},
              'numerics': details, 'input_audit': input_report,
              'allow_legacy_data': bool(allow_legacy_data),
              'scaler_mode': 'legacy' if legacy_scaler else 'explicit_compatibility'}
    output.with_suffix('.provenance.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=RELEASE/'local_data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', default='20250405')
    parser.add_argument('--forecast-year', help='Preserved legacy argument; does not restrict output years')
    parser.add_argument('--tmax', type=float, default=0.)
    parser.add_argument('--et', type=float, default=0.)
    parser.add_argument('--pine', type=float, default=0.)
    parser.add_argument('--wassi-scenario', default='0', help='Precomputed file percentage; "uploaded" reads wassi_REVISION.csv')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--allow-legacy-data', action='store_true', help='Explicitly allow audited legacy input defects for reproduction only')
    parser.add_argument('--legacy-scaler', action='store_true', help='Reproduce original no-op scaler under pandas 2.2.3')
    args = parser.parse_args()
    scenario = None if args.wassi_scenario == 'uploaded' else int(args.wassi_scenario)
    try:
        result = run_forecast(args.data_dir, args.output, args.revision, args.forecast_year,
                              args.tmax, args.et, args.pine, scenario, args.force,
                              args.allow_legacy_data, args.legacy_scaler)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2,str(error)+'\n')
    print(result)
if __name__ == '__main__':
    main()
