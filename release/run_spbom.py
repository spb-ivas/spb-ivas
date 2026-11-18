"""Prepare county inputs, fit a model, or predict using an exported calibration."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from prepared import prepare
from fitted import export_identity, fit, predict, validate_model

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'examples/demo'
DEMO_MODEL = ROOT / 'examples/demo-model.json'


def atomic_output(path, write):
    """Readers see either the previous complete file or the new complete file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8', newline='') as stream:
        staged = Path(stream.name)
        try:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            staged.unlink()
            raise
    try:
        staged.replace(path)
    finally:
        if staged.exists():
            staged.unlink()


def save_json(path, obj):
    def write(stream):
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')
    atomic_output(path, write)


def read_model(path):
    with path.open(encoding='utf-8') as stream:
        model = json.load(stream)
    validate_model(model)
    return model


def add_data_options(parser):
    parser.add_argument('--data-dir', type=Path, default=DEMO, help='Directory of revision-tagged prepared county tables (default: synthetic example)')
    parser.add_argument('--revision', default='20250101', help='Input revision (default: 20250101, the synthetic example)')


def main():
    if sys.argv[1:2] == ['--legacy']:
        path = ROOT / 'legacy/release/run_spbom.py'
        os.execv(sys.executable, [sys.executable, str(path)] + sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__, epilog='For historical reproduction use --legacy followed by the archived launcher options.')
    commands = parser.add_subparsers(dest='action', required=True)
    for name in ('audit', 'fit', 'predict'):
        command = commands.add_parser(name)
        add_data_options(command)
        command.add_argument('--output', type=Path, required=True)
        if name == 'fit':
            command.add_argument('--through-year', type=int, required=True, help='Last inclusive calibration year')
        if name == 'predict':
            command.add_argument('--model', required=True, type=Path)
            command.add_argument('--year', type=int, required=True, help='Only this year receives scenario changes')
            command.add_argument('--tmax', type=float, default=0, help='Add degrees Celsius to April maximum temperature')
            command.add_argument('--et', type=float, default=0, help='Percentage change in mean monthly March–October ET')
            command.add_argument('--pine', type=float, default=0, help='Relative percent change in pine-area fraction predictor, with reporting area held fixed')
    args = parser.parse_args()
    try:
        # Keep both inputs and any existing model separate from output destinations.
        if args.output.resolve().parent == args.data_dir.resolve() and args.output.exists():
            raise ValueError('Choose an output outside the input directory to replace an existing file')
        if args.action == 'predict' and args.model.resolve() == args.output.resolve():
            raise ValueError('Model and prediction output must use distinct paths')
        frame, audit = prepare(args.data_dir, args.revision)
        if args.action == 'audit':
            save_json(args.output, audit)
        elif args.action == 'fit':
            provenance = {'revision': args.revision, 'input_sha256': audit['input_sha256']}
            save_json(args.output, fit(frame, args.through_year, provenance))
        else:
            model = read_model(args.model)
            result = predict(frame, model, args.year, tmax=args.tmax, et=args.et, pine=args.pine)
            for key, value in export_identity(frame, model).items():
                result[key] = value
            for name in ('tmax', 'et', 'pine'):
                result[name + '_adjustment'] = result.YEAR.eq(args.year).astype(int) * getattr(args, name)
            atomic_output(args.output, lambda stream: result.to_csv(stream, index=False))
    except (ValueError, OSError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
