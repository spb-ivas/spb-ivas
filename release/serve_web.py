"""Launch the fitted-model scenario explorer on the local machine."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

from run_spbom import DEMO, DEMO_MODEL, ROOT, add_data_options, prepare, read_model


def read_geometry(path, tolerance=None):
    """Simplify large geographic boundaries for display only; inference uses county keys."""
    if tolerance is None:
        tolerance = 0.003 if path.stat().st_size > 8 * 1024 ** 2 else 0.
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Geometry tolerance must be a finite, nonnegative number of degrees')
    geometry = json.loads(path.read_text())
    if geometry.get('type') != 'FeatureCollection':
        raise ValueError('County geometry must be a GeoJSON FeatureCollection')
    if tolerance:
        from shapely.geometry import shape, mapping
        for feature in geometry['features']:
            boundary = feature.get('geometry')
            if boundary and boundary.get('type') in ('Polygon', 'MultiPolygon'):
                polygon = shape(boundary).simplify(tolerance, preserve_topology=True)
                feature['geometry'] = mapping(polygon)
    return geometry


def main():
    if sys.argv[1:2] == ['--legacy']:
        path = ROOT / 'legacy/release/serve_web.py'
        os.execv(sys.executable, [sys.executable, str(path)] + sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_options(parser)
    parser.add_argument('--model', type=Path, help='Initial fitted JSON model. The synthetic example has a bundled model.')
    parser.add_argument('--geometry', type=Path, help='Local county GeoJSON with five-digit FIPS feature IDs')
    parser.add_argument('--geometry-tolerance', type=float, help='Display simplification in degrees; 0 keeps all vertices. Default: 0.003 for files over 8 MiB, otherwise 0.')
    parser.add_argument('--port', type=int, default=8050)
    args = parser.parse_args()
    from scenario_web import create_app
    try:
        if args.model is None and (args.data_dir.resolve() != DEMO.resolve() or args.revision != '20250101'):
            raise ValueError('Select a fitted --model for these inputs')
        frame, _ = prepare(args.data_dir, args.revision)
        if args.data_dir.resolve() == DEMO.resolve() and args.revision == '20250101':
            frame.attrs['display_context'] = 'Fictional county demonstration'
        model = read_model(args.model or DEMO_MODEL)
        geometry_path = args.geometry or args.data_dir / 'counties.geojson'
        geometry = None
        if args.geometry or geometry_path.is_file():
            geometry = read_geometry(geometry_path, args.geometry_tolerance)
        app = create_app(frame, model, geojson=geometry)
        app.run(host='127.0.0.1', port=args.port, debug=False)
    except (ValueError, OSError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
