"""Generate the fictional SPB-IVAS counties, observations, and rectangular boundaries."""
import argparse
import csv
import json
from math import cos, sin
from pathlib import Path
import random

REVISION = '20250101'


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(48119)
    climate, hydro, forest, spots, features = [], [], [], [], []
    for i in range(12):
        fips = f'99{i+1:03d}'
        lon, lat = -84 + i % 4 * 0.6, 34 + i // 4 * 0.45
        ring = [[lon,lat],[lon+0.58,lat],[lon+0.58,lat+0.43],[lon,lat+0.43],[lon,lat]]
        features.append({'type':'Feature', 'id':fips, 'properties':{'FIPS':fips, 'GEOID':fips, 'NAME':f'Example {i+1}'},
                         'geometry':{'type':'Polygon','coordinates':[ring]}})
        previous = rng.randrange(8)
        for year in range(2014,2026):
            fraction = round(0.3 + rng.uniform(0,0.4), 6)
            acres = round((32000 + 913*i)*fraction, 6)
            origin = round(rng.uniform(0.05,0.9), 6)
            forest.append([fips, year, acres, 32000+913*i, fraction, origin])
            summer = rng.uniform(-2.5,2.5)
            for month in range(1,13):
                seasonal = sin((month-3)*3.141592653589793/6)
                tmax = round(19+summer+9*seasonal+i/4,4)
                rain = round(70+rng.uniform(0,35)+7*cos(month),4)
                et = round(44+13*seasonal+0.04*rain+rng.uniform(-4,4),4)
                climate.append([fips,year,month,rain,tmax,tmax-6,tmax-11,'synthetic'])
                hydro.append([fips,year,month,et,0.4*rain,0.55,1.0])
            count = max(0,round(3+0.28*previous+2*summer-2*fraction+2*origin+rng.uniform(-3,4)))
            if year < 2025:
                spots.append([fips,year,count])
            previous=count
    def table(name, cols, rows):
        with (directory/f'{name}_{REVISION}.csv').open('w',newline='') as stream:
            writer=csv.writer(stream, lineterminator='\n')
            writer.writerow(cols.split(','))
            writer.writerows(rows)
    names='FIPS,YEAR,MONTH,PPT_MM,TMAX_C,TAVG_C,TMIN_C,STATUS'
    table('prism_stable',names,climate)
    table('prism_provisional',names,[])
    table('cpc','FIPS,YEAR,TMEAN_4,PPT_4',[])
    table('fia','FIPS,YEAR,PINE_ACRES,TOTAL_ACRES,PINE_DENSITY,STD_1',forest)
    table('wassi','FIPS,YEAR,MONTH,ET,RO,SOLPSAT,GEP',hydro)
    table('spots','FIPS,YEAR,SPOTS',spots)
    (directory/'counties.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'demo')
    generate(parser.parse_args().output)
