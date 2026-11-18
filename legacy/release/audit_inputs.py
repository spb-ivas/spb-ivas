#!/usr/bin/env python3
"""Report input defects without modifying inputs or changing legacy model behavior."""
import argparse
import json
from pathlib import Path
import pandas as pd
import geopandas as gpd

def audit(data_dir,revision='20250405'):
    data_dir=Path(data_dir)
    report={'revision':revision,'tables':{},'blocking_issues':[]}
    frames={}
    for prefix in ['cpc','fia','prism_stable','prism_provisional','spots']:
        frame=pd.read_csv(data_dir/f'{prefix}_{revision}.csv');frames[prefix]=frame
        key=['FIPS','YEAR']+(['MONTH'] if prefix.startswith('prism') else [])
        duplicates=int(frame.duplicated(key).sum())
        report['tables'][prefix]={'rows':len(frame),'counties':int(frame.FIPS.nunique()),
                                 'years':[int(frame.YEAR.min()),int(frame.YEAR.max())],
                                 'key':key,'duplicate_key_rows':duplicates,
                                 'null_key_rows':int(frame[key].isna().any(axis=1).sum())}
        if duplicates:report['blocking_issues'].append(f'{prefix}: {duplicates} duplicate key rows')
    geo=gpd.read_file(data_dir/'source/county/county.shp')
    raw_climate=pd.concat([frames['prism_stable'],frames['prism_provisional']])
    climate=set(raw_climate.dropna(subset=['TMAX_C','PPT_MM','TAVG_C'],how='all').FIPS)
    report['all_missing_climate_counties']=sorted(int(x) for x in set(raw_climate.FIPS)-climate)
    counties=climate & set(geo.FIPSasnum)
    missing=sorted(int(x) for x in counties-set(frames['fia'].FIPS))
    report['climate_geometry_counties']=len(counties)
    report['counties_without_any_fia_records']=missing
    if missing:report['blocking_issues'].append(f'{len(missing)} climate/geometry counties have no FIA records; legacy ungrouped ffill can cross counties')
    report['publication_grade_inputs']=not report['blocking_issues']
    report['mode']='diagnostic only: no deduplication, imputation changes, or source writes'
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--revision',default='20250405')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--strict',action='store_true')
    args=parser.parse_args();result=audit(args.data_dir,args.revision)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    else:print(text,end='')
    if args.strict and not result['publication_grade_inputs']:raise SystemExit(2)
