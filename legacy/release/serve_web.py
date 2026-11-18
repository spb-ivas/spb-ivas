#!/usr/bin/env python3
"""Launch the portable copy of the existing GUI for local review."""
import argparse
import os
from pathlib import Path
import sys
RELEASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--port',type=int,default=8050)
parser.add_argument('--allow-legacy-data',action='store_true',help='Allow supplied input defects for local review only')
args=parser.parse_args()
if args.allow_legacy_data:os.environ['SPBOM_ALLOW_LEGACY_DATA']='1'
sys.path.insert(0,str(RELEASE/'runtime/web'))
import static
static.app.run(host='127.0.0.1',port=args.port,debug=False)
