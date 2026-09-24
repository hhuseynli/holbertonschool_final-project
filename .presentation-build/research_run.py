import sys, os, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'.presentation-build/research-deps'),str(root)]
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
from bakuml.pipeline import run_pipeline
for geometry in ['kdtree','h3']:
    out=root/'.presentation-build'/('research-'+geometry)
    result=run_pipeline(outdir=out,fast=False,prefer_forecaster='ridge',tessellation=geometry)
    print(geometry,json.dumps({k:v for k,v in result.items() if k not in ['walk_forward','spatial_cv']}),flush=True)
