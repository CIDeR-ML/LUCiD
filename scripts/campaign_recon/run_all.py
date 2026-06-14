"""Dispatch the 100-event seeded recon across NGPU GPUs (contiguous event blocks, one job/GPU).
Usage: NGPU=5 NEVENTS=100 python scripts/campaign_recon/run_all.py"""
import os, sys, subprocess, time

_HERE = os.path.dirname(os.path.abspath(__file__))
NGPU = int(os.environ.get('NGPU', '5')); NEV = int(os.environ.get('NEVENTS', '100'))
OUT = os.environ.get('OUT', os.path.join(_HERE, 'out')); os.makedirs(OUT, exist_ok=True)
per = (NEV + NGPU - 1) // NGPU
jobs = [(g, g * per, min(per, NEV - g * per)) for g in range(NGPU) if g * per < NEV]
procs = []
for g, start, count in jobs:
    e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), EVENT_START=str(start),
             EVENT_COUNT=str(count), OUT=OUT)
    e.pop('JAX_PLATFORM_NAME', None)
    lf = open(os.path.join(OUT, f'gpu{g}.log'), 'w')
    p = subprocess.Popen([sys.executable, os.path.join(_HERE, 'worker.py')],
                         env=e, stdout=lf, stderr=subprocess.STDOUT)
    procs.append((g, p, lf, start, count)); print(f'[gpu{g}] events {start}..{start+count-1}', flush=True)
t0 = time.time()
for g, p, lf, start, count in procs:
    p.wait(); lf.close()
    print(f'[gpu{g}] done ({p.returncode}) events {start}..{start+count-1} [{(time.time()-t0)/60:.1f} min]', flush=True)
print(f'ALL DONE in {(time.time()-t0)/60:.1f} min', flush=True)
