"""Dispatch a grid of gridfit.py configs across all GPUs (one process per GPU, round-robin).

Usage: NGPU=5 PHASE=<phase> python campaign/run_grid.py
Phases: 'crb_nscan', 'crb_srcscan', 'recover_shot_nscan', 'all_crb'.
Each job runs `CUDA_VISIBLE_DEVICES=<g> TAG=<tag> <env> python gridfit.py`, logging to
grid_out/<tag>.log; results land in grid_out/<tag>.json.
"""
import os, sys, subprocess, time

_HERE = os.path.dirname(os.path.abspath(__file__))
NGPU = int(os.environ.get('NGPU', '5'))
PHASE = os.environ.get('PHASE', 'all_crb')
OUT = os.path.join(_HERE, 'grid_out'); os.makedirs(OUT, exist_ok=True)

SRC_KEYS = ['laser_down', 'laser_up', 'laser_wall', 'laser_diag', 'iso_center',
            'iso_off', 'iso_top', 'laser_iso', 'multi_laser', 'multi_laser_iso',
            'iso_ring', 'all']
N_LIST = ['1e5', '3e5', '1e6', '3e6', '1e7']     # 1e7 probes the 11GB ceiling (may OOM)


def J(tag, **env):
    return (tag, {k: str(v) for k, v in env.items()})


def grid(phase):
    # GRID='0' (reduced geometry) for 11GB memory headroom + consistency across N.
    if phase == 'crb_nscan':
        return [J(f'crb_laser_iso_N{n}', NPH=n, SRC='laser_iso', GRID='0', NB_H='3') for n in N_LIST]
    if phase == 'crb_srcscan':
        return [J(f'crb_{s}_N1e6', NPH='1e6', SRC=s, GRID='0', NB_H='3') for s in SRC_KEYS]
    if phase == 'all_crb':
        return grid('crb_nscan') + grid('crb_srcscan')
    if phase == 'recover_shot_nscan':
        # implicit recovery + shot-noise scatter (stabilized recipe) vs N, two source sets
        out = []
        for s in ['laser_iso', 'multi_laser_iso']:
            for n in N_LIST:
                out.append(J(f'rs_{s}_N{n}', NPH=n, SRC=s, GRID='1', NB_H='2',
                             RECOVER='1', SHOT='1', M='4', STEPS='60',
                             BAKE_K='1', POLYAK='12', EPS='0.375'))
        return out
    raise SystemExit(f'unknown phase {phase}')


def main():
    jobs = grid(PHASE)
    print(f'PHASE={PHASE}: {len(jobs)} jobs across {NGPU} GPUs', flush=True)
    queue = list(jobs)
    running = {}            # gpu -> (proc, tag, logfile, t0)
    free = list(range(NGPU))
    t_start = time.time()
    while queue or running:
        while queue and free:
            g = free.pop(0)
            tag, env = queue.pop(0)
            e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), TAG=tag)
            e.update(env)
            e.pop('JAX_PLATFORM_NAME', None)             # ensure GPU
            lf = open(os.path.join(OUT, f'{tag}.log'), 'w')
            p = subprocess.Popen([sys.executable, os.path.join(_HERE, 'gridfit.py')],
                                 env=e, stdout=lf, stderr=subprocess.STDOUT)
            running[g] = (p, tag, lf, time.time())
            print(f'  [gpu{g}] launch {tag}', flush=True)
        for g, (p, tag, lf, t0) in list(running.items()):
            if p.poll() is not None:
                lf.close()
                ok = 'ok' if p.returncode == 0 else f'FAIL({p.returncode})'
                print(f'  [gpu{g}] {tag} {ok} ({time.time()-t0:.0f}s) '
                      f'[{len(queue)} queued]', flush=True)
                del running[g]; free.append(g)
        time.sleep(5)
    print(f'PHASE={PHASE} complete in {(time.time()-t_start)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
