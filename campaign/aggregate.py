"""Aggregate grid_out/*.json into markdown tables → GRID_RESULTS.md."""
import os, sys, json, glob

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, 'grid_out')
REPORT = os.path.join(_HERE, 'GRID_RESULTS.md')
PARAMS = ['g', 'L_R', 'L_M', 'L_abs', 'wall', 'sensor', 'qe']


def load():
    d = {}
    for f in sorted(glob.glob(os.path.join(OUT, '*.json'))):
        try:
            d[os.path.basename(f)[:-5]] = json.load(open(f))
        except Exception as e:
            print(f'skip {f}: {e}')
    return d


def pct(x):
    return f'{x*100:.2f}%' if x < 10 else f'{x*100:.0f}%'


def main():
    d = load()
    L = []
    def emit(s=''): L.append(s)

    emit('# Grid results — CRB & fit vs N and source/location')
    emit('')

    # --- N-scan CRB (tags crb_laser_iso_N*) ---
    ns = sorted([(j['nph'], j) for k, j in d.items() if k.startswith('crb_laser_iso_N')])
    if ns:
        emit('## CRB vs N (laser_iso) — fractional σ')
        emit('')
        emit('| N_photons | ' + ' | '.join(PARAMS) + ' |')
        emit('|' + '---|' * (len(PARAMS) + 1))
        for nph, j in ns:
            emit(f'| {nph:.0e} | ' + ' | '.join(pct(j['crb'][p]) for p in PARAMS) + ' |')
        emit('')
        # 1/√N check: ratio of σ at successive N vs √(N2/N1)
        emit('√N check (σ ratio vs √(N₂/N₁)=1.73 per ×3, 1.0=perfect 1/√N, >1=flatter):')
        for p in PARAMS:
            ratios = []
            for i in range(1, len(ns)):
                n1, j1 = ns[i-1]; n2, j2 = ns[i]
                exp = (n2/n1) ** 0.5
                got = j1['crb'][p] / max(j2['crb'][p], 1e-12)
                ratios.append(got/exp)
            emit(f'  {p}: ' + ' '.join(f'{r:.2f}' for r in ratios))
        emit('')

    # --- source-combo CRB (tags crb_<src>_N1e6) ---
    sc = [(k.replace('crb_', '').replace('_N1e6', ''), j) for k, j in d.items()
          if k.startswith('crb_') and k.endswith('_N1e6') and not k.startswith('crb_laser_iso_N')]
    sc = sorted(sc, key=lambda x: x[0])
    if sc:
        emit('## CRB vs source/location combo (N=1e6) — fractional σ')
        emit('')
        emit('| source combo | nsrc | ' + ' | '.join(PARAMS) + ' |')
        emit('|' + '---|' * (len(PARAMS) + 2))
        for name, j in sc:
            emit(f'| {name} | {j.get("n_sources","?")} | '
                 + ' | '.join(pct(j['crb'][p]) for p in PARAMS) + ' |')
        emit('')

    # --- recover + shot (tags rs_*) ---
    rs = sorted([(j['nph'], k, j) for k, j in d.items() if k.startswith('rs_')])
    if rs:
        emit('## Recovery + shot-noise vs N (stabilized: bake_k+polyak+Anscombe)')
        emit('')
        emit('| tag | N | param | CRB | implicit ferr | shot bias | shot σ | σ/CRB |')
        emit('|---|---|---|---|---|---|---|---|')
        for nph, k, j in rs:
            for p in PARAMS:
                rc = j.get('recover', {}).get(p, {})
                sh = j.get('shot', {}).get(p, {})
                emit(f'| {k.replace("rs_","")} | {nph:.0e} | {p} | '
                     f'{pct(j["crb"][p])} | {pct(rc.get("ferr",float("nan"))) if rc else "—"} | '
                     f'{(sh.get("bias",0)*100):+.1f}% | {pct(sh.get("sigma",float("nan"))) if sh else "—"} | '
                     f'{(sh.get("sigma",0)/max(sh.get("crb",1e-9),1e-9)):.2f} |')
        emit('')

    with open(REPORT, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'wrote {REPORT} ({len(d)} configs)')


if __name__ == '__main__':
    main()
