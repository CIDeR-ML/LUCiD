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

    def budget(tag, j):
        if 'intens' in j:
            return float(j['intens'])
        return float(tag.split('_I')[-1]) if '_I' in tag else float(j.get('nph', 0))

    # --- N-scan CRB: the physical-budget scan (tags crb_laser_iso_I*) ---
    ns = sorted([(budget(k, j), j) for k, j in d.items() if k.startswith('crb_laser_iso_I')])
    if ns:
        emit('## CRB vs photon budget (laser_iso, NPH=1e6 fixed) — fractional σ')
        emit('')
        emit('| budget | ' + ' | '.join(PARAMS) + ' |')
        emit('|' + '---|' * (len(PARAMS) + 1))
        for b, j in ns:
            emit(f'| {b:.0e} | ' + ' | '.join(pct(j['crb'][p]) for p in PARAMS) + ' |')
        emit('')
        emit('√budget check (σ-ratio / √(b₂/b₁); 1.0=perfect 1/√budget, >1=flatter/'
             'systematics-limited, <1=steeper):')
        for p in PARAMS:
            ratios = []
            for i in range(1, len(ns)):
                b1, j1 = ns[i-1]; b2, j2 = ns[i]
                exp = (b2/b1) ** 0.5
                ratios.append((j1['crb'][p] / max(j2['crb'][p], 1e-12)) / exp)
            emit(f'  {p}: ' + ' '.join(f'{r:.2f}' for r in ratios))
        emit('')

    # --- source-combo CRB (tags ending _I<budget>, the src scan at the common budget) ---
    src_budget = '1e7'
    sc = [(k.replace('crb_', '').rsplit('_I', 1)[0], j) for k, j in d.items()
          if k.startswith('crb_') and k.endswith(f'_I{src_budget}')]
    sc = sorted(sc, key=lambda x: x[0])
    if sc:
        emit(f'## CRB vs source/location combo (budget {src_budget}, NPH=1e6) — fractional σ')
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
