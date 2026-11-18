"""Build lucid-data-v1's overlap tables through the simulator's own path (DetectorGeometry.from_config,
sigma = temperature * the detector's S_radius), so a shipped table is the table a cold first use
would build. Each (config, temperature) builds in its own cache dir, in parallel; same-named tables
from different configs must be byte-identical, and one copy of each goes to <staging>/spatial_overlap_integrals.
Usage: python tools/build_overlap_tables.py <repo> <staging dir>   (a CPU node: one process per entry)"""
import filecmp, glob, json, os, shutil, sys
from multiprocessing import Pool

REPO, STAGE = sys.argv[1:3]
WORK = os.path.join(STAGE, 'work')
BOX = dict(length=8.0, width=20.0, height=10.0, n_sensors=6480, sensor_radius=0.11)  # fig_charge_displays.BOX


def jobs():
    js = [(p, 0.2) for p in sorted(glob.glob(f'{REPO}/config/*_geom_config.json'))]  # library default
    js.append((f'{REPO}/tests/helpers/tiny_geom_config.json', 0.2))                  # default-lane tests
    sk, juno = f'{REPO}/config/SK_like_geom_config.json', f'{REPO}/config/JUNO_geom_config.json'
    js.append((sk, 0.1))                    # lucid.fitting.recon.MODEL_RECIPE: the paper's reconstruction
    js += [(sk, 0.05), (juno, 0.05)]        # fig_charge_displays / fig_cylinder_2d_displays --temperature
    cfg = json.load(open(f'{REPO}/config/MidBox_geom_config.json'))
    cfg['geometry_definitions'].update(BOX)
    os.makedirs(WORK, exist_ok=True)
    box = os.path.join(WORK, 'MidBox_paper_box.json')
    json.dump(cfg, open(box, 'w'), indent=2)
    js.append((box, 0.05))                  # fig_charge_displays box
    return js


def build(arg):
    i, (path, t) = arg
    from lucid import overlap
    from lucid.geometry.detector_geometry import DetectorGeometry
    d = os.path.join(WORK, f'w{i:02d}')
    overlap.set_cache_dir(d)
    DetectorGeometry.from_config(path, temperature=t)
    return path, t, sorted(os.listdir(os.path.join(d, 'spatial_overlap_integrals')))


if __name__ == '__main__':
    out = os.path.join(STAGE, 'spatial_overlap_integrals')
    os.makedirs(out, exist_ok=True)
    js = list(enumerate(jobs()))
    with Pool(len(js)) as p:
        for path, t, names in p.imap_unordered(build, js):
            print(f'{os.path.basename(path):34s} T={t:<5} -> {names}', flush=True)
    for i, _ in js:
        for f in glob.glob(os.path.join(WORK, f'w{i:02d}', 'spatial_overlap_integrals', '*.json')):
            dst = os.path.join(out, os.path.basename(f))
            if os.path.exists(dst):
                assert filecmp.cmp(f, dst, shallow=False), f'{f} differs from {dst}'
            else:
                shutil.copy(f, dst)
    print('\n'.join(sorted(os.listdir(out))))
