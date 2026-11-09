"""Generator inputs that set physics are stated, not defaulted (supernova burst, particle bomb)."""
import json
from pathlib import Path

import pytest

from lucid.production.dataprod.generators import macro, supernova
from tests.helpers.paths import REPO


def test_an_unknown_sntools_pdg_raises_instead_of_going_massless():
    """``_p4_gev`` recovers |p| from total energy; an unlisted PDG got mass 0 and so |p| = E."""
    px, py, pz, e = supernova._p4_gev(2112, 1000.0, 0.0, 0.0, 1.0)
    assert e == 1.0 and pz == pytest.approx((1000.0 ** 2 - 939.565420 ** 2) ** 0.5 * 1e-3)
    assert supernova._p4_gev(-12, 20.0, 1.0, 0.0, 0.0)[0] == pytest.approx(0.02)
    with pytest.raises(supernova.SupernovaError, match="no rest mass for PDG 1000080160"):
        supernova._p4_gev(1000080160, 15000.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize("drop", ["detector", "distance_kpc"])
def test_supernova_burst_size_keys_are_required(tmp_path, drop):
    """sntools' --detector and --distance set the burst size; they defaulted to SuperK / 10 kpc
    (the shipped config is HyperK at 50 kpc)."""
    sn = {"detector": "HyperK", "distance_kpc": 50.0,
          "models": [{"name": "m", "format": "gamma", "flux_file": "absent.txt"}]}
    del sn[drop]
    with pytest.raises(supernova.SupernovaError, match=f"needs \\['{drop}'\\]"):
        supernova.run_sntools(config={"supernova": sn}, output_dir=tmp_path, job_id=1)


@pytest.mark.parametrize("drop", ["min_particles", "max_particles"])
def test_bomb_multiplicity_is_required(drop):
    """max_particles defaulted to 5, so a bomb config without it threw up to 5 particles."""
    bomb = {"min_particles": 1, "max_particles": 1,
            "candidates": [{"type": "mu-", "energy_min_MeV": 100, "energy_max_MeV": 200}]}
    cfg = {"name": "t", "material": "water", "primary_source": "bomb", "bomb": bomb}
    assert "Uniform[1, 1]" in macro.generate_macro(cfg, output_root_file="o.root", n_events=2)
    del bomb[drop]
    with pytest.raises(ValueError, match=f"bomb.{drop}"):
        macro.generate_macro(cfg, output_root_file="o.root", n_events=2)


# --------------------------------------------------------------------------
# supernova mass ordering: one key (supernova.orderings), required
# --------------------------------------------------------------------------
SN_CONFIG = Path(REPO) / "lucid/production/dataprod/configs/SN/01_supernova_50kpc.json"


@pytest.mark.parametrize("sn, match", [
    ({}, "supernova.orderings must be a non-empty list"),
    ({"orderings": []}, "supernova.orderings must be a non-empty list"),
    ({"orderings": "NMO"}, "supernova.orderings must be a non-empty list"),
    ({"orderings": ["NMO", "NMO"]}, "repeats an entry"),
    ({"orderings": ["SNEWPY-Foo"]}, "Unknown mass ordering 'SNEWPY-Foo'"),
    ({"ordering": "IMO"}, "supernova.ordering is no longer read"),
    ({"ordering": "IMO", "orderings": ["IMO"]}, "supernova.ordering is no longer read"),
])
def test_the_supernova_mass_ordering_is_stated_not_defaulted(sn, match):
    """``run_sntools`` read only the singular ``ordering`` and fell back to NMO, the fan-out only
    the plural ``orderings``; sntools' own default is NoTransformation."""
    with pytest.raises(ValueError, match=match) as err:
        supernova.sn_orderings(sn)
    if "non-empty" in match or "Unknown" in match:   # the message names what is allowed
        for name in ("NMO", "IMO", "noosc") + supernova.SNTOOLS_TRANSFORMATIONS:
            assert name in str(err.value)


def test_a_job_takes_its_ordering_from_the_list_or_from_sn_ordering():
    assert supernova.resolve_ordering({"orderings": ["IMO"]}) == ("IMO", "AdiabaticMSW_IMO")
    assert supernova.resolve_ordering({"orderings": ["NMO", "noosc"]}, "noosc") == (
        "noosc", "NoTransformation")
    assert supernova.resolve_ordering({"orderings": ["SNEWPY-CompleteExchange"]}) == (
        "SNEWPY-CompleteExchange", "SNEWPY-CompleteExchange")
    with pytest.raises(ValueError, match="pass --sn-ordering"):
        supernova.resolve_ordering({"orderings": ["NMO", "IMO"]})
    with pytest.raises(ValueError, match="'NMO' is not in supernova.orderings"):
        supernova.resolve_ordering({"orderings": ["IMO"]}, "NMO")
    shipped = json.loads(SN_CONFIG.read_text())["supernova"]
    assert supernova.resolve_ordering(shipped) == ("NMO", "AdiabaticMSW_NMO")


def test_run_sntools_runs_the_ordering_the_config_lists(tmp_path, monkeypatch):
    """``"orderings": ["IMO"]`` and no ``--sn-ordering`` ran sntools under AdiabaticMSW_NMO."""
    flux = tmp_path / "flux.txt"
    flux.write_text("")
    calls = []
    monkeypatch.setattr(supernova.subprocess, "call", lambda cmd, cwd: calls.append(cmd) or 1)
    sn = {"detector": "HyperK", "distance_kpc": 50.0, "orderings": ["IMO"],
          "models": [{"name": "m", "format": "gamma", "flux_file": str(flux)}]}
    with pytest.raises(supernova.SupernovaError, match="sntools failed"):
        supernova.run_sntools(config={"supernova": sn}, output_dir=tmp_path, job_id=1)
    cmd = calls[0]
    assert cmd[cmd.index("--transformation") + 1] == "AdiabaticMSW_IMO"
    with pytest.raises(ValueError, match="supernova.orderings"):
        supernova.run_sntools(config={"supernova": dict(sn, orderings=None)},
                              output_dir=tmp_path, job_id=1)


def _user_paths(tmp_path):
    image = tmp_path / "lucid.sif"
    image.touch()
    up = tmp_path / "user_paths.sh"
    up.write_text(f'export CLUSTER="slurm"\nexport LUCID_IMAGE_PATH="{image}"\n'
                  f'export OUTPUT_BASE_PATH="{tmp_path / "prod"}"\n'
                  'export SLURM_PARTITION="milano"\nexport SLURM_ACCOUNT="a:b"\n'
                  'export APPTAINER_BINDS="/sdf"\n')
    return str(up)


def test_the_fanout_and_the_job_read_the_same_orderings_key(tmp_path, capsys):
    """The fan-out writes one ``<model>/<ordering>/`` subcase per listed ordering, each job passed
    its ``--sn-ordering``; a config that lists none, or uses the singular key, is refused."""
    from lucid.production.dataprod import fanout

    cfg = json.loads(SN_CONFIG.read_text())
    cfg["supernova"]["orderings"] = ["NMO", "IMO"]
    (tmp_path / "SN").mkdir()
    path = tmp_path / "SN" / "01_sn.json"
    path.write_text(json.dumps(cfg))
    assert fanout.main(["-c", str(path), "--user-paths", _user_paths(tmp_path)]) == 0
    subs = sorted((tmp_path / "prod").rglob("submit_job_*"))
    n = cfg["nominal_test"]
    assert [p.parent.name for p in subs] == ["IMO"] * n + ["NMO"] * n
    for p in subs:
        assert f"--sn-ordering {p.parent.name}" in p.read_text()

    for sn in ({k: v for k, v in cfg["supernova"].items() if k != "orderings"},
               dict(cfg["supernova"], orderings=None, ordering="NMO")):
        sn = {k: v for k, v in sn.items() if v is not None}
        path.write_text(json.dumps(dict(cfg, supernova=sn)))
        capsys.readouterr()
        assert fanout.main(["-c", str(path), "--user-paths", _user_paths(tmp_path)]) == 1
        assert "supernova.orderings" in capsys.readouterr().err
