"""Edge cases of composable physics config loading, over every shipped config.

Two sets of tests. The first walks the whole catalogue --- every
``*physics_config*.json`` and every ``*_geom_config.json``, found by glob, so a
new detector is covered the moment it is added.

The second set is the original per-detector depth: field presence, projection,
medium and QE-curve resolution and ``qe_corrections`` expansion, on the twelve
detectors whose sensor counts are written down here. The third is the strict
loader: unknown, missing and mode-ignored keys raise, and the angular reflection model's block.
"""

import glob
import json
import os

import jax.numpy as jnp
import pytest

from lucid.detector_params import (
    DetectorParams,
    load_physics_config,
    load_detector_params,
)
from tests.helpers.paths import REPO

CONFIG_DIR = os.path.join(
    str(REPO), "config"
)

# The whole shipped catalogue, by glob rather than by hand.
PHYSICS_CONFIGS = sorted(glob.glob(os.path.join(CONFIG_DIR, "**", "*physics_config*.json"),
                                   recursive=True))
GEOM_CONFIGS = sorted(glob.glob(os.path.join(CONFIG_DIR, "**", "*_geom_config.json"),
                                recursive=True))

# What a geometry config has to carry for the production template to pair it
# with its physics file and build a detector from it.
_GEOM_REQUIRED_KEYS = ("detector_type", "geometry_definitions", "material")

# The two IceCube86 geometries are the only ones the production template cannot
# pair: it builds config/{detector}_physics_config.json from one --detector
# string, and the shipped physics files are IceCube86(_ice), not
# IceCube86_{full,simple}. The string telescope is driven from tests and
# notebooks, not from run_job, so this is a naming gap rather than a broken run
# --- but it is a gap, and naming it here is how it stops being invisible.
_GEOM_WITHOUT_TEMPLATE_PARTNER = ("IceCube86_full", "IceCube86_simple")


def _name(path):
    return os.path.basename(path)


def _raw(path):
    """The config's leaf values, the scalar_mode block's included."""
    with open(path) as f:
        raw = json.load(f)
    return {**raw, **raw.get("scalar_mode", {})}


_FRACTIONS = ("qe", "wall_reflection_rate", "sensor_reflection_rate")
_LENGTHS = ("scatter_length", "absorption_length", "mie_scatter_length")


@pytest.mark.parametrize("cfg_path", PHYSICS_CONFIGS, ids=_name)
def test_every_physics_config_loads_its_numbers_onto_the_right_leaves(cfg_path):
    """Over the whole catalogue: every scalar the JSON writes arrives, unchanged, on the
    DetectorParams leaf of the same name; the only NaN leaves left are reflection rates the
    file omits; and what is set is physical (qe and reflection rates in [0, 1] with qe > 0,
    optical lengths > 0).

    ``test_field_presence`` below checks the same mapping on twelve named detectors; this one
    holds it for every config the glob finds, including the SK_WAND variants. A loader that
    scaled a value, dropped one to the default or put it on another leaf fails here.
    """
    from lucid.detector_params import _flatten_detector_params
    raw = _raw(cfg_path)
    params, _, _ = load_physics_config(cfg_path, num_sensors=None)
    assert isinstance(params, DetectorParams)
    flat = _flatten_detector_params(params)

    written = {k: v for k, v in raw.items()
               if k in flat and isinstance(v, (int, float)) and not isinstance(v, bool)}
    for field, value in written.items():
        got = jnp.asarray(flat[field])
        assert got.ndim == 0, f"{field}: scalar {value} became shape {got.shape}"
        assert float(got) == float(jnp.float32(value)), f"{field}: JSON {value}, loaded {float(got)}"

    for field, value in flat.items():
        v = jnp.asarray(value)
        assert bool(jnp.all(jnp.isfinite(v))), f"{field} has a non-finite entry"
        if field in _FRACTIONS:
            assert 0.0 <= float(v) <= 1.0, f"{field} = {float(v)} is not a fraction"
        if field in _LENGTHS:
            assert float(v) > 0.0, f"{field} = {float(v)} m"
    assert float(flat["qe"]) > 0.0


def _raw_qe_corrections(path):
    with open(path) as f:
        return json.load(f).get("qe_corrections")


# configs whose qe_corrections is a per-sensor vector (a file or an inline list)
FILE_QE_CONFIGS = [p for p in PHYSICS_CONFIGS
                   if isinstance(_raw_qe_corrections(p), (str, list))]
SCALAR_QE_CONFIGS = [p for p in PHYSICS_CONFIGS if p not in FILE_QE_CONFIGS]


@pytest.mark.parametrize("cfg_path", SCALAR_QE_CONFIGS, ids=_name)
def test_scalar_qe_corrections_expand_to_one_per_sensor(cfg_path):
    """A scalar (or absent -> neutral 1.0) qe_corrections becomes one value per sensor.

    The simulator multiplies it against the per-sensor charge, so any other
    length broadcasts into a wrong answer or raises deep inside the forward.
    """
    n = 777
    params, _, _ = load_physics_config(cfg_path, num_sensors=n)
    qe_c = params.per_pmt.qe_corrections
    raw = _raw_qe_corrections(cfg_path)
    assert qe_c.shape == (n,), f"{_name(cfg_path)} expanded to {qe_c.shape}"
    assert jnp.allclose(qe_c, float(raw))


@pytest.mark.parametrize("cfg_path", FILE_QE_CONFIGS, ids=_name)
def test_file_qe_corrections_of_the_wrong_length_are_refused(cfg_path):
    n_file = load_physics_config(cfg_path, num_sensors=None)[0].per_pmt.qe_corrections.shape[0]
    with pytest.raises(ValueError):
        load_physics_config(cfg_path, num_sensors=n_file // 2)


def test_some_config_has_a_file_qe_vector():
    """The length check above must have something to run on (JUNO ships one)."""
    assert "JUNO_physics_config.json" in map(_name, FILE_QE_CONFIGS)


@pytest.mark.parametrize("cfg_path", GEOM_CONFIGS, ids=_name)
def test_every_geom_config_parses_and_names_its_type(cfg_path):
    """Every shipped geometry config is JSON and declares a detector_type.

    The build itself is slow for the large detectors, so the catalogue-wide
    check is the parse; ``tests/production/test_wand_detector_pair.py`` builds the two
    WAND detectors through the production template.
    """
    with open(cfg_path) as f:
        raw = json.load(f)
    for key in _GEOM_REQUIRED_KEYS:
        assert key in raw, f"{_name(cfg_path)} has no {key!r}"


@pytest.mark.parametrize("cfg_path", GEOM_CONFIGS, ids=_name)
def test_every_geom_config_has_a_physics_partner(cfg_path):
    """The production template pairs the two by name, so both files must exist.

    ``run_job.py`` builds ``config/{detector}_{geom,physics}_config.json`` from
    one ``--detector`` string: a geometry file whose physics partner is missing
    is a detector nobody can run.
    """
    detector = _name(cfg_path)[:-len("_geom_config.json")]
    partner = os.path.join(CONFIG_DIR, f"{detector}_physics_config.json")
    if detector in _GEOM_WITHOUT_TEMPLATE_PARTNER:
        assert not os.path.isfile(partner), (
            f"{detector} now has a physics partner — drop it from "
            f"_GEOM_WITHOUT_TEMPLATE_PARTNER")
        return
    assert os.path.isfile(partner), (
        f"{_name(cfg_path)} has no {detector}_physics_config.json beside it")


# Map: detector name -> (physics_config filename, num_sensors from geom config)
DETECTOR_INFO = {
    "BigHK":   ("BigHK_physics_config.json",   20000),
    "EOS":     ("EOS_physics_config.json",       200),
    "HK":      ("HK_physics_config.json",      19746),
    "IWCD":    ("IWCD_physics_config.json",     10000),
    "JUNO":    ("JUNO_physics_config.json",     10000),
    "MidBox":  ("MidBox_physics_config.json",    9000),
    "SK":      ("SK_physics_config.json",       11096),
    "SK_like": ("SK_like_physics_config.json",  11000),
    "TAO":     ("TAO_physics_config.json",       4000),
    "WCTE":    ("WCTE_physics_config.json",      1995),
    "WCTE_like": ("WCTE_like_physics_config.json", 2500),
    "nuSCOPE": ("nuSCOPE_physics_config.json",   5000),
}

# DetectorParams is now nested by physics. These tests check the FLAT JSON
# schema -> leaf-field projection semantics, so iterate the ORIGINAL flat leaf
# fields (the new response/per-PMT placeholders carry neutral defaults, not the
# NaN "missing" semantics asserted below).
DETECTOR_PARAMS_FIELDS = (
    "scatter_length", "g", "mie_scatter_length",
    "wall_reflection_rate", "sensor_reflection_rate",
    "absorption_length", "qe", "qe_corrections",
)

# Map each flat leaf field to its sub-tuple so getattr resolves through nesting.
_LEAF_PARENT = {
    "scatter_length": "scattering", "mie_scatter_length": "scattering", "g": "scattering",
    "absorption_length": "absorption",
    "wall_reflection_rate": "reflection", "sensor_reflection_rate": "reflection",
    "qe": "response", "tts": "response",
    "qe_corrections": "per_pmt", "t0": "per_pmt",
}


def _leaf(params, field):
    """Fetch a flat leaf field from the nested DetectorParams pytree."""
    return getattr(getattr(params, _LEAF_PARENT[field]), field)


@pytest.fixture(scope="module")
def loaded_configs():
    """Load every physics config once per module."""
    out = {}
    for det_name, (cfg_file, n_sensors) in DETECTOR_INFO.items():
        cfg_path = os.path.join(CONFIG_DIR, cfg_file)
        params, medium, qe_curve = load_physics_config(cfg_path, num_sensors=n_sensors)
        out[det_name] = (params, medium, qe_curve, cfg_path, n_sensors)
    return out


# ---------------------------------------------------------------------------
# 2. Field presence: populated fields carry their JSON values; omitted
#    projectable fields are filled from wavelength curves at the reference λ.
# ---------------------------------------------------------------------------
_PROJECTABLE = ("scatter_length", "absorption_length", "qe")


@pytest.mark.parametrize("det_name", list(DETECTOR_INFO.keys()))
def test_field_presence(det_name, loaded_configs):
    params, _, _, cfg_path, n_sensors = loaded_configs[det_name]
    raw = _raw(cfg_path)

    for field in DETECTOR_PARAMS_FIELDS:
        val = _leaf(params, field)
        if field in raw:
            raw_val = raw[field]
            if isinstance(raw_val, (int, float)):
                if field == "qe_corrections" and n_sensors is not None:
                    assert val.shape == (n_sensors,), (
                        f"{det_name}.{field} shape mismatch: "
                        f"got {val.shape}, expected ({n_sensors},)")
                    assert jnp.allclose(val, raw_val)
                else:
                    assert jnp.allclose(val, jnp.asarray(float(raw_val))), (
                        f"{det_name}.{field}={float(val)}, expected {raw_val}")
            elif isinstance(raw_val, str):
                assert val.ndim >= 1, (
                    f"{det_name}.{field} loaded from file '{raw_val}' "
                    f"should be array, got shape {val.shape}")
        else:
            # An omitted leaf is projected from the medium / QE curve at λ_ref (the loader
            # raises for any other omitted one): finite, and positive except g.
            assert field in _PROJECTABLE + ("mie_scatter_length", "g"), (det_name, field)
            assert val.ndim == 0 and bool(jnp.isfinite(val)), (det_name, field, float(val))
            assert float(val) > 0 or field == "g", (det_name, field, float(val))


def test_sk_missing_fields_project_from_curves(loaded_configs):
    """SK_physics_config has no scalar scatter/absorption/qe: they are projected
    from the referenced curves at 400 nm, in metres and as a fraction.

    Independent references, evaluated here from the curve files:
    * qe: SK_QE.json is tabulated in percent; linear interpolation at 400 nm
      gives 22.62% -> 0.2262.
    * scatter: water.json carries arXiv:1307.0162 Eq. 17 (Rayleigh + symmetric
      Mie), alpha = P4 / lambda^4 (1 + P5 / lambda^2) = 5.69e-3 /m -> 175.7 m.
    * absorption: Eq. 16 is P0 P1 / lambda^4 plus a non-negative continuum, so
      the length is at most 1 / (P0 P1 / lambda^4) = 1386 m; a continuum below
      0.02 /m bounds it below at 48 m. A cm/m or percent mix-up is x100.
    """
    import numpy as np
    sk_params, medium, _, cfg_path, _ = loaded_configs["SK"]
    with open(cfg_path) as f:
        sk_raw = json.load(f)
    for field in ["scatter_length", "absorption_length", "qe"]:
        assert field not in sk_raw, f"SK config unexpectedly has '{field}'"
    lam = 400.0
    with open(os.path.join(CONFIG_DIR, sk_raw["qe_curve"])) as f:
        qe_tab = json.load(f)
    qe_ref = np.interp(lam, qe_tab["wavelengths_nm"], qe_tab["qe_percent"]) / 100.0
    assert float(sk_params.response.qe) == pytest.approx(qe_ref, rel=1e-4)
    water = medium.model
    sym = water["scattering"]["symmetric"]
    alpha_sym = sym["P4"] / lam ** 4 * (1.0 + sym["P5"] / lam ** 2)
    assert float(sk_params.scattering.scatter_length) == pytest.approx(1.0 / alpha_sym, rel=1e-4)
    ab = water["absorption"]
    alpha_r = ab["P0"] * ab["P1"] / lam ** 4
    L_abs = float(sk_params.absorption.absorption_length)
    assert 1.0 / (alpha_r + 0.02) < L_abs <= 1.0 / alpha_r


# ---------------------------------------------------------------------------
# 3. medium_model resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("det_name", list(DETECTOR_INFO.keys()))
def test_medium_model_path(det_name, loaded_configs):
    _, medium, _, cfg_path, _ = loaded_configs[det_name]
    with open(cfg_path) as f:
        raw = json.load(f)
    assert (medium is not None) == (raw.get("medium_model") is not None), det_name


# ---------------------------------------------------------------------------
# 4. qe_curve resolution and 400nm check
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("det_name", list(DETECTOR_INFO.keys()))
def test_qe_curve_path(det_name, loaded_configs):
    _, _, qe_curve, cfg_path, _ = loaded_configs[det_name]
    with open(cfg_path) as f:
        raw = json.load(f)

    if raw.get("qe_curve") is not None:
        qe_400 = float(qe_curve(400.0))
        assert 0.0 < qe_400 < 1.0, (
            f"{det_name} QE at 400nm = {qe_400}, expected 0 < QE < 1")
    else:
        assert qe_curve is None, det_name


# ---------------------------------------------------------------------------
# 5. qe_corrections handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("det_name", ["WCTE", "EOS", "SK", "SK_like", "HK"])
def test_scalar_qe_corrections_expansion(det_name, loaded_configs):
    params, _, _, _, n_sensors = loaded_configs[det_name]
    qe_c = params.per_pmt.qe_corrections
    assert qe_c.shape == (n_sensors,), (
        f"{det_name} qe_corrections shape {qe_c.shape}, expected ({n_sensors},)")
    assert jnp.allclose(qe_c, 1.0)


def test_juno_file_qe_corrections(loaded_configs):
    """JUNO has qe_corrections loaded from file — 10000 elements, all finite/positive,
    the same with num_sensors = 10000 or None."""
    params = loaded_configs["JUNO"][0]
    qe_c = params.per_pmt.qe_corrections
    assert qe_c.shape == (10000,)
    assert bool(jnp.all(jnp.isfinite(qe_c)))
    assert bool(jnp.all(qe_c > 0))
    juno_path = os.path.join(CONFIG_DIR, "JUNO_physics_config.json")
    p, _, _ = load_physics_config(juno_path, num_sensors=None)
    assert jnp.array_equal(p.per_pmt.qe_corrections, qe_c)


def test_scalar_qe_stays_scalar_without_num_sensors():
    wcte_path = os.path.join(CONFIG_DIR, "WCTE_physics_config.json")
    params, _, _ = load_physics_config(wcte_path, num_sensors=None)
    assert params.per_pmt.qe_corrections.ndim == 0
    assert jnp.allclose(params.per_pmt.qe_corrections, 1.0)


def test_load_detector_params_is_load_physics_config(loaded_configs):
    """The backward-compatible entry point returns the same parameters."""
    import jax
    _, _, _, cfg_path, n = loaded_configs["SK"]
    a = load_detector_params(cfg_path, num_sensors=n)
    b = load_physics_config(cfg_path, num_sensors=n)[0]
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        assert jnp.array_equal(x, y, equal_nan=True)


def test_missing_required_scalar_without_curve_raises(tmp_path):
    """If a scalar is missing and no curve is available to project from,
    loading must fail loudly rather than silently substitute a placeholder."""
    bad_cfg = tmp_path / "bad_physics.json"
    cfg = {k: v for k, v in _MINIMAL.items() if k != "scalar_mode"}
    bad_cfg.write_text(json.dumps({**cfg, "qe": 0.2}))     # no scalar_mode, no medium_model
    with pytest.raises(ValueError, match="'scatter_length'.*and no 'medium_model'"):
        load_detector_params(str(bad_cfg))


# ---------------------------------------------------------------------------
# 6. The strict loader
# ---------------------------------------------------------------------------
_MINIMAL = {"wall_reflection_rate": 0.2, "sensor_reflection_rate": 0.2,
            "wall_fspec": 0.55, "sensor_fspec": 0.9, "qe_corrections": 1.0, "g": 0.0,
            "scalar_mode": {"scatter_length": 50.0, "absorption_length": 50.0,
                            "mie_scatter_length": 1.0e9}}


def _load(tmp_path, cfg):
    path = tmp_path / "physics_config.json"
    path.write_text(json.dumps(cfg))
    return load_physics_config(str(path))[0]


def test_a_minimal_config_loads(tmp_path):
    dp = _load(tmp_path, {**_MINIMAL, "qe": 0.2, "_comment": "comments pass"})
    assert float(dp.scattering.scatter_length) == 50.0 and float(dp.response.tts) == 0.0


def test_an_unknown_key_raises_naming_the_closest(tmp_path):
    with pytest.raises(ValueError, match="unknown key 'wall_reflection'.*'wall_reflection_rate'"):
        _load(tmp_path, {**_MINIMAL, "qe": 0.2, "wall_reflection": 0.1})
    with pytest.raises(ValueError, match="unknown key 'scatter_lenght'"):
        _load(tmp_path, {**_MINIMAL, "qe": 0.2,
                         "scalar_mode": {**_MINIMAL["scalar_mode"], "scatter_lenght": 1.0}})


@pytest.mark.parametrize("field", ["wall_reflection_rate", "sensor_reflection_rate",
                                   "wall_fspec", "sensor_fspec", "qe_corrections"])
def test_a_required_key_left_out_raises(tmp_path, field):
    cfg = {**_MINIMAL, "qe": 0.2}
    del cfg[field]
    with pytest.raises(ValueError, match=f"does not state.*{field}"):
        _load(tmp_path, cfg)


def test_a_scalar_mode_length_at_the_top_level_raises(tmp_path):
    """Wavelength mode would ignore it, so the config must say which mode it is for."""
    with pytest.raises(ValueError, match="read only in scalar mode"):
        _load(tmp_path, {**_MINIMAL, "qe": 0.2, "scatter_length": 50.0})


_ANGULAR = {"wall_R0": 0.1, "wall_p": 5.0, "cathode_nr": 2.6, "cathode_nk": 1.3}


def test_the_angular_block_lands_on_its_leaves(tmp_path):
    r = _load(tmp_path, {**_MINIMAL, "qe": 0.2, "angular_reflection": _ANGULAR}).reflection
    assert {k: float(getattr(r, k)) for k in _ANGULAR} == {
        k: float(jnp.float32(v)) for k, v in _ANGULAR.items()}


def test_an_angular_leaf_at_the_top_level_or_a_partial_block_raises(tmp_path):
    """Only reflection_model='angular' reads them, so the config says so with the block."""
    with pytest.raises(ValueError, match="read only by reflection_model='angular'"):
        _load(tmp_path, {**_MINIMAL, "qe": 0.2, "wall_R0": 0.1})
    partial = {k: v for k, v in _ANGULAR.items() if k != "cathode_nk"}
    with pytest.raises(ValueError, match="'angular_reflection' does not state.*cathode_nk"):
        _load(tmp_path, {**_MINIMAL, "qe": 0.2, "angular_reflection": partial})


def test_the_angular_model_requires_the_block(tmp_path):
    path = tmp_path / "physics_config.json"
    path.write_text(json.dumps({**_MINIMAL, "qe": 0.2}))
    load_physics_config(str(path), reflection_model="scalar_mix")
    with pytest.raises(ValueError, match="no \"angular_reflection\" block"):
        load_physics_config(str(path), reflection_model="angular")


def test_null_is_not_a_value(tmp_path):
    with pytest.raises(ValueError, match="'qe'.*not a number"):
        _load(tmp_path, {**_MINIMAL, "qe": None})


@pytest.mark.slow                 # builds a detector and propagator
def test_a_physics_medium_unlike_the_geometry_material_is_refused(tmp_path, small_cylinder_config):
    """The geometry JSON's material sets n, c and the emission processes; the physics config's
    medium_model sets the optical curves and the scintillation defaults. A water detector with an
    ice medium model is two media at once, so the simulator refuses it."""
    import json
    from lucid.simulation import setup_event_simulator
    repo_config = os.path.join(str(REPO), 'config')
    phys = json.load(open(os.path.join(repo_config, 'SK_like_physics_config.json')))
    phys['medium_model'] = os.path.join(repo_config, 'materials', 'ice.json')
    for key in ('qe_curve',):
        if key in phys and not os.path.isabs(phys[key]):
            phys[key] = os.path.join(repo_config, phys[key])
    path = tmp_path / 'ice_medium_physics_config.json'
    path.write_text(json.dumps(phys))
    with pytest.raises(ValueError, match="'water' but the physics config's medium_model"):
        setup_event_simulator(small_cylinder_config, 1000, source='particle',
                              physics_config=str(path),
                              default_detector_params=True, wavelength_mode=False)
