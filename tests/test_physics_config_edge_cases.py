"""
Comprehensive edge-case tests for composable physics config loading.
Tests all 11 physics configs, field presence, path resolution,
qe_corrections handling, cross-validation, and backward compatibility.
"""

import json
import os

import jax.numpy as jnp
import pytest

from lucid.detector_params import (
    DetectorParams,
    load_physics_config,
    load_detector_params,
)
from lucid.wavelength.medium import load_qe_curve

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"
)

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
    "qe": "response", "spe_width": "response", "tts": "response",
    "qe_corrections": "per_pmt", "gain": "per_pmt", "t0": "per_pmt", "walk": "per_pmt",
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
        params, mm_path, qe_path = load_physics_config(cfg_path, num_sensors=n_sensors)
        out[det_name] = (params, mm_path, qe_path, cfg_path, n_sensors)
    return out


# ---------------------------------------------------------------------------
# 1. All 11 physics configs load without error
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("det_name,cfg_file,n_sensors", [
    (k, v[0], v[1]) for k, v in DETECTOR_INFO.items()
])
def test_config_loads(det_name, cfg_file, n_sensors):
    cfg_path = os.path.join(CONFIG_DIR, cfg_file)
    params, mm_path, qe_path = load_physics_config(cfg_path, num_sensors=n_sensors)
    assert isinstance(params, DetectorParams)


# ---------------------------------------------------------------------------
# 2. Field presence: populated fields carry their JSON values; missing
#    projectable fields are filled from wavelength curves at the reference λ;
#    missing non-projectable fields are left as NaN (loud failure if used).
# ---------------------------------------------------------------------------
_PROJECTABLE = ("scatter_length", "absorption_length", "qe")


@pytest.mark.parametrize("det_name", list(DETECTOR_INFO.keys()))
def test_field_presence(det_name, loaded_configs):
    params, _, _, cfg_path, n_sensors = loaded_configs[det_name]
    with open(cfg_path) as f:
        raw = json.load(f)

    for field in DETECTOR_PARAMS_FIELDS:
        val = _leaf(params, field)
        if field in raw and raw[field] is not None:
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
        elif field in _PROJECTABLE:
            # Missing scatter/absorption/qe should be projected from curves —
            # finite, positive, not the placeholder NaN.
            assert val.ndim == 0 and bool(jnp.isfinite(val)) and float(val) > 0, (
                f"{det_name}.{field} should be projected from curves at λ_ref, "
                f"got {float(val)}")
        elif field == "qe_corrections":
            # qe_corrections defaults to neutral 1.0 (optionally expanded).
            assert bool(jnp.all(jnp.isclose(val, 1.0))), (
                f"{det_name}.qe_corrections missing → should be neutral 1.0, "
                f"got {val}")
        elif field in ("mie_scatter_length", "g"):
            # Mie scalars are projected from the medium when one is referenced,
            # else default to "no Mie" (mie_scatter_length huge, g=0). Always
            # finite — a NaN here would poison the expected-value photon step.
            assert val.ndim == 0 and bool(jnp.isfinite(val)), (
                f"{det_name}.{field} should be filled (projected or no-Mie "
                f"default), not NaN; got {float(val)}")
        else:
            # Non-projectable scalars (reflections): NaN placeholder if absent.
            assert val.ndim == 0 and bool(jnp.isnan(val)), (
                f"{det_name}.{field} should be NaN when missing, got {float(val)}")


def test_sk_missing_fields_project_from_curves(loaded_configs):
    """SK_physics_config has no scalar scatter/absorption/qe — they must be
    projected from the referenced curves (medium_model, qe_curve), not NaN,
    not 1.0."""
    sk_params, _, _, cfg_path, _ = loaded_configs["SK"]
    with open(cfg_path) as f:
        sk_raw = json.load(f)

    for field in ["scatter_length", "absorption_length", "qe"]:
        assert field not in sk_raw, f"SK config unexpectedly has '{field}'"
        val = _leaf(sk_params, field)
        assert val.ndim == 0, f"SK.{field} should be scalar, got shape {val.shape}"
        assert bool(jnp.isfinite(val)) and float(val) > 0, (
            f"SK.{field} should be projected (finite, positive), got {float(val)}")
        assert not jnp.allclose(val, 1.0), (
            f"SK.{field} = {float(val):.4f} suspiciously equal to 1.0 placeholder")


# ---------------------------------------------------------------------------
# 3. medium_model path resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("det_name", list(DETECTOR_INFO.keys()))
def test_medium_model_path(det_name, loaded_configs):
    _, mm_path, _, cfg_path, _ = loaded_configs[det_name]
    with open(cfg_path) as f:
        raw = json.load(f)

    if "medium_model" in raw and raw["medium_model"] is not None:
        assert mm_path is not None and os.path.isfile(mm_path), (
            f"{det_name} medium_model path does not exist: {mm_path}")
    else:
        assert mm_path is None, (
            f"{det_name} medium_model_path should be None, got {mm_path}")


# ---------------------------------------------------------------------------
# 4. qe_curve path resolution and 400nm check
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("det_name", list(DETECTOR_INFO.keys()))
def test_qe_curve_path(det_name, loaded_configs):
    _, _, qe_path, cfg_path, _ = loaded_configs[det_name]
    with open(cfg_path) as f:
        raw = json.load(f)

    if "qe_curve" in raw and raw["qe_curve"] is not None:
        assert qe_path is not None and os.path.isfile(qe_path), (
            f"{det_name} qe_curve path does not exist: {qe_path}")
        qe_fn = load_qe_curve(qe_path)
        qe_400 = float(qe_fn(400.0))
        assert 0.0 < qe_400 < 1.0, (
            f"{det_name} QE at 400nm = {qe_400}, expected 0 < QE < 1")
    else:
        assert qe_path is None, (
            f"{det_name} qe_curve_path should be None, got {qe_path}")


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
    """JUNO has qe_corrections loaded from file — 10000 elements, all finite/positive."""
    params = loaded_configs["JUNO"][0]
    qe_c = params.per_pmt.qe_corrections
    assert qe_c.shape == (10000,)
    assert bool(jnp.all(jnp.isfinite(qe_c)))
    assert bool(jnp.all(qe_c > 0))


def test_scalar_qe_stays_scalar_without_num_sensors():
    wcte_path = os.path.join(CONFIG_DIR, "WCTE_physics_config.json")
    params, _, _ = load_physics_config(wcte_path, num_sensors=None)
    assert params.per_pmt.qe_corrections.ndim == 0
    assert jnp.allclose(params.per_pmt.qe_corrections, 1.0)


# ---------------------------------------------------------------------------
# 6. Cross-validation with different num_sensors
# ---------------------------------------------------------------------------
def test_sk_with_real_num_sensors():
    sk_path = os.path.join(CONFIG_DIR, "SK_physics_config.json")
    p, _, _ = load_physics_config(sk_path, num_sensors=11146)
    assert p.per_pmt.qe_corrections.shape == (11146,)


def test_sk_with_alt_num_sensors():
    """SK config has scalar qe_corrections, so it expands to any requested size."""
    sk_path = os.path.join(CONFIG_DIR, "SK_physics_config.json")
    p, _, _ = load_physics_config(sk_path, num_sensors=11000)
    assert p.per_pmt.qe_corrections.shape == (11000,)


def test_juno_file_qe_corrections_keep_length():
    """JUNO's file-based qe_corrections keep their original length regardless of
    num_sensors — no automatic truncation/padding. This is expected behavior."""
    juno_path = os.path.join(CONFIG_DIR, "JUNO_physics_config.json")
    p, _, _ = load_physics_config(juno_path, num_sensors=5000)
    assert p.per_pmt.qe_corrections.shape[0] == 10000  # original file length, not 5000


def test_juno_with_correct_num_sensors():
    juno_path = os.path.join(CONFIG_DIR, "JUNO_physics_config.json")
    p, _, _ = load_physics_config(juno_path, num_sensors=10000)
    assert p.per_pmt.qe_corrections.shape == (10000,)


def test_juno_with_no_num_sensors():
    juno_path = os.path.join(CONFIG_DIR, "JUNO_physics_config.json")
    p, _, _ = load_physics_config(juno_path, num_sensors=None)
    assert p.per_pmt.qe_corrections.ndim >= 1


# ---------------------------------------------------------------------------
# 7. Backward compatibility: load_detector_params
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("det_name", list(DETECTOR_INFO.keys()))
def test_load_detector_params_returns_container(det_name):
    cfg_file, n_sensors = DETECTOR_INFO[det_name]
    cfg_path = os.path.join(CONFIG_DIR, cfg_file)
    params = load_detector_params(cfg_path, num_sensors=n_sensors)
    assert isinstance(params, DetectorParams)
    for field in DETECTOR_PARAMS_FIELDS:
        val = _leaf(params, field)
        assert hasattr(val, "shape"), f"{det_name}.{field} is not a jax array"


@pytest.mark.parametrize("det_name", ["SK", "SK_like", "HK", "BigHK"])
def test_missing_projectable_fields_are_projected(det_name):
    """load_detector_params should project missing scatter/absorption/qe from
    the referenced wavelength curves. The result must be finite and positive."""
    cfg_file, n_sensors = DETECTOR_INFO[det_name]
    cfg_path = os.path.join(CONFIG_DIR, cfg_file)
    with open(cfg_path) as f:
        raw = json.load(f)

    params = load_detector_params(cfg_path, num_sensors=n_sensors)
    for field in _PROJECTABLE:
        if field in raw:
            continue
        val = _leaf(params, field)
        assert val.ndim == 0 and bool(jnp.isfinite(val)) and float(val) > 0, (
            f"load_detector_params({det_name}).{field} should be projected "
            f"from curves (finite, positive), got {float(val)}")


def test_missing_required_scalar_without_curve_raises(tmp_path):
    """If a scalar is missing and no curve is available to project from,
    loading must fail loudly rather than silently substitute a placeholder."""
    bad_cfg = tmp_path / "bad_physics.json"
    bad_cfg.write_text(json.dumps({
        "wall_reflection_rate": 0.2,
        "sensor_reflection_rate": 0.2,
        "qe_corrections": 1.0,
        # no scatter_length / absorption_length / qe / medium_model / qe_curve
    }))
    with pytest.raises(ValueError, match="no scalar"):
        load_detector_params(str(bad_cfg))
