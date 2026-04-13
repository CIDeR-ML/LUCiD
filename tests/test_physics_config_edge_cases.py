"""
Comprehensive edge-case tests for composable physics config loading.
Tests all 11 physics configs, field presence, path resolution,
qe_corrections handling, cross-validation, and backward compatibility.
"""

import json
import os
import sys
import traceback

import jax.numpy as jnp
import numpy as np

# Ensure LUCiD is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lucid.detector_params import (
    DetectorParams,
    load_physics_config,
    load_detector_params,
)
from lucid.wavelength.medium import load_qe_curve

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

# Map: detector name -> (physics_config filename, num_sensors from geom config)
DETECTOR_INFO = {
    "BigHK":   ("BigHK_physics_config.json",   20000),
    "EOS":     ("EOS_physics_config.json",       200),
    "HK":      ("HK_physics_config.json",      20000),
    "IWCD":    ("IWCD_physics_config.json",     10000),
    "JUNO":    ("JUNO_physics_config.json",     10000),
    "MidBox":  ("MidBox_physics_config.json",    9000),
    "SK":      ("SK_physics_config.json",       11146),
    "SK_like": ("SK_like_physics_config.json",  11000),
    "TAO":     ("TAO_physics_config.json",       4000),
    "WCTE":    ("WCTE_physics_config.json",      2500),
    "nuSCOPE": ("nuSCOPE_physics_config.json",   5000),
}

DETECTOR_PARAMS_FIELDS = DetectorParams._fields  # all 6 fields

passed = 0
failed = 0
errors = []


def record(test_name, ok, detail=""):
    global passed, failed, errors
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {test_name}" + (f"  -- {detail}" if detail else ""))
    if ok:
        passed += 1
    else:
        failed += 1
        errors.append(f"{test_name}: {detail}")


# =========================================================================
# 1. All 11 physics configs load without error
# =========================================================================
print("=" * 72)
print("TEST 1: All 11 physics configs load without error")
print("=" * 72)

loaded_configs = {}
for det_name, (cfg_file, n_sensors) in DETECTOR_INFO.items():
    cfg_path = os.path.join(CONFIG_DIR, cfg_file)
    try:
        params, mm_path, qe_path = load_physics_config(cfg_path, num_sensors=n_sensors)
        loaded_configs[det_name] = (params, mm_path, qe_path)

        # Read raw JSON to determine which fields are explicitly set
        with open(cfg_path) as f:
            raw = json.load(f)

        populated = []
        defaulted = []
        for field in DETECTOR_PARAMS_FIELDS:
            if field in raw and raw[field] is not None:
                populated.append(field)
            else:
                defaulted.append(field)

        print(f"\n  {det_name} (num_sensors={n_sensors}):")
        print(f"    Populated : {populated}")
        print(f"    Defaulted : {defaulted}")
        record(f"{det_name} loads successfully", True)
    except Exception as e:
        record(f"{det_name} loads successfully", False, str(e))
        traceback.print_exc()

print(f"\n  Loaded {len(loaded_configs)}/11 configs")


# =========================================================================
# 2. Field presence check
# =========================================================================
print("\n" + "=" * 72)
print("TEST 2: Field presence -- correct values vs 1.0 defaults")
print("=" * 72)

for det_name, (cfg_file, n_sensors) in DETECTOR_INFO.items():
    cfg_path = os.path.join(CONFIG_DIR, cfg_file)
    if det_name not in loaded_configs:
        continue

    params, _, _ = loaded_configs[det_name]
    with open(cfg_path) as f:
        raw = json.load(f)

    print(f"\n  {det_name}:")
    for field in DETECTOR_PARAMS_FIELDS:
        val = getattr(params, field)
        if field in raw and raw[field] is not None:
            # Field IS in JSON -- check it has the correct value
            raw_val = raw[field]
            if isinstance(raw_val, (int, float)):
                # Scalar value check
                if field == "qe_corrections" and n_sensors is not None:
                    # Should have been expanded to array
                    expected_shape = (n_sensors,)
                    ok = val.shape == expected_shape and jnp.allclose(val, raw_val)
                    record(
                        f"{det_name}.{field} correct (expanded scalar {raw_val})",
                        bool(ok),
                        f"shape={val.shape}, expected {expected_shape}",
                    )
                else:
                    ok = jnp.allclose(val, jnp.asarray(float(raw_val)))
                    record(
                        f"{det_name}.{field} = {float(val):.4f} (expected {raw_val})",
                        bool(ok),
                    )
            elif isinstance(raw_val, str):
                # File reference -- just verify it loaded as non-scalar array
                record(
                    f"{det_name}.{field} loaded from file '{raw_val}'",
                    val.ndim >= 1,
                    f"shape={val.shape}",
                )
            else:
                record(f"{det_name}.{field} loaded (list/other)", True)
        else:
            # Field NOT in JSON -- must default to 1.0
            ok = val.ndim == 0 and jnp.allclose(val, 1.0)
            record(
                f"{det_name}.{field} defaults to 1.0",
                bool(ok),
                f"actual={float(val):.4f}" if val.ndim == 0 else f"shape={val.shape}",
            )

# Special check: SK should NOT have scatter_length, absorption_length, or qe
print("\n  --- Special: SK_physics_config should NOT have scatter_length/absorption_length/qe ---")
sk_path = os.path.join(CONFIG_DIR, "SK_physics_config.json")
with open(sk_path) as f:
    sk_raw = json.load(f)

for field in ["scatter_length", "absorption_length", "qe"]:
    absent = field not in sk_raw
    record(
        f"SK_physics_config does NOT have '{field}'",
        absent,
        f"present={not absent}",
    )

if "SK" in loaded_configs:
    sk_params = loaded_configs["SK"][0]
    for field in ["scatter_length", "absorption_length", "qe"]:
        val = getattr(sk_params, field)
        is_default = val.ndim == 0 and jnp.allclose(val, 1.0)
        record(
            f"SK.{field} correctly defaulted to 1.0",
            bool(is_default),
            f"actual={float(val):.4f}",
        )


# =========================================================================
# 3. medium_model path resolution
# =========================================================================
print("\n" + "=" * 72)
print("TEST 3: medium_model path resolution")
print("=" * 72)

for det_name, (cfg_file, n_sensors) in DETECTOR_INFO.items():
    if det_name not in loaded_configs:
        continue

    _, mm_path, _ = loaded_configs[det_name]
    cfg_path = os.path.join(CONFIG_DIR, cfg_file)
    with open(cfg_path) as f:
        raw = json.load(f)

    has_medium = "medium_model" in raw and raw["medium_model"] is not None
    if has_medium:
        exists = mm_path is not None and os.path.isfile(mm_path)
        record(
            f"{det_name} medium_model path exists",
            exists,
            f"path={mm_path}",
        )
    else:
        is_none = mm_path is None
        record(
            f"{det_name} medium_model_path is None (no medium_model in config)",
            is_none,
            f"actual={mm_path}",
        )


# =========================================================================
# 4. qe_curve path resolution
# =========================================================================
print("\n" + "=" * 72)
print("TEST 4: qe_curve path resolution and 400nm check")
print("=" * 72)

for det_name, (cfg_file, n_sensors) in DETECTOR_INFO.items():
    if det_name not in loaded_configs:
        continue

    _, _, qe_path = loaded_configs[det_name]
    cfg_path = os.path.join(CONFIG_DIR, cfg_file)
    with open(cfg_path) as f:
        raw = json.load(f)

    has_qe_curve = "qe_curve" in raw and raw["qe_curve"] is not None
    if has_qe_curve:
        exists = qe_path is not None and os.path.isfile(qe_path)
        record(
            f"{det_name} qe_curve path exists",
            exists,
            f"path={qe_path}",
        )
        if exists:
            try:
                qe_fn = load_qe_curve(qe_path)
                qe_400 = float(qe_fn(400.0))
                reasonable = 0.0 < qe_400 < 1.0
                record(
                    f"{det_name} QE at 400nm = {qe_400:.4f}",
                    reasonable,
                    "expected 0 < QE < 1",
                )
            except Exception as e:
                record(f"{det_name} QE curve loads and evaluates", False, str(e))
    else:
        is_none = qe_path is None
        record(
            f"{det_name} qe_curve_path is None (no qe_curve in config)",
            is_none,
            f"actual={qe_path}",
        )


# =========================================================================
# 5. qe_corrections handling
# =========================================================================
print("\n" + "=" * 72)
print("TEST 5: qe_corrections handling")
print("=" * 72)

# 5a. Scalar 1.0 -> expands to ones(num_sensors)
print("\n  --- 5a: Scalar qe_corrections expansion ---")
for det_name in ["WCTE", "EOS", "SK", "SK_like", "HK"]:
    if det_name not in loaded_configs:
        continue
    params = loaded_configs[det_name][0]
    n = DETECTOR_INFO[det_name][1]
    qe_c = params.qe_corrections
    ok = qe_c.shape == (n,) and jnp.allclose(qe_c, 1.0)
    record(
        f"{det_name} scalar 1.0 expanded to ones({n})",
        bool(ok),
        f"shape={qe_c.shape}, all_ones={bool(jnp.allclose(qe_c, 1.0))}",
    )

# 5b. JSON file reference -> correct array length
print("\n  --- 5b: JSON file qe_corrections ---")
if "JUNO" in loaded_configs:
    juno_params = loaded_configs["JUNO"][0]
    juno_qe_c = juno_params.qe_corrections
    record(
        f"JUNO qe_corrections shape = {juno_qe_c.shape}",
        juno_qe_c.shape == (10000,),
        f"expected (10000,)",
    )
    record(
        f"JUNO qe_corrections all finite",
        bool(jnp.all(jnp.isfinite(juno_qe_c))),
    )
    record(
        f"JUNO qe_corrections all positive",
        bool(jnp.all(juno_qe_c > 0)),
    )

# 5c. Without num_sensors, scalar stays scalar
print("\n  --- 5c: Without num_sensors, scalar qe_corrections stays scalar ---")
try:
    wcte_path = os.path.join(CONFIG_DIR, "WCTE_physics_config.json")
    params_no_ns, _, _ = load_physics_config(wcte_path, num_sensors=None)
    qe_c = params_no_ns.qe_corrections
    record(
        "WCTE (no num_sensors) qe_corrections is scalar",
        qe_c.ndim == 0,
        f"ndim={qe_c.ndim}, value={float(qe_c):.1f}",
    )
except Exception as e:
    record("WCTE (no num_sensors) loads", False, str(e))


# =========================================================================
# 6. Cross-validation
# =========================================================================
print("\n" + "=" * 72)
print("TEST 6: Cross-validation with different num_sensors")
print("=" * 72)

sk_path = os.path.join(CONFIG_DIR, "SK_physics_config.json")

# 6a. SK with real 11146
print("\n  --- 6a: SK with num_sensors=11146 (real SK) ---")
try:
    p, mm, qe = load_physics_config(sk_path, num_sensors=11146)
    record(
        "SK with 11146 sensors loads",
        True,
        f"qe_corrections shape={p.qe_corrections.shape}",
    )
    record(
        "SK with 11146 qe_corrections shape correct",
        p.qe_corrections.shape == (11146,),
    )
except Exception as e:
    record("SK with 11146 sensors loads", False, str(e))

# 6b. SK with 11000 (SK_like count)
print("\n  --- 6b: SK with num_sensors=11000 (SK_like) ---")
try:
    p, mm, qe = load_physics_config(sk_path, num_sensors=11000)
    record(
        "SK with 11000 sensors loads (scalar qe_corrections)",
        True,
        f"qe_corrections shape={p.qe_corrections.shape}",
    )
    record(
        "SK with 11000 qe_corrections shape correct",
        p.qe_corrections.shape == (11000,),
    )
except Exception as e:
    record("SK with 11000 sensors loads", False, str(e))

# 6c. JUNO with wrong num_sensors
print("\n  --- 6c: JUNO with wrong num_sensors=5000 ---")
juno_path = os.path.join(CONFIG_DIR, "JUNO_physics_config.json")
try:
    p, mm, qe = load_physics_config(juno_path, num_sensors=5000)
    # JUNO loads qe_corrections from JSON file (10000 elements).
    # Since the loaded array is NOT scalar, the auto-expand code does NOT
    # run. So the array keeps its original length 10000 regardless of
    # num_sensors=5000. This is a potential issue.
    actual_len = p.qe_corrections.shape[0]
    record(
        f"JUNO with 5000: qe_corrections length = {actual_len}",
        True,
        "NOTE: file-based qe_corrections keep original length (10000) "
        "regardless of num_sensors -- no automatic truncation/padding",
    )
    mismatch = actual_len != 5000
    record(
        "JUNO with 5000: length mismatch detected (expected behavior)",
        mismatch,
        f"actual={actual_len}, requested num_sensors=5000",
    )
except Exception as e:
    record("JUNO with 5000 sensors", False, str(e))
    traceback.print_exc()

# 6d. JUNO with correct num_sensors
print("\n  --- 6d: JUNO with correct num_sensors=10000 ---")
try:
    p, mm, qe = load_physics_config(juno_path, num_sensors=10000)
    record(
        f"JUNO with 10000: qe_corrections length = {p.qe_corrections.shape[0]}",
        p.qe_corrections.shape == (10000,),
    )
except Exception as e:
    record("JUNO with 10000 sensors", False, str(e))

# 6e. Load with num_sensors=None
print("\n  --- 6e: JUNO with num_sensors=None ---")
try:
    p, mm, qe = load_physics_config(juno_path, num_sensors=None)
    record(
        f"JUNO no num_sensors: qe_corrections shape = {p.qe_corrections.shape}",
        p.qe_corrections.ndim >= 1,
        "file-based array is not expanded/modified",
    )
except Exception as e:
    record("JUNO no num_sensors", False, str(e))


# =========================================================================
# 7. Backward compatibility: load_detector_params
# =========================================================================
print("\n" + "=" * 72)
print("TEST 7: Backward compatibility -- load_detector_params with new format")
print("=" * 72)

for det_name, (cfg_file, n_sensors) in DETECTOR_INFO.items():
    cfg_path = os.path.join(CONFIG_DIR, cfg_file)
    try:
        params = load_detector_params(cfg_path, num_sensors=n_sensors)
        ok = isinstance(params, DetectorParams)
        record(
            f"load_detector_params({det_name}) returns DetectorParams",
            ok,
            f"type={type(params).__name__}",
        )
        # Verify all fields are present and are jax arrays
        all_fields_ok = True
        for field in DETECTOR_PARAMS_FIELDS:
            val = getattr(params, field)
            if not hasattr(val, "shape"):
                all_fields_ok = False
        record(
            f"load_detector_params({det_name}) all fields are arrays",
            all_fields_ok,
        )
    except Exception as e:
        record(f"load_detector_params({det_name})", False, str(e))
        traceback.print_exc()

# 7b. Missing fields handled gracefully
print("\n  --- 7b: Missing fields default to 1.0 ---")
for det_name in ["SK", "SK_like", "HK", "BigHK"]:
    if det_name not in DETECTOR_INFO:
        continue
    cfg_file, n_sensors = DETECTOR_INFO[det_name]
    cfg_path = os.path.join(CONFIG_DIR, cfg_file)
    with open(cfg_path) as f:
        raw = json.load(f)

    params = load_detector_params(cfg_path, num_sensors=n_sensors)
    for field in DETECTOR_PARAMS_FIELDS:
        if field not in raw:
            val = getattr(params, field)
            if field == "qe_corrections":
                # qe_corrections might be expanded
                continue
            ok = val.ndim == 0 and jnp.allclose(val, 1.0)
            record(
                f"load_detector_params({det_name}).{field} defaults to 1.0",
                bool(ok),
                f"actual={float(val):.4f}",
            )


# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 72)
print(f"SUMMARY: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 72)

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  - {e}")

sys.exit(0 if failed == 0 else 1)
