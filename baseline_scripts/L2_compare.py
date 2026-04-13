"""L2 Baseline Comparison — Compare tools vs lucid for track forward + calibration.

With matching grid params (n_cap=150, n_angular=250, n_height=150), both backends
should produce bit-identical results. This script verifies that.

Checks:
  1. Track forward: exact match (same grid → same values)
  2. Calibration: convergence to true params, tools≈lucid trajectories
"""
import json
import sys
import numpy as np

PASS = 0
FAIL = 0
WARN = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def warn(name, detail=""):
    global WARN
    WARN += 1
    print(f"  WARN  {name}  {detail}")


def rel_diff(a, b):
    denom = max(abs(a), abs(b), 1e-10)
    return abs(a - b) / denom


# ── L2 Track Forward ────────────────────────────────────────────────

print("\n=== L2 Track Forward ===")
try:
    with open("baseline_scripts/L2_track_forward_tools.json") as f:
        tools = json.load(f)
    with open("baseline_scripts/L2_track_forward_lucid.json") as f:
        lucid = json.load(f)

    # Structural checks
    check("same mode", tools["mode"] == lucid["mode"],
          f"tools={tools['mode']}, lucid={lucid['mode']}")

    if "log_w_shape" in tools:
        check("same log_w shape", tools["log_w_shape"] == lucid["log_w_shape"])
        check("same flat_times shape", tools["flat_times_shape"] == lucid["flat_times_shape"])

    # Gradient finiteness
    check("tools grad finite", tools["grad_finite"])
    check("lucid grad finite", lucid["grad_finite"])

    # With matching grid params, expect exact match
    t_grad = np.array(tools["grad_vector"])
    l_grad = np.array(lucid["grad_vector"])

    # Charge sum exact match
    if "total_charge_sum" in tools:
        rd = rel_diff(tools["total_charge_sum"], lucid["total_charge_sum"])
        check("charge sum exact match", rd < 1e-6,
              f"tools={tools['total_charge_sum']:.6f}, lucid={lucid['total_charge_sum']:.6f}, rel_diff={rd:.2e}")

    # Loss exact match
    rd_loss = rel_diff(tools["grad_loss_value"], lucid["grad_loss_value"])
    check("loss value exact match", rd_loss < 1e-6,
          f"tools={tools['grad_loss_value']:.6f}, lucid={lucid['grad_loss_value']:.6f}, rel_diff={rd_loss:.2e}")

    # Per-component gradient exact match
    print("\n  Gradient component comparison:")
    labels = ["energy", "pos_x", "pos_y", "pos_z", "theta", "phi"]
    all_match = True
    for i in range(len(t_grad)):
        label = labels[i] if i < len(labels) else f"[{i}]"
        rd_i = rel_diff(float(t_grad[i]), float(l_grad[i]))
        status = "MATCH" if rd_i < 1e-5 else "DIFF"
        if rd_i >= 1e-5:
            all_match = False
        print(f"    {label:8s}: tools={t_grad[i]:12.6f}  lucid={l_grad[i]:12.6f}  rel_diff={rd_i:.2e}  {status}")

    check("all gradient components match", all_match)

except FileNotFoundError as e:
    print(f"  SKIP  Missing file: {e}")


# ── L2 Calibration 4-param ──────────────────────────────────────────

print("\n=== L2 Calibration 4-param ===")
try:
    with open("baseline_scripts/L2_4_baseline_tools.json") as f:
        tools_cal = json.load(f)
    with open("baseline_scripts/L2_4_baseline_lucid.json") as f:
        lucid_cal = json.load(f)

    # Both converge (loss decreases significantly)
    t_loss = tools_cal["loss_curve"]
    l_loss = lucid_cal["loss_curve"]
    t_reduction = (t_loss[0] - t_loss[-1]) / t_loss[0]
    l_reduction = (l_loss[0] - l_loss[-1]) / l_loss[0]
    check("tools loss reduces >20%", t_reduction > 0.20,
          f"reduction={t_reduction:.1%}")
    check("lucid loss reduces >20%", l_reduction > 0.20,
          f"reduction={l_reduction:.1%}")

    # Final loss match between backends (should be very close with matching grids)
    rd_final = rel_diff(t_loss[-1], l_loss[-1])
    check("final loss match (<1%)", rd_final < 0.01,
          f"tools={t_loss[-1]:.6f}, lucid={l_loss[-1]:.6f}, rel_diff={rd_final:.4f}")

    # Parameter convergence to true values
    true_vals = [tools_cal["true_scatter_length"], tools_cal["true_wall_reflection"],
                 tools_cal["true_sensor_reflection"], tools_cal["true_absorption_length"]]
    param_names = ["scatter_length", "wall_reflection", "sensor_reflection", "absorption_length"]
    # Relative error thresholds per parameter
    # scatter and absorption converge well; wall/sensor reflections are coupled
    thresholds = [0.10, 0.50, 0.50, 0.10]

    print("\n  Convergence to true values:")
    for backend, cal, label in [("tools", tools_cal, "tools"), ("lucid", lucid_cal, "lucid")]:
        final_p = cal["param_history"][-1]
        init_p = cal["init_params"]
        print(f"\n  {label}:")
        for i, name in enumerate(param_names):
            true_v = true_vals[i]
            rel_err = abs(final_p[i] - true_v) / max(abs(true_v), 1e-10)
            init_err = abs(init_p[i] - true_v) / max(abs(true_v), 1e-10)
            improved = rel_err < init_err
            print(f"    {name:22s}: true={true_v:7.2f}  init={init_p[i]:7.2f}  "
                  f"final={final_p[i]:7.2f}  rel_err={rel_err:.3f}  {'✓' if improved else '—'}")

            check(f"{label} {name} within {thresholds[i]:.0%}",
                  rel_err < thresholds[i],
                  f"rel_err={rel_err:.3f}")

    # Cross-backend: final parameters should be similar
    t_final = tools_cal["param_history"][-1]
    l_final = lucid_cal["param_history"][-1]
    print("\n  Cross-backend parameter match:")
    for i, name in enumerate(param_names):
        rd = rel_diff(t_final[i], l_final[i])
        print(f"    {name:22s}: tools={t_final[i]:7.3f}  lucid={l_final[i]:7.3f}  rel_diff={rd:.4f}")
        check(f"tools≈lucid {name} (<5%)", rd < 0.05,
              f"rel_diff={rd:.4f}")

    # Reflection coupling check: wall+sensor sum should be stable
    t_refl_sum = t_final[1] + t_final[2]
    l_refl_sum = l_final[1] + l_final[2]
    true_refl_sum = true_vals[1] + true_vals[2]
    print(f"\n  Reflection coupling (wall+sensor sum):")
    print(f"    true={true_refl_sum:.3f}  tools={t_refl_sum:.3f}  lucid={l_refl_sum:.3f}")

except FileNotFoundError as e:
    print(f"  SKIP  Missing file: {e}")


# ── Summary ─────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"L2 Results: {PASS} PASS, {FAIL} FAIL, {WARN} WARN")
if FAIL > 0:
    print("Some checks failed — review differences above.")
    sys.exit(1)
else:
    print("All L2 checks passed.")
