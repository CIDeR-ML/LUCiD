# Wavelength-Dependent Calibration: Findings & Next Steps

## What we did

Ran 4-parameter calibration (scatter_length, wall_reflection_rate, sensor_reflection_rate, absorption_length) at three laser wavelengths using `wavelength_mode=False`, setting the "true" scatter/absorption values from the water medium physics at each wavelength.

Settings: SK_like detector, Nphot=500k (opt) / 5M (truth), K=10, 500 Adam iterations.

## Results

| wavelength | L_scat (m) | L_abs (m) | scatter err | wall err | sensor err | absorb err |
|------------|-----------|-----------|-------------|----------|------------|------------|
| 350 nm     | 91        | 608       | 0.7%        | 9.6%     | 9.5%       | **20%**    |
| 405 nm     | 187       | 368       | 0.2%        | 1.3%     | 1.7%       | 2.3%       |
| 500 nm     | 504       | 39        | **10%**     | 10.8%    | 6.3%       | 4.7%       |

## Key finding: parameter identifiability depends on length-to-detector ratio

The detector diameter is ~34m. A parameter is well-constrained when its associated length scale is within roughly 1-10x the detector size. When L >> detector, the per-bounce effect is too small to distinguish from noise or from correlated parameters.

- **405nm is the sweet spot**: both L_scat=187m and L_abs=368m are in a measurable regime. All 4 params converge within 2.3%.
- **350nm**: L_abs=608m (~18x detector). Absorption effect per chord is only ~5.4%. Absorption error is 20%, anti-correlated with wall_reflection error (9.6%), suggesting degeneracy.
- **500nm**: L_scat=504m (~15x detector). Scatter effect per chord is only ~6.5%. Scatter error is 10%.

## Open question: statistics vs degeneracy

When a length scale >> detector, two effects limit convergence:

1. **Statistics**: Finite Nphot creates a frozen noise pattern whose minimum is shifted from truth. Error should scale as ~1/sqrt(Nphot).
2. **Parameter degeneracy**: wall_reflection and absorption are partially degenerate (both multiplicatively reduce per-bounce survival). When d/L_abs is small, exp(-d/L_abs) ~ 1 - d/L_abs, so a small change in wall_refl compensates a change in L_abs. The 350nm errors are anti-correlated (wall_refl high, absorption low), consistent with this degeneracy.

The degeneracy is broken by geometric diversity (different photons travel different path lengths), but this breaking is weak when d/L is small.

## Proposed experiments (not yet run)

**A. Nphot scaling**: Run 350nm at Nphot = 500k, 2M, 8M, 32M. If absorption error ~ 1/sqrt(Nphot), statistics is the bottleneck. If it plateaus, degeneracy dominates.

**B. Break the degeneracy**: At 350nm, fix wall_reflection=0.200 and optimize only scatter + absorption. If absorption converges, degeneracy was the bottleneck.

**C. Hessian analysis**: Compute d^2 loss / d theta_i d theta_j at the true parameters. Small eigenvalue in the (wall_refl, absorption) subspace confirms a flat valley. If that eigenvalue scales linearly with Nphot, more photons sharpen the valley.

**Expectation**: Degeneracy is the primary bottleneck at 350nm (the correlated errors are the signature). Experiment B should confirm this most cleanly.

## Notebook

`good_notebooks/wavelength_calibration.ipynb` — runs the full comparison with convergence and loss plots. Uses `wavelength_mode=False` with true values derived from the medium, so scatter/absorption are free optimization parameters.

## Relation to the wavelength_mode=True refactoring

In `wavelength_mode=True`, scatter and absorption come from the medium and are **not** optimized. Only wall_reflection and sensor_reflection are free. This sidesteps the degeneracy entirely — the wavelength mode converges faster and more accurately (both reflections within 0.5% at 405nm and 350nm).

The calibration notebook tests a different question: can we *recover* the medium values from data if we don't know them? The answer is yes at 405nm, partially at 350nm/500nm (limited by degeneracy with reflections).
