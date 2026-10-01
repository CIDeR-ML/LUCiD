# Combined Rayleigh and Mie TOF comparison

## Configuration

This test enables Rayleigh and Mie water scattering simultaneously while
keeping bulk-water absorption and all optical reflection disabled.

- SKDetSim: 20,000 pulses, water factors `(abs, ray, mie) = (0, 1, 1)`.
- LUCiD: 32 independent batches of one million rays with four propagation
  steps and the average-response estimator.
- Both use the 405 nm laser source and 8,090 photons per pulse.
- LUCiD uses the corrected rotationally symmetric Rayleigh sampler and the
  SK-IV `SGMIES` law, `p(mu) = 2 mu` for `0 <= mu <= 1`.
- The comparison uses measured PMT directions, the stable local
  sphere-plus-torus solve, wall-selected PMT routing, the local-angle PMT
  response, the SK group speed, and the SK-IV timing response.

The SKDetSim run contains 13,365,111 accepted photoelectrons and 6,016,713
digitized hits. The LUCiD result contains 15,717,703 weighted deposition
records.

## TOF results

Both accepted-PE prompts are 147 ns before display alignment. With each curve
normalized by its own detector-wide accepted-PE total, the full seven-section
TOF total-variation distance is **0.001904**.

| Section | TOF TV contribution |
|---|---:|
| Top | 0.000344 |
| B1 | 0.000137 |
| B2 | 0.000134 |
| B3 | 0.000089 |
| B4 | 0.000154 |
| B5 | 0.000123 |
| Bottom | 0.000923 |

The total accepted yields are:

- SKDetSim: **668.2556 ± 0.1755 PE/pulse**.
- LUCiD: **728.9511 ± 0.1778 PE/pulse**.

LUCiD is therefore 60.696 PE/pulse, or 9.083%, higher. The detector-total
normalization removes this global yield difference from the canonical shape
plot; the PE-per-pulse plot retains it.

## Statistical uncertainty

SKDetSim treats each pulse as one independent sample. LUCiD treats each
one-million-ray batch as one independent sample. Individual LUCiD depositions
are not counted equally. For time bin `j` and batch `b`, the bin content is

`S_bj = sum_i w_i`,

where the sum includes the actual average-response weight of every deposition
in that batch and bin. The detector total for the same batch is `T_b`. For the
normalized fraction `f_j = sum_b S_bj / sum_b T_b`, the ratio-of-means delta
method uses the batch distribution of

`S_bj - f_j T_b`.

This retains unequal weights and the covariance between a bin and the total
used to normalize it. The combined LUCiD deposition weights have an 18.1%
RMS/mean spread. Exact per-deposition batch indices are stored in this output;
the weighted batch sums reproduce the saved batch charges to better than
`9e-8` relative.

The pull is `(f_SK - f_LUCiD) / sqrt(SEM_SK^2 + SEM_LUCiD^2)`. B1, B2, and B3
have RMS pulls 1.15, 1.03, and 0.95, respectively, with no bin above 3 sigma.
Small but statistically resolved structure remains in isolated Top, B4, B5,
and bottom bins. The largest deviations are Top 1180–1190 ns at +6.33 sigma
and Bottom 1490–1500 ns at +5.15 sigma. These differences are small in
absolute detector fraction but are resolved by the high statistics.

## Artifacts

- `results_rayleigh_mie_20000_lucid32m/canonical_tof.png`
- `results_rayleigh_mie_20000_lucid32m/canonical_tof_pe_per_pulse.png`
- `results_rayleigh_mie_20000_lucid32m/tof_statistical_agreement.png`
- `results_rayleigh_mie_20000_lucid32m/tof_statistical_agreement.json`
- `sk_rayleigh_mie_response_20000.npz`
- `lucid_sk20inch_angle_rayleigh_mie_groupspeed_wall_stable_local_timing_32m.npz`

Large raw ZBS, HBOOK, and ROOT conversion intermediates were removed after the
compact SK and LUCiD caches were validated.
