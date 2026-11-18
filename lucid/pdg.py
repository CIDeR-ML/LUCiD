"""Particle rest masses, by PDG code, in MeV: one table for the whole repository.

PDG Review of Particle Physics 2022 (R. L. Workman et al., Prog. Theor. Exp. Phys. 2022, 083C01).
Free of every import, so a submit host without the scientific stack (the fiTQun fan-out) can use it.
"""

PDG_MASS_MEV = {
    11: 0.51099895,       # e-
    13: 105.6583755,      # mu-
    111: 134.9768,        # pi0
    211: 139.57039,       # pi+
    2112: 939.56542052,   # n
    2212: 938.27208816,   # p
}
