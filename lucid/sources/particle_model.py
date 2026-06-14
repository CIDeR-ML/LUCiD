"""ParticleModel container — SIREN model + Cherenkov context + cubic t0 params."""
from typing import NamedTuple

from lucid.utils import unpack_t0_params, unpack_siren_params
from lucid.sources.siren_rays import build_cherenkov_context, make_cherenkov_surrogate_fn
from lucid.siren.training.inference import SIRENPredictor


class ParticleModel(NamedTuple):
    """Everything needed to generate Cherenkov photon rays for a given particle type.

    Built via ``from_config(particle, material)`` which loads the s/s_max-trained SIREN
    model (whose metadata carries the smax/nphot count+range model), the ray-sampling
    knobs, and the cubic t0 coefficients from the data directory.
    """
    siren_predictor: object         # SIRENPredictor instance
    cherenkov_get_rays: object      # jitted emitter from make_cherenkov_surrogate_fn
    model_params: dict              # Flax model params (traced arg to the emitter)
    t0_params: tuple                # (a_coeffs, l_coeffs, b_coeffs) cubic stretched_exp
    particle: str                   # e.g. 'muon', 'electron'

    @staticmethod
    def from_config(particle: str = 'muon', material: str = 'water') -> 'ParticleModel':
        """Load the SIREN model + emitter + cubic t0 for a particle/material combination."""
        cfg = unpack_siren_params(particle, material)
        predictor = SIRENPredictor(cfg['siren_model_path'])
        ctx = build_cherenkov_context(predictor, cfg['ray_sampling'])
        return ParticleModel(
            siren_predictor=predictor,
            cherenkov_get_rays=make_cherenkov_surrogate_fn(ctx),
            model_params=predictor.params,
            t0_params=unpack_t0_params(particle, material),
            particle=particle,
        )
