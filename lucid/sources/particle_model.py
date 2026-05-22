"""ParticleModel container — SIREN inference context + model params + t0 params."""
from __future__ import annotations

from typing import NamedTuple

from lucid.utils import unpack_t0_params, unpack_siren_params
from lucid.siren.core import build_photonsim_context
from lucid.siren.training.inference import SIRENPredictor


class ParticleModel(NamedTuple):
    """Everything needed to generate photon rays for a given particle type.

    Built via ``from_config(particle, material)`` which loads the SIREN model,
    its inference context, and the t0 timing parameters from the data directory.

    Does NOT store ``material`` — validation happens at load time only.
    """
    siren_predictor: object         # SIRENPredictor instance
    context: object                 # PhotonSimContext from build_photonsim_context
    model_params: dict              # Flax model params
    t0_params: tuple                # (baseline_slope, baseline_intercept, A_slope, ...)
    particle: str                   # e.g. 'muon', 'electron'

    @staticmethod
    def from_config(particle: str = 'muon',
                    material: str = 'water') -> 'ParticleModel':
        """Load the SIREN model and parameters for a particle/material combination.

        Parameters
        ----------
        particle : str
            Particle type (e.g. 'muon', 'electron').
        material : str
            Medium material. Used to locate the correct SIREN model directory.
            Not stored on the resulting object.

        Returns
        -------
        ParticleModel

        Raises
        ------
        FileNotFoundError
            If SIREN model data is not found for this particle/material.
        """
        # Resolve the SIREN model path, load it, and build the inference context
        # (domain ranges + s_max(E) / N_photons(E) closures + ray-sampling knobs).
        siren_params = unpack_siren_params(particle, material)
        predictor = SIRENPredictor(siren_params['siren_model_path'])
        context = build_photonsim_context(predictor, siren_params['ray_sampling'])

        # t0 timing parameters
        t0_params = unpack_t0_params(particle, material)

        return ParticleModel(
            siren_predictor=predictor,
            context=context,
            model_params=predictor.params,
            t0_params=t0_params,
            particle=particle,
        )
