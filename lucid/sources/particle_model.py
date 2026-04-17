"""ParticleModel container — SIREN model + normalization + t0 params."""
from typing import NamedTuple

from lucid.utils import (
    unpack_t0_params, unpack_photonsim_params, base_dir_path,
)
from lucid.siren.core import create_photonsim_siren_grid
from lucid.siren.training.inference import SIRENPredictor


class ParticleModel(NamedTuple):
    """Everything needed to generate photon rays for a given particle type.

    Built via ``from_config(particle, material)`` which loads the SIREN model,
    t0 parameters, and photon normalization from the data directory.

    Does NOT store ``material`` — validation happens at load time only.
    """
    siren_predictor: object         # SIRENPredictor instance
    grid_data: object               # jnp.ndarray from create_photonsim_siren_grid
    model_params: dict              # Flax model params
    t0_params: tuple                # (baseline_slope, baseline_intercept, A_slope, ...)
    normalization: tuple            # (a, b, c) for tot_n_photons = a * E^b + c
    num_seeds: tuple                # (a, b, c) for num_seeds = a * E^b + c
    particle: str                   # e.g. 'muon', 'electron'

    @staticmethod
    def from_config(particle: str = 'muon',
                    material: str = 'water') -> 'ParticleModel':
        """Load SIREN model and parameters for a particle/material combination.

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
        # Load photonsim parameters (normalization, seeds, model path)
        photonsim_params = unpack_photonsim_params(particle, material)

        # Load SIREN model
        siren_model_path = photonsim_params['siren_model_path']
        predictor = SIRENPredictor(siren_model_path)
        grid_data = create_photonsim_siren_grid(predictor)

        # t0 timing parameters
        t0_params = unpack_t0_params(particle, material)

        return ParticleModel(
            siren_predictor=predictor,
            grid_data=grid_data,
            model_params=predictor.params,
            t0_params=t0_params,
            normalization=photonsim_params['tot_n_photons_normalization'],
            num_seeds=photonsim_params['num_seeds'],
            particle=particle,
        )
