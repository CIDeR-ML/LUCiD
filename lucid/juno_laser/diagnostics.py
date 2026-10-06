"""Numerical credibility checks for JUNO laser calibration objectives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from lucid.sources.diffuser_ball import bernstein_angular_density

from .source import set_diffuser_angular_shape


def response_jacobian(forward, coordinates, keys):
    """Return one source response and its optimizer-coordinate Jacobian."""
    theta = jnp.asarray(coordinates)
    if theta.ndim != 1:
        raise ValueError("coordinates must be one-dimensional")
    value = forward(theta, keys)
    jacobian = jax.jacfwd(forward)(theta, keys)
    if value.ndim < 1 or jacobian.shape != value.shape + theta.shape:
        raise ValueError("forward Jacobian has an unexpected shape")
    value.block_until_ready()
    jacobian.block_until_ready()
    return np.asarray(value), np.asarray(jacobian)


def summarize_fisher_matrix(fisher):
    """Return eigenvalue, rank, condition and CRB diagnostics."""
    matrix = np.asarray(fisher, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or np.any(~np.isfinite(matrix))
    ):
        raise ValueError("Fisher matrix must be finite and square")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    rank = int(np.linalg.matrix_rank(matrix))
    condition = (
        float(np.max(eigenvalues) / np.min(eigenvalues))
        if eigenvalues.size and np.min(eigenvalues) > 0.0
        else float("inf")
    )
    covariance = np.linalg.pinv(matrix, hermitian=True)
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(standard_deviation, standard_deviation)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )
    return {
        "fisher": matrix,
        "eigenvalues": eigenvalues,
        "rank": rank,
        "condition_number": condition,
        "crb_standard_deviation": (
            standard_deviation if rank == matrix.shape[0] else None
        ),
        "pseudoinverse_standard_deviation": standard_deviation,
        "pseudoinverse_correlation": correlation,
    }


def poisson_basis_fisher(
    bin_masses,
    jacobian,
    *,
    event_exposures=(1.0,),
    background_mass=0.0,
):
    """Build expected Poisson Fisher information on a numerical basis.

    This is a numerical-time-basis identifiability diagnostic, not yet the
    final continuous-time PPP Fisher.  No rate floor is applied.  A cell with
    zero expected mass and a nonzero derivative is reported as non-regular.
    """
    masses = np.asarray(bin_masses, dtype=np.float64)
    derivatives = np.asarray(jacobian, dtype=np.float64)
    if derivatives.shape[:-1] != masses.shape or derivatives.ndim < 2:
        raise ValueError("Jacobian must have mass shape plus parameter axis")
    if np.any(~np.isfinite(masses)) or np.any(masses < 0.0):
        raise ValueError("bin masses must be finite and non-negative")
    if np.any(~np.isfinite(derivatives)):
        raise ValueError("Jacobian must be finite")
    exposures = np.asarray(event_exposures, dtype=np.float64)
    if exposures.ndim == 0:
        exposures = exposures.reshape(1)
    if (
        exposures.ndim != 1
        or not exposures.size
        or np.any(~np.isfinite(exposures))
        or np.any(exposures <= 0.0)
    ):
        raise ValueError("event exposures must be finite and positive")
    background = np.broadcast_to(
        np.asarray(background_mass, dtype=np.float64), masses.shape
    )
    if np.any(~np.isfinite(background)) or np.any(background < 0.0):
        raise ValueError("background mass must be finite and non-negative")
    flat_mass = masses.reshape(-1)
    flat_background = background.reshape(-1)
    flat_jacobian = derivatives.reshape(-1, derivatives.shape[-1])
    fisher = np.zeros(
        (derivatives.shape[-1], derivatives.shape[-1]),
        dtype=np.float64,
    )
    nonregular = 0
    for exposure in exposures:
        expected = exposure * flat_mass + flat_background
        event_jacobian = exposure * flat_jacobian
        active_derivative = np.any(event_jacobian != 0.0, axis=1)
        invalid = (expected <= 0.0) & active_derivative
        nonregular += int(np.count_nonzero(invalid))
        valid = expected > 0.0
        weighted = event_jacobian[valid] / np.sqrt(expected[valid, None])
        fisher += weighted.T @ weighted
    if nonregular:
        raise ValueError(
            "non-regular Poisson support: zero expected mass with a "
            f"nonzero derivative in {nonregular} event-cells"
        )
    summary = summarize_fisher_matrix(fisher)
    axes = tuple(range(derivatives.ndim - 1))
    jacobian_summary = {
        "l2_norm": np.sqrt(np.sum(derivatives**2, axis=axes)),
        "max_abs": np.max(np.abs(derivatives), axis=axes),
        "mean_abs": np.mean(np.abs(derivatives), axis=axes),
        "nonzero_fraction": np.mean(derivatives != 0.0, axis=axes),
    }
    return {
        **summary,
        "regular": True,
        "n_response_cells": int(masses.size),
        "jacobian_summary": jacobian_summary,
    }


def split_source_indices(n_sources, heldout_indices):
    """Return disjoint training and held-out source-index tuples."""
    if n_sources < 2:
        raise ValueError("held-out validation requires at least two sources")
    heldout = tuple(sorted(set(int(item) for item in heldout_indices)))
    if not heldout:
        raise ValueError("at least one held-out source is required")
    if heldout[0] < 0 or heldout[-1] >= n_sources:
        raise ValueError("held-out source index is outside the source range")
    train = tuple(index for index in range(n_sources) if index not in heldout)
    if not train:
        raise ValueError("at least one training source must remain")
    return train, heldout


def summarize_closure_scan(records):
    """Summarize gradient variation over photon, ensemble and seed scans."""
    records = tuple(records)
    if not records:
        raise ValueError("closure scan requires at least one result")
    completed = [record for record in records if record.get("completed")]
    if not completed:
        raise ValueError("closure scan has no completed results")
    names = tuple(completed[0]["parameter_names"])
    settings = {}
    for record in completed:
        if tuple(record["parameter_names"]) != names:
            raise ValueError("scan results use different parameter orders")
        key = (
            int(record["n_photons"]),
            int(record["fit_ensemble_size"]),
        )
        settings.setdefault(key, []).append(record)
    summaries = {}
    for (n_photons, ensemble_size), group in sorted(settings.items()):
        gradients = np.asarray([
            [record["initial_gradient"][name] for name in names]
            for record in group
        ])
        losses = np.asarray([
            record["initial_loss_per_pe"] for record in group
        ])
        gradient_mean = np.mean(gradients, axis=0)
        gradient_std = np.std(gradients, axis=0)
        gradient_norms = np.linalg.norm(gradients, axis=1)
        safe_norms = np.maximum(gradient_norms, 1.0e-30)
        unit_gradients = gradients / safe_norms[:, None]
        if len(group) > 1:
            cosine_matrix = unit_gradients @ unit_gradients.T
            upper = cosine_matrix[np.triu_indices(len(group), k=1)]
        else:
            upper = np.ones(1)
        data_seeds = sorted(set(int(record["seed"]) for record in group))
        if len(data_seeds) != 1:
            raise ValueError(
                "each transport scan setting must keep one data seed"
            )
        seed_offsets = [
            int(record["fit_seed_offset"]) for record in group
        ]
        label = f"photons_{n_photons}_ensemble_{ensemble_size}"
        summaries[label] = {
            "n_photons": n_photons,
            "ensemble_size": ensemble_size,
            "n_seeds": len(group),
            "data_seed": data_seeds[0],
            "fit_seed_offsets": seed_offsets,
            "gradient_mean": gradient_mean,
            "gradient_std": gradient_std,
            "gradient_relative_std": gradient_std / np.maximum(
                np.abs(gradient_mean), 1.0e-30
            ),
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "gradient_norm_std": float(np.std(gradient_norms)),
            "pairwise_gradient_cosine_mean": float(np.mean(upper)),
            "pairwise_gradient_cosine_min": float(np.min(upper)),
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
        }
    return {
        "schema": "lucid_photon_ensemble_seed_scan_v1",
        "parameter_names": list(names),
        "n_completed": len(completed),
        "n_failed": len(records) - len(completed),
        "settings": summaries,
    }


def objective_ad_fd_comparison(
    objective,
    coordinates,
    keys,
    *,
    relative_step: float = 1.0e-3,
    minimum_step: float = 1.0e-4,
    value_and_grad_fn=None,
):
    """Compare automatic and centered finite-difference gradients.

    The objective is evaluated with identical transport keys for every
    perturbation.  This common-random-number condition is essential: finite
    differences of independently sampled Monte Carlo objectives would mostly
    measure sampling noise instead of a derivative.
    """
    if relative_step <= 0.0 or minimum_step <= 0.0:
        raise ValueError("finite-difference steps must be positive")
    theta = jnp.asarray(coordinates)
    if theta.ndim != 1:
        raise ValueError("coordinates must be one-dimensional")
    if value_and_grad_fn is None:
        value, ad_gradient = jax.value_and_grad(objective)(theta, keys)
        value.block_until_ready()
    else:
        value, ad_gradient = value_and_grad_fn(theta, keys)
    ad = np.asarray(ad_gradient, dtype=np.float64)
    host_theta = np.asarray(theta, dtype=np.float64)
    steps = np.maximum(np.abs(host_theta) * relative_step, minimum_step)
    finite_difference = []
    plus_values = []
    minus_values = []
    for index, step in enumerate(steps):
        displacement = jnp.zeros_like(theta).at[index].set(
            jnp.asarray(step, dtype=theta.dtype)
        )
        plus = float(objective(theta + displacement, keys))
        minus = float(objective(theta - displacement, keys))
        plus_values.append(plus)
        minus_values.append(minus)
        finite_difference.append((plus - minus) / (2.0 * step))
    fd = np.asarray(finite_difference, dtype=np.float64)
    plus_array = np.asarray(plus_values, dtype=np.float64)
    minus_array = np.asarray(minus_values, dtype=np.float64)
    invalid_perturbations = np.flatnonzero(
        ~np.isfinite(plus_array) | ~np.isfinite(minus_array)
    )
    if invalid_perturbations.size:
        details = ", ".join(
            f"index {index}: plus={plus_array[index]}, "
            f"minus={minus_array[index]}"
            for index in invalid_perturbations
        )
        raise FloatingPointError(
            "AD-FD perturbation left the strict PPP support: " + details
        )
    scale = np.maximum.reduce(
        (np.abs(ad), np.abs(fd), np.full_like(ad, 1.0e-12))
    )
    relative_difference = np.abs(ad - fd) / scale
    if not (
        np.isfinite(float(value))
        and np.all(np.isfinite(ad))
        and np.all(np.isfinite(fd))
    ):
        raise FloatingPointError("AD-FD audit produced non-finite values")
    return {
        "objective": float(value),
        "relative_step": float(relative_step),
        "minimum_step": float(minimum_step),
        "steps": steps,
        "ad_gradient": ad,
        "finite_difference_gradient": fd,
        "absolute_difference": np.abs(ad - fd),
        "relative_difference": relative_difference,
        "plus_objective": plus_array,
        "minus_objective": minus_array,
        "max_relative_difference": float(np.max(relative_difference)),
    }


def summarize_key_closure_pair(same, independent):
    """Compare same-key and independent-key closure result mappings."""
    fields = (
        "source_positions_m",
        "n_events",
        "n_photons",
        "k",
        "intensity",
        "background_rate_hz_per_pmt",
        "parameter_names",
        "truth",
        "data_importance_normalization",
        "fit_importance_normalization",
    )
    for field in fields:
        if same.get(field) != independent.get(field):
            raise ValueError(
                "closure pair differs in non-key field " + field
            )
    names = tuple(same["parameter_names"])
    truth = np.asarray([same["truth"][name] for name in names])
    same_fit = np.asarray([same["fit"][name] for name in names])
    independent_fit = np.asarray([
        independent["fit"][name] for name in names
    ])
    scale = np.maximum(np.abs(truth), 1.0)
    return {
        "schema": "lucid_same_independent_key_comparison_v1",
        "parameter_names": list(names),
        "same_fit": dict(zip(names, same_fit.tolist())),
        "independent_fit": dict(zip(names, independent_fit.tolist())),
        "same_minus_truth_scaled": dict(
            zip(names, ((same_fit - truth) / scale).tolist())
        ),
        "independent_minus_truth_scaled": dict(
            zip(names, ((independent_fit - truth) / scale).tolist())
        ),
        "independent_minus_same_scaled": dict(
            zip(names, ((independent_fit - same_fit) / scale).tolist())
        ),
        "same_final_loss_per_pe": same["final_loss_per_pe"],
        "independent_final_loss_per_pe": independent[
            "final_loss_per_pe"
        ],
        "same_fit_seed_offset": same["fit_seed_offset"],
        "independent_fit_seed_offset": independent["fit_seed_offset"],
    }


def diffuser_importance_audit(
    source,
    photon_counts,
    seeds,
    *,
    n_angular_bins: int = 8,
):
    """Compare analytic and self-normalized diffuser importance estimators.

    A deterministic quadrature supplies the nominal angular-bin truth.  For
    every photon budget and random seed, both estimators reuse exactly the
    same proposal rays.  The reported NLL gradient is with respect to the
    three identifiable LSPhase coordinates at the nominal truth.
    """
    photon_counts = tuple(int(value) for value in photon_counts)
    seeds = tuple(int(value) for value in seeds)
    if not photon_counts or min(photon_counts) <= 0:
        raise ValueError("photon counts must be positive and non-empty")
    if not seeds or n_angular_bins <= 1:
        raise ValueError("seeds must be non-empty and angular bins > 1")
    nodes, quadrature_weights = np.polynomial.legendre.leggauss(512)
    density = np.asarray(
        bernstein_angular_density(nodes, source.angular_logits)
    )
    target_bins = np.clip(
        ((nodes + 1.0) * 0.5 * n_angular_bins).astype(int),
        0,
        n_angular_bins - 1,
    )
    probabilities = np.bincount(
        target_bins,
        weights=quadrature_weights * density,
        minlength=n_angular_bins,
    )
    probabilities /= np.sum(probabilities)
    target = jnp.asarray(
        probabilities * float(source.intensity), dtype=jnp.float32
    )
    origin = jnp.zeros(3, dtype=jnp.float32)
    results = {}
    for n_photons in photon_counts:
        by_mode = {}
        for mode, code in (("analytic", 0), ("self_normalized", 1)):
            selected_source = source._replace(
                importance_normalization=code
            )

            def objective(coordinates, key):
                varied = set_diffuser_angular_shape(
                    selected_source, coordinates
                )
                batch = varied(n_photons, key)
                normals = (
                    batch.positions - varied.center[None, :]
                ) / varied.radius
                cosine = normals @ varied.axis
                indices = jnp.clip(
                    jnp.floor(
                        0.5 * (cosine + 1.0) * n_angular_bins
                    ).astype(jnp.int32),
                    0,
                    n_angular_bins - 1,
                )
                predicted = jax.ops.segment_sum(
                    batch.weights,
                    indices,
                    num_segments=n_angular_bins,
                )
                return jnp.sum(
                    predicted - target * jnp.log(predicted)
                ) / jnp.sum(target)

            value_and_grad = jax.jit(jax.value_and_grad(objective))
            records = []
            for seed in seeds:
                key = jax.random.PRNGKey(seed)
                batch = selected_source(n_photons, key)
                normals = np.asarray(
                    (batch.positions - selected_source.center[None, :])
                    / selected_source.radius
                )
                cosine = normals @ np.asarray(selected_source.axis)
                occupancy = np.histogram(
                    cosine, bins=n_angular_bins, range=(-1.0, 1.0)
                )[0]
                if np.any(occupancy == 0):
                    raise ValueError(
                        "importance audit has an empty angular bin; "
                        "increase the photon budget or reduce bin count"
                    )
                value, gradient = value_and_grad(origin, key)
                value.block_until_ready()
                records.append({
                    "seed": seed,
                    "total_weight_ratio": float(
                        jnp.sum(batch.weights) / selected_source.intensity
                    ),
                    "nll_per_expected_photon": float(value),
                    "shape_gradient": np.asarray(
                        gradient, dtype=np.float64
                    ),
                })
            gradients = np.stack([
                record["shape_gradient"] for record in records
            ])
            totals = np.asarray([
                record["total_weight_ratio"] for record in records
            ])
            by_mode[mode] = {
                "records": records,
                "total_weight_ratio_mean": float(np.mean(totals)),
                "total_weight_ratio_std": float(np.std(totals)),
                "gradient_mean": np.mean(gradients, axis=0),
                "gradient_std": np.std(gradients, axis=0),
                "gradient_mean_norm": float(
                    np.linalg.norm(np.mean(gradients, axis=0))
                ),
            }
        paired = np.stack([
            analytic["shape_gradient"] - normalized["shape_gradient"]
            for analytic, normalized in zip(
                by_mode["analytic"]["records"],
                by_mode["self_normalized"]["records"],
            )
        ])
        results[str(n_photons)] = {
            **by_mode,
            "paired_gradient_difference_mean": np.mean(paired, axis=0),
            "paired_gradient_difference_std": np.std(paired, axis=0),
        }
    return {
        "schema": "lucid_diffuser_importance_audit_v1",
        "n_angular_bins": n_angular_bins,
        "seeds": list(seeds),
        "target_bin_probabilities": probabilities,
        "photon_budgets": results,
    }
