"""
JAX-based SIREN Trainer for PhotonSim Data

The training working point is data: :class:`Recipe`, read from ``nominal.json`` (the recipe of
every shipped emitter) with any field overridden. :class:`SIRENTrainer` trains a SIREN on a
:class:`~lucid.siren.training.dataset.PhotonSimDataset` and writes the result as an emitter
(``emitter.json`` + ``weights.npz``) through :func:`lucid.siren.emitter.write_emitter`, the
loader's own schema, with the recipe, losses and commit recorded in it.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import matplotlib.pyplot as plt

from lucid.siren.emitter import T0_FORM, read_weights, write_emitter, write_weights
from lucid.siren.net import SIREN
from lucid.siren.training.dataset import table_curves

logger = logging.getLogger(__name__)

NOMINAL = Path(__file__).with_name('nominal.json')
GRID_BINS = 250        # the ray samplers' first-pass grid, recorded into every new emitter


@dataclass(frozen=True)
class Recipe:
    """A training working point; the fields are those of ``nominal.json``."""
    # network
    hidden_features: int
    hidden_layers: int
    w0: float
    # optimiser: 'adam', 'adamw' or 'sgd' (momentum 0.9); weight_decay is AdamW's or, for the
    # others, an added decay term; grad_clip_norm 0 = no clipping
    optimizer: str
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    # plateau schedule: cut the LR by lr_reduction_factor after `patience` validations without a
    # new best, down to min_lr (off: a constant LR)
    use_patience_scheduler: bool
    patience: int
    lr_reduction_factor: float
    min_lr: float
    batch_size: int
    num_steps: int
    val_every: int
    # dataset (PhotonSimDataset)
    zero_threshold: float
    zero_keep_frac: float
    energy_balance: str
    target_importance: float
    val_split: float
    seed: int

    @classmethod
    def load(cls, path=NOMINAL, **overrides) -> 'Recipe':
        """The recipe in ``path`` with ``overrides``; an unknown or missing field raises."""
        values = {**json.loads(Path(path).read_text()), **overrides}
        names = {f.name for f in dataclasses.fields(cls)}
        unknown, missing = sorted(set(values) - names), sorted(names - set(values))
        if unknown or missing:
            raise ValueError(f"{path}: unknown recipe fields {unknown}, missing {missing}")
        if values['optimizer'] not in ('adam', 'adamw', 'sgd'):
            raise ValueError(f"optimizer must be 'adam', 'adamw' or 'sgd', got {values['optimizer']!r}")
        return cls(**values)


class SIRENTrainer:
    """
    Trains a SIREN on a PhotonSim dataset with a :class:`Recipe`.

    ```python
    from lucid.siren.training.dataset import PhotonSimDataset
    from lucid.siren.training.trainer import Recipe, SIRENTrainer

    recipe = Recipe.load(num_steps=5000)
    dataset = PhotonSimDataset(h5, val_split=recipe.val_split, zero_threshold=recipe.zero_threshold,
                               zero_keep_frac=recipe.zero_keep_frac,
                               energy_balance=recipe.energy_balance,
                               target_importance=recipe.target_importance)
    trainer = SIRENTrainer(dataset, recipe, output_dir='runs/x')
    trainer.train()
    trainer.save_emitter('runs/x')
    ```

    ``init`` is a ``weights.npz`` to start from (fine-tuning an emitter for a new material, or
    continuing a run past its wall time from a checkpoint; the optimiser state starts fresh).
    Every ``checkpoint_every`` steps (0: never) the weights are written to ``output_dir`` as
    ``checkpoint_step_<N>.npz``, a valid ``init``.
    """

    def __init__(
        self,
        dataset,
        recipe: Recipe,
        output_dir: Optional[Path] = None,
        init: Optional[Path] = None,
        log_every: int = 10,
        checkpoint_every: int = 0,
    ):
        self.dataset = dataset
        self.recipe = recipe
        self.output_dir = Path(output_dir) if output_dir else None
        self.init = init
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every

        self.model = SIREN(
            hidden_features=recipe.hidden_features,
            hidden_layers=recipe.hidden_layers,
            out_features=1,
            w0=recipe.w0
        )

        # Check JAX devices
        self.device = self._check_devices()

        # Initialize state variables
        self.state = None
        self.rng = jax.random.PRNGKey(recipe.seed)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'step': []
        }

        # Patience scheduler state
        self.current_lr = recipe.learning_rate
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.lr_reductions = 0

        # Callbacks
        self.callbacks = []

    def _check_devices(self):
        """Check and log available JAX devices."""
        devices = jax.devices()
        logger.info(f"JAX devices available: {len(devices)}")
        for i, device in enumerate(devices):
            logger.info(f"  Device {i}: {device.device_kind}")
        return devices[0]

    def _make_optimizer(self, lr: float):
        """The recipe's optimiser at learning rate ``lr``."""
        r = self.recipe
        if r.optimizer == 'adamw':
            base = optax.adamw(learning_rate=lr, weight_decay=r.weight_decay)
        elif r.optimizer == 'sgd':
            base = optax.sgd(learning_rate=lr, momentum=0.9)
        else:
            base = optax.adam(learning_rate=lr)
        parts = [optax.clip_by_global_norm(r.grad_clip_norm)] if r.grad_clip_norm > 0 else []
        parts.append(base)
        if r.optimizer != 'adamw' and r.weight_decay > 0:
            parts.append(optax.add_decayed_weights(r.weight_decay))
        return optax.chain(*parts)

    def _init_training_state(self):
        """Initialize model parameters (or read ``init``) and optimizer state."""
        if self.init:
            params = read_weights(self.init)
            logger.info(f"Starting from the weights in {self.init}")
        else:
            params = self.model.init(self.rng, self.dataset.get_sample_input())
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=self._make_optimizer(self.current_lr))

    def _create_train_step(self):
        """Create JIT-compiled training step function."""
        @jax.jit
        def train_step(state, batch):
            """Single training step."""
            inputs, targets = batch

            def loss_fn(params):
                output = state.apply_fn(params, inputs)
                # MSE on the log-normalised targets, scaled by 1000 for numerical range
                return jnp.mean((output - targets) ** 2) * 1000.0

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)

            return state, loss

        return train_step

    def _create_eval_step(self):
        """Create JIT-compiled evaluation step function."""
        @jax.jit
        def eval_step(state, batch):
            """Single evaluation step (the training loss)."""
            inputs, targets = batch
            output = state.apply_fn(state.params, inputs)
            return jnp.mean((output - targets) ** 2) * 1000.0

        return eval_step

    def _update_learning_rate(self, new_lr: float):
        """Swap in the optimiser at ``new_lr``; the optimiser state (Adam's moments) is kept."""
        self.current_lr = new_lr
        self.state = self.state.replace(tx=self._make_optimizer(new_lr))
        logger.info(f"Learning rate set to {new_lr:.2e}")

    def _check_patience_and_update_lr(self, val_loss: float):
        """Check patience and update learning rate if needed."""
        if not self.recipe.use_patience_scheduler:
            # Just track best loss for monitoring
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                logger.info(f"✅ Validation loss improved to {val_loss:.6f}")
            return

        # Check if validation loss improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            logger.info(f"✅ Validation loss improved to {val_loss:.6f} - patience reset")
        else:
            self.patience_counter += 1
            logger.info(f"⏳ No improvement for {self.patience_counter}/{self.recipe.patience} checks")

            # Check if we should reduce learning rate
            if self.patience_counter >= self.recipe.patience:
                new_lr = self.current_lr * self.recipe.lr_reduction_factor

                if new_lr >= self.recipe.min_lr:
                    self._update_learning_rate(new_lr)
                    self.patience_counter = 0
                    self.lr_reductions += 1
                    logger.info(f"📉 Reduced learning rate (reduction #{self.lr_reductions})")
                else:
                    logger.info(f"⚠️ Learning rate already at minimum ({self.recipe.min_lr:.2e})")

    def add_callback(self, callback: Callable):
        """Add a callback, called as ``callback(trainer, step)`` after every step."""
        self.callbacks.append(callback)

    def train(self) -> Dict[str, Any]:
        """Train for ``recipe.num_steps`` steps; returns the history."""
        total_steps = self.recipe.num_steps
        if self.state is None:
            self._init_training_state()
        train_step = self._create_train_step()
        eval_step = self._create_eval_step()

        logger.info(f"Starting training for {total_steps} steps...")
        start_time = time.time()

        for step in range(total_steps):
            self.rng, batch_rng, val_rng = jax.random.split(self.rng, 3)
            train_batch = self.dataset.get_batch(
                self.recipe.batch_size, rng=batch_rng, split='train', normalized=True)
            self.state, train_loss = train_step(self.state, train_batch)

            if step % self.log_every == 0:
                self.history['train_loss'].append(float(train_loss))
                self.history['step'].append(step)
                self.history['learning_rate'].append(float(self.current_lr))
                logger.info(f"Step {step:4d}/{total_steps}: Loss={train_loss:.6f}, LR={self.current_lr:.2e}")

            # Validation and patience checking
            if step % self.recipe.val_every == 0 and self.dataset.has_validation:
                val_batch = self.dataset.get_batch(
                    self.recipe.batch_size, rng=val_rng, split='val', normalized=True)
                val_loss = eval_step(self.state, val_batch)
                self.history['val_loss'].append(float(val_loss))
                logger.info(f"       Val Loss: {val_loss:.6f}")
                self._check_patience_and_update_lr(float(val_loss))

            if self.output_dir and self.checkpoint_every and step % self.checkpoint_every == 0:
                self.save_checkpoint(step)

            for callback in self.callbacks:
                callback(self, step)

        # Final callback pass at step == total_steps so the periodic hooks (prediction plots)
        # see the final weights: the loop's last step is total_steps - 1.
        for callback in self.callbacks:
            callback(self, total_steps)

        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
        return self.history

    def save_checkpoint(self, step: int):
        """``checkpoint_step_<step>.npz`` (a valid ``init``) and the history so far."""
        self.output_dir.mkdir(exist_ok=True, parents=True)
        write_weights(self.output_dir / f'checkpoint_step_{step}.npz', self.state.params)
        self.save_history(self.output_dir)
        logger.info(f"Saved checkpoint at step {step}")

    def save_history(self, output_dir: Path):
        """``training_history.json``: loss, learning rate and step."""
        (Path(output_dir) / 'training_history.json').write_text(json.dumps(self.history, indent=2))

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """The net's (log-normalised) output on normalised ``inputs``, shape ``(n, 1)``."""
        return np.asarray(self.state.apply_fn(self.state.params, jnp.asarray(inputs)))

    def plot_training_history(self, save_path: Optional[Path] = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(self.history['step'], self.history['train_loss'], label='Train')
        if self.history['val_loss']:
            val_steps = self.history['step'][::self.recipe.val_every // self.log_every]
            ax1.plot(val_steps[:len(self.history['val_loss'])],
                    self.history['val_loss'], label='Validation')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.set_title('Training Loss')

        # Learning rate plot
        ax2.plot(self.history['step'], self.history['learning_rate'])
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.set_title('Learning Rate Schedule')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved training plot to {save_path}")
        else:
            plt.show()

        return fig

    def emitter_record(self, t0: Optional[dict] = None) -> dict:
        """The ``emitter.json`` content of the trained emitter.

        ``t0`` is a ``calculate_t0.py`` fit (its ``t0.json``), recorded into a Cherenkov emitter
        for the simulator's emission time.
        """
        from lucid.utils import git_commit
        ds, r = self.dataset, self.recipe
        axis2 = 'angle_rad' if ds.kind == 'cherenkov' else 'dedx_kev_per_mm'
        (e_lo, e_hi), (a_lo, a_hi), (s_lo, s_hi) = ds.domain
        s_max_mm, n_photons = table_curves(ds.metadata, ds.kind)
        rec = {'kind': ds.kind,
               'net': {'hidden_features': r.hidden_features, 'hidden_layers': r.hidden_layers,
                       'w0': r.w0},
               'domain': {'energy_mev': [e_lo, e_hi], axis2: [a_lo, a_hi], 's_over_smax': [s_lo, s_hi]},
               'target': {'log10_min': float(ds.normalized_bounds['target_min']),
                          'log10_max': float(ds.normalized_bounds['target_max']),
                          'offset': r.zero_threshold},
               'grid_bins': GRID_BINS,
               's_max_mm': s_max_mm}
        if ds.kind == 'cherenkov':
            rec['n_photons'] = n_photons
            if t0 is not None:
                rec['t0'] = _t0_block(t0)
        meta = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in ds.metadata.items()}
        h = self.history
        rec['provenance'] = {
            'material': meta['material'], 'particle': meta['particle'], 'table': str(ds.data_path),
            'steps': r.num_steps,
            'losses': {'final_train': h['train_loss'][-1],
                       'final_val': h['val_loss'][-1] if h['val_loss'] else None,
                       'best_val': min(h['val_loss']) if h['val_loss'] else None},
            'init': str(self.init) if self.init else None,
            'commit': git_commit(),
            'trained_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
        rec['training'] = {'recipe': dataclasses.asdict(r)}
        return rec

    def save_emitter(self, output_dir: Path, t0: Optional[dict] = None):
        """Write the trained emitter (``emitter.json`` + ``weights.npz``) into ``output_dir``."""
        write_emitter(output_dir, self.emitter_record(t0), self.state.params)
        logger.info(f"Saved the emitter to {output_dir}")


def _t0_block(t0: dict) -> dict:
    """A ``calculate_t0.py`` fit (``t0.json``) as the emitter's ``t0`` block."""
    if t0['form'] != T0_FORM:
        raise ValueError(f"t0 fit of form {t0['form']!r}; expected {T0_FORM!r}")
    source = dict(t0.get('provenance', {}))
    if 'source' in source:
        source['tool'] = source.pop('source')
    return {'form': t0['form'], 'c_mm_per_ns': t0['c_mm_per_ns'],
            'log10_A': t0['A']['log10_poly_logE'], 'log10_lambda_mm': t0['lambda']['log10_poly_logE'],
            'beta': t0['beta']['poly_logE'], 'source': source}
