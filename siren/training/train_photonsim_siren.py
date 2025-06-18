"""
JAX-based SIREN Training for PhotonSim Data

This module provides the trainer class for SIREN networks on PhotonSim data.
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import time

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from flax.core import freeze, unfreeze
import matplotlib.pyplot as plt

from siren import SIREN, SineLayer

# Set up logging
logger = logging.getLogger(__name__)

# Print JAX device info
def print_device_info():
    """Print information about JAX devices."""
    devices = jax.devices()
    logger.info(f"JAX devices available: {len(devices)}")
    for i, device in enumerate(devices):
        logger.info(f"  Device {i}: {device.device_kind} - {device}")
    
    # Check which device will be used
    default_device = jax.devices()[0]
    logger.info(f"Default device: {default_device.device_kind} - {default_device}")
    
    # Test a simple computation to verify device usage
    test_array = jnp.array([1.0, 2.0, 3.0])
    logger.info(f"Test array device: {test_array.device()}")
    return default_device

class PhotonSimSIRENTrainer:
    """
    Trainer class for SIREN networks on PhotonSim data.
    """
    
    def __init__(
        self,
        dataset,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        w0: float = 30.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        scheduler_step_size: int = 2000,
        scheduler_gamma: float = 0.1,
    ):
        """
        Initialize SIREN trainer.
        
        Args:
            dataset: PhotonSim dataset (either PhotonSimDataset or PhotonSimSampledDataset)
            hidden_features: Number of hidden units
            hidden_layers: Number of hidden layers
            w0: Frequency parameter for SIREN
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            scheduler_step_size: Steps for learning rate scheduler
            scheduler_gamma: Multiplicative factor for learning rate decay
        """
        self.dataset = dataset
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.w0 = w0
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        
        # Initialize model
        self.model = SIREN(
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=1,
            w0=w0
        )
        
        # Print device information
        self.device = print_device_info()
        
        # Initialize training state
        self._init_training_state()
        
        # Create JIT-compiled functions
        self._init_jit_functions()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.step_times = []
    
    def _init_training_state(self):
        """Initialize training state with model and optimizer."""
        # Initialize model parameters
        key = jax.random.PRNGKey(42)
        sample_input = jnp.ones((1, 3))  # [energy, angle, distance]
        
        variables = self.model.init(key, sample_input)
        params = variables['params']
        
        # Create optimizer with learning rate schedule
        schedule = optax.exponential_decay(
            init_value=self.learning_rate,
            transition_steps=self.scheduler_step_size,
            decay_rate=self.scheduler_gamma
        )
        
        optimizer = optax.adamw(
            learning_rate=schedule,
            weight_decay=self.weight_decay
        )
        
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )
        
        logger.info(f"Initialized model with {self._count_parameters():,} parameters")
    
    def _init_jit_functions(self):
        """Initialize JIT-compiled functions."""
        @jax.jit
        def train_step_jit(state, inputs, targets):
            def loss_fn(params):
                predictions = self.model.apply({'params': params}, inputs)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                predictions = predictions.squeeze()
                mse_loss = jnp.mean((predictions - targets) ** 2)
                return 1000.0 * mse_loss
            
            def loss_and_grad(params):
                loss = loss_fn(params)
                return loss, loss
            
            grad_fn = jax.value_and_grad(loss_and_grad, has_aux=True)
            (loss, _), grads = grad_fn(state.params)
            
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        @jax.jit
        def eval_step_jit(params, inputs, targets):
            predictions = self.model.apply({'params': params}, inputs)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            predictions = predictions.squeeze()
            mse_loss = jnp.mean((predictions - targets) ** 2)
            return 1000.0 * mse_loss
        
        self.train_step_jit = train_step_jit
        self.eval_step_jit = eval_step_jit
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        def count_params(params):
            return sum(x.size for x in jax.tree.leaves(params))
        return count_params(self.state.params)
    
    
    def train(
        self,
        num_steps: int = 2000,
        log_every: int = 100,
        eval_every: int = 200,
        save_every: int = 500,
        output_dir: str = "output/siren_training",
    ):
        """
        Train the SIREN model.
        
        Args:
            num_steps: Number of training steps
            log_every: Log training progress every N steps
            eval_every: Evaluate on validation set every N steps
            save_every: Save checkpoint every N steps
            output_dir: Directory to save outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Get validation data
        val_inputs, val_targets = self.dataset.get_validation_data()
        
        logger.info(f"Starting training for {num_steps} steps...")
        logger.info(f"Validation samples: {len(val_targets):,}")
        
        start_time = time.time()
        
        for step in range(num_steps):
            step_start_time = time.time()
            
            # Get training batch
            batch_iter = self.dataset.get_batch_iterator(shuffle=True)
            train_inputs, train_targets = next(batch_iter)
            
            # Training step
            self.state, train_loss = self.train_step_jit(
                self.state, train_inputs, train_targets
            )
            
            step_time = time.time() - step_start_time
            self.step_times.append(step_time)
            self.train_losses.append(float(train_loss))
            
            # Logging
            if step % log_every == 0:
                try:
                    lr = self.state.opt_state[1].hyperparams['learning_rate']
                    lr_str = f", lr = {lr:.2e}"
                except:
                    lr_str = ""
                logger.info(
                    f"Step {step:4d}: train_loss = {train_loss:.6f}{lr_str}, time = {step_time:.3f}s"
                )
            
            # Evaluation
            if step % eval_every == 0:
                val_loss = self.eval_step_jit(self.state.params, val_inputs, val_targets)
                self.val_losses.append(float(val_loss))
                logger.info(f"Step {step:4d}: val_loss = {val_loss:.6f}")
                
                # Plot progress
                if step > 0:
                    self._plot_training_progress(output_path / f"training_progress_step_{step}.png")
            
            # Save checkpoint
            if step % save_every == 0:
                self._save_checkpoint(output_path / f"checkpoint_step_{step}.npz")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s ({total_time/num_steps:.3f}s/step)")
        
        # Final evaluation and save
        final_val_loss = self.eval_step_jit(self.state.params, val_inputs, val_targets)
        logger.info(f"Final validation loss: {final_val_loss:.6f}")
        
        # Save final model
        self._save_model(output_path / "final_model.npz")
        self._save_training_config(output_path / "config.json")
        self._plot_training_progress(output_path / "final_training_progress.png")
        
        logger.info(f"Training outputs saved to {output_path}")
    
    def _save_checkpoint(self, path: Path):
        """Save training checkpoint."""
        checkpoint = {
            'params': self.state.params,
            'step': self.state.step,
            'train_losses': np.array(self.train_losses),
            'val_losses': np.array(self.val_losses),
            'step_times': np.array(self.step_times),
        }
        
        # Convert JAX arrays to numpy for saving
        checkpoint_np = jax.tree.map(lambda x: np.array(x) if hasattr(x, 'shape') else x, checkpoint)
        
        np.savez(path, **checkpoint_np)
        logger.info(f"Saved checkpoint to {path}")
    
    def _save_model(self, path: Path):
        """Save final trained model with metadata."""
        model_data = {
            'params': self.state.params,
            'model_config': {
                'hidden_features': self.hidden_features,
                'hidden_layers': self.hidden_layers,
                'w0': self.w0,
                'out_features': 1
            },
            'normalization_params': getattr(self.dataset, 'normalization_params', {}),
            'dataset_stats': self.dataset.get_stats(),
            'final_step': self.state.step,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
        }
        
        # Convert JAX arrays to numpy
        model_data_np = jax.tree.map(lambda x: np.array(x) if hasattr(x, 'shape') else x, model_data)
        
        np.savez(path, **model_data_np)
        logger.info(f"Saved final model to {path}")
    
    def _save_training_config(self, path: Path):
        """Save training configuration."""
        dataset_config = {}
        if hasattr(self.dataset, 'table_path'):
            dataset_config['table_path'] = str(self.dataset.table_path)
        if hasattr(self.dataset, 'dataset_path'):
            dataset_config['dataset_path'] = str(self.dataset.dataset_path)
        if hasattr(self.dataset, 'batch_size'):
            dataset_config['batch_size'] = self.dataset.batch_size
        
        config = {
            'hidden_features': self.hidden_features,
            'hidden_layers': self.hidden_layers,
            'w0': self.w0,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'scheduler_step_size': self.scheduler_step_size,
            'scheduler_gamma': self.scheduler_gamma,
            'dataset_config': dataset_config,
            'n_parameters': self._count_parameters(),
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved training config to {path}")
    
    def _plot_training_progress(self, path: Path):
        """Plot and save training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training loss
        if self.train_losses:
            axes[0, 0].plot(self.train_losses)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Validation loss
        if self.val_losses:
            val_steps = np.arange(0, len(self.train_losses), len(self.train_losses) // len(self.val_losses))[:len(self.val_losses)]
            axes[0, 1].plot(val_steps, self.val_losses, 'orange')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Step times
        if self.step_times:
            axes[1, 0].plot(self.step_times)
            axes[1, 0].set_title('Step Times')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Time (s)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Combined losses
        if self.train_losses and self.val_losses:
            axes[1, 1].plot(self.train_losses, label='Train', alpha=0.7)
            val_steps = np.arange(0, len(self.train_losses), len(self.train_losses) // len(self.val_losses))[:len(self.val_losses)]
            axes[1, 1].plot(val_steps, self.val_losses, 'orange', label='Validation')
            axes[1, 1].set_title('Training Progress')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def evaluate_model(self, n_samples: int = 10000) -> Dict[str, float]:
        """
        Evaluate the trained model on a sample of data.
        
        Args:
            n_samples: Number of samples to evaluate
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Get sample data
        inputs, targets = self.dataset.get_sample_batch(n_samples)
        
        # Make predictions
        predictions = self.model.apply({'params': self.state.params}, inputs)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = predictions.squeeze()
        
        # Convert back to numpy for metrics calculation
        predictions_np = np.array(predictions)
        targets_np = np.array(targets)
        
        # Calculate metrics
        mse = np.mean((predictions_np - targets_np) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions_np - targets_np))
        
        # R-squared
        ss_res = np.sum((targets_np - predictions_np) ** 2)
        ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Relative error
        relative_error = np.mean(np.abs(predictions_np - targets_np) / (targets_np + 1e-8))
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'relative_error': float(relative_error),
            'n_samples': n_samples
        }
        
        logger.info("Model evaluation metrics:")
        for key, value in metrics.items():
            if key != 'n_samples':
                logger.info(f"  {key}: {value:.6f}")
        
        return metrics


def load_trained_model(model_path: str) -> Tuple[SIREN, Dict, Dict]:
    """
    Load a trained SIREN model.
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        Tuple of (model, params, metadata)
    """
    data = np.load(model_path, allow_pickle=True)
    
    # Extract model configuration
    model_config = data['model_config'].item()
    
    # Create model
    model = SIREN(**model_config)
    
    # Extract parameters (convert back to JAX format)
    params = jax.tree.map(lambda x: jnp.array(x), data['params'].item())
    
    # Extract metadata
    metadata = {
        'normalization_params': data['normalization_params'].item(),
        'dataset_stats': data['dataset_stats'].item(),
        'final_step': int(data['final_step']),
        'final_train_loss': float(data['final_train_loss']) if data['final_train_loss'] is not None else None,
        'final_val_loss': float(data['final_val_loss']) if data['final_val_loss'] is not None else None,
    }
    
    return model, params, metadata