"""
JAX-based SIREN Trainer for PhotonSim Data

This module provides a reusable trainer class for SIREN networks on PhotonSim data,
designed to be easily used from Jupyter notebooks or scripts.
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Callable
import time
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from flax.core import freeze, unfreeze
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for SIREN training."""
    hidden_features: int = 256
    hidden_layers: int = 3
    w0: float = 30.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0  # Disabled - might have been interfering
    batch_size: int = 8192
    num_steps: int = 10000
    checkpoint_every: int = 1000
    log_every: int = 100
    val_every: int = 100
    seed: int = 42
    # Patience-based scheduler (currently disabled due to optimizer state reset issues)
    use_patience_scheduler: bool = False
    patience: int = 10  # Reduce LR after 10 evals with no improvement
    lr_reduction_factor: float = 0.5  # Multiply LR by this when reducing
    min_lr: float = 1e-7  # Minimum learning rate
    # Stability options
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd' - back to plain Adam
    grad_clip_norm: float = 0.0  # Gradient clipping disabled
    # Legacy options (only used if use_patience_scheduler = False)
    scheduler_step_size: int = 0  # Disabled by default
    scheduler_gamma: float = 0.5


class SIRENTrainer:
    """
    Reusable trainer class for SIREN networks on PhotonSim data.
    
    Example usage in notebook:
    ```python
    from diffCherenkov.siren.training import SIRENTrainer, TrainingConfig
    
    config = TrainingConfig(
        hidden_features=256,
        num_steps=5000,
        learning_rate=1e-4
    )
    
    trainer = SIRENTrainer(dataset, config)
    history = trainer.train()
    ```
    """
    
    def __init__(
        self,
        dataset,
        config: TrainingConfig,
        model_class=None,
        output_dir: Optional[Path] = None,
        resume_from_checkpoint: bool = True
    ):
        """
        Initialize SIREN trainer.
        
        Args:
            dataset: PhotonSim dataset object
            config: Training configuration
            model_class: Optional custom SIREN model class
            output_dir: Optional output directory for checkpoints
            resume_from_checkpoint: Whether to resume from existing checkpoint if available
        """
        self.dataset = dataset
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        self.resume_from_checkpoint = resume_from_checkpoint
        
        from tools.siren import SIREN
        model_class = siren_model
            
        # Initialize model
        self.model = model_class(
            hidden_features=config.hidden_features,
            hidden_layers=config.hidden_layers,
            out_features=1,
            w0=config.w0
        )
        
        # Check JAX devices
        self.device = self._check_devices()
        
        # Initialize state variables
        self.state = None
        self.rng = jax.random.PRNGKey(config.seed)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'step': []
        }
        self.start_step = 0
        
        # Patience scheduler state
        self.current_lr = config.learning_rate
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.lr_reductions = 0
        
        # Callbacks
        self.callbacks = []
        
        # Try to load existing checkpoint if resume is enabled
        if self.resume_from_checkpoint and self.output_dir and self.output_dir.exists():
            self._try_load_existing_checkpoint()
        
    def _check_devices(self):
        """Check and log available JAX devices."""
        devices = jax.devices()
        logger.info(f"JAX devices available: {len(devices)}")
        for i, device in enumerate(devices):
            logger.info(f"  Device {i}: {device.device_kind}")
        return devices[0]
        
    def _try_load_existing_checkpoint(self):
        """Try to load the most recent checkpoint if available."""
        if not self.output_dir or not self.output_dir.exists():
            return
            
        # Look for final model first, then latest checkpoint
        final_model_path = self.output_dir / 'final_model.npz'
        history_path = self.output_dir / 'training_history.json'
        config_path = self.output_dir / 'config.json'
        
        checkpoint_path = None
        
        if final_model_path.exists():
            checkpoint_path = final_model_path
            logger.info("Found final model checkpoint")
        else:
            # Look for latest numbered checkpoint
            checkpoint_files = list(self.output_dir.glob('checkpoint_step_*.npz'))
            if checkpoint_files:
                # Sort by step number and take the latest
                try:
                    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
                    checkpoint_path = checkpoint_files[-1]
                    step_num = int(checkpoint_path.stem.split('_')[-1])
                    logger.info(f"Found checkpoint at step {step_num}")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing checkpoint filenames: {e}")
                    checkpoint_path = None
        
        if checkpoint_path and checkpoint_path.exists():
            logger.info(f"Attempting to resume training from {checkpoint_path}")
            
            # Load training history first
            if history_path.exists():
                try:
                    import json
                    with open(history_path, 'r') as f:
                        self.history = json.load(f)
                        if self.history.get('step'):
                            self.start_step = max(self.history['step']) + 1
                            logger.info(f"Training history loaded, resuming from step {self.start_step}")
                            
                            # Restore the last learning rate from history
                            if self.history.get('learning_rate'):
                                last_lr = self.history['learning_rate'][-1]
                                self.current_lr = last_lr
                                logger.info(f"Restored learning rate: {last_lr:.2e}")
                            
                except Exception as e:
                    logger.warning(f"Failed to load training history: {e}")
            
            # Initialize model first (needed for loading weights)
            self._init_training_state()
            
            # Load the checkpoint weights
            self.load_checkpoint(checkpoint_path)
            
            if self.state is not None:
                logger.info(f"Successfully resumed training from checkpoint")
            else:
                logger.warning("Failed to load checkpoint, starting from scratch")
        else:
            logger.info("No existing checkpoint found, starting from scratch")
            
    def _init_training_state(self):
        """Initialize model parameters and optimizer state."""
        # Initialize model parameters
        sample_input = self.dataset.get_sample_input()
        logger.info(f"Initializing model with input shape: {sample_input.shape}")
        
        try:
            params = self.model.init(self.rng, sample_input)
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            # Try with a different shape
            sample_input = jnp.ones((1, 3), dtype=jnp.float32)
            logger.info(f"Retrying with default input shape: {sample_input.shape}")
            params = self.model.init(self.rng, sample_input)
        
        # CRITICAL: Force parameters to be concrete (not traced)
        # This prevents the traced array issue that causes checkpoint loading errors
        def make_concrete(tree):
            return jax.tree_map(lambda x: jax.block_until_ready(x) if hasattr(x, 'shape') else x, tree)
        
        params = make_concrete(params)
        logger.info("✅ Model parameters forced to concrete arrays")
        
        # Create optimizer with fixed learning rate
        # The patience LR updates were causing issues by resetting optimizer state
        if self.config.optimizer.lower() == 'adamw':
            base_optimizer = optax.adamw(learning_rate=self.current_lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == 'adam':
            base_optimizer = optax.adam(learning_rate=self.current_lr)
        elif self.config.optimizer.lower() == 'sgd':
            base_optimizer = optax.sgd(learning_rate=self.current_lr, momentum=0.9)
        else:
            base_optimizer = optax.adamw(learning_rate=self.current_lr, weight_decay=self.config.weight_decay)
        
        # Add gradient clipping and weight decay (if not using AdamW)
        optimizer_components = []
        
        if self.config.grad_clip_norm > 0:
            optimizer_components.append(optax.clip_by_global_norm(self.config.grad_clip_norm))
            
        optimizer_components.append(base_optimizer)
        
        if self.config.optimizer.lower() != 'adamw' and self.config.weight_decay > 0:
            optimizer_components.append(optax.add_decayed_weights(self.config.weight_decay))
            
        optimizer = optax.chain(*optimizer_components)
            
        # Create training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )
        
    def _create_train_step(self):
        """Create JIT-compiled training step function."""
        @jax.jit
        def train_step(state, batch):
            """Single training step."""
            inputs, targets = batch
            
            def loss_fn(params):
                predictions = state.apply_fn(params, inputs)
                
                # Handle SIREN output format (tuple with output and coords)
                if isinstance(predictions, (tuple, list)):
                    output = predictions[0]  # Take first element like CProfSiren
                else:
                    output = predictions
                
                # Ensure proper shape
                if output.ndim == 1:
                    output = output[:, None]
                    
                # CProfSiren-style MSE loss with 1000x scaling for better numerical stability
                loss = jnp.mean((output - targets) ** 2) * 1000.0
                return loss
                
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            
            return state, loss
            
        return train_step
        
    def _create_eval_step(self):
        """Create JIT-compiled evaluation step function."""
        @jax.jit
        def eval_step(state, batch):
            """Single evaluation step."""
            inputs, targets = batch
            predictions = state.apply_fn(state.params, inputs)
            
            # Handle SIREN output format (tuple with output and coords)
            if isinstance(predictions, (tuple, list)):
                output = predictions[0]  # Take first element like CProfSiren
            else:
                output = predictions
            
            # Ensure proper shape
            if output.ndim == 1:
                output = output[:, None]
                
            # CProfSiren-style MSE loss with 1000x scaling (same as training)
            loss = jnp.mean((output - targets) ** 2) * 1000.0
            return loss
            
        return eval_step
        
    def _update_learning_rate(self, new_lr: float):
        """Update the learning rate of the optimizer."""
        if not self.config.use_patience_scheduler:
            logger.info(f"LR update skipped - patience scheduler disabled")
            return
            
        # Update learning rate using optax's inject_hyperparams
        # This preserves optimizer state (momentum, etc.)
        self.current_lr = new_lr
        
        # Create new optimizer with updated LR while preserving state
        if self.config.optimizer.lower() == 'adamw':
            base_optimizer = optax.adamw(learning_rate=new_lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == 'adam':
            base_optimizer = optax.adam(learning_rate=new_lr)
        elif self.config.optimizer.lower() == 'sgd':
            base_optimizer = optax.sgd(learning_rate=new_lr, momentum=0.9)
        else:
            base_optimizer = optax.adam(learning_rate=new_lr)
        
        # Rebuild optimizer chain with new learning rate
        optimizer_components = []
        if self.config.grad_clip_norm > 0:
            optimizer_components.append(optax.clip_by_global_norm(self.config.grad_clip_norm))
        optimizer_components.append(base_optimizer)
        if self.config.optimizer.lower() != 'adamw' and self.config.weight_decay > 0:
            optimizer_components.append(optax.add_decayed_weights(self.config.weight_decay))
            
        new_optimizer = optax.chain(*optimizer_components)
        
        # Create new state with updated optimizer but SAME parameters and opt_state
        # This preserves momentum and other optimizer statistics
        import copy
        old_opt_state = copy.deepcopy(self.state.opt_state)
        
        # Update only the learning rate in the optimizer state
        self.state = self.state.replace(tx=new_optimizer)
        
        logger.info(f"✅ Learning rate updated: {self.current_lr:.2e} → {new_lr:.2e}")
        
    def _check_patience_and_update_lr(self, val_loss: float):
        """Check patience and update learning rate if needed."""
        if not self.config.use_patience_scheduler:
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
            logger.info(f"⏳ No improvement for {self.patience_counter}/{self.config.patience} checks")
            
            # Check if we should reduce learning rate
            if self.patience_counter >= self.config.patience:
                new_lr = self.current_lr * self.config.lr_reduction_factor
                
                if new_lr >= self.config.min_lr:
                    self._update_learning_rate(new_lr)
                    self.patience_counter = 0
                    self.lr_reductions += 1
                    logger.info(f"📉 Reduced learning rate (reduction #{self.lr_reductions})")
                else:
                    logger.info(f"⚠️ Learning rate already at minimum ({self.config.min_lr:.2e})")
        
    def add_callback(self, callback: Callable):
        """Add a callback function to be called during training."""
        self.callbacks.append(callback)
        
    def train(self, num_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the SIREN model.
        
        Args:
            num_steps: Optional override for number of training steps
            
        Returns:
            Dictionary containing training history
        """
        num_steps = num_steps or self.config.num_steps
        
        # Initialize training state if not already done
        if self.state is None:
            self._init_training_state()
            
        # Create training functions
        train_step = self._create_train_step()
        eval_step = self._create_eval_step()
        
        # Training loop
        total_steps = num_steps
        remaining_steps = total_steps - self.start_step
        logger.info(f"Starting training from step {self.start_step} for {remaining_steps} more steps (total: {total_steps})...")
        start_time = time.time()
        
        for step in range(self.start_step, total_steps):
            # Get training batch
            self.rng, batch_rng = jax.random.split(self.rng)
            train_batch = self.dataset.get_batch(
                self.config.batch_size, 
                rng=batch_rng,
                split='train',
                normalized=True  # Use normalized inputs and log targets
            )
            
            # Training step
            self.state, train_loss = train_step(self.state, train_batch)
            
            # Logging
            if step % self.config.log_every == 0:
                self.history['train_loss'].append(float(train_loss))
                self.history['step'].append(step)
                self.history['learning_rate'].append(float(self.current_lr))
                
                # DEBUG: Print target and prediction statistics
                inputs, targets = train_batch
                predictions = self.state.apply_fn(self.state.params, inputs)
                if isinstance(predictions, (tuple, list)):
                    predictions = predictions[0]
                predictions = jnp.atleast_2d(predictions)
                if predictions.ndim == 1:
                    predictions = predictions[:, None]
                
                logger.info(f"Step {step:4d}/{total_steps}: Loss={train_loss:.6f}, LR={self.current_lr:.2e}")
                logger.info(f"       Targets: min={targets.min():.6f}, max={targets.max():.6f}, mean={targets.mean():.6f}")
                logger.info(f"       Predictions: min={predictions.min():.6f}, max={predictions.max():.6f}, mean={predictions.mean():.6f}")
                
            # Validation and patience checking
            if step % self.config.val_every == 0 and hasattr(self.dataset, 'has_validation'):
                val_batch = self.dataset.get_batch(
                    self.config.batch_size,
                    rng=batch_rng,
                    split='val',
                    normalized=True  # Use normalized inputs and log targets
                )
                val_loss = eval_step(self.state, val_batch)
                self.history['val_loss'].append(float(val_loss))
                logger.info(f"       Val Loss: {val_loss:.6f}")
                
                # Check patience and update learning rate if needed
                self._check_patience_and_update_lr(float(val_loss))
                
            # Checkpointing
            if self.output_dir and step % self.config.checkpoint_every == 0:
                self.save_checkpoint(step)
                
            # Run callbacks
            for callback in self.callbacks:
                callback(self, step)
                
        # Final checkpoint
        if self.output_dir:
            self.save_checkpoint(total_steps, final=True)
            
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time:.2f} seconds")
        
        return self.history
        
    def save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint."""
        if not self.output_dir:
            return
            
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model parameters using pickle for Flax compatibility
        checkpoint_name = 'final_model.npz' if final else f'checkpoint_step_{step}.npz'
        checkpoint_path = self.output_dir / checkpoint_name
        
        # CRITICAL: Force evaluation of traced arrays by blocking until concrete
        # This ensures we don't save traced arrays from JAX transformations
        import jax
        
        # Get parameters and force them to be concrete (not traced)
        params_dict = unfreeze(self.state.params)
        
        # Convert JAX arrays to numpy arrays recursively and force evaluation
        def jax_to_numpy_concrete(tree):
            def convert_leaf(x):
                if hasattr(x, 'shape'):
                    # Force evaluation by using jax.block_until_ready
                    concrete_x = jax.block_until_ready(x)
                    return np.asarray(concrete_x)
                else:
                    return x
            return jax.tree_map(convert_leaf, tree)
        
        params_numpy = jax_to_numpy_concrete(params_dict)
        
        # Double-check: ensure no traced arrays remain
        def check_no_traced(tree):
            def check_leaf(x):
                if hasattr(x, '__class__') and 'Traced' in str(type(x)):
                    raise ValueError(f"Still have traced array: {type(x)}")
                return x
            return jax.tree_map(check_leaf, tree, is_leaf=lambda x: hasattr(x, '__class__'))
        
        try:
            check_no_traced(params_numpy)
        except ValueError as e:
            logger.error(f"ERROR: Traced arrays detected during save: {e}")
            logger.error("Attempting to force concrete evaluation...")
            # Fallback: try to force evaluation again
            params_numpy = jax.tree_map(
                lambda x: np.array(jax.block_until_ready(x)) if hasattr(x, 'shape') else x, 
                params_dict
            )
        
        np.savez(checkpoint_path, params=params_numpy)
            
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
            
        # Save config
        if step == 0 or final:
            config_path = self.output_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
                
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        try:
            # Try loading with pickle support for Flax parameters
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = np.load(f, allow_pickle=True)
                
                if 'params' in checkpoint_data:
                    # New format with single 'params' key
                    params_dict = checkpoint_data['params'].item()
                else:
                    # Old format with multiple keys (backward compatibility)
                    params_dict = {k: checkpoint_data[k] for k in checkpoint_data.files}
            
            # Convert numpy arrays back to JAX arrays and freeze
            def numpy_to_jax(tree):
                return jax.tree_map(lambda x: jnp.asarray(x) if hasattr(x, 'shape') else x, tree)
            
            params_jax = numpy_to_jax(params_dict)
            loaded_params = freeze(params_jax)
            
            # CRITICAL FIX: Reinitialize the entire training state with loaded parameters
            # This ensures the optimizer state matches the parameter tree structure
            
            # Get the current optimizer
            current_optimizer = self.state.tx
            
            # Create new training state with loaded parameters and fresh optimizer state
            self.state = train_state.TrainState.create(
                apply_fn=self.model.apply,
                params=loaded_params,
                tx=current_optimizer
            )
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            logger.info("✅ Training state rebuilt with fresh optimizer state")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            logger.info("Will continue training from scratch")
            # Reset start_step if loading fails
            self.start_step = 0
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'learning_rate': [],
                'step': []
            }
        
    def clear_checkpoints(self):
        """Clear all existing checkpoints and training history."""
        if not self.output_dir or not self.output_dir.exists():
            return
            
        # Remove all checkpoint files
        checkpoint_files = list(self.output_dir.glob('*.npz'))
        for checkpoint_file in checkpoint_files:
            checkpoint_file.unlink()
            logger.info(f"Removed checkpoint: {checkpoint_file.name}")
            
        # Remove training history and config
        for filename in ['training_history.json', 'config.json', 'monitoring_data.json', 'analysis_results.json']:
            file_path = self.output_dir / filename
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Removed: {filename}")
                
        # Remove plot files
        plot_files = list(self.output_dir.glob('*.png'))
        for plot_file in plot_files:
            plot_file.unlink()
            logger.info(f"Removed plot: {plot_file.name}")
            
        # Reset training state
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'step': []
        }
        self.start_step = 0
        self.state = None
        
        logger.info("All checkpoints and training data cleared")
        
    def evaluate(self, num_batches: int = 100) -> Dict[str, float]:
        """
        Evaluate model on validation/test data.
        
        Args:
            num_batches: Number of batches to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        eval_step = self._create_eval_step()
        total_loss = 0.0
        
        for i in range(num_batches):
            self.rng, batch_rng = jax.random.split(self.rng)
            val_batch = self.dataset.get_batch(
                self.config.batch_size,
                rng=batch_rng,
                split='val',
                normalized=True  # Use normalized inputs and log targets
            )
            loss = eval_step(self.state, val_batch)
            total_loss += float(loss)
            
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
        
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            inputs: Input array of shape (n_samples, input_dim)
            
        Returns:
            Predictions array of shape (n_samples, 1)
        """
        inputs = jnp.array(inputs)
        predictions = self.state.apply_fn(self.state.params, inputs)
        
        # Handle case where model returns tuple or other structure
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]  # Take first element
        elif hasattr(predictions, '__getitem__') and not hasattr(predictions, 'shape'):
            predictions = predictions[0]  # Handle other sequence types
        
        # Ensure predictions are properly shaped and converted
        predictions = jnp.atleast_2d(predictions)
        if predictions.ndim == 1:
            predictions = predictions[:, None]
            
        return np.asarray(predictions)
        
    def plot_training_history(self, save_path: Optional[Path] = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.history['step'], self.history['train_loss'], label='Train')
        if self.history['val_loss']:
            val_steps = self.history['step'][::self.config.val_every // self.config.log_every]
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
    
    def save_trained_model(self, output_dir: Path, model_name: str = "siren_model"):
        """
        Save the trained SIREN model with all necessary metadata for inference.
        
        This saves:
        - Model weights (.npz file)
        - Model metadata including normalization parameters (.json file)
        
        Args:
            output_dir: Directory to save the model
            model_name: Base name for the model files (without extension)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Helper function to ensure JSON serializable values
        def make_json_serializable(obj):
            """Convert numpy/JAX arrays and types to JSON serializable Python types."""
            if hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif hasattr(obj, '__float__'):  # JAX arrays/scalars
                return float(obj)
            else:
                return obj

        # Prepare model metadata with proper type conversion
        metadata = {
            'model_config': {
                'hidden_features': int(self.config.hidden_features),
                'hidden_layers': int(self.config.hidden_layers),
                'out_features': 1,
                'w0': float(self.config.w0)
            },
            'input_normalization': {
                'scheme': 'linear_to_range',
                'target_range': [-1, 1],
                'input_min': make_json_serializable(self.dataset.normalized_bounds['input_min']),
                'input_max': make_json_serializable(self.dataset.normalized_bounds['input_max'])
            },
            'dataset_info': {
                'data_type': getattr(self.dataset, 'data_type', 'unknown'),
                'energy_range': [float(self.dataset.energy_range[0]), float(self.dataset.energy_range[1])],
                'angle_range': [float(self.dataset.angle_range[0]), float(self.dataset.angle_range[1])],
                'distance_range': [float(self.dataset.distance_range[0]), float(self.dataset.distance_range[1])],
                'units': {
                    'energy': 'MeV',
                    'angle': 'radians',
                    'distance': 'mm',
                    'photon_density': 'photons/mm^2'
                }
            },
            'training_info': {
                'final_step': int(self.state.step) if self.state else 0,
                'final_train_loss': float(self.history['train_loss'][-1]) if self.history['train_loss'] else None,
                'final_val_loss': float(self.history['val_loss'][-1]) if self.history['val_loss'] else None,
                'trained_on_log_scale': True,  # Now using consistent log-normalized targets by default
                'loss_scaling_factor': 1000.0  # CProfSiren-style loss scaling
            }
        }
        
        # If the dataset has denormalization info for targets, include it
        if hasattr(self.dataset, 'normalized_bounds'):
            if 'target_min' in self.dataset.normalized_bounds:
                metadata['target_normalization'] = {
                    'scheme': 'log_normalized_to_01',
                    'log_min': make_json_serializable(self.dataset.normalized_bounds['target_min']),
                    'log_max': make_json_serializable(self.dataset.normalized_bounds['target_max'])
                }
            # Check for linear target normalization (from LinearPhotonSimDataset)
            if 'linear_target_min' in self.dataset.normalized_bounds:
                metadata['target_normalization'] = {
                    'scheme': 'linear_normalized_to_01',
                    'linear_min': make_json_serializable(self.dataset.normalized_bounds['linear_target_min']),
                    'linear_max': make_json_serializable(self.dataset.normalized_bounds['linear_target_max'])
                }
        
        # Save metadata as JSON
        metadata_path = output_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save model weights - flatten structure for reliable saving
        weights_path = output_dir / f"{model_name}_weights.npz"
        
        # Flatten the parameter tree to avoid nested structure issues
        def flatten_params(params):
            """Flatten nested parameter dict with descriptive keys."""
            flat_params = {}
            
            def _flatten(tree, prefix=""):
                for key, value in tree.items():
                    new_key = f"{prefix}_{key}" if prefix else key
                    if isinstance(value, dict):
                        _flatten(value, new_key)
                    else:
                        # Convert to numpy array
                        flat_params[new_key] = np.array(value)
            
            _flatten(params)
            return flat_params
        
        # Get parameters and flatten them
        params_dict = unfreeze(self.state.params)
        flat_params = flatten_params(params_dict)
        
        # Save flattened parameters
        np.savez(weights_path, **flat_params)
        
        logger.info(f"Saved trained model to:")
        logger.info(f"  Weights: {weights_path}")
        logger.info(f"  Metadata: {metadata_path}")
        
        return weights_path, metadata_path