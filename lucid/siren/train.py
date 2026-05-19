#!/usr/bin/env python3
"""
PhotonSim SIREN Training Script

Train SIREN models for photon simulation using lookup tables from PhotonSim.
Supports different particle types and materials with automatic path detection.

Usage:
    python train.py [OPTIONS]
    
    # Train with default settings (water/muon)
    python train.py
    
    # Train for specific material/particle
    python train.py --material water --particle muon
    python train.py --material ice --particle electron
    
    # Custom settings
    python train.py --batch-size 32768 --num-steps 50000
    python train.py --learning-rate 5e-5 --hidden-layers 4
    
    # Resume training
    python train.py --resume
"""

import sys
import os
from pathlib import Path
import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import training modules
from lucid.siren.training.trainer import SIRENTrainer, TrainingConfig
from lucid.siren.training.dataset import PhotonSimDataset
from lucid.siren.training.monitor import TrainingMonitor, LiveTrainingCallback
from lucid.siren.training.analyzer import TrainingAnalyzer
from lucid.utils import base_dir_path, setup_matplotlib_for_notebook


class PhotonSimTrainer:
    """Main training class for PhotonSim SIREN models."""

    def __init__(self, material='water', particle='muon', resume=False,
                 data_type='photon', h5_path_override=None,
                 output_dir_override=None):
        """Initialize trainer with material, particle type, and data type.

        ``h5_path_override`` lets callers point at an arbitrary .h5 file
        without renaming directories. ``output_dir_override`` redirects every
        artifact the trainer writes (checkpoints, prediction plots, trained
        model, history) to a caller-chosen folder — used by the batch-scan
        tool to land each hyperparameter combo in its own sub-folder.
        """
        self.material = material
        self.particle = particle
        self.resume = resume
        self.data_type = data_type

        # Set up paths
        self.base_dir = Path(base_dir_path())
        self.data_dir = self.base_dir / 'data' / material / particle

        # Training output and h5 file depend on data type
        if data_type == 'dedx':
            self.h5_path = self.data_dir / 'dedx_lookup_table.h5'
            self.output_dir = self.data_dir / 'dedx_siren_training'
            self.model_name = 'dedx_siren'
        else:
            self.h5_path = self.data_dir / 'photon_lookup_table.h5'
            self.output_dir = self.data_dir / 'siren_training'
            self.model_name = 'photonsim_siren'

        if h5_path_override is not None:
            self.h5_path = Path(h5_path_override)
        if output_dir_override is not None:
            self.output_dir = Path(output_dir_override)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🎯 PhotonSim SIREN Training")
        print(f"  Material: {material}")
        print(f"  Particle: {particle}")
        print(f"  Data type: {data_type}")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Output directory: {self.output_dir}")

        # Check for h5 file
        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"HDF5 lookup table not found at {self.h5_path}\n"
                f"Please ensure the {data_type} table exists for {material}/{particle}"
            )

        print(f"✓ Found HDF5 file: {self.h5_path}")
        print(f"✓ JAX devices: {jax.devices()}")
    
    def setup_dataset(self, val_split=0.1, zero_threshold=1e-2,
                      zero_keep_frac=0.002, energy_balance='none',
                      target_importance=0.0):
        """Load and configure the dataset."""
        print("\n📊 Loading dataset...")
        self.dataset = PhotonSimDataset(
            self.h5_path, val_split=val_split,
            zero_threshold=zero_threshold,
            zero_keep_frac=zero_keep_frac,
            energy_balance=energy_balance,
            target_importance=target_importance,
        )

        print(f"\nDataset info:")
        print(f"  Data type: {self.dataset.data_type}")
        print(f"  Table type: {getattr(self.dataset, 'table_type', 'unknown')}")
        print(f"  Total samples: {len(self.dataset.data['inputs']):,}")
        print(f"  Train samples: {len(self.dataset.train_indices):,}")
        print(f"  Val samples: {len(self.dataset.val_indices):,}")
        print(f"  Energy range: {self.dataset.energy_range[0]:.0f}-{self.dataset.energy_range[1]:.0f} MeV")
        if self.dataset.angle_range is not None:
            print(f"  Angle range: {np.degrees(self.dataset.angle_range[0]):.1f}-{np.degrees(self.dataset.angle_range[1]):.1f} degrees")
        if getattr(self.dataset, 'dedx_range', None) is not None:
            print(f"  dE/dx range: {self.dataset.dedx_range[0]:.1f}-{self.dataset.dedx_range[1]:.1f} keV/mm")
        print(f"  Distance range: {self.dataset.distance_range[0]:.0f}-{self.dataset.distance_range[1]:.0f} mm")
        
        # Verify normalization
        rng = jax.random.PRNGKey(42)
        sample_inputs, sample_targets = self.dataset.get_batch(100, rng, 'train', normalized=True)
        
        print(f"\n🧪 Dataset verification:")
        print(f"  Sample inputs shape: {sample_inputs.shape}")
        print(f"  Sample targets shape: {sample_targets.shape}")
        print(f"  Input range: [{sample_inputs.min():.3f}, {sample_inputs.max():.3f}]")
        print(f"  Target range: [{sample_targets.min():.3f}, {sample_targets.max():.3f}]")
        print(f"  ✅ Normalization working correctly")
        
        return self.dataset
    
    def setup_training(self, config):
        """Initialize trainer with configuration."""
        print("\n🚀 Initializing trainer...")
        
        # Check for existing training
        existing_checkpoints = list(self.output_dir.glob('*.npz'))
        existing_history = self.output_dir / 'training_history.json'
        
        if existing_checkpoints or existing_history.exists():
            print(f"\n🔍 Found existing training data:")
            if existing_history.exists():
                with open(existing_history, 'r') as f:
                    history = json.load(f)
                    if history.get('step'):
                        last_step = max(history['step'])
                        print(f"  - Training history up to step {last_step}")
            
            for checkpoint in existing_checkpoints[:3]:  # Show first 3
                print(f"  - Checkpoint: {checkpoint.name}")
            if len(existing_checkpoints) > 3:
                print(f"  - ... and {len(existing_checkpoints) - 3} more checkpoints")
            
            if self.resume:
                print(f"\n▶️  Resuming from latest checkpoint")
            else:
                print(f"\n🔄 Starting fresh (use --resume to continue)")
        else:
            print(f"\n✨ No existing training data found. Starting fresh.")
        
        # Initialize trainer
        self.trainer = SIRENTrainer(
            dataset=self.dataset,
            config=config,
            output_dir=self.output_dir,
            resume_from_checkpoint=self.resume
        )
        
        # Clear checkpoints if not resuming
        if not self.resume and (existing_checkpoints or existing_history.exists()):
            print("🧹 Clearing existing checkpoints...")
            self.trainer.clear_checkpoints()
            print("✅ Starting with clean slate")
        
        print(f"✓ Trainer initialized")
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ JAX device: {self.trainer.device}")
        
        if self.trainer.start_step > 0:
            print(f"✓ Resuming from step {self.trainer.start_step}")
            print(f"✓ Training history loaded with {len(self.trainer.history['train_loss'])} entries")
        else:
            print(f"✓ Starting fresh training from step 0")
        
        return self.trainer
    
    def train(self, config, enable_monitoring=True,
              prediction_plot_opts=None, val_split=0.01,
              zero_threshold=1e-2, zero_keep_frac=0.002,
              energy_balance='none', target_importance=0.0):
        """Run the training process."""
        # Setup dataset
        dataset = self.setup_dataset(val_split=val_split,
                                     zero_threshold=zero_threshold,
                                     zero_keep_frac=zero_keep_frac,
                                     energy_balance=energy_balance,
                                     target_importance=target_importance)
        
        # Setup trainer
        trainer = self.setup_training(config)
        
        # Setup monitoring if enabled
        if enable_monitoring:
            print("\n📊 Setting up monitoring...")
            monitor = TrainingMonitor(self.output_dir, live_plotting=True)
            
            # Create live callback
            live_callback = LiveTrainingCallback(
                monitor, 
                update_every=config.log_every * 5,    # Update data every 5 log intervals
                plot_every=config.log_every * 20      # Update plots every 20 log intervals
            )
            
            # Add callback to trainer
            trainer.add_callback(live_callback)
            print("✓ Live plotting enabled")

        # Truth-vs-prediction snapshot callback
        if prediction_plot_opts is not None:
            from lucid.siren.training.prediction_plot import (
                PredictionComparisonCallback,
            )
            print("\n🖼️  Setting up prediction-vs-truth plots...")
            cb = PredictionComparisonCallback(
                h5_path=self.h5_path,
                dataset=dataset,
                output_dir=self.output_dir,
                energies_mev=prediction_plot_opts['energies'],
                distance_slices=prediction_plot_opts['distance_slices'],
                xaxis_slices=prediction_plot_opts['xaxis_slices'],
                every=prediction_plot_opts['every'],
                resume=self.resume,
            )
            trainer.add_callback(cb)
            print(f"✓ Prediction plots → {cb.plot_dir} (every "
                  f"{prediction_plot_opts['every']} steps)")

        # Start training
        print("\n🏃 Starting SIREN training...")
        print("=" * 60)
        history = trainer.train()
        
        print("\n✅ Training completed!")
        print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        if history['val_loss']:
            print(f"Final val loss: {history['val_loss'][-1]:.6f}")
        
        # Plot final training history
        print("\n📈 Generating final plots...")
        fig = trainer.plot_training_history(save_path=self.output_dir / 'final_training_progress.png')
        plt.close(fig)  # Close to avoid memory issues
        
        # Save trained model
        print("\n💾 Saving trained model...")
        model_save_dir = self.output_dir / 'trained_model'
        weights_path, metadata_path = trainer.save_trained_model(model_save_dir, self.model_name)
        
        print(f"✅ Model saved successfully!")
        print(f"  Weights: {weights_path}")
        print(f"  Metadata: {metadata_path}")
        
        # Display metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📋 Model Metadata:")
        print(f"  Material: {self.material}")
        print(f"  Particle: {self.particle}")
        print(f"  Data type: {self.data_type}")
        print(f"  Energy range: {metadata['dataset_info']['energy_range']} MeV")
        if 'angle_range' in metadata['dataset_info'] and metadata['dataset_info']['angle_range'] is not None:
            print(f"  Angle range: {[f'{np.degrees(a):.1f}°' for a in metadata['dataset_info']['angle_range']]}")
        if 'dedx_range' in metadata['dataset_info'] and metadata['dataset_info']['dedx_range'] is not None:
            print(f"  dE/dx range: {metadata['dataset_info']['dedx_range']} keV/mm")
        print(f"  Distance range: {metadata['dataset_info']['distance_range']} mm")
        print(f"  Architecture: {metadata['model_config']['hidden_layers']} layers × {metadata['model_config']['hidden_features']} features")
        print(f"  Final training loss: {metadata['training_info']['final_train_loss']:.6f}")
        if metadata['training_info']['final_val_loss']:
            print(f"  Final validation loss: {metadata['training_info']['final_val_loss']:.6f}")
        
        print(f"\n🎉 Training complete! Model saved to: {model_save_dir}")
        
        return history, trainer


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='Train PhotonSim SIREN models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data configuration
    parser.add_argument('--material', type=str, default='water',
                        help='Material type (default: water)')
    parser.add_argument('--particle', type=str, default='muon',
                        help='Particle type (default: muon)')
    parser.add_argument('--data-type', type=str, default='photon',
                        choices=['photon', 'dedx'],
                        help='Type of data to train on: photon (Cherenkov) or dedx (default: photon)')
    
    # Model architecture
    parser.add_argument('--hidden-features', type=int, default=256,
                        help='Hidden layer size (default: 256)')
    parser.add_argument('--hidden-layers', type=int, default=3,
                        help='Number of hidden layers (default: 3)')
    parser.add_argument('--w0', type=float, default=30.0,
                        help='SIREN frequency parameter (default: 30.0)')
    
    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=65536,
                        help='Batch size (default: 65536)')
    parser.add_argument('--num-steps', type=int, default=30000,
                        help='Total training steps (default: 30000)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--grad-clip-norm', type=float, default=0.0,
                        help='Clip the global gradient norm to this value '
                             '(default: 0.0 = no clipping). Try 1.0 or 5.0 '
                             'to tame SIREN loss spikes.')
    
    # Scheduler settings
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for LR reduction (default: 20)')
    parser.add_argument('--lr-reduction-factor', type=float, default=0.5,
                        help='LR reduction factor (default: 0.5)')
    parser.add_argument('--min-lr', type=float, default=1e-7,
                        help='Minimum learning rate (default: 1e-7)')
    
    # Logging settings
    parser.add_argument('--log-every', type=int, default=10,
                        help='Log frequency (default: 10)')
    parser.add_argument('--val-every', type=int, default=50,
                        help='Validation frequency (default: 50)')
    parser.add_argument('--checkpoint-every', type=int, default=500,
                        help='Checkpoint frequency (default: 500)')
    
    # Other options
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--no-monitoring', action='store_true',
                        help='Disable live monitoring')
    parser.add_argument('--notebook-mode', action='store_true',
                        help='Force notebook mode for plot display')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--val-split', type=float, default=0.01,
                        help='Validation split fraction (default: 0.01)')
    parser.add_argument('--zero-threshold', type=float, default=1e-2,
                        help='Targets <= this value are treated as "zero" '
                             '(0.2%% are kept as anchors, the rest dropped). '
                             'Doubles as the log offset in '
                             'log10(target + threshold). Must be > 0. '
                             'Default 1e-2; try 1e-6 to keep the long tail.')
    parser.add_argument('--zero-keep-frac', type=float, default=0.002,
                        help='Fraction of below-threshold ("zero") samples '
                             'to keep as training anchors. 0 drops them all; '
                             '1 keeps everything. Default 0.002 (0.2%%).')
    parser.add_argument('--energy-balance', type=str, default='none',
                        choices=['none', 'uniform', 'log_uniform'],
                        help='How to weight samples across the energy grid. '
                             "'none' (default) — uniform-across-rows (low-E "
                             'over-represented because the grid is denser '
                             "there). 'uniform' — each energy point gets "
                             "equal probability. 'log_uniform' — uniform in "
                             "log(E), so each decade is equally weighted.")
    parser.add_argument('--target-importance', type=float, default=0.0,
                        help='Mixture coefficient β ∈ [0, 1] for '
                             'importance-sampling on the target value '
                             'WITHIN each energy. Within each energy the '
                             'per-sample weight is (1-β)·uniform + '
                             'β·target_normalised. β=0 (default) leaves '
                             'sampling uniform across (angle, s/s_max); β=1 '
                             'fully weights by target (high-target bins like '
                             'the Cherenkov ring get most of the samples). '
                             'Combines cleanly with --energy-balance.')

    # Data source override
    parser.add_argument('--h5-path', type=str, default=None,
                        help='Override the conventional .h5 path '
                             '(default: data/<material>/<particle>/<table>.h5). '
                             'Material/particle/data-type still drive the '
                             'output directory.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Redirect every training artifact (checkpoints, '
                             'prediction plots, trained model, history) to '
                             'this folder instead of '
                             'data/<material>/<particle>/<siren_training|dedx_siren_training>. '
                             'Used by the batch-scan tool to land each '
                             'hyperparameter run in its own folder.')

    # In-training prediction-vs-truth plots
    parser.add_argument('--prediction-plots', dest='prediction_plots',
                        action='store_true', default=True,
                        help='Save predicted-vs-truth comparison PNGs every '
                             '--prediction-plot-every steps (default: on).')
    parser.add_argument('--no-prediction-plots', dest='prediction_plots',
                        action='store_false',
                        help='Disable the predicted-vs-truth PNG stream.')
    parser.add_argument('--prediction-plot-every', type=int, default=500,
                        help='Refresh interval in steps (default: 500).')
    parser.add_argument('--prediction-plot-energies', type=str,
                        default='1000,2000,5000,10000,80000',
                        help='Comma-separated representative energies in MeV '
                             '(default: 1000,2000,5000,10000,80000).')
    parser.add_argument('--prediction-plot-distance-slices', type=str,
                        default='0.1,0.25,0.5,0.75',
                        help='Comma-separated s/s_max slice values, each in '
                             '[0, 1] (default: 0.1,0.25,0.5,0.75).')
    parser.add_argument('--prediction-plot-angle-slices-deg', type=str,
                        default='21,41,61,90',
                        help='Photon variant only: angle slices for the new '
                             'middle row, in degrees. Default 21,41,61,90 '
                             '(Cherenkov θ_C ~ 41° in water, plus '
                             '±20° and 90°).')
    parser.add_argument('--prediction-plot-dedx-slices', type=str,
                        default='100,250,500,800',
                        help='dEdx variant only: dE/dx slice values for the '
                             'new middle row, in keV/mm '
                             '(default: 100,250,500,800).')

    args = parser.parse_args()
    
    # Configure matplotlib for notebook display if needed
    setup_matplotlib_for_notebook(force_notebook_mode=getattr(args, 'notebook_mode', False))
    
    # Create training configuration
    config = TrainingConfig(
        # Model architecture
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        w0=args.w0,
        
        # Training parameters
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        
        # Scheduler settings
        use_patience_scheduler=True,
        patience=args.patience,
        lr_reduction_factor=args.lr_reduction_factor,
        min_lr=args.min_lr,
        
        # Optimizer
        optimizer='adam',
        grad_clip_norm=args.grad_clip_norm,
        
        # Logging
        log_every=args.log_every,
        val_every=args.val_every,
        checkpoint_every=args.checkpoint_every,
        
        seed=args.seed
    )
    
    # Handle data_type argument (argparse converts dashes to underscores)
    data_type = getattr(args, 'data_type', 'photon')

    print("📊 Training Configuration:")
    print(f"  • Material/Particle: {args.material}/{args.particle}")
    print(f"  • Data type: {data_type}")
    print(f"  • Architecture: {config.hidden_layers} layers × {config.hidden_features} features")
    print(f"  • Learning Rate: {config.learning_rate:.2e} (with patience-based scheduling)")
    print(f"  • Batch Size: {config.batch_size:,}")
    print(f"  • Total Steps: {config.num_steps:,}")
    print(f"  • Validation Split: {args.val_split:.1%}")

    # Initialize trainer
    trainer_manager = PhotonSimTrainer(
        material=args.material,
        particle=args.particle,
        resume=args.resume,
        data_type=data_type,
        h5_path_override=args.h5_path,
        output_dir_override=args.output_dir,
    )

    # Wire up predicted-vs-truth comparison plots if enabled.
    prediction_plot_opts = None
    if args.prediction_plots:
        from lucid.siren.training.prediction_plot import (
            parse_float_list, parse_int_list,
        )
        # The x-axis slice values live in the variant's native units (radians
        # for photon angle, keV/mm for dedx). The CLI exposes degrees for the
        # photon case for readability, so convert here.
        if data_type == 'photon':
            xaxis_slices = [float(np.radians(float(x)))
                            for x in args.prediction_plot_angle_slices_deg.split(',')
                            if x.strip()]
        else:
            xaxis_slices = parse_float_list(args.prediction_plot_dedx_slices)
        prediction_plot_opts = {
            'energies':       parse_int_list(args.prediction_plot_energies),
            'distance_slices': parse_float_list(args.prediction_plot_distance_slices),
            'xaxis_slices':   xaxis_slices,
            'every':          args.prediction_plot_every,
        }

    # Run training
    try:
        history, trainer = trainer_manager.train(
            config=config,
            enable_monitoring=not args.no_monitoring,
            prediction_plot_opts=prediction_plot_opts,
            val_split=args.val_split,
            zero_threshold=args.zero_threshold,
            zero_keep_frac=args.zero_keep_frac,
            energy_balance=args.energy_balance,
            target_importance=args.target_importance,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Progress has been saved. Use --resume to continue.")
        return 1
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())