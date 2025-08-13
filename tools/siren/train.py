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
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directories to path
script_dir = Path(__file__).parent
tools_dir = script_dir.parent
project_root = tools_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(tools_dir))

# Import training modules
from siren.training.trainer import SIRENTrainer, TrainingConfig
from siren.training.dataset import PhotonSimDataset
from siren.training.monitor import TrainingMonitor, LiveTrainingCallback
from siren.training.analyzer import TrainingAnalyzer
from utils import base_dir_path, setup_matplotlib_for_notebook


class PhotonSimTrainer:
    """Main training class for PhotonSim SIREN models."""
    
    def __init__(self, material='water', particle='muon', resume=False):
        """Initialize trainer with material and particle type."""
        self.material = material
        self.particle = particle
        self.resume = resume
        
        # Set up paths
        self.base_dir = Path(base_dir_path())
        self.data_dir = self.base_dir / 'data' / material / particle
        # Training output goes to siren_training subdirectory
        self.output_dir = self.data_dir / 'siren_training'
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 PhotonSim SIREN Training")
        print(f"  Material: {material}")
        print(f"  Particle: {particle}")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Output directory: {self.output_dir}")
        
        # Check for h5 file
        self.h5_path = self.data_dir / 'photon_lookup_table.h5'
        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"HDF5 lookup table not found at {self.h5_path}\n"
                f"Please ensure the PhotonSim table exists for {material}/{particle}"
            )
        
        print(f"✓ Found HDF5 file: {self.h5_path}")
        
        # Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ PyTorch device: {device}")
        print(f"✓ JAX devices: {jax.devices()}")
    
    def setup_dataset(self, val_split=0.1):
        """Load and configure the dataset."""
        print("\n📊 Loading dataset...")
        self.dataset = PhotonSimDataset(self.h5_path, val_split=val_split)
        
        print(f"\nDataset info:")
        print(f"  Data type: {self.dataset.data_type}")
        print(f"  Total samples: {len(self.dataset.data['inputs']):,}")
        print(f"  Train samples: {len(self.dataset.train_indices):,}")
        print(f"  Val samples: {len(self.dataset.val_indices):,}")
        print(f"  Energy range: {self.dataset.energy_range[0]:.0f}-{self.dataset.energy_range[1]:.0f} MeV")
        print(f"  Angle range: {np.degrees(self.dataset.angle_range[0]):.1f}-{np.degrees(self.dataset.angle_range[1]):.1f} degrees")
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
    
    def train(self, config, enable_monitoring=True):
        """Run the training process."""
        # Setup dataset
        dataset = self.setup_dataset()
        
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
        model_name = f'photonsim_siren'
        weights_path, metadata_path = trainer.save_trained_model(model_save_dir, model_name)
        
        print(f"✅ Model saved successfully!")
        print(f"  Weights: {weights_path}")
        print(f"  Metadata: {metadata_path}")
        
        # Display metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📋 Model Metadata:")
        print(f"  Material: {self.material}")
        print(f"  Particle: {self.particle}")
        print(f"  Energy range: {metadata['dataset_info']['energy_range']} MeV")
        print(f"  Angle range: {[f'{np.degrees(a):.1f}°' for a in metadata['dataset_info']['angle_range']]}")
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
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split fraction (default: 0.1)')
    
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
        grad_clip_norm=0.0,
        
        # Logging
        log_every=args.log_every,
        val_every=args.val_every,
        checkpoint_every=args.checkpoint_every,
        
        seed=args.seed
    )
    
    print("📊 Training Configuration:")
    print(f"  • Material/Particle: {args.material}/{args.particle}")
    print(f"  • Architecture: {config.hidden_layers} layers × {config.hidden_features} features")
    print(f"  • Learning Rate: {config.learning_rate:.2e} (with patience-based scheduling)")
    print(f"  • Batch Size: {config.batch_size:,}")
    print(f"  • Total Steps: {config.num_steps:,}")
    print(f"  • Validation Split: {args.val_split:.1%}")
    
    # Initialize trainer
    trainer_manager = PhotonSimTrainer(
        material=args.material,
        particle=args.particle,
        resume=args.resume
    )
    
    # Run training
    try:
        history, trainer = trainer_manager.train(
            config=config,
            enable_monitoring=not args.no_monitoring
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