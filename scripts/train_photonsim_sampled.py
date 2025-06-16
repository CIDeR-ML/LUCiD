#!/usr/bin/env python3
"""
Train JAXSiren on PhotonSim sampled dataset.

This script trains a SIREN network on pre-sampled PhotonSim density data,
using the dataset created by PhotonSim's create_siren_dataset.py tool.
"""

import os
import sys
import logging
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from photonsim_sampled_dataset import PhotonSimSampledDataset
from train_photonsim_siren import PhotonSimSIRENTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SIREN on PhotonSim sampled data')
    parser.add_argument('--dataset-path', required=True,
                       help='Path to PhotonSim sampled dataset directory')
    parser.add_argument('--output-dir', default='output/photonsim_siren_training',
                       help='Output directory for training results')
    parser.add_argument('--num-steps', type=int, default=2000,
                       help='Number of training steps')
    parser.add_argument('--hidden-features', type=int, default=256,
                       help='Number of hidden features')
    parser.add_argument('--hidden-layers', type=int, default=3,
                       help='Number of hidden layers')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16384,
                       help='Batch size')
    parser.add_argument('--w0', type=float, default=30.0,
                       help='SIREN frequency parameter')
    
    args = parser.parse_args()
    
    # Create dataset
    logger.info(f"Loading PhotonSim sampled dataset from {args.dataset_path}")
    dataset = PhotonSimSampledDataset(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
    )
    
    # Print dataset stats
    stats = dataset.get_stats()
    logger.info("Dataset statistics:")
    for key, value in stats.items():
        if key not in ['input_ranges', 'output_stats']:
            logger.info(f"  {key}: {value}")
    
    logger.info("Input ranges:")
    for dim, range_val in stats['input_ranges'].items():
        logger.info(f"  {dim}: {range_val}")
    
    logger.info("Output statistics:")
    for stat, value in stats['output_stats'].items():
        logger.info(f"  {stat}: {value:.6f}")
    
    # Create trainer
    logger.info("Initializing SIREN trainer...")
    trainer = PhotonSimSIRENTrainer(
        dataset=dataset,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        w0=args.w0,
        learning_rate=args.learning_rate,
    )
    
    # Train model
    logger.info(f"Starting training for {args.num_steps} steps...")
    trainer.train(
        num_steps=args.num_steps,
        log_every=100,
        eval_every=200,
        save_every=500,
        output_dir=args.output_dir,
    )
    
    # Final evaluation
    logger.info("Performing final evaluation...")
    metrics = trainer.evaluate_model(n_samples=10000)
    
    logger.info("Training completed successfully!")
    logger.info("Final evaluation metrics:")
    for key, value in metrics.items():
        if key != 'n_samples':
            logger.info(f"  {key}: {value:.6f}")

if __name__ == "__main__":
    main()