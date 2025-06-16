#!/usr/bin/env python3
"""
Test script for the PhotonSim → diffCherenkov pipeline.

This script tests the complete pipeline:
1. Load PhotonSim 3D lookup table
2. Train a small SIREN model
3. Test inference with the trained model
"""

import sys
import logging
from pathlib import Path
import numpy as np
import jax.numpy as jnp

# Add tools directory to path
sys.path.append(str(Path(__file__).parent.parent / "tools"))

from photonsim_dataset import PhotonSimDataset
from train_photonsim_siren import PhotonSimSIRENTrainer
from siren import PhotonSimSIREN, load_siren_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_loading(table_path: str):
    """Test loading PhotonSim dataset."""
    logger.info("=== Testing Dataset Loading ===")
    
    try:
        dataset = PhotonSimDataset(
            table_path=table_path,
            batch_size=1024,
            min_photon_count=1.0,
            max_distance=2000.0,
            max_angle=0.05,  # Very small for testing
        )
        
        # Print stats
        stats = dataset.get_stats()
        logger.info("Dataset Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Test batch iterator
        logger.info("Testing batch iterator...")
        batch_count = 0
        for inputs, outputs in dataset.get_batch_iterator():
            logger.info(f"  Batch {batch_count}: inputs {inputs.shape}, outputs {outputs.shape}")
            logger.info(f"    Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
            logger.info(f"    Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
            batch_count += 1
            if batch_count >= 2:
                break
        
        logger.info("✅ Dataset loading test passed!")
        return dataset
        
    except Exception as e:
        logger.error(f"❌ Dataset loading test failed: {e}")
        raise

def test_training(dataset: PhotonSimDataset, output_dir: str = "test_training"):
    """Test SIREN training."""
    logger.info("=== Testing SIREN Training ===")
    
    try:
        # Create trainer with small configuration for testing
        trainer = PhotonSimSIRENTrainer(
            dataset=dataset,
            hidden_features=64,  # Smaller for testing
            hidden_layers=2,     # Fewer layers for testing
            learning_rate=1e-3,  # Higher LR for faster convergence
        )
        
        # Train for just a few steps
        trainer.train(
            num_steps=50,
            log_every=10,
            eval_every=20,
            save_every=25,
            output_dir=output_dir,
        )
        
        # Check if model was saved
        model_path = Path(output_dir) / "final_model.npz"
        if model_path.exists():
            logger.info(f"✅ Model saved successfully at {model_path}")
        else:
            raise FileNotFoundError(f"Model not saved at {model_path}")
        
        logger.info("✅ Training test passed!")
        return str(model_path)
        
    except Exception as e:
        logger.error(f"❌ Training test failed: {e}")
        raise

def test_inference(model_path: str, dataset: PhotonSimDataset):
    """Test model inference."""
    logger.info("=== Testing Model Inference ===")
    
    try:
        # Load trained model
        model = PhotonSimSIREN(model_path)
        
        # Print model info
        info = model.get_model_info()
        logger.info("Model Information:")
        logger.info(f"  Hidden features: {info['model_config']['hidden_features']}")
        logger.info(f"  Hidden layers: {info['model_config']['hidden_layers']}")
        logger.info(f"  Final training loss: {info['metadata']['final_train_loss']:.6f}")
        
        # Test inference on sample data
        sample_inputs, sample_targets = dataset.get_sample_batch(100)
        
        # Denormalize inputs for model (it expects physical units)
        physical_inputs = dataset.denormalize_inputs(sample_inputs)
        
        # Make predictions
        predictions = model(physical_inputs)
        
        # Compare with targets (denormalized)
        physical_targets = dataset.denormalize_output(sample_targets)
        
        logger.info(f"Sample predictions vs targets:")
        logger.info(f"  Predictions shape: {predictions.shape}")
        logger.info(f"  Targets shape: {physical_targets.shape}")
        logger.info(f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        logger.info(f"  Target range: [{physical_targets.min():.3f}, {physical_targets.max():.3f}]")
        
        # Calculate simple metrics
        mse = jnp.mean((predictions - physical_targets) ** 2)
        mae = jnp.mean(jnp.abs(predictions - physical_targets))
        
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        
        logger.info("✅ Inference test passed!")
        
    except Exception as e:
        logger.error(f"❌ Inference test failed: {e}")
        raise

def test_model_loading_compatibility(model_path: str):
    """Test automatic model loading."""
    logger.info("=== Testing Model Loading Compatibility ===")
    
    try:
        # Test automatic model type detection
        model = load_siren_model(model_path, model_type="auto")
        
        logger.info(f"Model type automatically detected and loaded")
        logger.info(f"Model class: {type(model).__name__}")
        
        # Test with specific type
        model2 = load_siren_model(model_path, model_type="photonsim")
        
        logger.info("✅ Model loading compatibility test passed!")
        
    except Exception as e:
        logger.error(f"❌ Model loading compatibility test failed: {e}")
        raise

def main():
    """Run all pipeline tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PhotonSim → diffCherenkov pipeline')
    parser.add_argument('--data-path', 
                       default='../../PhotonSim/3d_lookup_table',
                       help='Path to PhotonSim 3D lookup table directory')
    parser.add_argument('--output-dir', 
                       default='test_pipeline_output',
                       help='Output directory for test results')
    
    args = parser.parse_args()
    
    # Check if data path exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        logger.info("Please run the PhotonSim table creation first or adjust the path")
        return 1
    
    logger.info("🚀 Starting PhotonSim → diffCherenkov Pipeline Test")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Test 1: Dataset loading
        dataset = test_dataset_loading(str(data_path))
        
        # Test 2: Training
        model_path = test_training(dataset, args.output_dir)
        
        # Test 3: Inference
        test_inference(model_path, dataset)
        
        # Test 4: Model loading compatibility
        test_model_loading_compatibility(model_path)
        
        logger.info("🎉 All tests passed! Pipeline is working correctly.")
        logger.info(f"Test outputs saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"💥 Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())