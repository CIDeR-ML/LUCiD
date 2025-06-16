#!/usr/bin/env python3
"""
Analyze trained SIREN model and compare with PhotonSim lookup table.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import jax.numpy as jnp

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from siren import PhotonSimSIREN
from photonsim_sampled_dataset import PhotonSimSampledDataset

def load_trained_model(model_path):
    """Load the trained SIREN model."""
    print(f"Loading trained model from {model_path}")
    model = PhotonSimSIREN(model_path)
    
    info = model.get_model_info()
    print(f"Model info:")
    print(f"  Hidden features: {info['model_config']['hidden_features']}")
    print(f"  Hidden layers: {info['model_config']['hidden_layers']}")
    print(f"  w0: {info['model_config']['w0']}")
    print(f"  Final step: {info['metadata']['final_step']}")
    print(f"  Final train loss: {info['metadata']['final_train_loss']:.6f}")
    print(f"  Final val loss: {info['metadata']['final_val_loss']:.6f}")
    
    return model

def compare_model_vs_data(model, dataset, n_samples=10000):
    """Compare model predictions with ground truth data."""
    print(f"\nComparing model predictions with ground truth on {n_samples} samples...")
    
    # Get sample data
    inputs, targets = dataset.get_sample_batch(n_samples)
    
    # Make predictions
    predictions = model(inputs)
    
    # Convert to numpy for analysis
    inputs_np = np.array(inputs)
    targets_np = np.array(targets)
    predictions_np = np.array(predictions)
    
    # Calculate metrics
    mse = np.mean((predictions_np - targets_np) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_np - targets_np))
    
    # R-squared
    ss_res = np.sum((targets_np - predictions_np) ** 2)
    ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Relative error (avoid division by very small numbers)
    relative_error = np.mean(np.abs(predictions_np - targets_np) / (np.abs(targets_np) + 1e-8))
    
    print(f"Evaluation metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    print(f"  Relative error: {relative_error:.6f}")
    
    return {
        'inputs': inputs_np,
        'targets': targets_np,
        'predictions': predictions_np,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'relative_error': relative_error
    }

def visualize_comparison(comparison_data, dataset, output_path="output/siren_analysis.png"):
    """Create comprehensive visualization of model vs data."""
    
    inputs = comparison_data['inputs']
    targets = comparison_data['targets']
    predictions = comparison_data['predictions']
    
    # Denormalize inputs for interpretation
    input_ranges = dataset.input_ranges
    
    # Inputs are normalized to [-1, 1], denormalize them
    energy = (inputs[:, 0] + 1) * (input_ranges['energy'][1] - input_ranges['energy'][0]) / 2 + input_ranges['energy'][0]
    angle_rad = (inputs[:, 1] + 1) * (input_ranges['angle'][1] - input_ranges['angle'][0]) / 2 + input_ranges['angle'][0]
    angle_deg = np.degrees(angle_rad)
    distance = (inputs[:, 2] + 1) * (input_ranges['distance'][1] - input_ranges['distance'][0]) / 2 + input_ranges['distance'][0]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Predictions vs Targets
    ax = axes[0, 0]
    ax.scatter(targets, predictions, alpha=0.5, s=1)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax.set_xlabel('Ground Truth (log₁₀ density)')
    ax.set_ylabel('SIREN Prediction')
    ax.set_title(f'Predictions vs Ground Truth\nR² = {comparison_data["r2"]:.4f}')
    ax.grid(True, alpha=0.3)
    
    # 2. Residuals vs Targets
    ax = axes[0, 1]
    residuals = predictions - targets
    ax.scatter(targets, residuals, alpha=0.5, s=1)
    ax.axhline(0, color='r', linestyle='--', alpha=0.8)
    ax.set_xlabel('Ground Truth (log₁₀ density)')
    ax.set_ylabel('Residuals (pred - truth)')
    ax.set_title(f'Residuals vs Ground Truth\nMAE = {comparison_data["mae"]:.4f}')
    ax.grid(True, alpha=0.3)
    
    # 3. Error vs Energy
    ax = axes[0, 2]
    abs_error = np.abs(residuals)
    ax.scatter(energy, abs_error, alpha=0.5, s=1)
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error vs Energy')
    ax.grid(True, alpha=0.3)
    
    # 4. Error vs Angle
    ax = axes[1, 0]
    ax.scatter(angle_deg, abs_error, alpha=0.5, s=1)
    ax.set_xlabel('Opening Angle (degrees)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error vs Opening Angle')
    ax.grid(True, alpha=0.3)
    
    # 5. Error vs Distance
    ax = axes[1, 1]
    ax.scatter(distance, abs_error, alpha=0.5, s=1)
    ax.set_xlabel('Distance (mm)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error vs Distance')
    ax.grid(True, alpha=0.3)
    
    # 6. Error distribution
    ax = axes[1, 2]
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Residuals (pred - truth)')
    ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution\nRMSE = {comparison_data["rmse"]:.4f}')
    ax.axvline(0, color='r', linestyle='--', alpha=0.8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('SIREN Model Analysis: Predictions vs PhotonSim Lookup Table', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Analysis plot saved: {output_path}")
    plt.show()

def test_siren_interpolation(model, dataset, output_path="output/siren_interpolation.png"):
    """Test SIREN's interpolation capability on a regular grid."""
    print("\nTesting SIREN interpolation on regular grid...")
    
    # Create a small regular grid for testing
    energy_test = [250, 300, 350, 400]  # MeV
    angle_test = np.linspace(0, 60, 10)  # degrees  
    distance_test = np.linspace(100, 2000, 10)  # mm
    
    # Convert to normalized inputs
    input_ranges = dataset.input_ranges
    
    results = {}
    for energy in energy_test:
        print(f"  Testing energy: {energy} MeV")
        
        # Create grid for this energy
        angle_grid, dist_grid = np.meshgrid(angle_test, distance_test)
        
        # Prepare inputs
        n_points = angle_grid.size
        inputs = np.zeros((n_points, 3))
        
        # Normalize inputs to [-1, 1]
        inputs[:, 0] = 2 * (energy - input_ranges['energy'][0]) / (input_ranges['energy'][1] - input_ranges['energy'][0]) - 1
        inputs[:, 1] = 2 * (np.radians(angle_grid.flatten()) - input_ranges['angle'][0]) / (input_ranges['angle'][1] - input_ranges['angle'][0]) - 1
        inputs[:, 2] = 2 * (dist_grid.flatten() - input_ranges['distance'][0]) / (input_ranges['distance'][1] - input_ranges['distance'][0]) - 1
        
        # Get predictions
        inputs_jax = jnp.array(inputs)
        predictions = model(inputs_jax)
        
        # Reshape back to grid
        pred_grid = np.array(predictions).reshape(angle_grid.shape)
        
        results[energy] = {
            'angle_grid': angle_grid,
            'distance_grid': dist_grid,
            'predictions': pred_grid
        }
    
    # Plot interpolation results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, energy in enumerate(energy_test):
        ax = axes[i]
        result = results[energy]
        
        im = ax.contourf(result['angle_grid'], result['distance_grid'], 
                        result['predictions'], levels=20, cmap='viridis')
        plt.colorbar(im, ax=ax, label='log₁₀(density)')
        
        ax.set_xlabel('Opening Angle (degrees)')
        ax.set_ylabel('Distance (mm)')
        ax.set_title(f'{energy} MeV')
        
        # Mark Cherenkov angle
        ax.axvline(43, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.suptitle('SIREN Interpolation: Photon Density Predictions', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Interpolation plot saved: {output_path}")
    plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze trained SIREN model')
    parser.add_argument('--model-path', 
                       default='output/training_output/final_model.npz',
                       help='Path to trained model')
    parser.add_argument('--dataset-path',
                       default='../PhotonSim/output/siren_dataset',
                       help='Path to original dataset')
    parser.add_argument('--n-samples', type=int, default=50000,
                       help='Number of samples for comparison')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load trained model
    model = load_trained_model(args.model_path)
    
    # Load dataset for comparison
    print(f"\nLoading dataset from {args.dataset_path}")
    dataset = PhotonSimSampledDataset(args.dataset_path, batch_size=args.n_samples)
    
    # Compare model vs data
    comparison_data = compare_model_vs_data(model, dataset, args.n_samples)
    
    # Create output directory if needed
    from pathlib import Path
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create visualizations
    visualize_comparison(comparison_data, dataset, output_path / "siren_analysis.png")
    
    # Test interpolation
    test_siren_interpolation(model, dataset, output_path / "siren_interpolation.png")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()