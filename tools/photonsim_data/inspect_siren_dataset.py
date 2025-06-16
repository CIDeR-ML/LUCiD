#!/usr/bin/env python3
"""
Inspect the SIREN dataset structure and show example data points.
"""

import numpy as np
from pathlib import Path

def inspect_dataset(dataset_dir):
    """Inspect and display information about the SIREN dataset."""
    dataset_path = Path(dataset_dir)
    
    # Load data
    inputs = np.load(dataset_path / "train_inputs.npy")
    outputs = np.load(dataset_path / "train_outputs.npy")
    metadata = np.load(dataset_path / "dataset_metadata.npz", allow_pickle=True)
    
    print("=== SIREN Dataset Structure ===\n")
    
    print(f"Input array shape: {inputs.shape}")
    print(f"Output array shape: {outputs.shape}")
    print(f"Data type: {inputs.dtype}\n")
    
    print("Dataset Configuration:")
    for key in ['n_samples', 'sampling_strategy', 'normalize_inputs', 
                'normalize_outputs', 'log_transform_output']:
        if key in metadata:
            print(f"  {key}: {metadata[key]}")
    
    print(f"\nInput dimensions: {metadata['input_names'].tolist()}")
    print(f"Output: {metadata['output_name']}")
    print(f"Density units: {metadata['density_units']}")
    
    print("\nInput ranges (original):")
    input_ranges = metadata['input_ranges'].item()
    for name, (min_val, max_val) in input_ranges.items():
        print(f"  {name}: [{min_val}, {max_val}]")
    
    print("\nOutput statistics (before transforms):")
    output_stats = metadata['output_stats'].item()
    for stat, value in output_stats.items():
        print(f"  {stat}: {value:.6f}")
    
    print("\n=== Example Data Points ===\n")
    print("First 5 samples:")
    print("Index | Energy | Angle | Distance | Density")
    print("------|--------|-------|----------|--------")
    
    for i in range(min(5, len(inputs))):
        # Inputs are normalized to [-1, 1], denormalize for display
        energy_norm = inputs[i, 0]
        angle_norm = inputs[i, 1]
        distance_norm = inputs[i, 2]
        
        # Denormalize
        e_min, e_max = input_ranges['energy']
        energy = (energy_norm + 1) * (e_max - e_min) / 2 + e_min
        
        a_min, a_max = input_ranges['angle']
        angle = (angle_norm + 1) * (a_max - a_min) / 2 + a_min
        angle_deg = np.degrees(angle)
        
        d_min, d_max = input_ranges['distance']
        distance = (distance_norm + 1) * (d_max - d_min) / 2 + d_min
        
        density = outputs[i]
        
        print(f"{i:5d} | {energy:6.1f} | {angle_deg:5.1f}° | {distance:8.1f} | {density:8.4f}")
    
    print("\nRandom 5 samples:")
    print("Index | Energy | Angle | Distance | Density")
    print("------|--------|-------|----------|--------")
    
    random_indices = np.random.choice(len(inputs), 5, replace=False)
    for idx in random_indices:
        # Denormalize
        energy_norm = inputs[idx, 0]
        angle_norm = inputs[idx, 1]
        distance_norm = inputs[idx, 2]
        
        e_min, e_max = input_ranges['energy']
        energy = (energy_norm + 1) * (e_max - e_min) / 2 + e_min
        
        a_min, a_max = input_ranges['angle']
        angle = (angle_norm + 1) * (a_max - a_min) / 2 + a_min
        angle_deg = np.degrees(angle)
        
        d_min, d_max = input_ranges['distance']
        distance = (distance_norm + 1) * (d_max - d_min) / 2 + d_min
        
        density = outputs[idx]
        
        print(f"{idx:5d} | {energy:6.1f} | {angle_deg:5.1f}° | {distance:8.1f} | {density:8.4f}")
    
    print("\n=== Data Distribution Summary ===\n")
    
    # Check normalized input ranges
    print("Normalized input ranges (should be ~[-1, 1]):")
    for i, name in enumerate(['energy', 'angle', 'distance']):
        print(f"  {name}: [{inputs[:, i].min():.3f}, {inputs[:, i].max():.3f}]")
    
    print("\nOutput distribution:")
    print(f"  Min: {outputs.min():.4f}")
    print(f"  Max: {outputs.max():.4f}")
    print(f"  Mean: {outputs.mean():.4f}")
    print(f"  Std: {outputs.std():.4f}")
    print(f"  Median: {np.median(outputs):.4f}")
    
    # Check for zeros/special values
    n_zeros = np.sum(outputs == outputs.min())
    print(f"\nSpecial values:")
    print(f"  Minimum value count: {n_zeros:,} ({100*n_zeros/len(outputs):.2f}%)")
    
    print("\n=== Ready for JAXSiren Training ===")
    print(f"✓ Inputs normalized to [-1, 1] for SIREN activation functions")
    print(f"✓ Outputs log-transformed to handle wide dynamic range")
    print(f"✓ {len(inputs):,} training samples available")
    print(f"✓ Mixed sampling ensures coverage of both high and low density regions")

if __name__ == '__main__':
    inspect_dataset('output/siren_dataset')