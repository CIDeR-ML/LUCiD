#!/usr/bin/env python3
"""
Simple debug script to check scaling in HDF5 file.
"""

import numpy as np
import h5py
from pathlib import Path

def simple_debug():
    """Simple debugging without JAX dependencies."""
    
    # Load dataset (same path as in notebook)
    data_path = Path('/sdf/home/c/cjesus/Dev/PhotonSim/output/photon_lookup_table.h5')
    
    if not data_path.exists():
        print(f"❌ File not found: {data_path}")
        return
        
    print("=" * 60)
    print("SIMPLE SCALING DEBUG")
    print("=" * 60)
    
    # Check what's in the original HDF5 file
    print("\n1. ORIGINAL HDF5 LOOKUP TABLE:")
    with h5py.File(data_path, 'r') as f:
        density_table = f['data/photon_table_density'][:]
        energy_centers = f['coordinates/energy_centers'][:]
        angle_centers = f['coordinates/angle_centers'][:]
        distance_centers = f['coordinates/distance_centers'][:]
        
    print(f"   Table shape: {density_table.shape}")
    print(f"   Table value range: {density_table.min():.2e} to {density_table.max():.2e}")
    print(f"   Table mean: {density_table.mean():.2e}")
    print(f"   Non-zero entries: {np.sum(density_table > 0)}/{density_table.size}")
    
    # Look at some specific energy slices
    print(f"\n2. ENERGY SLICES:")
    energies_to_check = [200, 400, 600, 800, 1000]  # Same as plots
    
    for energy in energies_to_check:
        # Find closest energy
        energy_idx = np.argmin(np.abs(energy_centers - energy))
        actual_energy = energy_centers[energy_idx]
        
        # Get 2D slice
        slice_2d = density_table[energy_idx, :, :]
        
        print(f"   {actual_energy:.0f} MeV slice:")
        print(f"     Range: {slice_2d.min():.2e} to {slice_2d.max():.2e}")
        print(f"     Mean: {slice_2d.mean():.2e}")
        print(f"     Valid points: {np.sum(slice_2d > 1e-10)}/{slice_2d.size}")
        
        # Check around Cherenkov angle (43 degrees ≈ 0.75 radians)
        cherenkov_angle_rad = np.radians(43)
        angle_idx = np.argmin(np.abs(angle_centers - cherenkov_angle_rad))
        actual_angle = np.degrees(angle_centers[angle_idx])
        
        # Get angular profile at this energy
        angular_profile = slice_2d[angle_idx, :]
        
        print(f"     At {actual_angle:.1f}° (Cherenkov): {angular_profile.min():.2e} to {angular_profile.max():.2e}")
    
    # 3. Check after log transform (what original training would see)
    print(f"\n3. AFTER LOG TRANSFORM:")
    log_table = np.log10(density_table + 1e-10)
    print(f"   Log range: {log_table.min():.3f} to {log_table.max():.3f}")
    print(f"   Log mean: {log_table.mean():.3f}")
    
    # Check if values are being filtered (mask for very small values)
    mask = density_table > 1e-10
    filtered_table = density_table[mask]
    print(f"\n4. AFTER FILTERING (> 1e-10):")
    print(f"   Filtered count: {len(filtered_table)}/{density_table.size}")
    print(f"   Filtered range: {filtered_table.min():.2e} to {filtered_table.max():.2e}")
    print(f"   Filtered mean: {filtered_table.mean():.2e}")
    
    print("=" * 60)

if __name__ == "__main__":
    simple_debug()