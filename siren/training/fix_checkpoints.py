#!/usr/bin/env python3
"""
Quick script to clear incompatible checkpoints.
Run this if you get pickle/loading errors with existing checkpoints.
"""

from pathlib import Path
import shutil

def clear_checkpoints():
    """Clear all checkpoints in the training output directory."""
    # Get the project root (diffCherenkov) from current script location
    script_dir = Path(__file__).parent  # siren/training
    project_root = script_dir.parent.parent  # diffCherenkov
    output_dir = project_root / 'notebooks/output/photonsim_siren_training'
    
    if not output_dir.exists():
        print(f"No output directory found at {output_dir}")
        return
    
    print(f"Clearing checkpoints in {output_dir}")
    
    # Remove all .npz files (checkpoints)
    checkpoint_files = list(output_dir.glob('*.npz'))
    for f in checkpoint_files:
        f.unlink()
        print(f"Removed: {f.name}")
    
    # Remove training data files
    for filename in ['training_history.json', 'config.json', 'monitoring_data.json', 'analysis_results.json']:
        file_path = output_dir / filename
        if file_path.exists():
            file_path.unlink()
            print(f"Removed: {filename}")
    
    # Remove plots
    plot_files = list(output_dir.glob('*.png'))
    for f in plot_files:
        f.unlink()
        print(f"Removed: {f.name}")
    
    print("✅ All checkpoints cleared. You can now run the training notebook fresh.")

if __name__ == "__main__":
    clear_checkpoints()