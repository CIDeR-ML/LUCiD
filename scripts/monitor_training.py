#!/usr/bin/env python3
"""
Monitor SIREN training progress in real-time.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path
import json

def load_training_progress(training_dir):
    """Load training progress from checkpoints."""
    training_path = Path(training_dir)
    
    progress_data = {
        'steps': [],
        'train_losses': [],
        'val_losses': [],
        'timestamps': []
    }
    
    # Load config if available
    config_file = training_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"Training configuration:")
        print(f"  Hidden features: {config.get('hidden_features', 'N/A')}")
        print(f"  Hidden layers: {config.get('hidden_layers', 'N/A')}")
        print(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
        print(f"  w0: {config.get('w0', 'N/A')}")
        print(f"  Total parameters: {config.get('n_parameters', 'N/A'):,}")
    
    # Find all checkpoint files
    checkpoint_files = sorted(training_path.glob("checkpoint_step_*.npz"))
    
    for checkpoint_file in checkpoint_files:
        try:
            data = np.load(checkpoint_file)
            step = int(data['step'])
            
            if 'train_losses' in data:
                train_losses = data['train_losses']
                progress_data['train_losses'].extend(train_losses)
                progress_data['steps'].extend(range(len(train_losses)))
            
            if 'val_losses' in data:
                val_losses = data['val_losses']
                # Validation losses are sampled at evaluation intervals
                val_steps = np.linspace(0, len(train_losses)-1, len(val_losses), dtype=int)
                for i, val_loss in enumerate(val_losses):
                    progress_data['val_losses'].append((val_steps[i], val_loss))
            
            # Use file modification time as timestamp
            timestamp = checkpoint_file.stat().st_mtime
            progress_data['timestamps'].append(timestamp)
            
        except Exception as e:
            print(f"Error loading {checkpoint_file}: {e}")
    
    return progress_data

def plot_training_progress(progress_data, save_path=None):
    """Plot current training progress."""
    if not progress_data['train_losses']:
        print("No training data found.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Training loss
    ax = axes[0, 0]
    if progress_data['train_losses']:
        ax.plot(progress_data['train_losses'], 'b-', alpha=0.7, linewidth=1)
        ax.set_title('Training Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Show recent progress
        recent_steps = min(100, len(progress_data['train_losses']))
        recent_losses = progress_data['train_losses'][-recent_steps:]
        recent_improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0] * 100
        ax.text(0.02, 0.98, f'Recent improvement: {recent_improvement:.2f}%', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Validation loss
    ax = axes[0, 1]
    if progress_data['val_losses']:
        val_steps, val_losses = zip(*progress_data['val_losses'])
        ax.plot(val_steps, val_losses, 'o-', color='orange', linewidth=2, markersize=4)
        ax.set_title('Validation Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Show validation trend
        if len(val_losses) > 1:
            val_improvement = (val_losses[0] - val_losses[-1]) / val_losses[0] * 100
            ax.text(0.02, 0.98, f'Val improvement: {val_improvement:.2f}%', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Combined losses
    ax = axes[1, 0]
    if progress_data['train_losses']:
        ax.plot(progress_data['train_losses'], 'b-', alpha=0.7, linewidth=1, label='Train')
    if progress_data['val_losses']:
        val_steps, val_losses = zip(*progress_data['val_losses'])
        ax.plot(val_steps, val_losses, 'o-', color='orange', linewidth=2, markersize=4, label='Validation')
    ax.set_title('Training Progress')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = "Training Statistics:\n\n"
    
    if progress_data['train_losses']:
        n_steps = len(progress_data['train_losses'])
        current_loss = progress_data['train_losses'][-1]
        initial_loss = progress_data['train_losses'][0]
        improvement = (initial_loss - current_loss) / initial_loss * 100
        
        stats_text += f"Steps completed: {n_steps:,}\n"
        stats_text += f"Current train loss: {current_loss:.1f}\n"
        stats_text += f"Initial train loss: {initial_loss:.1f}\n"
        stats_text += f"Total improvement: {improvement:.2f}%\n\n"
    
    if progress_data['val_losses']:
        val_steps, val_losses = zip(*progress_data['val_losses'])
        current_val = val_losses[-1]
        initial_val = val_losses[0]
        val_improvement = (initial_val - current_val) / initial_val * 100
        
        stats_text += f"Current val loss: {current_val:.1f}\n"
        stats_text += f"Initial val loss: {initial_val:.1f}\n"
        stats_text += f"Val improvement: {val_improvement:.2f}%\n\n"
    
    # Add convergence status
    if len(progress_data['train_losses']) > 50:
        recent_variance = np.var(progress_data['train_losses'][-50:])
        if recent_variance < 1000:  # Arbitrary threshold
            stats_text += "Status: ✅ Converging\n"
        else:
            stats_text += "Status: 🔄 Still learning\n"
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Progress plot saved: {save_path}")
    
    plt.show()
    
    return fig

def monitor_training_live(training_dir, refresh_interval=30):
    """Monitor training progress in real-time."""
    print(f"Monitoring training in {training_dir}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            print(f"\n{'='*50}")
            print(f"Training Progress Update - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            
            # Load current progress
            progress_data = load_training_progress(training_dir)
            
            if progress_data['train_losses']:
                plot_training_progress(progress_data, 
                                     save_path=Path(training_dir) / "live_progress.png")
                
                # Print summary
                n_steps = len(progress_data['train_losses'])
                current_loss = progress_data['train_losses'][-1]
                print(f"\nCurrent status:")
                print(f"  Steps: {n_steps:,}")
                print(f"  Train loss: {current_loss:.1f}")
                
                if progress_data['val_losses']:
                    val_loss = progress_data['val_losses'][-1][1]
                    print(f"  Val loss: {val_loss:.1f}")
            else:
                print("No training data found yet...")
            
            print(f"\nNext update in {refresh_interval} seconds...")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

def view_existing_plots(training_dir):
    """View existing training progress plots."""
    training_path = Path(training_dir)
    
    # Find all progress plots
    plot_files = sorted(training_path.glob("*progress*.png"))
    
    if not plot_files:
        print("No training progress plots found.")
        return
    
    print(f"Found {len(plot_files)} training progress plots:")
    for i, plot_file in enumerate(plot_files):
        print(f"  {i+1}. {plot_file.name}")
    
    # Show the final training progress
    final_plot = training_path / "final_training_progress.png"
    if final_plot.exists():
        print(f"\nDisplaying final training progress...")
        from IPython.display import Image, display
        try:
            display(Image(str(final_plot)))
        except:
            print(f"Open this file to view: {final_plot}")

def main():
    parser = argparse.ArgumentParser(description='Monitor SIREN training progress')
    parser.add_argument('--training-dir', default='output/training_output',
                       help='Training output directory')
    parser.add_argument('--live', action='store_true',
                       help='Monitor training in real-time')
    parser.add_argument('--refresh', type=int, default=30,
                       help='Refresh interval for live monitoring (seconds)')
    parser.add_argument('--view-plots', action='store_true',
                       help='View existing training plots')
    
    args = parser.parse_args()
    
    if args.view_plots:
        view_existing_plots(args.training_dir)
    elif args.live:
        monitor_training_live(args.training_dir, args.refresh)
    else:
        # Just show current progress
        progress_data = load_training_progress(args.training_dir)
        plot_training_progress(progress_data, 
                             save_path=Path(args.training_dir) / "current_progress.png")

if __name__ == "__main__":
    main()