#!/usr/bin/env python3
"""
SIREN Training Results Visualization Tool

Load and visualize SIREN training results without retraining.
Creates publication-quality plots with customizable aesthetics.

Usage:
    python plot_training_results.py --training-dir /path/to/training/output
    python plot_training_results.py --training-dir /path/to/output --style seaborn-v0_8
    python plot_training_results.py --training-dir /path/to/output --save-format pdf --dpi 300
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import seaborn as sns

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from tools.utils import base_dir_path

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

class SIRENTrainingVisualizer:
    """Load and visualize SIREN training results."""
    
    def __init__(self, training_dir: Path):
        """Initialize with training directory."""
        self.training_dir = Path(training_dir)
        self.history = {}
        self.config = {}
        self.metadata = {}
        
        # Load all available data
        self._load_training_data()
        
    def _load_training_data(self):
        """Load training history, config, and metadata."""
        print(f"Loading training data from: {self.training_dir}")
        
        # Load training history
        history_path = self.training_dir / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.history = json.load(f)
            print(f"✓ Loaded training history with {len(self.history.get('step', []))} data points")
        else:
            print("✗ No training history found")
            
        # Load training config
        config_path = self.training_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✓ Loaded training configuration")
        else:
            print("⚠ No training config found")
            
        # Load model metadata (if available)
        metadata_files = list(self.training_dir.glob('*_metadata.json'))
        if metadata_files:
            with open(metadata_files[0], 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ Loaded model metadata from {metadata_files[0].name}")
        else:
            print("⚠ No model metadata found")
    
    def plot_training_curves(
        self, 
        figsize=(15, 5), 
        style='seaborn-v0_8-darkgrid',
        save_path: Optional[Path] = None,
        format='png',
        dpi=150
    ) -> plt.Figure:
        """
        Create comprehensive training curves plot.
        
        Args:
            figsize: Figure size (width, height)
            style: Matplotlib style name
            save_path: Optional path to save figure
            format: Save format ('png', 'pdf', 'svg')
            dpi: Resolution for raster formats
        """
        if not self.history:
            raise ValueError("No training history loaded")
            
        # Set style
        plt.style.use(style)
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Extract data
        steps = np.array(self.history['step'])
        train_loss = np.array(self.history['train_loss'])
        val_loss = np.array(self.history.get('val_loss', []))
        learning_rates = np.array(self.history.get('learning_rate', []))
        
        # Plot 1: Training Loss
        ax1 = axes[0]
        ax1.plot(steps, train_loss, label='Training Loss', linewidth=2, alpha=0.8)
        
        if len(val_loss) > 0:
            # Calculate validation steps (assuming val_every from config)
            val_every = self.config.get('val_every', 50)
            log_every = self.config.get('log_every', 10)
            val_steps = steps[::val_every // log_every][:len(val_loss)]
            ax1.plot(val_steps, val_loss, label='Validation Loss', linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.set_title('Training Progress')
        ax1.grid(True, alpha=0.3)
        
        # Add final loss values as text
        final_train = train_loss[-1]
        ax1.text(0.02, 0.98, f'Final Train: {final_train:.4f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        if len(val_loss) > 0:
            final_val = val_loss[-1]
            ax1.text(0.02, 0.88, f'Final Val: {final_val:.4f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Plot 2: Learning Rate Schedule
        ax2 = axes[1]
        if len(learning_rates) > 0:
            ax2.plot(steps, learning_rates, linewidth=2, color='orange')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Learning Rate')
            ax2.set_yscale('log')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
            
            # Mark LR reductions
            if len(learning_rates) > 1:
                lr_changes = np.where(np.diff(learning_rates) < 0)[0]
                for change_idx in lr_changes:
                    ax2.axvline(steps[change_idx], color='red', linestyle='--', alpha=0.6)
                    
        else:
            ax2.text(0.5, 0.5, 'No learning rate data', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Learning Rate Schedule')
        
        # Plot 3: Loss Improvement Rate
        ax3 = axes[2]
        if len(train_loss) > 10:
            # Calculate smoothed loss improvement
            window = min(50, len(train_loss) // 10)
            smooth_loss = np.convolve(train_loss, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
            
            # Calculate derivative (improvement rate)
            if len(smooth_loss) > 1:
                improvement_rate = -np.gradient(smooth_loss, smooth_steps)
                ax3.plot(smooth_steps, improvement_rate, linewidth=2, color='green')
                ax3.set_xlabel('Training Step')
                ax3.set_ylabel('Loss Improvement Rate')
                ax3.set_title('Training Speed')
                ax3.grid(True, alpha=0.3)
                
                # Mark regions of fast/slow improvement
                median_rate = np.median(improvement_rate[improvement_rate > 0])
                ax3.axhline(median_rate, color='red', linestyle='--', alpha=0.6, 
                           label=f'Median: {median_rate:.2e}')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'Insufficient data\nfor improvement rate', 
                        ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, 'Insufficient data\nfor improvement rate', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Training Speed')
        
        # Add main title with metadata
        model_info = ""
        if self.metadata:
            model_config = self.metadata.get('model_config', {})
            model_info = f" - {model_config.get('hidden_layers', '?')}×{model_config.get('hidden_features', '?')} SIREN"
        
        fig.suptitle(f'SIREN Training Results{model_info}', fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, format=format, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved training curves to: {save_path}")
        
        return fig
    
    def plot_loss_distribution(
        self,
        figsize=(12, 4),
        style='seaborn-v0_8-whitegrid',
        save_path: Optional[Path] = None,
        format='png',
        dpi=150
    ) -> plt.Figure:
        """Plot loss value distributions and statistics."""
        if not self.history:
            raise ValueError("No training history loaded")
            
        plt.style.use(style)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        train_loss = np.array(self.history['train_loss'])
        val_loss = np.array(self.history.get('val_loss', []))
        
        # Plot 1: Loss histogram
        ax1 = axes[0]
        ax1.hist(train_loss, bins=50, alpha=0.7, label='Training Loss', density=True)
        if len(val_loss) > 0:
            ax1.hist(val_loss, bins=30, alpha=0.7, label='Validation Loss', density=True)
        
        ax1.set_xlabel('Loss Value')
        ax1.set_ylabel('Density')
        ax1.set_xscale('log')
        ax1.set_title('Loss Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training phases analysis
        ax2 = axes[1]
        if len(train_loss) > 100:
            # Divide training into phases
            n_phases = 5
            phase_size = len(train_loss) // n_phases
            phase_means = []
            phase_stds = []
            phase_labels = []
            
            for i in range(n_phases):
                start_idx = i * phase_size
                end_idx = (i + 1) * phase_size if i < n_phases - 1 else len(train_loss)
                phase_losses = train_loss[start_idx:end_idx]
                
                phase_means.append(np.mean(phase_losses))
                phase_stds.append(np.std(phase_losses))
                phase_labels.append(f'Phase {i+1}')
            
            x_pos = np.arange(len(phase_labels))
            ax2.bar(x_pos, phase_means, yerr=phase_stds, capsize=5, alpha=0.7)
            ax2.set_xlabel('Training Phase')
            ax2.set_ylabel('Mean Loss')
            ax2.set_yscale('log')
            ax2.set_title('Loss by Training Phase')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(phase_labels)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data\nfor phase analysis', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Loss by Training Phase')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=format, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved loss distribution to: {save_path}")
        
        return fig
    
    def plot_training_summary(
        self,
        figsize=(16, 10),
        style='seaborn-v0_8-white',
        save_path: Optional[Path] = None,
        format='png',
        dpi=150
    ) -> plt.Figure:
        """Create a comprehensive training summary dashboard."""
        if not self.history:
            raise ValueError("No training history loaded")
            
        plt.style.use(style)
        fig = plt.figure(figsize=figsize)
        
        # Create complex subplot layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Extract data
        steps = np.array(self.history['step'])
        train_loss = np.array(self.history['train_loss'])
        val_loss = np.array(self.history.get('val_loss', []))
        learning_rates = np.array(self.history.get('learning_rate', []))
        
        # Main training curve (spans 2 columns)
        ax_main = fig.add_subplot(gs[0, :2])
        ax_main.plot(steps, train_loss, label='Training Loss', linewidth=2)
        if len(val_loss) > 0:
            val_every = self.config.get('val_every', 50)
            log_every = self.config.get('log_every', 10)
            val_steps = steps[::val_every // log_every][:len(val_loss)]
            ax_main.plot(val_steps, val_loss, label='Validation Loss', linewidth=2)
        ax_main.set_xlabel('Training Step')
        ax_main.set_ylabel('Loss')
        ax_main.set_yscale('log')
        ax_main.legend()
        ax_main.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax_main.grid(True, alpha=0.3)
        
        # Learning rate schedule
        ax_lr = fig.add_subplot(gs[0, 2])
        if len(learning_rates) > 0:
            ax_lr.plot(steps, learning_rates, color='orange', linewidth=2)
            ax_lr.set_yscale('log')
        ax_lr.set_xlabel('Step')
        ax_lr.set_ylabel('Learning Rate')
        ax_lr.set_title('LR Schedule')
        ax_lr.grid(True, alpha=0.3)
        
        # Training statistics
        ax_stats = fig.add_subplot(gs[0, 3])
        ax_stats.axis('off')
        
        # Compile statistics
        stats_text = "Training Statistics\n" + "="*20 + "\n"
        stats_text += f"Total Steps: {len(steps):,}\n"
        stats_text += f"Final Train Loss: {train_loss[-1]:.4f}\n"
        if len(val_loss) > 0:
            stats_text += f"Final Val Loss: {val_loss[-1]:.4f}\n"
            stats_text += f"Best Val Loss: {min(val_loss):.4f}\n"
        
        if self.config:
            stats_text += f"\nConfiguration\n" + "-"*15 + "\n"
            stats_text += f"Batch Size: {self.config.get('batch_size', 'N/A'):,}\n"
            stats_text += f"Learning Rate: {self.config.get('learning_rate', 'N/A')}\n"
            stats_text += f"Hidden Layers: {self.config.get('hidden_layers', 'N/A')}\n"
            stats_text += f"Hidden Features: {self.config.get('hidden_features', 'N/A')}\n"
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Loss histogram
        ax_hist = fig.add_subplot(gs[1, 0])
        ax_hist.hist(train_loss, bins=30, alpha=0.7, color='blue', density=True)
        if len(val_loss) > 0:
            ax_hist.hist(val_loss, bins=20, alpha=0.7, color='orange', density=True)
        ax_hist.set_xlabel('Loss Value')
        ax_hist.set_ylabel('Density')
        ax_hist.set_xscale('log')
        ax_hist.set_title('Loss Distribution')
        ax_hist.grid(True, alpha=0.3)
        
        # Convergence analysis
        ax_conv = fig.add_subplot(gs[1, 1])
        if len(train_loss) > 10:
            # Moving average convergence
            window_sizes = [10, 50, 100]
            for window in window_sizes:
                if len(train_loss) > window:
                    smooth_loss = np.convolve(train_loss, np.ones(window)/window, mode='valid')
                    smooth_steps = steps[window-1:]
                    ax_conv.plot(smooth_steps, smooth_loss, label=f'MA({window})', alpha=0.8)
            
            ax_conv.set_xlabel('Step')
            ax_conv.set_ylabel('Smoothed Loss')
            ax_conv.set_yscale('log')
            ax_conv.set_title('Convergence Analysis')
            ax_conv.legend()
            ax_conv.grid(True, alpha=0.3)
        else:
            ax_conv.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                        transform=ax_conv.transAxes)
            ax_conv.set_title('Convergence Analysis')
        
        # Learning rate vs loss correlation
        ax_corr = fig.add_subplot(gs[1, 2])
        if len(learning_rates) > 0 and len(learning_rates) == len(train_loss):
            # Sample points to avoid overcrowding
            sample_size = min(1000, len(train_loss))
            indices = np.linspace(0, len(train_loss)-1, sample_size, dtype=int)
            ax_corr.scatter(learning_rates[indices], train_loss[indices], alpha=0.6, s=10)
            ax_corr.set_xlabel('Learning Rate')
            ax_corr.set_ylabel('Training Loss')
            ax_corr.set_xscale('log')
            ax_corr.set_yscale('log')
            ax_corr.set_title('LR vs Loss')
            ax_corr.grid(True, alpha=0.3)
        else:
            ax_corr.text(0.5, 0.5, 'No LR data\navailable', ha='center', va='center',
                        transform=ax_corr.transAxes)
            ax_corr.set_title('LR vs Loss')
        
        # Training efficiency
        ax_eff = fig.add_subplot(gs[1, 3])
        if len(train_loss) > 50:
            # Calculate loss improvement per step
            step_diffs = np.diff(steps)
            loss_diffs = np.diff(train_loss)
            improvement_rate = -loss_diffs / step_diffs  # Negative because loss should decrease
            
            # Use only positive improvements for histogram
            positive_improvements = improvement_rate[improvement_rate > 0]
            if len(positive_improvements) > 10:
                ax_eff.hist(np.log10(positive_improvements), bins=30, alpha=0.7, color='green')
                ax_eff.set_xlabel('log10(Improvement Rate)')
                ax_eff.set_ylabel('Frequency')
                ax_eff.set_title('Training Efficiency')
                ax_eff.grid(True, alpha=0.3)
            else:
                ax_eff.text(0.5, 0.5, 'Limited\nimprovement data', ha='center', va='center',
                           transform=ax_eff.transAxes)
                ax_eff.set_title('Training Efficiency')
        else:
            ax_eff.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                       transform=ax_eff.transAxes)
            ax_eff.set_title('Training Efficiency')
        
        # Timeline overview (spans full width)
        ax_timeline = fig.add_subplot(gs[2, :])
        
        # Create timeline with different phases
        ax_timeline.plot(steps, train_loss, linewidth=1, alpha=0.7, color='blue')
        
        # Mark important events
        if len(learning_rates) > 1:
            lr_changes = np.where(np.diff(learning_rates) < 0)[0]
            for i, change_idx in enumerate(lr_changes):
                ax_timeline.axvline(steps[change_idx], color='red', linestyle='--', alpha=0.8)
                ax_timeline.text(steps[change_idx], train_loss[change_idx], 
                               f'LR↓{i+1}', rotation=90, va='bottom', ha='right')
        
        # Mark training phases
        if len(train_loss) > 500:
            phase_boundaries = [len(train_loss)//4, len(train_loss)//2, 3*len(train_loss)//4]
            phase_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
            phase_names = ['Initial', 'Early', 'Middle', 'Final']
            
            prev_x = 0
            for i, boundary in enumerate(phase_boundaries + [len(steps)]):
                ax_timeline.axvspan(steps[prev_x], steps[boundary-1], 
                                  alpha=0.2, color=phase_colors[i], label=phase_names[i])
                prev_x = boundary
            
            ax_timeline.legend(loc='upper right', fontsize=8)
        
        ax_timeline.set_xlabel('Training Step')
        ax_timeline.set_ylabel('Training Loss')
        ax_timeline.set_yscale('log')
        ax_timeline.set_title('Training Timeline Overview', fontsize=12, fontweight='bold')
        ax_timeline.grid(True, alpha=0.3)
        
        # Main title
        title = "SIREN Training Dashboard"
        if self.training_dir.name:
            title += f" - {self.training_dir.name}"
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.95)
        
        if save_path:
            plt.savefig(save_path, format=format, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved training summary to: {save_path}")
        
        return fig
    
    def plot_training_timeline(
        self,
        figsize=(5, 3),
        style='seaborn-v0_8-darkgrid',
        save_path: Optional[Path] = None,
        format='png',
        dpi=300
    ) -> plt.Figure:
        """
        Create a focused training timeline plot with both training and validation loss.
        
        Args:
            figsize: Figure size (width, height)
            style: Matplotlib style name
            save_path: Optional path to save figure
            format: Save format ('png', 'pdf', 'svg')
            dpi: Resolution for raster formats
        """
        if not self.history:
            raise ValueError("No training history loaded")
            
        # Use default matplotlib style
        plt.style.use('default')
        
        # Set LaTeX rendering after style reset
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Extract data
        steps = np.array(self.history['step'])
        train_loss = np.array(self.history['train_loss'])
        val_loss = np.array(self.history.get('val_loss', []))
        learning_rates = np.array(self.history.get('learning_rate', []))
        
        # Plot training loss
        ax.plot(steps, train_loss, linewidth=2, alpha=0.8, color='blue', label='Training')
        
        # Plot validation loss if available
        if len(val_loss) > 0:
            val_every = self.config.get('val_every', 50)
            log_every = self.config.get('log_every', 10)
            val_steps = steps[::val_every // log_every][:len(val_loss)]
            ax.plot(val_steps, val_loss, linewidth=2, alpha=0.8, color='orange', label='Validation')
        
        # Mark learning rate reductions (simplified)
        if len(learning_rates) > 1:
            lr_changes = np.where(np.diff(learning_rates) < 0)[0]
            for change_idx in lr_changes:
                ax.axvline(steps[change_idx], color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Formatting
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_yscale('log')
        #ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='upper right', frameon=False, handlelength=1)
        
        # # Simple title
        # ax.set_title("Training Progress", fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=format, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved training timeline to: {save_path}")
        
        return fig
    
    def generate_all_plots(
        self, 
        output_dir: Optional[Path] = None,
        format='png',
        dpi=150,
        style='seaborn-v0_8-darkgrid'
    ):
        """Generate all available plots and save them."""
        if output_dir is None:
            output_dir = self.training_dir / 'plots'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\nGenerating all plots in: {output_dir}")
        print("="*50)
        
        # Training curves
        try:
            fig1 = self.plot_training_curves(
                style=style,
                save_path=output_dir / f'training_curves.{format}',
                format=format, dpi=dpi
            )
            plt.close(fig1)
        except Exception as e:
            print(f"✗ Failed to generate training curves: {e}")
        
        # Loss distribution
        try:
            fig2 = self.plot_loss_distribution(
                style=style,
                save_path=output_dir / f'loss_distribution.{format}',
                format=format, dpi=dpi
            )
            plt.close(fig2)
        except Exception as e:
            print(f"✗ Failed to generate loss distribution: {e}")
        
        # Training summary dashboard
        try:
            fig3 = self.plot_training_summary(
                style=style,
                save_path=output_dir / f'training_summary.{format}',
                format=format, dpi=dpi
            )
            plt.close(fig3)
        except Exception as e:
            print(f"✗ Failed to generate training summary: {e}")
        
        # Training timeline
        try:
            fig4 = self.plot_training_timeline(
                style=style,
                save_path=output_dir / f'training_timeline.{format}',
                format=format, dpi=dpi
            )
            plt.close(fig4)
        except Exception as e:
            print(f"✗ Failed to generate training timeline: {e}")
        
        print("\n✓ Plot generation complete!")
        print(f"Find all plots in: {output_dir}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='Visualize SIREN training results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--training-dir', type=str, required=True,
                        help='Directory containing training results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: training-dir/plots)')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-darkgrid',
                        help='Matplotlib style (default: seaborn-v0_8-darkgrid)')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'],
                        help='Output format (default: png)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Resolution for raster formats (default: 150)')
    parser.add_argument('--plot-type', type=str, default='all',
                        choices=['all', 'curves', 'distribution', 'summary', 'timeline'],
                        help='Type of plot to generate (default: all)')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    training_dir = Path(args.training_dir)
    if not training_dir.exists():
        print(f"Error: Training directory not found: {training_dir}")
        return 1
    
    try:
        visualizer = SIRENTrainingVisualizer(training_dir)
    except Exception as e:
        print(f"Error initializing visualizer: {e}")
        return 1
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else training_dir / 'plots'
    
    # Generate requested plots
    if args.plot_type == 'all':
        visualizer.generate_all_plots(
            output_dir=output_dir,
            format=args.format,
            dpi=args.dpi,
            style=args.style
        )
    else:
        # Generate specific plot type
        if args.plot_type == 'curves':
            fig = visualizer.plot_training_curves(
                style=args.style,
                save_path=output_dir / f'training_curves.{args.format}',
                format=args.format, dpi=args.dpi
            )
        elif args.plot_type == 'distribution':
            fig = visualizer.plot_loss_distribution(
                style=args.style,
                save_path=output_dir / f'loss_distribution.{args.format}',
                format=args.format, dpi=args.dpi
            )
        elif args.plot_type == 'summary':
            fig = visualizer.plot_training_summary(
                style=args.style,
                save_path=output_dir / f'training_summary.{args.format}',
                format=args.format, dpi=args.dpi
            )
        elif args.plot_type == 'timeline':
            fig = visualizer.plot_training_timeline(
                style=args.style,
                save_path=output_dir / f'training_timeline.{args.format}',
                format=args.format, dpi=args.dpi
            )
        
        plt.show()  # Show the plot
        plt.close(fig)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())