"""
Training Monitor Module for SIREN Training

This module provides classes for monitoring training progress in real-time,
designed to be used from Jupyter notebooks or scripts.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import threading

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    Monitor for tracking SIREN training progress.
    
    Example usage in notebook:
    ```python
    from diffCherenkov.siren.training import TrainingMonitor
    
    monitor = TrainingMonitor('output/training_run')
    monitor.start_monitoring(update_interval=10)  # Update every 10 seconds
    
    # Or for one-time monitoring:
    monitor.plot_progress()
    ```
    """
    
    def __init__(self, training_dir: Path, live_plotting: bool = True):
        """
        Initialize training monitor.
        
        Args:
            training_dir: Directory containing training outputs
            live_plotting: Whether to enable live plotting in notebooks
        """
        self.training_dir = Path(training_dir)
        self.live_plotting = live_plotting
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Progress data
        self.progress = {
            'steps': [],
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'timestamps': [],
            'last_update': None
        }
        
        # Configuration
        self.config = {}
        self._load_config()
        
    def _load_config(self):
        """Load training configuration if available."""
        config_file = self.training_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                self.config = json.load(f)
                
    def load_progress(self) -> Dict[str, Any]:
        """
        Load current training progress from saved files.
        
        Returns:
            Dictionary with training progress data
        """
        # Load training history if available
        history_file = self.training_dir / "training_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                
            self.progress.update({
                'steps': history.get('step', []),
                'train_losses': history.get('train_loss', []),
                'val_losses': history.get('val_loss', []),
                'learning_rates': history.get('learning_rate', []),
                'last_update': time.time()
            })
            
        return self.progress
        
    def get_latest_metrics(self) -> Dict[str, float]:
        """
        Get latest training metrics.
        
        Returns:
            Dictionary with latest metrics
        """
        metrics = {}
        
        if self.progress['train_losses']:
            metrics['latest_train_loss'] = self.progress['train_losses'][-1]
            
        if self.progress['val_losses']:
            metrics['latest_val_loss'] = self.progress['val_losses'][-1]
            
        if self.progress['learning_rates']:
            metrics['current_lr'] = self.progress['learning_rates'][-1]
            
        if len(self.progress['train_losses']) > 100:
            # Calculate recent improvement
            recent_losses = self.progress['train_losses'][-100:]
            improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0] * 100
            metrics['recent_improvement_pct'] = improvement
            
        return metrics
        
    def estimate_remaining_time(self) -> Optional[float]:
        """
        Estimate remaining training time based on current progress.
        
        Returns:
            Estimated remaining time in seconds, or None if can't estimate
        """
        if not self.progress['steps'] or not self.config.get('num_steps'):
            return None
            
        current_step = max(self.progress['steps'])
        total_steps = self.config['num_steps']
        
        if current_step == 0:
            return None
            
        # Estimate based on elapsed time per step
        if self.progress['timestamps']:
            elapsed_time = time.time() - min(self.progress['timestamps'])
            time_per_step = elapsed_time / current_step
            remaining_steps = total_steps - current_step
            return remaining_steps * time_per_step
            
        return None
        
    def plot_progress(
        self, 
        save_path: Optional[Path] = None,
        show_recent: int = 1000,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Plot training progress.
        
        Args:
            save_path: Optional path to save plot
            show_recent: Number of recent steps to highlight
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Update progress
        self.load_progress()
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'SIREN Training Progress - {self.training_dir.name}', fontsize=16)
        
        # Training loss
        ax = axes[0, 0]
        if self.progress['train_losses']:
            steps = self.progress['steps']
            losses = self.progress['train_losses']
            
            ax.plot(steps, losses, 'b-', alpha=0.7, linewidth=1, label='Training')
            
            # Highlight recent progress
            if len(steps) > show_recent:
                recent_steps = steps[-show_recent:]
                recent_losses = losses[-show_recent:]
                ax.plot(recent_steps, recent_losses, 'b-', linewidth=2, alpha=1.0)
                
            ax.set_title('Training Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Show improvement
            if len(losses) > 100:
                recent_losses = losses[-100:]
                improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0] * 100
                ax.text(0.02, 0.98, f'Recent improvement: {improvement:.2f}%',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                       
        # Validation loss
        ax = axes[0, 1]
        if self.progress['val_losses']:
            # Assuming validation is evaluated at regular intervals
            val_steps = np.linspace(0, len(self.progress['steps'])-1, 
                                  len(self.progress['val_losses']), dtype=int)
            val_steps = np.array(self.progress['steps'])[val_steps]
            
            ax.plot(val_steps, self.progress['val_losses'], 'o-', 
                   color='orange', linewidth=2, markersize=4, label='Validation')
            ax.set_title('Validation Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
        # Learning rate schedule
        ax = axes[0, 2]
        if self.progress['learning_rates']:
            ax.plot(self.progress['steps'], self.progress['learning_rates'], 
                   'g-', linewidth=2, label='Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
        # Training statistics
        ax = axes[1, 0]
        ax.axis('off')
        stats_text = self._format_stats()
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, family='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))
        ax.set_title('Training Statistics', fontsize=10)
        
        # Loss distribution (recent)
        ax = axes[1, 1]
        if len(self.progress['train_losses']) > 100:
            recent_losses = self.progress['train_losses'][-1000:]
            ax.hist(np.log10(recent_losses), bins=50, alpha=0.7, color='blue')
            ax.set_title('Recent Loss Distribution (log10)')
            ax.set_xlabel('log10(Loss)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
        # Training speed
        ax = axes[1, 2]
        if len(self.progress['steps']) > 1:
            # Calculate steps per second (approximate)
            time_estimate = self.estimate_remaining_time()
            if time_estimate is not None:
                current_step = max(self.progress['steps'])
                total_steps = self.config.get('num_steps', current_step)
                
                progress_pct = current_step / total_steps * 100
                
                ax.barh(['Progress'], [progress_pct], color='green', alpha=0.7)
                ax.set_xlim(0, 100)
                ax.set_xlabel('Progress (%)')
                ax.set_title(f'Progress: {progress_pct:.1f}%')
                
                # Add time estimate
                hours, remainder = divmod(time_estimate, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                ax.text(0.5, 0.5, f'ETA: {time_str}', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                       
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training plot to {save_path}")
            
        return fig
        
    def _format_stats(self) -> str:
        """Format training statistics as text."""
        metrics = self.get_latest_metrics()
        config_stats = []
        
        # Configuration info
        if self.config:
            config_stats.extend([
                f"Features: {self.config.get('hidden_features', 'N/A')}",
                f"Layers: {self.config.get('hidden_layers', 'N/A')}",
                f"LR: {self.config.get('learning_rate', 'N/A'):.2e}",
                f"w0: {self.config.get('w0', 'N/A')}",
                f"Batch: {self.config.get('batch_size', 'N/A'):,}",
                ""
            ])
            
        # Current metrics
        metric_stats = []
        if metrics:
            if 'latest_train_loss' in metrics:
                metric_stats.append(f"Train Loss: {metrics['latest_train_loss']:.6f}")
            if 'latest_val_loss' in metrics:
                metric_stats.append(f"Val Loss: {metrics['latest_val_loss']:.6f}")
            if 'current_lr' in metrics:
                metric_stats.append(f"LR: {metrics['current_lr']:.2e}")
            if 'recent_improvement_pct' in metrics:
                metric_stats.append(f"Improve: {metrics['recent_improvement_pct']:.1f}%")
                
        # Progress info
        progress_stats = []
        if self.progress['steps']:
            current_step = max(self.progress['steps'])
            total_steps = self.config.get('num_steps', current_step)
            progress_stats.extend([
                f"",
                f"Step: {current_step:,}",
                f"Total: {total_steps:,}",
                f"Points: {len(self.progress['steps']):,}"
            ])
            
        return "\n".join(config_stats + metric_stats + progress_stats)
        
    def start_monitoring(self, update_interval: float = 10.0):
        """
        Start monitoring training progress in real-time.
        
        Args:
            update_interval: Update interval in seconds
        """
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    if self.live_plotting:
                        clear_output(wait=True)
                        fig = self.plot_progress()
                        plt.show()
                        
                    time.sleep(update_interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(update_interval)
                    
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Started monitoring with {update_interval}s intervals")
        
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        logger.info("Stopped monitoring")
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of training progress.
        
        Returns:
            Dictionary with training summary
        """
        self.load_progress()
        
        summary = {
            'config': self.config,
            'metrics': self.get_latest_metrics(),
            'progress': {
                'current_step': max(self.progress['steps']) if self.progress['steps'] else 0,
                'total_steps': self.config.get('num_steps', 0),
                'data_points': len(self.progress['steps'])
            }
        }
        
        # Add time estimates
        eta = self.estimate_remaining_time()
        if eta:
            summary['time_estimates'] = {
                'eta_seconds': eta,
                'eta_formatted': self._format_time(eta)
            }
            
        return summary
        
    def _format_time(self, seconds: float) -> str:
        """Format time duration as string."""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
    def export_data(self, output_path: Path):
        """
        Export training progress data to file.
        
        Args:
            output_path: Path to save data
        """
        self.load_progress()
        
        export_data = {
            'progress': self.progress,
            'config': self.config,
            'summary': self.get_summary(),
            'export_timestamp': time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported training data to {output_path}")


class LiveTrainingCallback:
    """
    Callback for live training monitoring that can be added to trainer.
    
    Example usage:
    ```python
    monitor = TrainingMonitor('output/training')
    callback = LiveTrainingCallback(monitor, update_every=100)
    trainer.add_callback(callback)
    ```
    """
    
    def __init__(
        self, 
        monitor: TrainingMonitor, 
        update_every: int = 100,
        plot_every: int = 500
    ):
        """
        Initialize live callback.
        
        Args:
            monitor: TrainingMonitor instance
            update_every: Update monitor data every N steps
            plot_every: Update plot every N steps
        """
        self.monitor = monitor
        self.update_every = update_every
        self.plot_every = plot_every
        
    def __call__(self, trainer, step: int):
        """Callback function called during training."""
        if step % self.update_every == 0:
            # Update monitor with current data
            self.monitor.progress.update({
                'steps': trainer.history['step'],
                'train_losses': trainer.history['train_loss'],
                'val_losses': trainer.history.get('val_loss', []),
                'learning_rates': trainer.history.get('learning_rate', []),
                'last_update': time.time()
            })
            
        if step % self.plot_every == 0 and self.monitor.live_plotting:
            clear_output(wait=True)
            fig = self.monitor.plot_progress()
            plt.show()