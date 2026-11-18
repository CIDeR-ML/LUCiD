"""
Analysis Module for Trained SIREN Models

This module provides classes for analyzing and evaluating trained SIREN models
on PhotonSim data, designed to be used from Jupyter notebooks or scripts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

logger = logging.getLogger(__name__)


class TrainingAnalyzer:
    """
    Analyzer for evaluating trained SIREN models.
    
    Example usage in notebook:
    ```python
    from lucid.siren.training.analyzer import TrainingAnalyzer
    
    analyzer = TrainingAnalyzer(trainer, dataset)
    results = analyzer.evaluate_model()
    analyzer.plot_comparison()
    ```
    """
    
    def __init__(self, trainer, dataset):
        """
        Initialize analyzer.
        
        Args:
            trainer: Trained SIRENTrainer instance
            dataset: Dataset used for training
        """
        self.trainer = trainer
        self.dataset = dataset
        
        # Results storage
        self.evaluation_results = {}
        self.comparison_data = {}
        
    def evaluate_model(
        self,
        n_samples: int = 10000,
        splits: List[str] = ['train', 'val'],
        return_raw: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the trained model.
        
        Args:
            n_samples: Number of samples to evaluate on
            splits: Data splits to evaluate ('train', 'val', 'all')
            return_raw: Whether to return raw predictions and targets
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating model on {n_samples} samples...")
        
        results = {}
        
        for split in splits:
            logger.info(f"Evaluating on {split} split...")
            
            # Get data
            inputs, targets = self.dataset.get_full_data(split=split, normalized=True)
            
            # Sample if needed
            if len(inputs) > n_samples:
                indices = np.random.choice(len(inputs), n_samples, replace=False)
                inputs = inputs[indices]
                targets = targets[indices]
                
            # Make predictions
            predictions = self.trainer.predict(inputs)
            
            # Calculate metrics
            metrics = self._calculate_metrics(targets, predictions)
            
            # Store results
            results[split] = {
                'metrics': metrics,
                'n_samples': len(inputs)
            }
            
            if return_raw:
                results[split].update({
                    'inputs': inputs,
                    'targets': targets,
                    'predictions': predictions
                })
                
            logger.info(f"{split} metrics: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.6f}")
            
        self.evaluation_results = results
        return results
        
    def _calculate_metrics(
        self, 
        targets: np.ndarray, 
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        # Flatten arrays
        targets = targets.flatten()
        predictions = predictions.flatten()
        
        # Basic metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Correlation coefficient
        correlation = np.corrcoef(targets, predictions)[0, 1]
        
        # Relative metrics
        relative_error = np.mean(np.abs(predictions - targets) / (np.abs(targets) + 1e-8))
        
        # Percentile errors
        errors = np.abs(predictions - targets)
        error_percentiles = {
            'p50': np.percentile(errors, 50),
            'p90': np.percentile(errors, 90),
            'p95': np.percentile(errors, 95),
            'p99': np.percentile(errors, 99)
        }
        
        # Log-scale metrics (if applicable)
        if np.all(targets > 0) and np.all(predictions > 0):
            log_targets = np.log10(targets)
            log_predictions = np.log10(predictions)
            log_rmse = np.sqrt(np.mean((log_predictions - log_targets) ** 2))
            log_mae = np.mean(np.abs(log_predictions - log_targets))
        else:
            log_rmse = np.nan
            log_mae = np.nan
            
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'correlation': correlation,
            'relative_error': relative_error,
            'log_rmse': log_rmse,
            'log_mae': log_mae,
            **error_percentiles
        }
        
    def analyze_error_patterns(
        self,
        split: str = 'val',
        n_samples: int = 10000
    ) -> Dict[str, Any]:
        """
        Analyze error patterns across different input dimensions.
        
        Args:
            split: Data split to analyze
            n_samples: Number of samples to analyze
            
        Returns:
            Dictionary with error analysis results
        """
        logger.info(f"Analyzing error patterns on {split} split...")
        
        # Get data
        inputs, targets = self.dataset.get_full_data(split=split, normalized=True)
        
        # Sample if needed
        if len(inputs) > n_samples:
            indices = np.random.choice(len(inputs), n_samples, replace=False)
            inputs = inputs[indices]
            targets = targets[indices]
            
        # Make predictions
        predictions = self.trainer.predict(inputs)
        
        # Calculate errors
        errors = np.abs(predictions.flatten() - targets.flatten())
        relative_errors = errors / (np.abs(targets.flatten()) + 1e-8)
        
        # Denormalize inputs for analysis
        inputs_denorm = self.dataset.denormalize_inputs(inputs)
        
        # Analyze errors by input dimensions
        error_analysis = {
            'energy_analysis': self._analyze_errors_by_dimension(
                inputs_denorm[:, 0], errors, 'Energy (MeV)'
            ),
            'angle_analysis': self._analyze_errors_by_dimension(
                np.degrees(inputs_denorm[:, 1]), errors, 'Angle (degrees)'
            ),
            'distance_analysis': self._analyze_errors_by_dimension(
                inputs_denorm[:, 2], errors, 'Distance (mm)'
            ),
            'target_magnitude_analysis': self._analyze_errors_by_dimension(
                targets.flatten(), errors, 'Target Value'
            )
        }
        
        # Store for plotting
        self.comparison_data = {
            'inputs': inputs,
            'inputs_denorm': inputs_denorm,
            'targets': targets,
            'predictions': predictions,
            'errors': errors,
            'relative_errors': relative_errors
        }
        
        return error_analysis
        
    def _analyze_errors_by_dimension(
        self,
        dimension_values: np.ndarray,
        errors: np.ndarray,
        dimension_name: str,
        n_bins: int = 20
    ) -> Dict[str, Any]:
        """Analyze how errors vary across a specific input dimension."""
        # Create bins
        bins = np.linspace(dimension_values.min(), dimension_values.max(), n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate error statistics per bin
        bin_indices = np.digitize(dimension_values, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_errors = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if np.any(mask):
                bin_errors.append(errors[mask])
                bin_counts.append(np.sum(mask))
            else:
                bin_errors.append(np.array([]))
                bin_counts.append(0)
                
        # Calculate statistics
        mean_errors = [np.mean(be) if len(be) > 0 else 0 for be in bin_errors]
        std_errors = [np.std(be) if len(be) > 0 else 0 for be in bin_errors]
        median_errors = [np.median(be) if len(be) > 0 else 0 for be in bin_errors]
        
        return {
            'dimension_name': dimension_name,
            'bin_centers': bin_centers,
            'bin_counts': bin_counts,
            'mean_errors': mean_errors,
            'std_errors': std_errors,
            'median_errors': median_errors,
            'dimension_range': (dimension_values.min(), dimension_values.max())
        }
        
    def plot_comparison(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (20, 15)
    ) -> plt.Figure:
        """
        Create comprehensive comparison plots.
        
        Args:
            save_path: Optional path to save plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.comparison_data:
            self.analyze_error_patterns()
            
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        fig.suptitle('SIREN Model Analysis', fontsize=16)
        
        inputs_denorm = self.comparison_data['inputs_denorm']
        targets = self.comparison_data['targets'].flatten()
        predictions = self.comparison_data['predictions'].flatten()
        errors = self.comparison_data['errors']
        
        # 1. Predictions vs Targets (scatter)
        ax = axes[0, 0]
        self._plot_predictions_vs_targets(ax, targets, predictions)
        
        # 2. Predictions vs Targets (density)
        ax = axes[0, 1]
        self._plot_predictions_vs_targets_density(ax, targets, predictions)
        
        # 3. Error distribution
        ax = axes[0, 2]
        ax.hist(np.log10(errors + 1e-12), bins=50, alpha=0.7, color='red')
        ax.set_xlabel('log10(Absolute Error)')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.grid(True, alpha=0.3)
        
        # 4. Relative error distribution
        ax = axes[0, 3]
        rel_errors = self.comparison_data['relative_errors']
        ax.hist(np.log10(rel_errors + 1e-12), bins=50, alpha=0.7, color='orange')
        ax.set_xlabel('log10(Relative Error)')
        ax.set_ylabel('Frequency')
        ax.set_title('Relative Error Distribution')
        ax.grid(True, alpha=0.3)
        
        # 5-7. Error vs Input Dimensions
        dimension_names = ['Energy (MeV)', 'Angle (degrees)', 'Distance (mm)']
        input_data = [
            inputs_denorm[:, 0],
            np.degrees(inputs_denorm[:, 1]),
            inputs_denorm[:, 2]
        ]
        
        for i, (name, data) in enumerate(zip(dimension_names, input_data)):
            ax = axes[1, i]
            self._plot_error_vs_dimension(ax, data, errors, name)
            
        # 8. Error vs Target Magnitude
        ax = axes[1, 3]
        self._plot_error_vs_dimension(ax, targets, errors, 'Target Value')
        
        # 9-11. Residual analysis
        residuals = predictions - targets
        
        # QQ plot
        ax = axes[2, 0]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Residuals Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        # Residuals vs predictions
        ax = axes[2, 1]
        ax.scatter(predictions, residuals, alpha=0.5, s=1)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel('Predictions')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predictions')
        ax.grid(True, alpha=0.3)
        
        # Error correlation matrix (if multiple outputs)
        ax = axes[2, 2]
        if len(self.evaluation_results) > 1:
            splits = list(self.evaluation_results.keys())
            metrics = ['r2', 'rmse', 'mae', 'relative_error']
            
            corr_data = []
            for split in splits:
                split_metrics = self.evaluation_results[split]['metrics']
                corr_data.append([split_metrics.get(m, 0) for m in metrics])
                
            corr_matrix = np.corrcoef(np.array(corr_data).T)
            vmax = np.abs(corr_matrix).max()
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-vmax, vmax=vmax)
            for (i, j), value in np.ndenumerate(corr_matrix):
                ax.text(j, i, f'{value:.2f}', ha='center', va='center')
            ax.set_xticks(range(len(metrics)), metrics)
            ax.set_yticks(range(len(metrics)), metrics)
            fig.colorbar(im, ax=ax)
            ax.set_title('Metric Correlations')
        else:
            ax.text(0.5, 0.5, 'Multiple splits needed\\nfor correlation analysis',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Metric Correlations')
            
        # Summary statistics
        ax = axes[2, 3]
        ax.axis('off')
        summary_text = self._format_summary_stats()
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=8, family='monospace', verticalalignment='top',
                wrap=True, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3))
        ax.set_title('Summary Statistics', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved analysis plot to {save_path}")
            
        return fig
        
    def _plot_predictions_vs_targets(self, ax, targets, predictions):
        """Plot predictions vs targets scatter plot."""
        # Sample for visualization if too many points
        if len(targets) > 10000:
            indices = np.random.choice(len(targets), 10000, replace=False)
            targets_plot = targets[indices]
            predictions_plot = predictions[indices]
        else:
            targets_plot = targets
            predictions_plot = predictions
            
        ax.scatter(targets_plot, predictions_plot, alpha=0.5, s=1)
        
        # Perfect prediction line
        min_val = min(targets_plot.min(), predictions_plot.min())
        max_val = max(targets_plot.max(), predictions_plot.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel('Targets')
        ax.set_ylabel('Predictions')
        ax.set_title('Predictions vs Targets')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R² to plot
        if self.evaluation_results:
            r2 = self.evaluation_results.get('val', {}).get('metrics', {}).get('r2', 0)
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                   
    def _plot_predictions_vs_targets_density(self, ax, targets, predictions):
        """Plot predictions vs targets as density plot."""
        ax.hexbin(targets, predictions, gridsize=50, cmap='Blues', mincnt=1)
        
        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax.set_xlabel('Targets')
        ax.set_ylabel('Predictions')
        ax.set_title('Predictions vs Targets (Density)')
        
    def _plot_error_vs_dimension(self, ax, dimension_values, errors, dimension_name):
        """Plot error vs input dimension."""
        # Sample for visualization
        if len(dimension_values) > 10000:
            indices = np.random.choice(len(dimension_values), 10000, replace=False)
            dim_plot = dimension_values[indices]
            err_plot = errors[indices]
        else:
            dim_plot = dimension_values
            err_plot = errors
            
        ax.scatter(dim_plot, err_plot, alpha=0.5, s=1)
        ax.set_xlabel(dimension_name)
        ax.set_ylabel('Absolute Error')
        ax.set_title(f'Error vs {dimension_name}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(dim_plot, np.log10(err_plot + 1e-12), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(dim_plot.min(), dim_plot.max(), 100)
        y_trend = 10 ** p(x_trend)
        ax.plot(x_trend, y_trend, 'r-', linewidth=2, alpha=0.7, label='Trend')
        ax.legend()
            
    def _format_summary_stats(self) -> str:
        """Format summary statistics as text."""
        if not self.evaluation_results:
            return "No evaluation results available"
            
        lines = ["Summary Statistics:", ""]
        
        for split, results in self.evaluation_results.items():
            metrics = results['metrics']
            lines.extend([
                f"{split.upper()} Split:",
                f"  R²: {metrics['r2']:.4f}",
                f"  RMSE: {metrics['rmse']:.6f}",
                f"  MAE: {metrics['mae']:.6f}",
                f"  Rel. Err: {metrics['relative_error']:.4f}",
                f"  Corr: {metrics['correlation']:.4f}",
                f"  N: {results['n_samples']:,}",
                ""
            ])
            
        return "\n".join(lines)
