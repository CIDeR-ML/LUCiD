"""
Analysis Module for Trained SIREN Models

This module provides classes for analyzing and evaluating trained SIREN models
on PhotonSim data, designed to be used from Jupyter notebooks or scripts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class TrainingAnalyzer:
    """
    Analyzer for evaluating trained SIREN models.
    
    Example usage in notebook:
    ```python
    from diffCherenkov.siren.training import TrainingAnalyzer
    
    analyzer = TrainingAnalyzer(trainer, dataset)
    results = analyzer.evaluate_model()
    analyzer.plot_comparison()
    ```
    """
    
    def __init__(
        self,
        trainer=None,
        dataset=None,
        model_path: Optional[Path] = None
    ):
        """
        Initialize analyzer.
        
        Args:
            trainer: Trained SIRENTrainer instance
            dataset: Dataset used for training
            model_path: Optional path to saved model (if trainer not provided)
        """
        self.trainer = trainer
        self.dataset = dataset
        self.model_path = model_path
        
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
            sns.heatmap(corr_matrix, annot=True, xticklabels=metrics, 
                       yticklabels=metrics, ax=ax, cmap='coolwarm', center=0)
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
        if hasattr(self, 'evaluation_results') and self.evaluation_results:
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
        try:
            z = np.polyfit(dim_plot, np.log10(err_plot + 1e-12), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(dim_plot.min(), dim_plot.max(), 100)
            y_trend = 10 ** p(x_trend)
            ax.plot(x_trend, y_trend, 'r-', linewidth=2, alpha=0.7, label='Trend')
            ax.legend()
        except:
            pass
            
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
        
    def export_results(self, output_path: Path):
        """
        Export analysis results to file.
        
        Args:
            output_path: Path to save results
        """
        import json
        
        export_data = {
            'evaluation_results': self.evaluation_results,
            'model_config': self.trainer.config.__dict__ if self.trainer else {},
            'dataset_info': {
                'data_type': getattr(self.dataset, 'data_type', 'unknown'),
                'n_samples': len(self.dataset.data.get('inputs', [])),
                'input_ranges': getattr(self.dataset, 'normalized_bounds', {})
            }
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.int32):
                return int(obj)
            return obj
            
        def serialize_dict(d):
            if isinstance(d, dict):
                return {k: serialize_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [serialize_dict(item) for item in d]
            else:
                return convert_numpy(d)
                
        serialized_data = serialize_dict(export_data)
        
        with open(output_path, 'w') as f:
            json.dump(serialized_data, f, indent=2)
            
        logger.info(f"Exported analysis results to {output_path}")
        
    def plot_lookup_table_slices(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (20, 12),
        n_samples_pred: int = 10000
    ) -> plt.Figure:
        """
        Plot angular profiles and 2D histograms for different energies.
        
        Shows:
        1. Angular profiles (distance averaged) for energies 200, 400, 600, 800, 1000 MeV
        2. Side-by-side 2D histograms (angle vs distance) for lookup table and SIREN predictions
        
        Args:
            save_path: Optional path to save plot
            figsize: Figure size
            n_samples_pred: Number of samples for prediction evaluation
            
        Returns:
            Matplotlib figure
        """
        if not self.comparison_data:
            self.analyze_error_patterns()
            
        # Get data
        inputs_denorm = self.comparison_data['inputs_denorm']
        targets = self.comparison_data['targets'].flatten()
        predictions = self.comparison_data['predictions'].flatten()
        
        # Create figure with subplots: 2 rows, 5 columns
        fig, axes = plt.subplots(2, 5, figsize=figsize)
        fig.suptitle('Lookup Table vs SIREN Model: Angular Profiles and 2D Comparisons', fontsize=16)
        
        # Define energy values
        energies = [200, 400, 600, 800, 1000]  # MeV
        
        # Row 1: Angular profiles (distance averaged) for each energy
        for i, energy in enumerate(energies):
            ax = axes[0, i]
            self._plot_angular_profile_for_energy(ax, energy, inputs_denorm, targets, predictions)
            
        # Row 2: 2D histograms (angle vs distance) for each energy
        for i, energy in enumerate(energies):
            ax = axes[1, i]
            self._plot_2d_angle_distance_for_energy(ax, energy, inputs_denorm, targets, predictions)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved slice comparison plot to {save_path}")
        
        return fig
        
    def _plot_angular_profile_for_energy(self, ax, energy, inputs_denorm, targets, predictions):
        """Plot angular profile (distance averaged) for a specific energy using original lookup table grid."""
        
        # Use the original lookup table grid from the dataset
        if not hasattr(self.dataset, 'data_type') or self.dataset.data_type != 'h5_lookup':
            ax.text(0.5, 0.5, 'Lookup table grid\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{energy} MeV')
            return
            
        # Load the original HDF5 file to get the full lookup table grid
        import h5py
        with h5py.File(self.dataset.data_path, 'r') as f:
            # Load full density table and coordinates
            density_table = f['data/photon_table_density'][:]  # Shape: (n_energy, n_angle, n_distance)
            energy_centers = f['coordinates/energy_centers'][:]
            angle_centers = f['coordinates/angle_centers'][:]
            distance_centers = f['coordinates/distance_centers'][:]
        
        # Find the closest energy index
        energy_idx = np.argmin(np.abs(energy_centers - energy))
        actual_energy = energy_centers[energy_idx]
        
        if np.abs(actual_energy - energy) > 100:  # More than 100 MeV difference
            ax.text(0.5, 0.5, f'No data near\n{energy} MeV', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{energy} MeV')
            return
        
        # Get the 2D slice for this energy and average over distance
        table_slice = density_table[energy_idx, :, :]  # Shape: (n_angle, n_distance)
        
        # Average over distance for each angle (distance-averaged angular profile)
        # Only average where values are significant
        mask_3d = table_slice > 1e-10
        table_profile = np.zeros(len(angle_centers))
        
        for i in range(len(angle_centers)):
            valid_distances = mask_3d[i, :]
            if np.sum(valid_distances) > 0:
                table_profile[i] = np.mean(table_slice[i, valid_distances])
            else:
                table_profile[i] = np.nan
        
        # Create coordinate grid for SIREN evaluation at this energy
        angle_grid = angle_centers
        distance_grid = distance_centers
        
        # Create full 2D grid
        angle_mesh, distance_mesh = np.meshgrid(angle_grid, distance_grid, indexing='ij')
        energy_grid = np.full_like(angle_mesh, actual_energy)
        
        # Flatten for SIREN prediction
        eval_coords = np.stack([
            energy_grid.flatten(),
            angle_mesh.flatten(),
            distance_mesh.flatten()
        ], axis=-1)
        
        # Normalize coordinates for SIREN
        input_min = self.dataset.normalized_bounds['input_min']
        input_max = self.dataset.normalized_bounds['input_max']
        eval_coords_norm = 2 * (
            (eval_coords - input_min) / (input_max - input_min)
        ) - 1
        
        # Get SIREN predictions and denormalize from log space to linear space
        siren_predictions_log = self.trainer.predict(eval_coords_norm)
        siren_predictions_linear = self.dataset.denormalize_targets(siren_predictions_log)
        siren_2d = siren_predictions_linear.reshape(angle_mesh.shape)
        
        # Average SIREN predictions over distance (same masking as table)
        siren_profile = np.zeros(len(angle_centers))
        for i in range(len(angle_centers)):
            valid_distances = mask_3d[i, :]
            if np.sum(valid_distances) > 0:
                siren_profile[i] = np.mean(siren_2d[i, valid_distances])
            else:
                siren_profile[i] = np.nan
        
        # Convert angles to degrees for plotting
        angle_degrees = np.degrees(angle_centers)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(table_profile) | np.isnan(siren_profile)) & (table_profile > 0) & (siren_profile > 0)
        
        if np.sum(valid_mask) > 5:
            # Plot both profiles
            ax.plot(angle_degrees[valid_mask], table_profile[valid_mask], 
                   'o-', alpha=0.8, label='Lookup Table', markersize=3, linewidth=2, color='blue')
            ax.plot(angle_degrees[valid_mask], siren_profile[valid_mask], 
                   's--', alpha=0.8, label='SIREN Model', markersize=2, linewidth=2, color='red')
            
            # Mark Cherenkov angle
            ax.axvline(43, color='black', linestyle=':', alpha=0.7, linewidth=1.5)
            ax.text(43.5, ax.get_ylim()[1]*0.8, 'Cherenkov', rotation=90, 
                   va='bottom', ha='left', fontsize=8, alpha=0.7)
            
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('Avg Photon Density')
            ax.set_title(f'{actual_energy:.0f} MeV')
            ax.set_yscale('log')
            ax.set_xlim(np.degrees(angle_centers.min()), np.degrees(angle_centers.max()))
            ax.set_ylim(1e-2, None)  # Set minimum y-axis to 10^-2
            ax.grid(True, alpha=0.3)
            if energy == 200:  # Only show legend for first plot
                ax.legend(fontsize=8, loc='upper right')
        else:
            ax.text(0.5, 0.5, f'No valid data\nnear {energy} MeV', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{energy} MeV')
            
    def _plot_2d_angle_distance_for_energy(self, ax, energy, inputs_denorm, targets, predictions):
        """Plot 2D histogram (angle vs distance) for a specific energy using original lookup table grid."""
        
        # Use the original lookup table grid from the dataset
        if not hasattr(self.dataset, 'data_type') or self.dataset.data_type != 'h5_lookup':
            ax.text(0.5, 0.5, 'Lookup table grid\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{energy} MeV - 2D Map')
            return
            
        # Load the original HDF5 file to get the full lookup table grid
        import h5py
        with h5py.File(self.dataset.data_path, 'r') as f:
            # Load full density table and coordinates
            density_table = f['data/photon_table_density'][:]  # Shape: (n_energy, n_angle, n_distance)
            energy_centers = f['coordinates/energy_centers'][:]
            angle_centers = f['coordinates/angle_centers'][:]
            distance_centers = f['coordinates/distance_centers'][:]
        
        # Find the closest energy index
        energy_idx = np.argmin(np.abs(energy_centers - energy))
        actual_energy = energy_centers[energy_idx]
        
        if np.abs(actual_energy - energy) > 100:  # More than 100 MeV difference
            ax.text(0.5, 0.5, f'No data near\n{energy} MeV', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{energy} MeV - 2D Map')
            return
        
        # Get the 2D slice for this energy
        table_slice = density_table[energy_idx, :, :]  # Shape: (n_angle, n_distance)
        
        # Create meshgrid for visualization (convert angles to degrees)
        angle_mesh, distance_mesh = np.meshgrid(
            np.degrees(angle_centers), 
            distance_centers, 
            indexing='ij'
        )
        
        # Create coordinate grid for SIREN evaluation
        angle_grid_flat = angle_mesh.flatten()
        distance_grid_flat = distance_mesh.flatten()
        energy_grid_flat = np.full_like(angle_grid_flat, actual_energy)
        
        # Stack coordinates for SIREN prediction
        eval_coords = np.stack([
            energy_grid_flat,
            np.radians(angle_grid_flat),  # Convert back to radians for SIREN
            distance_grid_flat
        ], axis=-1)
        
        # Normalize coordinates for SIREN (same logic as in dataset._normalize_data)
        input_min = self.dataset.normalized_bounds['input_min']
        input_max = self.dataset.normalized_bounds['input_max']
        eval_coords_norm = 2 * (
            (eval_coords - input_min) / (input_max - input_min)
        ) - 1
        
        # Get SIREN predictions and denormalize from log space to linear space
        siren_predictions_log = self.trainer.predict(eval_coords_norm)
        siren_predictions_linear = self.dataset.denormalize_targets(siren_predictions_log)
        siren_slice = siren_predictions_linear.reshape(angle_mesh.shape)
        
        # Compute ratio: SIREN/Table
        # Add small epsilon to avoid division by zero
        ratio_slice = siren_slice / (table_slice + 1e-12)
        
        # Mask out very low density regions for cleaner visualization
        mask = table_slice > 1e-10
        ratio_slice_masked = np.where(mask, ratio_slice, np.nan)
        
        # Plot using clean contour levels only (no fine grid visualization)
        valid_mask = ~np.isnan(ratio_slice_masked)
        if np.sum(valid_mask) > 10:  # Need at least some valid points
            # Clean contour visualization with multiple levels
            try:
                # Main contour levels
                levels = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]
                
                # Filled contours for background
                contourf = ax.contourf(angle_mesh, distance_mesh, ratio_slice_masked, 
                                     levels=levels, cmap='RdBu_r', alpha=0.6, extend='both')
                
                # Line contours for clarity
                contours = ax.contour(angle_mesh, distance_mesh, ratio_slice_masked, 
                                    levels=levels, colors='black', linewidths=0.8, alpha=0.8)
                
                # Label key contours (perfect match and significant deviations)
                key_levels = [0.5, 0.85, 1.0, 1.15, 1.5]
                ax.clabel(contours, levels=key_levels, inline=True, fontsize=9, fmt='%.2f')
                
                # Add colorbar for filled contours
                if energy == 1000:  # Only add colorbar for last plot
                    cbar = plt.colorbar(contourf, ax=ax)
                    cbar.set_label('SIREN/Table Ratio')
                    
            except Exception as e:
                # Fallback: simple scatter if contours fail
                valid_points = valid_mask
                if np.sum(valid_points) > 0:
                    scatter = ax.scatter(angle_mesh[valid_points], distance_mesh[valid_points], 
                                       c=ratio_slice_masked[valid_points], cmap='RdBu_r', 
                                       s=10, vmin=0.5, vmax=2.0, alpha=0.7)
                    if energy == 1000:
                        cbar = plt.colorbar(scatter, ax=ax)
                        cbar.set_label('SIREN/Table Ratio')
            
            # Mark Cherenkov angle (approximately 43 degrees for water)
            ax.axvline(43, color='white', linestyle='--', alpha=0.9, linewidth=2)
            ax.text(43.5, ax.get_ylim()[1]*0.95, 'Cherenkov', rotation=90, 
                   va='top', ha='left', color='white', fontweight='bold', fontsize=8)
            
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('Distance (mm)')
            ax.set_title(f'{actual_energy:.0f} MeV\nSIREN/Table Ratio')
            
            # Set limits based on actual data range
            ax.set_xlim(np.degrees(angle_centers.min()), np.degrees(angle_centers.max()))
            ax.set_ylim(distance_centers.min(), distance_centers.max())
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No valid data\nnear {energy} MeV', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{energy} MeV - 2D Map')
        
    def _plot_energy_slice(self, ax, energy, inputs_denorm, targets, predictions):
        """Plot angle profile for a specific energy at different distances."""
        energy_tolerance = 50  # MeV
        distances_to_plot = [1000, 2000, 3000, 5000]  # mm
        
        # Filter data near the specified energy
        energy_mask = np.abs(inputs_denorm[:, 0] - energy) < energy_tolerance
        
        for distance in distances_to_plot:
            dist_tolerance = 200  # mm
            dist_mask = np.abs(inputs_denorm[:, 2] - distance) < dist_tolerance
            mask = energy_mask & dist_mask
            
            if np.sum(mask) < 10:
                continue
                
            # Get angles and sort
            angles = np.degrees(inputs_denorm[mask, 1])
            targets_slice = targets[mask]
            predictions_slice = predictions[mask]
            
            # Sort by angle for clean plotting
            sort_idx = np.argsort(angles)
            angles_sorted = angles[sort_idx]
            targets_sorted = targets_slice[sort_idx]
            predictions_sorted = predictions_slice[sort_idx]
            
            # Plot with different styles
            alpha = 0.7
            ax.plot(angles_sorted, targets_sorted, 'o-', 
                   alpha=alpha, markersize=3, linewidth=1,
                   label=f'Data {distance}mm')
            ax.plot(angles_sorted, predictions_sorted, 's--', 
                   alpha=alpha, markersize=2, linewidth=1,
                   label=f'SIREN {distance}mm')
        
        # Mark Cherenkov angle
        ax.axvline(43, color='red', linestyle=':', alpha=0.5, label='Cherenkov angle')
        
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Photon Density')
        ax.set_title(f'Energy = {energy} MeV')
        ax.set_yscale('log')
        ax.set_xlim(20, 80)
        ax.grid(True, alpha=0.3)
        if energy == 200:  # Only show legend for first plot
            ax.legend(fontsize=8, loc='upper right')
            
    def _plot_distance_slice(self, ax, distance, inputs_denorm, targets, predictions):
        """Plot angle profile for a specific distance at different energies."""
        distance_tolerance = 200  # mm
        energies_to_plot = [200, 400, 600, 800]  # MeV
        
        # Filter data near the specified distance
        distance_mask = np.abs(inputs_denorm[:, 2] - distance) < distance_tolerance
        
        for energy in energies_to_plot:
            energy_tolerance = 50  # MeV
            energy_mask = np.abs(inputs_denorm[:, 0] - energy) < energy_tolerance
            mask = distance_mask & energy_mask
            
            if np.sum(mask) < 10:
                continue
                
            # Get angles and sort
            angles = np.degrees(inputs_denorm[mask, 1])
            targets_slice = targets[mask]
            predictions_slice = predictions[mask]
            
            # Sort by angle
            sort_idx = np.argsort(angles)
            angles_sorted = angles[sort_idx]
            targets_sorted = targets_slice[sort_idx]
            predictions_sorted = predictions_slice[sort_idx]
            
            # Plot
            alpha = 0.7
            ax.plot(angles_sorted, targets_sorted, 'o-', 
                   alpha=alpha, markersize=3, linewidth=1,
                   label=f'Data {energy}MeV')
            ax.plot(angles_sorted, predictions_sorted, 's--', 
                   alpha=alpha, markersize=2, linewidth=1,
                   label=f'SIREN {energy}MeV')
        
        ax.axvline(43, color='red', linestyle=':', alpha=0.5)
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Photon Density')
        ax.set_title(f'Distance = {distance} mm')
        ax.set_yscale('log')
        ax.set_xlim(20, 80)
        ax.grid(True, alpha=0.3)
        if distance == 1000:  # Only show legend for first plot
            ax.legend(fontsize=8, loc='upper right')
            
    def _plot_angular_slice(self, ax, angle_deg, inputs_denorm, targets, predictions):
        """Plot energy-distance projection at a specific angle."""
        angle_tolerance = 5  # degrees
        angle_rad = np.radians(angle_deg)
        angle_tolerance_rad = np.radians(angle_tolerance)
        
        # Filter data near the specified angle
        angle_mask = np.abs(inputs_denorm[:, 1] - angle_rad) < angle_tolerance_rad
        
        if np.sum(angle_mask) < 50:
            ax.text(0.5, 0.5, f'Insufficient data\nnear {angle_deg}°', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Angle = {angle_deg}°')
            return
        
        # Create 2D plot: energy vs distance
        energies = inputs_denorm[angle_mask, 0]
        distances = inputs_denorm[angle_mask, 2]
        targets_slice = targets[angle_mask]
        
        # Create scatter plot with color representing photon density
        scatter = ax.scatter(energies, distances, c=targets_slice, 
                           cmap='viridis', alpha=0.6, s=20)
        
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Distance (mm)')
        ax.set_title(f'Angle = {angle_deg}° (Data)')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Photon Density')
        
    def _plot_2d_energy_angle(self, ax, inputs_denorm, targets, predictions):
        """Plot 2D energy-angle comparison."""
        # Sample data for plotting
        n_plot = min(5000, len(targets))
        indices = np.random.choice(len(targets), n_plot, replace=False)
        
        energies = inputs_denorm[indices, 0]
        angles = np.degrees(inputs_denorm[indices, 1])
        errors = np.abs(predictions[indices] - targets[indices])
        
        # Create error map
        scatter = ax.scatter(energies, angles, c=errors, cmap='Reds', 
                           alpha=0.6, s=15, vmax=np.percentile(errors, 95))
        
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Prediction Error Map')
        ax.axhline(43, color='blue', linestyle='--', alpha=0.5, label='Cherenkov')
        
        plt.colorbar(scatter, ax=ax, label='|Error|')
        
    def _plot_distance_profiles(self, ax, inputs_denorm, targets, predictions):
        """Plot distance falloff profiles."""
        # Fix energy and angle, vary distance
        fixed_energy = 500  # MeV
        fixed_angle = 43    # degrees
        
        energy_tol = 100
        angle_tol = 10
        
        energy_mask = np.abs(inputs_denorm[:, 0] - fixed_energy) < energy_tol
        angle_mask = np.abs(np.degrees(inputs_denorm[:, 1]) - fixed_angle) < angle_tol
        mask = energy_mask & angle_mask
        
        if np.sum(mask) > 10:
            distances = inputs_denorm[mask, 2]
            targets_slice = targets[mask]
            predictions_slice = predictions[mask]
            
            # Sort by distance
            sort_idx = np.argsort(distances)
            distances_sorted = distances[sort_idx]
            targets_sorted = targets_slice[sort_idx]
            predictions_sorted = predictions_slice[sort_idx]
            
            ax.plot(distances_sorted, targets_sorted, 'o-', 
                   alpha=0.7, label='Lookup Table', markersize=4)
            ax.plot(distances_sorted, predictions_sorted, 's--', 
                   alpha=0.7, label='SIREN Model', markersize=3)
            
            ax.set_xlabel('Distance (mm)')
            ax.set_ylabel('Photon Density')
            ax.set_title(f'Distance Profile\n(E={fixed_energy}MeV, θ={fixed_angle}°)')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Distance Profile')
            
    def _plot_energy_scaling(self, ax, inputs_denorm, targets, predictions):
        """Plot energy scaling at Cherenkov angle."""
        cherenkov_angle = 43  # degrees
        fixed_distance = 2000  # mm
        
        angle_tol = 5
        dist_tol = 500
        
        angle_mask = np.abs(np.degrees(inputs_denorm[:, 1]) - cherenkov_angle) < angle_tol
        dist_mask = np.abs(inputs_denorm[:, 2] - fixed_distance) < dist_tol
        mask = angle_mask & dist_mask
        
        if np.sum(mask) > 10:
            energies = inputs_denorm[mask, 0]
            targets_slice = targets[mask]
            predictions_slice = predictions[mask]
            
            # Sort by energy
            sort_idx = np.argsort(energies)
            energies_sorted = energies[sort_idx]
            targets_sorted = targets_slice[sort_idx]
            predictions_sorted = predictions_slice[sort_idx]
            
            ax.plot(energies_sorted, targets_sorted, 'o-', 
                   alpha=0.7, label='Lookup Table', markersize=4)
            ax.plot(energies_sorted, predictions_sorted, 's--', 
                   alpha=0.7, label='SIREN Model', markersize=3)
            
            ax.set_xlabel('Energy (MeV)')
            ax.set_ylabel('Photon Density')
            ax.set_title(f'Energy Scaling\n(θ={cherenkov_angle}°, d={fixed_distance}mm)')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Energy Scaling')
            
    def _plot_cherenkov_enhancement(self, ax, inputs_denorm, targets, predictions):
        """Plot angular profile showing Cherenkov enhancement."""
        # Integrate over energy and distance to show pure angular dependence
        energy_bins = np.linspace(200, 800, 10)
        distance_bins = np.linspace(1000, 5000, 8)
        angle_bins = np.linspace(20, 80, 50)
        
        # Create binned averages
        angle_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
        data_profile = []
        siren_profile = []
        
        for i in range(len(angle_bins) - 1):
            angle_mask = ((np.degrees(inputs_denorm[:, 1]) >= angle_bins[i]) & 
                         (np.degrees(inputs_denorm[:, 1]) < angle_bins[i+1]))
            
            if np.sum(angle_mask) > 5:
                data_profile.append(np.mean(targets[angle_mask]))
                siren_profile.append(np.mean(predictions[angle_mask]))
            else:
                data_profile.append(np.nan)
                siren_profile.append(np.nan)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(data_profile) | np.isnan(siren_profile))
        if np.sum(valid_mask) > 5:
            ax.plot(angle_centers[valid_mask], np.array(data_profile)[valid_mask], 
                   'o-', alpha=0.7, label='Lookup Table', markersize=4)
            ax.plot(angle_centers[valid_mask], np.array(siren_profile)[valid_mask], 
                   's--', alpha=0.7, label='SIREN Model', markersize=3)
            
            ax.axvline(43, color='red', linestyle=':', alpha=0.7, label='Cherenkov angle')
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('Average Photon Density')
            ax.set_title('Angular Profile\n(Energy & Distance Averaged)')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Angular Profile')
            
    def _plot_accuracy_map(self, ax, inputs_denorm, targets, predictions):
        """Plot prediction accuracy as function of input parameters."""
        # Calculate relative errors
        rel_errors = np.abs(predictions - targets) / (np.abs(targets) + 1e-10)
        
        # Create 2D histogram of relative errors
        energies = inputs_denorm[:, 0]
        angles = np.degrees(inputs_denorm[:, 1])
        
        # Bin the data
        H, xedges, yedges = np.histogram2d(energies, angles, bins=[20, 20], 
                                          weights=rel_errors, density=False)
        counts, _, _ = np.histogram2d(energies, angles, bins=[20, 20])
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            H_avg = np.divide(H, counts, out=np.zeros_like(H), where=counts!=0)
        
        # Plot
        im = ax.imshow(H_avg.T, origin='lower', aspect='auto', cmap='Reds',
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Relative Error Map')
        ax.axhline(43, color='blue', linestyle='--', alpha=0.7)
        
        plt.colorbar(im, ax=ax, label='Avg Relative Error')
        
    def compare_with_baseline(
        self,
        baseline_predictions: np.ndarray,
        baseline_name: str = "Baseline"
    ) -> Dict[str, Any]:
        """
        Compare SIREN model with a baseline model.
        
        Args:
            baseline_predictions: Baseline model predictions
            baseline_name: Name for the baseline model
            
        Returns:
            Comparison results
        """
        if not self.comparison_data:
            self.analyze_error_patterns()
            
        targets = self.comparison_data['targets'].flatten()
        siren_predictions = self.comparison_data['predictions'].flatten()
        
        # Calculate metrics for both models
        siren_metrics = self._calculate_metrics(targets, siren_predictions)
        baseline_metrics = self._calculate_metrics(targets, baseline_predictions)
        
        # Calculate improvement
        improvements = {}
        for metric in ['r2', 'rmse', 'mae', 'relative_error']:
            if metric == 'r2':
                # Higher is better for R²
                improvements[metric] = siren_metrics[metric] - baseline_metrics[metric]
            else:
                # Lower is better for error metrics
                improvements[metric] = baseline_metrics[metric] - siren_metrics[metric]
                
        comparison = {
            'siren_metrics': siren_metrics,
            'baseline_metrics': baseline_metrics,
            'improvements': improvements,
            'baseline_name': baseline_name
        }
        
        logger.info(f"Comparison with {baseline_name}:")
        logger.info(f"  R² improvement: {improvements['r2']:.4f}")
        logger.info(f"  RMSE improvement: {improvements['rmse']:.6f}")
        
        return comparison