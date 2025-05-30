#!/usr/bin/env python
"""
Visualization utilities for biomass prediction results.

Author: najahpokkiri
Date: 2025-05-28
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr


def visualize_cv_results(fold_results, fold_metrics, fold_histories, output_dir):
    """Create visualizations for CV results."""
    print("\nCreating visualizations...")
    
    # Create combined predictions vs ground truth plot
    plt.figure(figsize=(10, 8))
    
    # Collect all predictions
    all_true = []
    all_pred = []
    all_sources = []
    all_valid = []
    
    for fold_idx, results_df in enumerate(fold_results):
        # Get values from dataframe
        y_true = results_df['y_true'].values
        y_pred = results_df['y_pred'].values
        sources = results_df['source'].values
        valid = results_df['valid'].values if 'valid' in results_df else np.ones_like(y_true, dtype=bool)
        
        all_true.extend(y_true[valid])
        all_pred.extend(y_pred[valid])
        all_sources.extend(sources[valid])
        all_valid.extend(valid)
    
    # Convert to arrays
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_sources = np.array(all_sources)
    
    # Plot by source
    unique_sources = np.unique(all_sources)
    cmap = plt.cm.get_cmap('tab10', len(unique_sources))
    
    for i, source in enumerate(unique_sources):
        mask = (all_sources == source)
        plt.scatter(
            all_true[mask], all_pred[mask], 
            alpha=0.7, s=30, color=cmap(i),
            label=f"Site {source}"
        )
    
    # Add 1:1 line
    min_val = min(np.min(all_true), np.min(all_pred))
    max_val = max(np.max(all_true), np.max(all_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    # Calculate overall metrics
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    r2 = r2_score(all_true, all_pred)
    mae = mean_absolute_error(all_true, all_pred)
    spearman, _ = spearmanr(all_true, all_pred)
    
    # Add metrics text
    plt.text(0.05, 0.95, 
             f"RMSE = {rmse:.4f}\n$R^2$ = {r2:.4f}\nMAE = {mae:.4f}\nSpearman = {spearman:.4f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labels and legend
    plt.xlabel('True Biomass (log scale)')
    plt.ylabel('Predicted Biomass (log scale)')
    plt.title('Cross-Validation Results: Predicted vs True Biomass')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_predictions_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot training histories
    if fold_histories:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        ax = axes[0, 0]
        for i, history in enumerate(fold_histories):
            if 'train_losses' in history and 'val_losses' in history:
                epochs = range(1, len(history['train_losses']) + 1)
                ax.plot(epochs, history['train_losses'], 'b-', alpha=0.6, label=f'Train Fold {i+1}' if i == 0 else "")
                ax.plot(epochs, history['val_losses'], 'r-', alpha=0.6, label=f'Val Fold {i+1}' if i == 0 else "")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate schedule
        ax = axes[0, 1]
        for i, history in enumerate(fold_histories):
            if 'learning_rates' in history:
                epochs = range(1, len(history['learning_rates']) + 1)
                ax.plot(epochs, history['learning_rates'], alpha=0.6, label=f'Fold {i+1}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Metrics distribution
        metrics_df = pd.DataFrame(fold_metrics)
        
        # RMSE boxplot
        ax = axes[1, 0]
        ax.boxplot([metrics_df['rmse']], labels=['RMSE'])
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE Distribution Across Folds')
        ax.grid(True, alpha=0.3)
        
        # R² boxplot
        ax = axes[1, 1]
        ax.boxplot([metrics_df['r2']], labels=['R²'])
        ax.set_ylabel('R²')
        ax.set_title('R² Distribution Across Folds')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Residual analysis
    plt.figure(figsize=(15, 5))
    
    # Residuals vs predicted
    plt.subplot(1, 3, 1)
    residuals = all_pred - all_true
    plt.scatter(all_pred, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Biomass')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Residual histogram
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=30, alpha=0.7, density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot
    plt.subplot(1, 3, 3)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normal)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Site-specific performance
    if len(unique_sources) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        site_metrics = []
        for source in unique_sources:
            mask = (all_sources == source)
            site_rmse = np.sqrt(mean_squared_error(all_true[mask], all_pred[mask]))
            site_r2 = r2_score(all_true[mask], all_pred[mask])
            site_mae = mean_absolute_error(all_true[mask], all_pred[mask])
            site_count = np.sum(mask)
            
            site_metrics.append({
                'Site': f'Site {source}',
                'RMSE': site_rmse,
                'R²': site_r2,
                'MAE': site_mae,
                'Count': site_count
            })
        
        site_df = pd.DataFrame(site_metrics)
        
        # Bar plots for each metric
        x_pos = np.arange(len(site_df))
        
        axes[0, 0].bar(x_pos, site_df['RMSE'])
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(site_df['Site'], rotation=45)
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('RMSE by Site')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].bar(x_pos, site_df['R²'])
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(site_df['Site'], rotation=45)
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].set_title('R² by Site')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].bar(x_pos, site_df['MAE'])
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(site_df['Site'], rotation=45)
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('MAE by Site')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(x_pos, site_df['Count'])
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(site_df['Site'], rotation=45)
        axes[1, 1].set_ylabel('Sample Count')
        axes[1, 1].set_title('Sample Count by Site')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'site_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to: {output_dir}")


def plot_data_distribution(X, y, sources, coordinates, output_path):
    """Plot data distribution and characteristics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Biomass distribution
    axes[0, 0].hist(y, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Biomass (log scale)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Biomass Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Biomass by site
    unique_sources = np.unique(sources)
    site_biomass = [y[sources == s] for s in unique_sources]
    axes[0, 1].boxplot(site_biomass, labels=[f'Site {s}' for s in unique_sources])
    axes[0, 1].set_ylabel('Biomass (log scale)')
    axes[0, 1].set_title('Biomass Distribution by Site')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sample count by site
    site_counts = [np.sum(sources == s) for s in unique_sources]
    axes[0, 2].bar(range(len(unique_sources)), site_counts)
    axes[0, 2].set_xticks(range(len(unique_sources)))
    axes[0, 2].set_xticklabels([f'Site {s}' for s in unique_sources], rotation=45)
    axes[0, 2].set_ylabel('Sample Count')
    axes[0, 2].set_title('Sample Count by Site')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Spatial distribution
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]
    
    scatter = axes[1, 0].scatter(x_coords, y_coords, c=sources, alpha=0.6, s=20, cmap='tab10')
    axes[1, 0].set_xlabel('X Coordinate')
    axes[1, 0].set_ylabel('Y Coordinate')
    axes[1, 0].set_title('Spatial Distribution of Samples')
    plt.colorbar(scatter, ax=axes[1, 0], label='Site ID')
    
    # Biomass spatial distribution
    scatter = axes[1, 1].scatter(x_coords, y_coords, c=y, alpha=0.6, s=20, cmap='viridis')
    axes[1, 1].set_xlabel('X Coordinate')
    axes[1, 1].set_ylabel('Y Coordinate')
    axes[1, 1].set_title('Spatial Distribution of Biomass')
    plt.colorbar(scatter, ax=axes[1, 1], label='Biomass (log scale)')
    
    # Band statistics (first few bands)
    n_bands = min(X.shape[1], 6)
    band_means = [np.mean(X[:, i, :, :]) for i in range(n_bands)]
    axes[1, 2].bar(range(n_bands), band_means)
    axes[1, 2].set_xticks(range(n_bands))
    axes[1, 2].set_xticklabels([f'Band {i+1}' for i in range(n_bands)])
    axes[1, 2].set_ylabel('Mean Value')
    axes[1, 2].set_title(f'Mean Values for First {n_bands} Bands')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Data distribution plot saved to: {output_path}")