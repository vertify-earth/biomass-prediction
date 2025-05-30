#!/usr/bin/env python
"""
visualisation utilities for biomass prediction results with dual-scale support.

Author: najahpokkiri
Date: 2025-05-30
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr


def visualise_cv_results(fold_results, fold_metrics, fold_histories, output_dir):
    """Create visualisations for CV results with dual-scale support."""
    print("\nCreating visualisations...")
    
    # Detect if we have dual-scale metrics
    has_dual_scale = len(fold_metrics) > 0 and 'original_scale' in fold_metrics[0]
    transform_type = fold_metrics[0].get('transform_type', 'none') if fold_metrics else 'none'
    
    # Create combined predictions vs ground truth plot (Original Scale)
    plt.figure(figsize=(12, 10))
    
    # Collect all predictions (use original scale if available)
    all_true = []
    all_pred = []
    all_sources = []
    all_valid = []
    
    for fold_idx, results_df in enumerate(fold_results):
        # Use original scale if available, otherwise log scale
        if 'y_true_original' in results_df.columns and has_dual_scale:
            y_true = results_df['y_true_original'].values
            y_pred = results_df['y_pred_original'].values
            scale_label = "Biomass (Mg/ha)"
        else:
            y_true = results_df['y_true'].values
            y_pred = results_df['y_pred'].values
            scale_label = "Biomass (log scale)"
        
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
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8)
    
    # Calculate overall metrics
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    r2 = r2_score(all_true, all_pred)
    mae = mean_absolute_error(all_true, all_pred)
    spearman, _ = spearmanr(all_true, all_pred)
    
    # Add metrics text with appropriate units
    if has_dual_scale and 'y_true_original' in fold_results[0].columns:
        metrics_text = f"RMSE = {rmse:.1f} Mg/ha\n$R^2$ = {r2:.3f}\nMAE = {mae:.1f} Mg/ha\nSpearman = {spearman:.3f}"
        plt.xlabel('True Biomass (Mg/ha)')
        plt.ylabel('Predicted Biomass (Mg/ha)')
        title_suffix = " - Original Scale (Mg/ha)"
    else:
        metrics_text = f"RMSE = {rmse:.4f}\n$R^2$ = {r2:.3f}\nMAE = {mae:.4f}\nSpearman = {spearman:.3f}"
        plt.xlabel('True Biomass (log scale)')
        plt.ylabel('Predicted Biomass (log scale)')
        title_suffix = " - Log Scale"
    
    plt.text(0.05, 0.95, metrics_text,
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labels and legend
    plt.title(f'Cross-Validation Results: Predicted vs True Biomass{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plot_name = 'cv_predictions_scatter_original.png' if has_dual_scale else 'cv_predictions_scatter.png'
    plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches='tight')
    plt.close()
    
    # If dual scale, also create log scale plot
    if has_dual_scale:
        plt.figure(figsize=(10, 8))
        
        # Collect log scale data
        all_true_log = []
        all_pred_log = []
        all_sources_log = []
        
        for results_df in fold_results:
            y_true_log = results_df['y_true_log'].values
            y_pred_log = results_df['y_pred_log'].values
            sources = results_df['source'].values
            valid = results_df['valid'].values
            
            all_true_log.extend(y_true_log[valid])
            all_pred_log.extend(y_pred_log[valid])
            all_sources_log.extend(sources[valid])
        
        all_true_log = np.array(all_true_log)
        all_pred_log = np.array(all_pred_log)
        all_sources_log = np.array(all_sources_log)
        
        # Plot log scale
        for i, source in enumerate(unique_sources):
            mask = (all_sources_log == source)
            plt.scatter(
                all_true_log[mask], all_pred_log[mask], 
                alpha=0.7, s=30, color=cmap(i),
                label=f"Site {source}"
            )
        
        # Add 1:1 line
        min_val_log = min(np.min(all_true_log), np.min(all_pred_log))
        max_val_log = max(np.max(all_true_log), np.max(all_pred_log))
        plt.plot([min_val_log, max_val_log], [min_val_log, max_val_log], 'k--', linewidth=2, alpha=0.8)
        
        # Calculate log scale metrics
        rmse_log = np.sqrt(mean_squared_error(all_true_log, all_pred_log))
        r2_log = r2_score(all_true_log, all_pred_log)
        mae_log = mean_absolute_error(all_true_log, all_pred_log)
        spearman_log, _ = spearmanr(all_true_log, all_pred_log)
        
        metrics_text_log = f"RMSE = {rmse_log:.4f}\n$R^2$ = {r2_log:.3f}\nMAE = {mae_log:.4f}\nSpearman = {spearman_log:.3f}"
        
        plt.text(0.05, 0.95, metrics_text_log,
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xlabel('True Biomass (log scale)')
        plt.ylabel('Predicted Biomass (log scale)')
        plt.title('Cross-Validation Results: Predicted vs True Biomass - Log Scale')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cv_predictions_scatter_log.png'), dpi=300, bbox_inches='tight')
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
        
        # Metrics distribution (use original scale if available)
        if has_dual_scale:
            metrics_to_plot = [m['original_scale'] for m in fold_metrics]
            rmse_label = 'RMSE (Mg/ha)'
        else:
            metrics_to_plot = [m.get('log_scale', m) for m in fold_metrics]
            rmse_label = 'RMSE (log scale)'
        
        metrics_df = pd.DataFrame(metrics_to_plot)
        
        # RMSE boxplot
        ax = axes[1, 0]
        ax.boxplot([metrics_df['rmse']], labels=[rmse_label])
        ax.set_ylabel(rmse_label)
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
    
    # Residual analysis (use original scale if available)
    plt.figure(figsize=(15, 5))
    
    if has_dual_scale and 'residual_original' in fold_results[0].columns:
        residuals = []
        predictions = []
        for results_df in fold_results:
            valid_mask = results_df['valid'].values
            residuals.extend(results_df['residual_original'].values[valid_mask])
            predictions.extend(results_df['y_pred_original'].values[valid_mask])
        
        residuals = np.array(residuals)
        predictions = np.array(predictions)
        xlabel_pred = 'Predicted Biomass (Mg/ha)'
        ylabel_resid = 'Residuals (Mg/ha)'
        title_suffix = ' (Mg/ha)'
    else:
        residuals = []
        predictions = []
        for results_df in fold_results:
            valid_mask = results_df.get('valid', np.ones(len(results_df), dtype=bool))
            if 'residual_log' in results_df.columns:
                residuals.extend(results_df['residual_log'].values[valid_mask])
                predictions.extend(results_df['y_pred_log'].values[valid_mask])
            else:
                # Fallback for older format
                y_true = results_df['y_true'].values[valid_mask]
                y_pred = results_df['y_pred'].values[valid_mask]
                residuals.extend(y_pred - y_true)
                predictions.extend(y_pred)
        
        residuals = np.array(residuals)
        predictions = np.array(predictions)
        xlabel_pred = 'Predicted Biomass (log scale)'
        ylabel_resid = 'Residuals (log scale)'
        title_suffix = ' (log scale)'
    
    # Residuals vs predicted
    plt.subplot(1, 3, 1)
    plt.scatter(predictions, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(xlabel_pred)
    plt.ylabel(ylabel_resid)
    plt.title(f'Residuals vs Predicted{title_suffix}')
    plt.grid(True, alpha=0.3)
    
    # Residual histogram
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=30, alpha=0.7, density=True, edgecolor='black')
    plt.xlabel(ylabel_resid)
    plt.ylabel('Density')
    plt.title(f'Residual Distribution{title_suffix}')
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot
    plt.subplot(1, 3, 3)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normal)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    residual_filename = 'residual_analysis_original.png' if has_dual_scale else 'residual_analysis.png'
    plt.savefig(os.path.join(output_dir, residual_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Site-specific performance
    if len(unique_sources) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        site_metrics = []
        
        # Use original scale metrics if available
        if has_dual_scale:
            for source in unique_sources:
                # Collect data for this source across all folds
                source_true = []
                source_pred = []
                
                for results_df in fold_results:
                    mask = (results_df['source'] == source) & results_df['valid']
                    source_true.extend(results_df['y_true_original'].values[mask])
                    source_pred.extend(results_df['y_pred_original'].values[mask])
                
                if len(source_true) > 0:
                    source_true = np.array(source_true)
                    source_pred = np.array(source_pred)
                    
                    site_rmse = np.sqrt(mean_squared_error(source_true, source_pred))
                    site_r2 = r2_score(source_true, source_pred)
                    site_mae = mean_absolute_error(source_true, source_pred)
                    site_count = len(source_true)
                    site_mean = np.mean(source_true)
                    
                    site_metrics.append({
                        'Site': f'Site {source}',
                        'RMSE': site_rmse,
                        'R²': site_r2,
                        'MAE': site_mae,
                        'Count': site_count,
                        'Mean_Biomass': site_mean
                    })
            
            rmse_unit = 'RMSE (Mg/ha)'
            mae_unit = 'MAE (Mg/ha)'
            title_suffix = ' (Mg/ha)'
        else:
            # Use log scale
            for source in unique_sources:
                source_true = []
                source_pred = []
                
                for results_df in fold_results:
                    mask = (results_df['source'] == source)
                    if 'valid' in results_df.columns:
                        mask = mask & results_df['valid']
                    
                    if 'y_true_log' in results_df.columns:
                        source_true.extend(results_df['y_true_log'].values[mask])
                        source_pred.extend(results_df['y_pred_log'].values[mask])
                    else:
                        source_true.extend(results_df['y_true'].values[mask])
                        source_pred.extend(results_df['y_pred'].values[mask])
                
                if len(source_true) > 0:
                    source_true = np.array(source_true)
                    source_pred = np.array(source_pred)
                    
                    site_rmse = np.sqrt(mean_squared_error(source_true, source_pred))
                    site_r2 = r2_score(source_true, source_pred)
                    site_mae = mean_absolute_error(source_true, source_pred)
                    site_count = len(source_true)
                    
                    site_metrics.append({
                        'Site': f'Site {source}',
                        'RMSE': site_rmse,
                        'R²': site_r2,
                        'MAE': site_mae,
                        'Count': site_count
                    })
            
            rmse_unit = 'RMSE (log scale)'
            mae_unit = 'MAE (log scale)'
            title_suffix = ' (log scale)'
        
        if site_metrics:
            site_df = pd.DataFrame(site_metrics)
            
            # Bar plots for each metric
            x_pos = np.arange(len(site_df))
            
            axes[0, 0].bar(x_pos, site_df['RMSE'])
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(site_df['Site'], rotation=45)
            axes[0, 0].set_ylabel(rmse_unit)
            axes[0, 0].set_title(f'RMSE by Site{title_suffix}')
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
            axes[1, 0].set_ylabel(mae_unit)
            axes[1, 0].set_title(f'MAE by Site{title_suffix}')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].bar(x_pos, site_df['Count'])
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(site_df['Site'], rotation=45)
            axes[1, 1].set_ylabel('Sample Count')
            axes[1, 1].set_title('Sample Count by Site')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            site_perf_filename = 'site_performance_original.png' if has_dual_scale else 'site_performance.png'
            plt.savefig(os.path.join(output_dir, site_perf_filename), dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"📊 visualisations saved to: {output_dir}")
    if has_dual_scale:
        print("🔄 Created plots for both log scale (training) and original scale (interpretable)")


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
