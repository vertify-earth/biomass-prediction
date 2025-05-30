#!/usr/bin/env python
"""
Data utility functions for loading and processing biomass data with dual-scale support.

Author: najahpokkiri
Date: 2025-05-30
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def load_preprocessed_data(config):
    """Load the latest preprocessed data."""
    print("\n==== Loading Preprocessed Data ====")
    
    # Find latest timestamp
    latest_path = os.path.join(config.preprocessed_dir, "latest.txt")
    
    if os.path.exists(latest_path):
        with open(latest_path, 'r') as f:
            timestamp = f.read().strip()
    else:
        # Find most recent files if latest.txt doesn't exist
        pattern = "X_*.npy"
        files = list(Path(config.preprocessed_dir).glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No preprocessed data found in {config.preprocessed_dir}")
        
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        timestamp = latest_file.name.split('_')[1].split('.')[0]
    
    # Load data
    X_path = os.path.join(config.preprocessed_dir, f"X_{timestamp}.npy")
    y_path = os.path.join(config.preprocessed_dir, f"y_{timestamp}.npy")
    sources_path = os.path.join(config.preprocessed_dir, f"sources_{timestamp}.npy")
    coord_path = os.path.join(config.preprocessed_dir, f"coordinates_{timestamp}.pkl")
    
    # Check core files exist
    for path in [X_path, y_path, sources_path, coord_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
    
    # Load arrays
    X = np.load(X_path)
    y = np.load(y_path)
    sources = np.load(sources_path)
    
    with open(coord_path, 'rb') as f:
        coordinates = pickle.load(f)
    
    # Get config file if it exists
    config_path = os.path.join(config.preprocessed_dir, f"preprocessing_config_{timestamp}.json")
    preprocess_config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            preprocess_config = json.load(f)
    
    print(f"Loaded data with timestamp: {timestamp}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"sources shape: {sources.shape}")
    print(f"coordinates: {len(coordinates)} points")
    
    # Check transformation type
    transform_type = "log" if preprocess_config.get('use_log_transform', False) else "none"
    if transform_type == "log":
        print("🔄 Data uses log transformation - results will be converted back to original scale")
    
    # Summarize site information
    unique_sites = np.unique(sources)
    site_counts = [np.sum(sources == s) for s in unique_sites]
    
    print("\nSite breakdown:")
    for site_id, count in zip(unique_sites, site_counts):
        site_name = f"Site_{site_id+1}"
        if 'site_names' in preprocess_config and site_id < len(preprocess_config['site_names']):
            site_name = preprocess_config['site_names'][site_id]
        print(f"  {site_name} (ID: {site_id}): {count} samples")
    
    # Create data dictionary
    data = {
        'X': X,
        'y': y,
        'coordinates': coordinates,
        'sources': sources,
        'timestamp': timestamp,
        'preprocess_config': preprocess_config
    }
    
    return data


def save_predictions(y_true, y_pred, coordinates, sources, output_path, transform_type="none"):
    """Save predictions to a CSV file with dual-scale support."""
    
    # Convert to original scale if needed
    if transform_type == "log":
        from ..models.hybrid_cv import inverse_transform_biomass
        y_true_original = inverse_transform_biomass(y_true, transform_type)
        y_pred_original = inverse_transform_biomass(y_pred, transform_type)
        
        # Create DataFrame with both scales
        results_df = pd.DataFrame({
            'y_true_log': y_true,
            'y_pred_log': y_pred,
            'y_true_original': y_true_original,
            'y_pred_original': y_pred_original,
            'residual_log': y_pred - y_true,
            'residual_original': y_pred_original - y_true_original,
            'source': sources,
            'x_coord': [coord[0] for coord in coordinates],
            'y_coord': [coord[1] for coord in coordinates]
        })
        
        print(f"💾 Predictions saved with dual scales:")
        print(f"  Log scale: RMSE = {np.sqrt(np.mean((y_pred - y_true)**2)):.4f}")
        print(f"  Original scale: RMSE = {np.sqrt(np.mean((y_pred_original - y_true_original)**2)):.1f} Mg/ha")
        
    else:
        # Single scale
        results_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'residual': y_pred - y_true,
            'source': sources,
            'x_coord': [coord[0] for coord in coordinates],
            'y_coord': [coord[1] for coord in coordinates]
        })
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"📁 Predictions saved to: {output_path}")
    
    return results_df


def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict


def export_results_summary(cv_summary_path, output_path):
    """Export a human-readable summary of CV results."""
    
    with open(cv_summary_path, 'r') as f:
        cv_summary = json.load(f)
    
    # Create summary text
    summary_lines = []
    summary_lines.append("🌳 BIOMASS PREDICTION MODEL RESULTS SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    
    # Check if we have dual-scale results
    if 'original_scale_metrics' in cv_summary:
        # Original scale results (primary)
        orig_metrics = cv_summary['original_scale_metrics']
        summary_lines.append("📊 MAIN RESULTS (Original Biomass Scale - Mg/ha):")
        summary_lines.append(f"  RMSE: {orig_metrics['mean_rmse']:.1f} ± {orig_metrics['std_rmse']:.1f} Mg/ha")
        summary_lines.append(f"  R²: {orig_metrics['mean_r2']:.3f} ± {orig_metrics['std_r2']:.3f}")
        summary_lines.append(f"  MAE: {orig_metrics['mean_mae']:.1f} ± {orig_metrics['std_mae']:.1f} Mg/ha")
        summary_lines.append(f"  Spearman Correlation: {orig_metrics['mean_spearman']:.3f} ± {orig_metrics['std_spearman']:.3f}")
        
        if 'mean_biomass' in orig_metrics:
            summary_lines.append(f"  Mean Biomass: {orig_metrics['mean_biomass']:.1f} Mg/ha")
            relative_rmse = (orig_metrics['mean_rmse'] / orig_metrics['mean_biomass']) * 100
            summary_lines.append(f"  Relative RMSE: {relative_rmse:.1f}% of mean biomass")
        
        # Performance assessment
        summary_lines.append("")
        if orig_metrics['mean_r2'] > 0.90:
            performance = "🌟 Excellent"
        elif orig_metrics['mean_r2'] > 0.85:
            performance = "✅ Very Good"
        elif orig_metrics['mean_r2'] > 0.75:
            performance = "👍 Good"
        else:
            performance = "⚠️ Needs Improvement"
        
        summary_lines.append(f"🎯 Overall Performance: {performance}")
        summary_lines.append("")
        
        # Technical details (log scale)
        log_metrics = cv_summary['log_scale_metrics']
        summary_lines.append("🔧 Technical Metrics (Log Scale - Training Scale):")
        summary_lines.append(f"  RMSE: {log_metrics['mean_rmse']:.4f} ± {log_metrics['std_rmse']:.4f}")
        summary_lines.append(f"  R²: {log_metrics['mean_r2']:.3f} ± {log_metrics['std_r2']:.3f}")
        
    else:
        # Single scale results
        if 'mean_rmse' in cv_summary:
            summary_lines.append("📊 CROSS-VALIDATION RESULTS:")
            summary_lines.append(f"  RMSE: {cv_summary['mean_rmse']:.4f} ± {cv_summary['std_rmse']:.4f}")
            summary_lines.append(f"  R²: {cv_summary['mean_r2']:.3f} ± {cv_summary['std_r2']:.3f}")
            summary_lines.append(f"  MAE: {cv_summary['mean_mae']:.4f} ± {cv_summary['std_mae']:.4f}")
            summary_lines.append(f"  Spearman: {cv_summary['mean_spearman']:.3f} ± {cv_summary['std_spearman']:.3f}")
    
    # Transformation info
    if 'transform_info' in cv_summary:
        transform_info = cv_summary['transform_info']
        summary_lines.append("")
        summary_lines.append("🔄 Data Transformation Info:")
        summary_lines.append(f"  Training Scale: {transform_info.get('training_scale', 'unknown')}")
        summary_lines.append(f"  Transform Type: {transform_info.get('transform_type', 'unknown')}")
        summary_lines.append(f"  Evaluation Scales: {', '.join(transform_info.get('evaluation_scales', []))}")
    
    summary_lines.append("")
    summary_lines.append("=" * 60)
    summary_lines.append("Generated by Biomass Prediction Pipeline")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"📄 Results summary exported to: {output_path}")
    
    # Also print to console
    for line in summary_lines:
        print(line)