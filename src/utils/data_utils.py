#!/usr/bin/env python
"""
Data utility functions for loading and processing biomass data.

Author: najahpokkiri
Date: 2025-05-28
"""

import os
import json
import pickle
import numpy as np
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


def save_predictions(y_true, y_pred, coordinates, sources, output_path):
    """Save predictions to a CSV file."""
    import pandas as pd
    
    # Create DataFrame
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
    print(f"Predictions saved to: {output_path}")
    
    return results_df


def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict