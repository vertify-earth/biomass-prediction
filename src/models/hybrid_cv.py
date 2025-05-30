#!/usr/bin/env python
"""
Hybrid Site-Spatial Cross-Validation for Biomass Prediction

This script implements a combined site and spatial approach for biomass prediction,
ensuring all sites are represented in training while respecting spatial autocorrelation.

Author: najahpokkiri
Date: 2025-05-28
"""

import os
import sys
import json
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from src.models.loss_functions import SpatialLoss
from .config import HybridCVConfig
from .cnn_models import create_model
from .loss_functions import create_loss_function

from src.utils.data_utils import load_preprocessed_data

from src.utils.visualization import visualize_cv_results

# Suppress warnings
warnings.filterwarnings('ignore')


# Add after the imports
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ======================================================================
# BIOMASS TRANSFORMATION UTILITIES
# ======================================================================

def inverse_transform_biomass(y_transformed, transform_type="log", transform_params=None):
    """
    Convert transformed biomass values back to original scale.
    
    Args:
        y_transformed: Transformed biomass values
        transform_type: Type of transformation used ("log", "none")
        transform_params: Additional parameters for transformation
    
    Returns:
        y_original: Biomass values in original scale (e.g., Mg/ha)
    """
    if transform_type == "log":
        # Inverse of log(biomass + 1) is exp(y) - 1
        y_original = np.exp(y_transformed) - 1
        # Ensure non-negative values
        y_original = np.maximum(y_original, 0)
    elif transform_type == "none":
        y_original = y_transformed
    else:
        print(f"Warning: Unknown transform type '{transform_type}', returning as-is")
        y_original = y_transformed
    
    return y_original


def detect_transform_type(preprocess_config):
    """
    Detect what transformation was used during preprocessing.
    
    Args:
        preprocess_config: Preprocessing configuration dict
    
    Returns:
        transform_type: String indicating transformation type
        transform_params: Dict with transformation parameters
    """
    if preprocess_config.get('use_log_transform', False):
        return "log", {"base": "natural"}
    else:
        return "none", {}


def calculate_dual_metrics(y_true_log, y_pred_log, transform_type="log"):
    """
    Calculate metrics in both log and original scales.
    
    Args:
        y_true_log: True values in log scale
        y_pred_log: Predicted values in log scale
        transform_type: Type of transformation used
    
    Returns:
        metrics_log: Metrics calculated on log scale
        metrics_original: Metrics calculated on original scale
    """
    # Handle NaN values
    valid_mask = ~(np.isnan(y_true_log) | np.isnan(y_pred_log))
    y_true_log_valid = y_true_log[valid_mask]
    y_pred_log_valid = y_pred_log[valid_mask]
    
    # Log scale metrics
    metrics_log = {
        'rmse': np.sqrt(mean_squared_error(y_true_log_valid, y_pred_log_valid)),
        'r2': r2_score(y_true_log_valid, y_pred_log_valid),
        'mae': mean_absolute_error(y_true_log_valid, y_pred_log_valid),
        'spearman': spearmanr(y_true_log_valid, y_pred_log_valid)[0] if len(y_true_log_valid) > 1 else 0.0
    }
    
    # Convert to original scale
    y_true_original = inverse_transform_biomass(y_true_log_valid, transform_type)
    y_pred_original = inverse_transform_biomass(y_pred_log_valid, transform_type)
    
    # Original scale metrics
    metrics_original = {
        'rmse': np.sqrt(mean_squared_error(y_true_original, y_pred_original)),
        'r2': r2_score(y_true_original, y_pred_original),
        'mae': mean_absolute_error(y_true_original, y_pred_original),
        'spearman': spearmanr(y_true_original, y_pred_original)[0] if len(y_true_original) > 1 else 0.0,
        'mean_true': np.mean(y_true_original),
        'mean_pred': np.mean(y_pred_original),
        'std_true': np.std(y_true_original),
        'std_pred': np.std(y_pred_original),
        'min_true': np.min(y_true_original),
        'max_true': np.max(y_true_original),
        'min_pred': np.min(y_pred_original),
        'max_pred': np.max(y_pred_original)
    }
    
    return metrics_log, metrics_original


# ======================================================================
# FEATURE ENGINEERING
# ======================================================================

def add_derived_features(X):
    """Add derived features like NDVI, EVI, etc."""
    print("\nAdding derived features...")
    
    # Check if we have enough bands
    if X.shape[1] < 5:
        print("Warning: Not enough bands for derived features, skipping")
        return X
    
    # Assuming standardized band positions:
    # Band 1: Blue
    # Band 2: Green
    # Band 3: Red
    # Band 4: NIR
    blue_idx, green_idx, red_idx, nir_idx = 1, 2, 3, 4
    
    # Make a copy to avoid modifying original
    X_new = X.copy()
    
    # Calculate NDVI: (NIR - Red) / (NIR + Red)
    ndvi = np.zeros((X.shape[0], 1, X.shape[2], X.shape[3]))
    epsilon = 1e-8  # To avoid division by zero
    
    nir = X[:, nir_idx, :, :]
    red = X[:, red_idx, :, :]
    denominator = nir + red + epsilon
    ndvi[:, 0, :, :] = (nir - red) / denominator
    
    # Calculate EVI: 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
    evi = np.zeros((X.shape[0], 1, X.shape[2], X.shape[3]))
    blue = X[:, blue_idx, :, :]
    denominator = nir + 6*red - 7.5*blue + 1 + epsilon
    evi[:, 0, :, :] = 2.5 * (nir - red) / denominator
    
    # Calculate SAVI: ((NIR - Red) / (NIR + Red + 0.5)) * (1.5)
    savi = np.zeros((X.shape[0], 1, X.shape[2], X.shape[3]))
    denominator = nir + red + 0.5 + epsilon
    savi[:, 0, :, :] = ((nir - red) / denominator) * 1.5
    
    # Calculate GNDVI: (NIR - Green) / (NIR + Green)
    gndvi = np.zeros((X.shape[0], 1, X.shape[2], X.shape[3]))
    green = X[:, green_idx, :, :]
    denominator = nir + green + epsilon
    gndvi[:, 0, :, :] = (nir - green) / denominator
    
    # Calculate NDWI: (Green - NIR) / (Green + NIR)
    ndwi = np.zeros((X.shape[0], 1, X.shape[2], X.shape[3]))
    denominator = green + nir + epsilon
    ndwi[:, 0, :, :] = (green - nir) / denominator
    
    # Add new features to X
    X_new = np.concatenate([X_new, ndvi, evi, savi, gndvi, ndwi], axis=1)
    
    # Replace any NaN values with 0
    X_new = np.nan_to_num(X_new, nan=0.0)
    
    print(f"Added 5 derived features, new shape: {X_new.shape}")
    return X_new


def standardize_features(X_train, X_val=None, X_test=None):
    """Standardize features by band, using training set statistics."""
    print("\nStandardizing features...")
    
    # Initialize output arrays
    X_train_std = np.zeros_like(X_train)
    X_val_std = None if X_val is None else np.zeros_like(X_val)
    X_test_std = None if X_test is None else np.zeros_like(X_test)
    
    # Standardize each band separately
    for b in range(X_train.shape[1]):
        # Get band data and reshape to 2D
        band_train = X_train[:, b, :, :].reshape(X_train.shape[0], -1)
        
        # Calculate mean and std
        band_mean = np.mean(band_train)
        band_std = np.std(band_train)
        
        # Handle constant bands
        if band_std == 0:
            band_std = 1.0
        
        # Standardize training data
        X_train_std[:, b, :, :] = ((X_train[:, b, :, :] - band_mean) / band_std)
        
        # Standardize validation data if provided
        if X_val is not None:
            X_val_std[:, b, :, :] = ((X_val[:, b, :, :] - band_mean) / band_std)
        
        # Standardize test data if provided
        if X_test is not None:
            X_test_std[:, b, :, :] = ((X_test[:, b, :, :] - band_mean) / band_std)
    
    # Replace any NaN values with 0
    X_train_std = np.nan_to_num(X_train_std, nan=0.0)
    
    if X_val is not None:
        X_val_std = np.nan_to_num(X_val_std, nan=0.0)
    
    if X_test is not None:
        X_test_std = np.nan_to_num(X_test_std, nan=0.0)
    
    return X_train_std, X_val_std, X_test_std


# ======================================================================
# DATA AUGMENTATION
# ======================================================================

def create_data_augmentation(config):
    """Create data augmentation transformations."""
    if not (config.use_geometric_aug or config.use_spectral_aug) or config.aug_probability <= 0:
        return None
    
    print("\nSetting up data augmentation:")
    transforms_list = []
    
    # Geometric augmentations
    if config.use_geometric_aug:
        print("  - Using geometric augmentation (flips, rotations)")
        
        # Custom transforms that work with 3D tensors (C, H, W)
        class RandomHorizontalFlip:
            def __init__(self, p=0.5):
                self.p = p
                
            def __call__(self, x):
                if torch.rand(1) < self.p:
                    return torch.flip(x, [1])  # Flip height dimension (dim 1 for 3D tensor)
                return x
        
        class RandomVerticalFlip:
            def __init__(self, p=0.5):
                self.p = p
                
            def __call__(self, x):
                if torch.rand(1) < self.p:
                    return torch.flip(x, [2])  # Flip width dimension (dim 2 for 3D tensor)
                return x
        
        class RandomRotation90:
            def __init__(self, p=0.5):
                self.p = p
                
            def __call__(self, x):
                if torch.rand(1) < self.p:
                    k = torch.randint(1, 4, (1,)).item()  # 1, 2, or 3 for 90, 180, 270 degrees
                    return torch.rot90(x, k, [1, 2])  # For 3D tensor, rotate dims 1 & 2
                return x
        
        transforms_list.extend([
            RandomHorizontalFlip(p=config.aug_probability),
            RandomVerticalFlip(p=config.aug_probability),
            RandomRotation90(p=config.aug_probability)
        ])
    
    # Spectral augmentations
    if config.use_spectral_aug:
        print("  - Using spectral augmentation (band jittering)")
        
        # Custom transform for multi-band spectral augmentation
        class SpectralJitter:
            def __init__(self, p=0.5, brightness_factor=0.1, contrast_factor=0.1):
                self.p = p
                self.brightness_factor = brightness_factor
                self.contrast_factor = contrast_factor
                
            def __call__(self, x):
                if torch.rand(1) < self.p:
                    # Select random bands to modify (1-5 bands)
                    num_bands = x.shape[0]  # First dimension is channels for 3D tensor
                    num_to_modify = torch.randint(1, min(6, max(2, num_bands // 3)), (1,)).item()
                    bands_to_modify = torch.randperm(num_bands)[:num_to_modify]
                    
                    # Make a copy
                    x_aug = x.clone()
                    
                    for band_idx in bands_to_modify:
                        # Random brightness adjustment
                        if torch.rand(1) < 0.5:
                            brightness_change = 1.0 + (torch.rand(1) * 2 - 1) * self.brightness_factor
                            x_aug[band_idx] = x_aug[band_idx] * brightness_change
                        
                        # Random contrast adjustment
                        if torch.rand(1) < 0.5:
                            contrast_change = 1.0 + (torch.rand(1) * 2 - 1) * self.contrast_factor
                            mean = torch.mean(x_aug[band_idx])
                            x_aug[band_idx] = (x_aug[band_idx] - mean) * contrast_change + mean
                    
                    return x_aug
                return x
        
        transforms_list.append(SpectralJitter(p=config.aug_probability))
    
    # Print summary
    print(f"  - Augmentation probability: {config.aug_probability}")
    print(f"  - Total transformations: {len(transforms_list)}")
    
    # Compose the transforms
    class SequentialTransforms:
        def __init__(self, transforms):
            self.transforms = transforms
        
        def __call__(self, x):
            for t in self.transforms:
                x = t(x)
            return x
    
    return SequentialTransforms(transforms_list)


# ======================================================================
# ADVANCED SAMPLING
# ======================================================================

class HardNegativeMiningDataset(Dataset):
    """Dataset with hard negative mining capabilities."""
    
    def __init__(self, X, y, coordinates, model=None, device=None, transforms=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.coordinates = coordinates
        self.transforms = transforms
        self.model = model
        self.device = device
        
        # Error tracking for hard negative mining
        self.errors = torch.zeros_like(self.y)
        self.has_errors = False
    
    def update_errors(self, model, device):
        """Update error metrics for each sample."""
        self.model = model
        self.device = device
        
        # Skip if model is not provided
        if model is None:
            self.has_errors = False
            return
        
        # Set model to evaluation mode
        model.eval()
        
        # Calculate errors for each sample
        errors = []
        with torch.no_grad():
            for i in range(len(self.X)):
                x = self.X[i:i+1].to(device)
                y_true = self.y[i].item()
                
                # Forward pass
                y_pred = model(x).item()
                
                # Calculate error
                error = abs(y_pred - y_true)
                errors.append(error)
        
        # Update errors tensor
        self.errors = torch.tensor(errors)
        self.has_errors = True
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        coords = self.coordinates[idx]
        
        # Apply augmentations if available
        if self.transforms:
            x = self.transforms(x)
        
        # Include error if available
        if self.has_errors:
            return x, y, coords, self.errors[idx]
        
        return x, y, coords


def create_hard_negative_sampler(dataset, config):
    """Create a sampler that prioritizes hard examples."""
    if not config.use_hard_negative_mining or not dataset.has_errors:
        return None
    
    print("\nCreating hard negative mining sampler:")
    
    # Calculate weights based on errors
    errors = dataset.errors.numpy()
    
    # Scale errors to [0,1] for numerical stability
    if np.max(errors) > np.min(errors):
        weights = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))
    else:
        weights = np.ones_like(errors)
    
    # Add small constant to ensure all samples get some weight
    weights = weights + 0.1
    
    # Square weights to emphasize harder examples
    weights = weights ** 2
    
    # Display error distribution
    print("  - Error distribution:")
    percentiles = [0, 25, 50, 75, 100]
    for p in percentiles:
        print(f"    {p}th percentile: {np.percentile(errors, p):.4f}")
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    # Create sampler
    num_samples = int(len(weights) * config.oversampling_factor)
    print(f"  - Using {num_samples} samples with hard negative mining")
    
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=num_samples)


# ======================================================================
# SPATIAL CROSS-VALIDATION SPLITS
# ======================================================================

def create_hybrid_site_spatial_split(coordinates, sources, config):
    """Create CV splits that ensure site representation while respecting spatial autocorrelation."""
    print("\n==== Creating Hybrid Site-Spatial CV Splits ====")
    
    unique_sites = np.unique(sources)
    n_sites = len(unique_sites)
    print(f"Found {n_sites} unique sites")
    
    # Create folds list
    folds = []
    
    # Set minimum required samples for each split
    min_train = 10  # Absolute minimum for training
    min_val = 5     # Absolute minimum for validation
    min_test = 5    # Absolute minimum for testing
    
    # Create a fixed number of folds
    for fold_idx in range(config.n_folds):
        # Initialize masks for this fold
        train_mask = np.zeros(len(coordinates), dtype=bool)
        val_mask = np.zeros(len(coordinates), dtype=bool)
        test_mask = np.zeros(len(coordinates), dtype=bool)
        
        # Process each site separately
        for site in unique_sites:
            # Get indices for this site
            site_indices = np.where(sources == site)[0]
            n_site_samples = len(site_indices)
            
            # For small sites, just add to training (skip splitting)
            if n_site_samples < (min_train + min_val + min_test):
                print(f"  Site {site} has only {n_site_samples} samples - adding all to training")
                train_mask[site_indices] = True
                continue
                
            # Get coordinates for this site
            site_coords = np.array([coordinates[i] for i in site_indices])
            
            # Use spatial clustering to ensure test samples are spatially coherent
            # Create more clusters than we need
            n_clusters = min(10, max(3, n_site_samples // 20))
            
            # Make sure we don't create too many clusters for small sites
            n_clusters = min(n_clusters, n_site_samples // (min_test * 2))
            n_clusters = max(2, n_clusters)  # At least 2 clusters
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=fold_idx + 42)
            clusters = kmeans.fit_predict(site_coords)
            
            # For this fold, select clusters that will go to test
            # Rotate which clusters based on fold index
            cluster_indices = np.arange(n_clusters)
            np.random.RandomState(fold_idx + 42).shuffle(cluster_indices)
            
            # Try different test cluster configurations until we find one that works
            valid_split = False
            for test_size_fraction in [0.2, 0.3, 0.4, 0.15, 0.25]:
                # Calculate how many clusters to assign to test
                n_test_clusters = max(1, int(round(n_clusters * test_size_fraction)))
                test_clusters = cluster_indices[:n_test_clusters]
                
                # Assign samples to appropriate splits
                site_test_indices = []
                site_train_val_indices = []
                
                for i, cluster in enumerate(clusters):
                    if cluster in test_clusters:
                        site_test_indices.append(site_indices[i])
                    else:
                        site_train_val_indices.append(site_indices[i])
                
                # Check if we have enough test samples
                if len(site_test_indices) < min_test:
                    continue
                    
                # Apply spatial buffer between train and test (with safety check)
                if config.spatial_buffer > 0 and len(site_test_indices) > 0:
                    # Get test coordinates
                    test_coords = np.array([coordinates[i] for i in site_test_indices])
                    
                    # Build KD tree for test coordinates
                    test_tree = cKDTree(test_coords)
                    
                    # Check each train/val point
                    filtered_train_val = []
                    for idx in site_train_val_indices:
                        point = coordinates[idx]
                        # Find distance to nearest test point
                        dist, _ = test_tree.query(point, k=1)
                        if dist >= config.spatial_buffer:
                            filtered_train_val.append(idx)
                    
                    # Only apply buffer if we'll have enough samples left
                    if len(filtered_train_val) >= (min_train + min_val):
                        site_train_val_indices = filtered_train_val
                
                # Check if we have enough train+val samples
                if len(site_train_val_indices) < (min_train + min_val):
                    continue
                
                # Split remaining samples into train/val
                np.random.RandomState(fold_idx + 42).shuffle(site_train_val_indices)
                n_val = max(min_val, int(len(site_train_val_indices) * 0.2))
                n_val = min(n_val, len(site_train_val_indices) - min_train)  # Ensure enough for training
                
                site_val_indices = site_train_val_indices[:n_val]
                site_train_indices = site_train_val_indices[n_val:]
                
                # Double-check we have enough samples in each split
                if len(site_train_indices) >= min_train and len(site_val_indices) >= min_val:
                    valid_split = True
                    break
            
            # If we couldn't find a valid split, use a simple ratio split with no spatial clustering
            if not valid_split:
                print(f"  Warning: Couldn't create spatial clusters for Site {site}, using simple split")
                np.random.RandomState(fold_idx + 42).shuffle(site_indices)
                n_test = max(min_test, int(len(site_indices) * 0.2))
                n_val = max(min_val, int(len(site_indices) * 0.2))
                n_train = len(site_indices) - n_test - n_val
                
                if n_train < min_train:  # Still not enough for a valid split
                    print(f"  Site {site} split problem - adding all to training")
                    train_mask[site_indices] = True
                    continue
                
                site_test_indices = site_indices[:n_test]
                site_val_indices = site_indices[n_test:n_test+n_val]
                site_train_indices = site_indices[n_test+n_val:]
            
            # Update masks
            test_mask[site_test_indices] = True
            val_mask[site_val_indices] = True
            train_mask[site_train_indices] = True
            
            print(f"  Site {site}: Train={len(site_train_indices)}, Val={len(site_val_indices)}, Test={len(site_test_indices)}")
        
        # Verify we have data in all splits before adding this fold
        if np.sum(train_mask) == 0 or np.sum(val_mask) == 0 or np.sum(test_mask) == 0:
            print(f"Warning: Fold {fold_idx+1} has an empty split, skipping")
            continue
            
        # Add fold to list
        folds.append({
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'fold_idx': fold_idx
        })
        
        print(f"Fold {fold_idx+1}: Train={np.sum(train_mask)}, Val={np.sum(val_mask)}, Test={np.sum(test_mask)}")
    
    # Make sure we have at least one fold
    if len(folds) == 0:
        raise ValueError("Could not create any valid folds with current settings. Try reducing min_site_samples or spatial_buffer.")
    
    return folds


# ======================================================================
# TRAINING AND EVALUATION
# ======================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                config, device, coordinates_val=None):
    """Train the model with early stopping and learning rate scheduling."""
    print("\nTraining model...")
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Lists to track metrics
    train_losses = []
    val_losses = []
    learning_rates = []
    
    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, 
                                 T_max=config.num_epochs,
                                 eta_min=config.base_learning_rate / 10)
    
    # Training loop
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Reset hard negative mining dataset if configured
        if isinstance(train_loader.dataset, HardNegativeMiningDataset) and epoch >= config.hard_negative_start_epoch:
            train_loader.dataset.update_errors(model, device)
            # Re-create sampler based on updated errors
            if config.use_hard_negative_mining:
                hard_negative_sampler = create_hard_negative_sampler(train_loader.dataset, config)
                if hard_negative_sampler is not None:
                    train_loader = DataLoader(
                        train_loader.dataset,
                        batch_size=config.batch_size,
                        sampler=hard_negative_sampler
                    )
        
        # Training batches
        batch_losses = []
        for batch in train_loader:
            # Unpack batch
            if len(batch) == 4:
                inputs, targets, coords, _ = batch  # With error info
            else:
                inputs, targets, coords = batch     # Without error info
            
            # Move to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            if isinstance(criterion, SpatialLoss):
                # For spatial loss, include coordinates
                batch_coords = torch.tensor(coords, dtype=torch.float32).to(device)
                loss = criterion(outputs, targets, batch_coords)
            else:
                loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            batch_losses.append(loss.item())
            running_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets, coords in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss - use standard MSE for validation
                loss = F.mse_loss(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
        
        # Calculate average validation loss
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Step learning rate scheduler
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"Epoch {epoch+1}/{config.num_epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}", end="")
        
        # Check for improvement
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print(f"\n  → New best validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print("")  # Newline
                
        # Early stopping check
        if epochs_no_improve >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "learning_rates": learning_rates,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses)
    }


def test_time_augmentation(model, X_test, config, device):
    """Apply test-time augmentation to improve prediction accuracy."""
    if not config.use_test_time_augmentation:
        return None
    
    print("\nApplying test-time augmentation...")
    
    # Create augmentation transforms
    transforms = create_data_augmentation(config)
    if transforms is None:
        return None
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_test)
    
    # Set model to evaluation mode
    model.eval()
    
    # Store predictions
    all_predictions = []
    
    # Make predictions with multiple augmentations
    with torch.no_grad():
        # Original prediction
        batch_size = 32
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            outputs = model(batch)
            all_predictions.append(outputs.cpu())
        
        # Augmented predictions
        for aug_idx in range(config.tta_samples):
            print(f"  - TTA iteration {aug_idx+1}/{config.tta_samples}")
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size].clone()
                
                # Apply augmentation to each sample
                for j in range(len(batch)):
                    batch[j] = transforms(batch[j])
                
                # Move to device and predict
                batch = batch.to(device)
                outputs = model(batch)
                all_predictions.append(outputs.cpu())
    
    # Combine predictions
    all_predictions = torch.cat(all_predictions).reshape(config.tta_samples + 1, len(X_test))
    y_pred = torch.mean(all_predictions, dim=0).numpy()
    
    print(f"  - Final predictions created from {config.tta_samples + 1} versions")
    
    return y_pred


def evaluate_model(model, X_test, y_test, coordinates_test, sources_test, config, device, preprocess_config=None):
    """Evaluate the model on test data with proper inverse transformation."""
    print("\nEvaluating model...")
    
    # Detect transformation type
    transform_type = "none"
    if preprocess_config and preprocess_config.get('use_log_transform', False):
        transform_type = "log"
        print(f"  Detected log transformation - will convert results to original biomass scale")
    
    # Convert test data to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions - use test-time augmentation if configured
    if config and config.use_test_time_augmentation:
        y_pred_log = test_time_augmentation(model, X_test, config, device)
    else:
        # Standard prediction
        with torch.no_grad():
            predictions = []
            batch_size = 32
            
            for i in range(0, len(X_test), batch_size):
                batch = X_test_tensor[i:i+batch_size]
                outputs = model(batch)
                predictions.append(outputs.cpu().numpy())
            
            y_pred_log = np.concatenate(predictions)
    
    # Calculate metrics in both scales
    metrics_log, metrics_original = calculate_dual_metrics(y_test, y_pred_log, transform_type)
    
    # Convert to original scale for results DataFrame
    y_test_original = inverse_transform_biomass(y_test, transform_type)
    y_pred_original = inverse_transform_biomass(y_pred_log, transform_type)
    
    # Handle NaN values for DataFrame
    valid_mask = ~(np.isnan(y_test) | np.isnan(y_pred_log))
    
    # Create results dataframe with both scales
    results_df = pd.DataFrame({
        'y_true_log': y_test,
        'y_pred_log': y_pred_log,
        'y_true_original': y_test_original,
        'y_pred_original': y_pred_original,
        'residual_log': np.where(valid_mask, y_pred_log - y_test, np.nan),
        'residual_original': np.where(valid_mask, y_pred_original - y_test_original, np.nan),
        'source': sources_test,
        'valid': valid_mask
    })
    
    # Add coordinates
    results_df['x_coord'] = [coord[0] for coord in coordinates_test]
    results_df['y_coord'] = [coord[1] for coord in coordinates_test]
    
    # Print metrics for both scales
    print(f"\n📊 Test Metrics (Log Scale - Training Scale):")
    print(f"RMSE: {metrics_log['rmse']:.4f}")
    print(f"R²: {metrics_log['r2']:.4f}")
    print(f"MAE: {metrics_log['mae']:.4f}")
    print(f"Spearman: {metrics_log['spearman']:.4f}")
    
    print(f"\n🌳 Test Metrics (Original Scale - Biomass in Mg/ha):")
    print(f"RMSE: {metrics_original['rmse']:.2f} Mg/ha")
    print(f"R²: {metrics_original['r2']:.4f}")
    print(f"MAE: {metrics_original['mae']:.2f} Mg/ha")
    print(f"Spearman: {metrics_original['spearman']:.4f}")
    print(f"Mean True Biomass: {metrics_original['mean_true']:.1f} Mg/ha")
    print(f"Mean Predicted Biomass: {metrics_original['mean_pred']:.1f} Mg/ha")
    print(f"Biomass Range: {metrics_original['min_true']:.1f} - {metrics_original['max_true']:.1f} Mg/ha")
    
    # Analyze by site if multiple sites present
    if len(np.unique(sources_test)) > 1:
        print("\n📍 Metrics by Site (Original Scale - Mg/ha):")
        for site in np.unique(sources_test):
            site_mask = (sources_test == site) & valid_mask
            if np.sum(site_mask) > 0:
                site_y_true = y_test_original[site_mask]
                site_y_pred = y_pred_original[site_mask]
                
                site_rmse = np.sqrt(mean_squared_error(site_y_true, site_y_pred))
                site_r2 = r2_score(site_y_true, site_y_pred)
                site_mae = mean_absolute_error(site_y_true, site_y_pred)
                site_mean_true = np.mean(site_y_true)
                
                print(f"  Site {site}: RMSE={site_rmse:.1f} Mg/ha, R²={site_r2:.3f}, "
                      f"MAE={site_mae:.1f} Mg/ha, Mean={site_mean_true:.1f} Mg/ha, n={np.sum(site_mask)}")
                
                # Add site-specific metrics (original scale)
                metrics_original[f'site_{site}_rmse'] = site_rmse
                metrics_original[f'site_{site}_r2'] = site_r2
                metrics_original[f'site_{site}_mae'] = site_mae
                metrics_original[f'site_{site}_mean_true'] = site_mean_true
    
    # Combine metrics for return
    combined_metrics = {
        'log_scale': metrics_log,
        'original_scale': metrics_original,
        'transform_type': transform_type
    }
    
    return results_df, combined_metrics


# ======================================================================
# MAIN CV CLASS
# ======================================================================

class HybridSpatialCV:
    """Main class for hybrid site-spatial cross-validation."""
    
    def __init__(self, config: HybridCVConfig = None):
        """Initialize with configuration."""
        self.config = config or HybridCVConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def run_cv_fold(self, fold, data):
        """Run a single fold of spatial cross-validation."""
        print(f"\n{'='*40}")
        print(f"Running Hybrid CV Fold {fold['fold_idx']+1}")
        print(f"{'='*40}")
        
        # Get train/val/test data
        X_train = data['X'][fold['train_mask']]
        y_train = data['y'][fold['train_mask']]
        coords_train = [data['coordinates'][i] for i, mask in enumerate(fold['train_mask']) if mask]
        sources_train = data['sources'][fold['train_mask']]
        
        X_val = data['X'][fold['val_mask']]
        y_val = data['y'][fold['val_mask']]
        coords_val = [data['coordinates'][i] for i, mask in enumerate(fold['val_mask']) if mask]
        sources_val = data['sources'][fold['val_mask']]
        
        X_test = data['X'][fold['test_mask']]
        y_test = data['y'][fold['test_mask']]
        coords_test = [data['coordinates'][i] for i, mask in enumerate(fold['test_mask']) if mask]
        sources_test = data['sources'][fold['test_mask']]
        
        # Feature engineering
        if self.config.add_derived_features:
            X_train = add_derived_features(X_train)
            X_val = add_derived_features(X_val)
            X_test = add_derived_features(X_test)
        
        # Feature standardization
        if self.config.standardize_features:
            X_train, X_val, X_test = standardize_features(X_train, X_val, X_test)
        
        # Create data augmentation
        transforms = create_data_augmentation(self.config)
        
        # Create dataset
        if self.config.use_hard_negative_mining:
            train_dataset = HardNegativeMiningDataset(X_train, y_train, coords_train, transforms=transforms)
        else:
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.FloatTensor(coords_train))
        
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val), torch.FloatTensor(coords_val))
        
        # Create data loaders with sensible batch sizes
        train_batch_size = min(self.config.batch_size, max(1, len(train_dataset) // 2))
        val_batch_size = min(self.config.batch_size, max(1, len(val_dataset) // 2))
            
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
            
        # Create model
        input_channels = X_train.shape[1]  # Number of channels
        height, width = X_train.shape[2], X_train.shape[3]  # Spatial dimensions
        
        model = create_model(self.config.model_type, input_channels, height, width, self.device)
        
        # Define optimizer
        optimizer = optim.AdamW(model.parameters(), 
                              lr=self.config.base_learning_rate, 
                              weight_decay=self.config.weight_decay)
        
        # Define loss function
        criterion = create_loss_function(self.config, self.device)
        
        # Train model
        trained_model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            self.config, self.device
        )
        
        # Evaluate model with proper inverse transformation
        results_df, metrics = evaluate_model(
            trained_model, X_test, y_test, coords_test, sources_test,
            self.config, self.device, data.get('preprocess_config', {})
        )
        
        return trained_model, results_df, metrics, history
    
    def run_cross_validation(self, data=None):
        """Run complete hybrid site-spatial cross-validation."""
        print("\n" + "=" * 80)
        print("HYBRID SITE-SPATIAL CROSS-VALIDATION")
        print("=" * 80)
        
        # Load data if not provided
        if data is None:
            data = load_preprocessed_data(self.config)
        
        # Create CV folds
        folds = create_hybrid_site_spatial_split(data['coordinates'], data['sources'], self.config)
        
        # Store results
        fold_models = []
        fold_results = []
        fold_metrics = []
        fold_histories = []
        
        # Run each fold
        for fold_idx, fold in enumerate(folds):
            # Set random seeds for reproducibility
            torch.manual_seed(42 + fold_idx)
            np.random.seed(42 + fold_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42 + fold_idx)
            
            # Run fold
            model, results_df, metrics, history = self.run_cv_fold(fold, data)
            
            # Store results
            fold_models.append(model)
            fold_results.append(results_df)
            fold_metrics.append(metrics)
            fold_histories.append(history)
        
        # Calculate overall metrics
        print("\n" + "="*60)
        print("🏆 HYBRID CV SUMMARY")
        print("="*60)
        
        # Extract metrics for both scales
        log_metrics = [m['log_scale'] for m in fold_metrics]
        original_metrics = [m['original_scale'] for m in fold_metrics]
        
        # Log scale summary
        log_rmse = [m['rmse'] for m in log_metrics]
        log_r2 = [m['r2'] for m in log_metrics]
        log_mae = [m['mae'] for m in log_metrics]
        log_spearman = [m['spearman'] for m in log_metrics]
        
        print("📊 Log Scale (Training Scale) Summary:")
        print(f"RMSE: {np.mean(log_rmse):.4f} ± {np.std(log_rmse):.4f}")
        print(f"R²: {np.mean(log_r2):.4f} ± {np.std(log_r2):.4f}")
        print(f"MAE: {np.mean(log_mae):.4f} ± {np.std(log_mae):.4f}")
        print(f"Spearman: {np.mean(log_spearman):.4f} ± {np.std(log_spearman):.4f}")
        
        # Original scale summary
        orig_rmse = [m['rmse'] for m in original_metrics]
        orig_r2 = [m['r2'] for m in original_metrics]
        orig_mae = [m['mae'] for m in original_metrics]
        orig_spearman = [m['spearman'] for m in original_metrics]
        orig_mean_true = [m['mean_true'] for m in original_metrics]
        
        print(f"\n🌳 Original Scale (Biomass Mg/ha) Summary:")
        print(f"RMSE: {np.mean(orig_rmse):.1f} ± {np.std(orig_rmse):.1f} Mg/ha")
        print(f"R²: {np.mean(orig_r2):.4f} ± {np.std(orig_r2):.4f}")
        print(f"MAE: {np.mean(orig_mae):.1f} ± {np.std(orig_mae):.1f} Mg/ha")
        print(f"Spearman: {np.mean(orig_spearman):.4f} ± {np.std(orig_spearman):.4f}")
        print(f"Mean True Biomass: {np.mean(orig_mean_true):.1f} Mg/ha")
        
        # Performance assessment
        mean_rmse_pct = (np.mean(orig_rmse) / np.mean(orig_mean_true)) * 100
        print(f"\n📈 Performance Assessment:")
        print(f"Relative RMSE: {mean_rmse_pct:.1f}% of mean biomass")
        
        if np.mean(orig_r2) > 0.90:
            performance = "🌟 Excellent"
        elif np.mean(orig_r2) > 0.85:
            performance = "✅ Very Good"
        elif np.mean(orig_r2) > 0.75:
            performance = "👍 Good"
        else:
            performance = "⚠️ Needs Improvement"
        
        print(f"Overall Performance: {performance}")
        
        # Save results
        self._save_cv_results(fold_models, fold_results, fold_metrics, fold_histories, data)
        
        return fold_models, fold_results, fold_metrics, fold_histories
    
    # def _save_cv_results(self, fold_models, fold_results, fold_metrics, fold_histories, data):
    #     """Save cross-validation results with dual-scale metrics."""
    #     # Create timestamp for results
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    #     # Create result directory
    #     result_dir = os.path.join(self.config.cv_dir, timestamp)
    #     os.makedirs(result_dir, exist_ok=True)
        
    #     # Extract metrics for both scales
    #     log_metrics = [m['log_scale'] for m in fold_metrics]
    #     original_metrics = [m['original_scale'] for m in fold_metrics]
        
    #     # Calculate summary statistics
    #     cv_summary = {
    #         'log_scale_metrics': {
    #             'fold_metrics': log_metrics,
    #             'mean_rmse': float(np.mean([m['rmse'] for m in log_metrics])),
    #             'std_rmse': float(np.std([m['rmse'] for m in log_metrics])),
    #             'mean_r2': float(np.mean([m['r2'] for m in log_metrics])),
    #             'std_r2': float(np.std([m['r2'] for m in log_metrics])),
    #             'mean_mae': float(np.mean([m['mae'] for m in log_metrics])),
    #             'std_mae': float(np.std([m['mae'] for m in log_metrics])),
    #             'mean_spearman': float(np.mean([m['spearman'] for m in log_metrics])),
    #             'std_spearman': float(np.std([m['spearman'] for m in log_metrics])),
    #         },
    #         'original_scale_metrics': {
    #             'fold_metrics': original_metrics,
    #             'mean_rmse': float(np.mean([m['rmse'] for m in original_metrics])),
    #             'std_rmse': float(np.std([m['rmse'] for m in original_metrics])),
    #             'mean_r2': float(np.mean([m['r2'] for m in original_metrics])),
    #             'std_r2': float(np.std([m['r2'] for m in original_metrics])),
    #             'mean_mae': float(np.mean([m['mae'] for m in original_metrics])),
    #             'std_mae': float(np.std([m['mae'] for m in original_metrics])),
    #             'mean_spearman': float(np.mean([m['spearman'] for m in original_metrics])),
    #             'std_spearman': float(np.std([m['spearman'] for m in original_metrics])),
    #             'mean_biomass': float(np.mean([m['mean_true'] for m in original_metrics])),
    #             'units': 'Mg/ha'
    #         },
    #         'transform_info': {
    #             'transform_type': fold_metrics[0].get('transform_type', 'unknown'),
    #             'training_scale': 'log' if fold_metrics[0].get('transform_type') == 'log' else 'original',
    #             'evaluation_scales': ['log', 'original']
    #         }
    #     }
        
    #     # Save CV summary
    #     summary_path = os.path.join(result_dir, "cv_summary.json")
    #     with open(summary_path, 'w') as f:
    #         json.dump(cv_summary, f, indent=2)
        
    #     # Save fold results
    #     for i, results in enumerate(fold_results):
    #         results_path = os.path.join(result_dir, f"fold_{i+1}_results.csv")
    #         results.to_csv(results_path, index=False)
        
    #     # Save fold models
    #     for i, model in enumerate(fold_models):
    #         model_path = os.path.join(result_dir, f"fold_{i+1}_model.pt")
    #         torch.save(model.state_dict(), model_path)
        
    #     # Save config
    #     config_dict = self.config.to_dict()
    #     config_path = os.path.join(result_dir, "config.json")
    #     with open(config_path, 'w') as f:
    #         json.dump(config_dict, f, indent=2, default=str)
        
    #     # Save preprocessing info
    #     if 'preprocess_config' in data:
    #         preprocess_path = os.path.join(result_dir, "preprocessing_info.json")
    #         with open(preprocess_path, 'w') as f:
    #             json.dump(data['preprocess_config'], f, indent=2)
        
    #     # Create visualizations
    #     visualize_cv_results(fold_results, fold_metrics, fold_histories, result_dir)
        
    #     print(f"\n💾 Hybrid CV complete. Results saved to {result_dir}")
    #     print(f"📊 Check cv_summary.json for detailed metrics in both scales")
    def _save_cv_results(self, fold_models, fold_results, fold_metrics, fold_histories, data):
        """Save cross-validation results with dual-scale metrics."""
        # Create timestamp for results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create result directory
        result_dir = os.path.join(self.config.cv_dir, timestamp)
        os.makedirs(result_dir, exist_ok=True)
        
        # Extract metrics for both scales
        log_metrics = [m['log_scale'] for m in fold_metrics]
        original_metrics = [m['original_scale'] for m in fold_metrics]
        
        # Calculate summary statistics
        cv_summary = {
            'log_scale_metrics': {
                'fold_metrics': log_metrics,
                'mean_rmse': float(np.mean([m['rmse'] for m in log_metrics])),
                'std_rmse': float(np.std([m['rmse'] for m in log_metrics])),
                'mean_r2': float(np.mean([m['r2'] for m in log_metrics])),
                'std_r2': float(np.std([m['r2'] for m in log_metrics])),
                'mean_mae': float(np.mean([m['mae'] for m in log_metrics])),
                'std_mae': float(np.std([m['mae'] for m in log_metrics])),
                'mean_spearman': float(np.mean([m['spearman'] for m in log_metrics])),
                'std_spearman': float(np.std([m['spearman'] for m in log_metrics])),
            },
            'original_scale_metrics': {
                'fold_metrics': original_metrics,
                'mean_rmse': float(np.mean([m['rmse'] for m in original_metrics])),
                'std_rmse': float(np.std([m['rmse'] for m in original_metrics])),
                'mean_r2': float(np.mean([m['r2'] for m in original_metrics])),
                'std_r2': float(np.std([m['r2'] for m in original_metrics])),
                'mean_mae': float(np.mean([m['mae'] for m in original_metrics])),
                'std_mae': float(np.std([m['mae'] for m in original_metrics])),
                'mean_spearman': float(np.mean([m['spearman'] for m in original_metrics])),
                'std_spearman': float(np.std([m['spearman'] for m in original_metrics])),
                'mean_biomass': float(np.mean([m['mean_true'] for m in original_metrics])),
                'units': 'Mg/ha'
            },
            'transform_info': {
                'transform_type': fold_metrics[0].get('transform_type', 'unknown'),
                'training_scale': 'log' if fold_metrics[0].get('transform_type') == 'log' else 'original',
                'evaluation_scales': ['log', 'original']
            }
        }
        
        # Save CV summary using the NumpyEncoder
        summary_path = os.path.join(result_dir, "cv_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(cv_summary, f, cls=NumpyEncoder, indent=2)
        
        # Save fold results
        for i, results in enumerate(fold_results):
            results_path = os.path.join(result_dir, f"fold_{i+1}_results.csv")
            results.to_csv(results_path, index=False)
        
        # Save fold models
        for i, model in enumerate(fold_models):
            model_path = os.path.join(result_dir, f"fold_{i+1}_model.pt")
            torch.save(model.state_dict(), model_path)
        
        # Save config using the NumpyEncoder
        config_dict = self.config.to_dict()
        config_path = os.path.join(result_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, cls=NumpyEncoder, indent=2)
        
        # Save preprocessing info using the NumpyEncoder
        if 'preprocess_config' in data:
            preprocess_path = os.path.join(result_dir, "preprocessing_info.json")
            with open(preprocess_path, 'w') as f:
                json.dump(data['preprocess_config'], f, cls=NumpyEncoder, indent=2)
        
        # Create visualizations
        visualize_cv_results(fold_results, fold_metrics, fold_histories, result_dir)
        
        print(f"\n💾 Hybrid CV complete. Results saved to {result_dir}")
        print(f"📊 Check cv_summary.json for detailed metrics in both scales")


def main():
    """Main function to run hybrid site-spatial cross-validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hybrid site-spatial cross-validation')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # TODO: Implement YAML config loading
        config = HybridCVConfig()
    else:
        config = HybridCVConfig()
    
    # Run cross-validation
    cv_trainer = HybridSpatialCV(config)
    cv_trainer.run_cross_validation()


if __name__ == "__main__":
    main()