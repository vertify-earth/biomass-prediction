#!/usr/bin/env python
"""
Configuration classes for hybrid site-spatial cross-validation.

Author: najahpokkiri
Date: 2025-05-28
"""

import os


class HybridCVConfig:
    """Configuration for hybrid site-spatial CV biomass model training."""
    
    def __init__(self):
        """Initialize configuration with default parameters."""
        
        # Paths
        self.preprocessed_dir = "data/processed"
        self.results_dir = "results/hybrid_results"
        self.visualization_dir = "results/visualizations"
        self.cv_dir = "results/cv_results"
        
        # Cross-validation settings
        self.n_folds = 5                      # Number of CV folds
        self.spatial_buffer = 20              # Buffer distance between train and test
        self.min_site_samples = 10            # Minimum samples required for site splitting
        
        # Model configuration
        self.model_type = "cnn_coordinate"    # Spatial-aware CNN model
        
        # Feature engineering
        self.add_derived_features = True      # Add vegetation indices
        self.standardize_features = True      # Standardize features
        
        # Loss function
        self.loss_function = "huber"          # Options: "mse", "huber", "spatial"
        self.huber_delta = 1.0                # Delta parameter for Huber loss
        self.spatial_loss_weight = 0.2        # Weight for spatial loss component
        
        # Training parameters
        self.batch_size = 16
        self.num_epochs = 100
        self.base_learning_rate = 0.001
        self.weight_decay = 5e-3              # Strong regularization
        self.early_stopping_patience = 20
        
        # Data augmentation
        self.use_geometric_aug = True
        self.use_spectral_aug = True
        self.aug_probability = 0.7
        
        # Advanced sampling
        self.use_hard_negative_mining = True   # Focus on difficult samples
        self.hard_negative_start_epoch = 20    # Start hard negative after this epoch
        self.oversampling_factor = 2.0         # Oversample by this factor
        
        # Test-time augmentation
        self.use_test_time_augmentation = True # Apply augmentation at test time
        self.tta_samples = 3                   # Number of augmentations per test sample
        
        # Create output directories
        self._create_directories()

    def _create_directories(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.cv_dir, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.
        
        Returns:
            dict: Configuration as dictionary
        """
        return {attr: getattr(self, attr) for attr in dir(self) 
                if not attr.startswith('_') and not callable(getattr(self, attr))}
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            HybridCVConfig: Configuration instance
        """
        config = cls()
        
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config._create_directories()
        return config