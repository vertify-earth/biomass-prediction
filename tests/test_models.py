#!/usr/bin/env python
"""
Tests for models module.

Author: najahpokkiri
Date: 2025-05-28
"""

import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.config import HybridCVConfig
from models.cnn_models import CNNCoordinateModel, create_model
from models.loss_functions import HuberLoss, SpatialLoss, create_loss_function


class TestHybridCVConfig:
    """Test the training configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HybridCVConfig()
        
        assert config.n_folds == 5
        assert config.model_type == "cnn_coordinate"
        assert config.batch_size == 16
        assert config.num_epochs == 100
        assert config.use_hard_negative_mining == True
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = HybridCVConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'n_folds' in config_dict
        assert 'model_type' in config_dict


class TestCNNCoordinateModel:
    """Test the CNN model architecture."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.input_channels = 10
        self.height = 24
        self.width = 24
        self.batch_size = 4
    
    def test_model_creation(self):
        """Test model creation."""
        model = CNNCoordinateModel(self.input_channels, self.height, self.width)
        
        assert isinstance(model, torch.nn.Module)
        assert model.input_channels == self.input_channels + 2  # +2 for coordinates
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = CNNCoordinateModel(self.input_channels, self.height, self.width)
        
        # Create dummy input
        x = torch.randn(self.batch_size, self.input_channels, self.height, self.width)
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (self.batch_size,)
        assert not torch.isnan(output).any()
    
    def test_create_model_function(self):
        """Test the create_model function."""
        model = create_model("cnn_coordinate", self.input_channels, 
                           self.height, self.width, self.device)
        
        assert isinstance(model, CNNCoordinateModel)
        
        # Test invalid model type
        with pytest.raises(ValueError):
            create_model("invalid_model", self.input_channels, 
                        self.height, self.width, self.device)


class TestLossFunctions:
    """Test custom loss functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 8
        
        # Create dummy data
        self.y_true = torch.randn(self.batch_size)
        self.y_pred = torch.randn(self.batch_size)
        self.coordinates = torch.randn(self.batch_size, 2)
    
    def test_huber_loss(self):
        """Test Huber loss function."""
        loss_fn = HuberLoss(delta=1.0)
        
        loss = loss_fn(self.y_pred, self.y_true)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_spatial_loss(self):
        """Test spatial loss function."""
        loss_fn = SpatialLoss(mse_weight=0.8, spatial_weight=0.2, device=self.device)
        
        # Test with coordinates
        loss = loss_fn(self.y_pred, self.y_true, self.coordinates)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        
        # Test without coordinates (should fall back to MSE)
        loss_no_coord = loss_fn(self.y_pred, self.y_true)
        assert isinstance(loss_no_coord, torch.Tensor)
    
    def test_create_loss_function(self):
        """Test loss function creation."""
        config = HybridCVConfig()
        
        # Test Huber loss
        config.loss_function = "huber"
        loss_fn = create_loss_function(config, self.device)
        assert isinstance(loss_fn, HuberLoss)
        
        # Test MSE loss
        config.loss_function = "mse"
        loss_fn = create_loss_function(config, self.device)
        assert isinstance(loss_fn, torch.nn.MSELoss)
        
        # Test spatial loss
        config.loss_function = "spatial"
        loss_fn = create_loss_function(config, self.device)
        assert isinstance(loss_fn, SpatialLoss)


def test_model_imports():
    """Test that all model modules can be imported."""
    try:
        from models.cnn_models import CNNCoordinateModel
        from models.loss_functions import HuberLoss, SpatialLoss
        from models.config import HybridCVConfig
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])