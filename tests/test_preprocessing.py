#!/usr/bin/env python
"""
Tests for preprocessing module.

Author: najahpokkiri
Date: 2025-05-28
"""

import sys
import os
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing.config import SpatialAwarePreprocessingConfig
from preprocessing.spatial_preprocessing import SpatialAwarePreprocessor


class TestSpatialAwarePreprocessingConfig:
    """Test the preprocessing configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SpatialAwarePreprocessingConfig()
        
        assert config.chip_size == 24
        assert config.overlap == 0.1
        assert config.use_log_transform == True
        assert config.min_valid_pixels == 0.7
        assert config.test_ratio == 0.2
        assert config.val_ratio == 0.15
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = SpatialAwarePreprocessingConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'chip_size' in config_dict
        assert 'raster_pairs' in config_dict
        assert config_dict['chip_size'] == 24
    
    def test_config_from_dict(self):
        """Test configuration from dictionary creation."""
        config_dict = {
            'chip_size': 32,
            'overlap': 0.2,
            'use_log_transform': False,
            'raster_pairs': []
        }
        
        config = SpatialAwarePreprocessingConfig.from_dict(config_dict)
        
        assert config.chip_size == 32
        assert config.overlap == 0.2
        assert config.use_log_transform == False


class TestSpatialAwarePreprocessor:
    """Test the preprocessing class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SpatialAwarePreprocessingConfig()
        self.config.output_dir = os.path.join(self.temp_dir, "output")
        self.config.processed_dir = os.path.join(self.temp_dir, "processed")
        self.config.raster_pairs = []  # Empty for testing
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_preprocessor_init(self):
        """Test preprocessor initialization."""
        # Ensure the output directory exists before checking
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        preprocessor = SpatialAwarePreprocessor(self.config)
        
        assert preprocessor.config == self.config
        assert os.path.exists(self.config.output_dir)
    
    def test_find_input_files_empty(self):
        """Test finding input files with empty configuration."""
        preprocessor = SpatialAwarePreprocessor(self.config)
        result = preprocessor.find_input_files()
        
        assert result is None or len(result) == 0


def test_imports():
    """Test that all modules can be imported."""
    try:
        from preprocessing.spatial_preprocessing import SpatialAwarePreprocessor
        from preprocessing.config import SpatialAwarePreprocessingConfig
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])