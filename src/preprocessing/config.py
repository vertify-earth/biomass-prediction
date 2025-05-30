#!/usr/bin/env python
"""
Configuration classes for spatial-aware biomass preprocessing.

Author: najahpokkiri
Date: 2025-05-28
"""

import os
from typing import List, Tuple


class SpatialAwarePreprocessingConfig:
    """Configuration for spatial-aware biomass preprocessing."""
    
    def __init__(self):
        """Initialize configuration with default parameters."""
        
        self.raster_pairs = [
            ("/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_dem_yellapur_2020.tif", 
             "/teamspace/studios/dl2/clean/data/agbd_yellapur_reprojected_1.tif", 
             "Yellapur"),
            ("/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_betul_2020_clipped.tif",
             "/teamspace/studios/dl2/clean/data/01_Betul_AGB40_band1_onImgGrid.tif", 
             "Betul"), 
            ("/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_goa_achankumar_2020_clipped.tif", 
             "/teamspace/studios/dl2/clean/data/02_Achanakmar_AGB40_band1_onImgGrid.tif", 
             "Achanakmar"),
            ("/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_goa_khaoyai_2020_clipped.tif",
             "/teamspace/studios/dl2/clean/data/05_Khaoyai_AGB40_band1_onImgGrid.tif", 
             "Khaoyai"),
            ("/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_goa_uppangala_2020_clipped.tif", 
             "/teamspace/studios/dl2/clean/data/04_Uppangala_AGB40_band1_onImgGrid.tif", 
             "Uppangala"),
        ]

        self.output_dir = "results/preprocessing"
        self.processed_dir = "data/processed"
        
        self.chip_size = 24  # Size of extracted chips (pixels)
        self.overlap = 0.1   # Overlap between chips (as fraction of chip_size)
        
        self.use_log_transform = True  # Log transform biomass data
        
        self.min_valid_pixels = 0.7  # Minimum fraction of valid pixels in a chip
        
        self.max_sat_nan_fraction = 0.3  # Maximum fraction of NaN allowed in satellite data
        
        self.test_ratio = 0.2   # Fraction of data for testing
        self.val_ratio = 0.15   # Fraction of training data for validation
        self.min_val_samples = 60  # Minimum number of validation samples
        
        self._create_directories()
    
    def _create_directories(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def update_raster_pairs(self, raster_pairs: List[Tuple[str, str, str]]):
        """Update raster pairs configuration.
        
        Args:
            raster_pairs: List of tuples (satellite_path, biomass_path, site_name)
        """
        self.raster_pairs = raster_pairs
    
    def validate_paths(self) -> bool:
        """Validate that all specified paths exist.
        
        Returns:
            bool: True if all paths exist, False otherwise
        """
        missing_files = []
        
        for sat_path, bio_path, site_name in self.raster_pairs:
            if not os.path.exists(sat_path):
                missing_files.append(f"Satellite file for {site_name}: {sat_path}")
            if not os.path.exists(bio_path):
                missing_files.append(f"Biomass file for {site_name}: {bio_path}")
        
        if missing_files:
            print("Missing files:")
            for file in missing_files:
                print(f"  - {file}")
            return False
        
        return True

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.
        
        Returns:
            dict: Configuration as dictionary
        """
        return {
            'raster_pairs': self.raster_pairs,
            'output_dir': self.output_dir,
            'processed_dir': self.processed_dir,
            'chip_size': self.chip_size,
            'overlap': self.overlap,
            'use_log_transform': self.use_log_transform,
            'min_valid_pixels': self.min_valid_pixels,
            'max_sat_nan_fraction': self.max_sat_nan_fraction,
            'test_ratio': self.test_ratio,
            'val_ratio': self.val_ratio,
            'min_val_samples': self.min_val_samples,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create configuration from dictionary."""
        config = cls()

        for key, value in config_dict.items():
            if key == "raster_pairs" and isinstance(value, list):
                converted_pairs = []
                for item in value:
                    if isinstance(item, dict):
                        converted_pairs.append((
                            item.get("satellite_path"),
                            item.get("biomass_path"),
                            item.get("site_name")
                        ))
                    else:
                        converted_pairs.append(item)
                setattr(config, key, converted_pairs)
            elif hasattr(config, key):
                setattr(config, key, value)

        config._create_directories()
        return config

        
        # Input raster pairs - users should update this
        # self.raster_pairs = [
        #     # Each tuple is (satellite_image_path, biomass_raster_path, site_name)
        #     ("data/raw/satellite/s1_s2_l8_palsar_ch_dem_yellapur_2020.tif", 
        #      "data/raw/biomass/agbd_yellapur_reprojected_1.tif", 
        #      "Site_1"),
        #     ("data/raw/satellite/s1_s2_l8_palsar_ch_betul_2020_clipped.tif",
        #      "data/raw/biomass/01_Betul_AGB40_band1_onImgGrid.tif", 
        #      "Site_2"), 
        #     ("data/raw/satellite/s1_s2_l8_palsar_ch_goa_achankumar_2020_clipped.tif", 
        #      "data/raw/biomass/02_Achanakmar_AGB40_band1_onImgGrid.tif", 
        #      "Site_3"),
        #     ("data/raw/satellite/s1_s2_l8_palsar_ch_goa_khaoyai_2020_clipped.tif",
        #      "data/raw/biomass/05_Khaoyai_AGB40_band1_onImgGrid.tif", 
        #      "Site_4"),
        #     ("data/raw/satellite/s1_s2_l8_palsar_ch_goa_uppangala_2020_clipped.tif", 
        #      "data/raw/biomass/04_Uppangala_AGB40_band1_onImgGrid.tif", 
        #      "Site_5"),
        # ]
        