#!/usr/bin/env python
"""
Script to run biomass preprocessing pipeline.

Author: najahpokkiri
Date: 2025-05-28
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.spatial_preprocessing import SpatialAwarePreprocessor
from src.preprocessing.config import SpatialAwarePreprocessingConfig
from src.utils.data_utils import load_yaml_config


def main():
    """Main function to run preprocessing."""
    parser = argparse.ArgumentParser(description='Run spatial-aware biomass preprocessing')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--output-dir', type=str, default='data/processed', 
                        help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        try:
            config_dict = load_yaml_config(args.config)
            config = SpatialAwarePreprocessingConfig.from_dict(config_dict)
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration...")
            config = SpatialAwarePreprocessingConfig()
    else:
        config = SpatialAwarePreprocessingConfig()
    
    # Override output directory if specified
    if args.output_dir:
        config.processed_dir = args.output_dir
    
    # Validate configuration
    print("Configuration:")
    print(f"  Output directory: {config.processed_dir}")
    print(f"  Chip size: {config.chip_size}")
    print(f"  Overlap: {config.overlap}")
    print(f"  Log transform: {config.use_log_transform}")
    print(f"  Min valid pixels: {config.min_valid_pixels}")
    print(f"  Number of raster pairs: {len(config.raster_pairs)}")
    
    # Validate paths
    if not config.validate_paths():
        print("\nError: Some input files are missing. Please check your configuration.")
        print("Update the raster_pairs in your config file or preprocessing/config.py")
        return 1
    
    # Run preprocessing
    preprocessor = SpatialAwarePreprocessor(config)
    
    try:
        success = preprocessor.run_preprocessing()
        
        if success:
            print("\n✅ Preprocessing completed successfully!")
            print(f"Processed data saved to: {config.processed_dir}")
            return 0
        else:
            print("\n❌ Preprocessing failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Preprocessing failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())