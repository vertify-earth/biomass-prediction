#!/usr/bin/env python
"""
Revised Spatial-Aware Biomass Preprocessing Script

This script preprocesses biomass and remote sensing data for machine learning while 
ensuring proper spatial handling and balanced data splits.

Author: najahpokkiri
Last Update: 2025-05-28
"""

import os
import sys
import json
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import geopandas as gpd
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from .config import SpatialAwarePreprocessingConfig

# Suppress warnings
warnings.filterwarnings('ignore')


class SpatialAwarePreprocessor:
    """Main class for spatial-aware biomass preprocessing."""
    
    def __init__(self, config: SpatialAwarePreprocessingConfig = None):
        """Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration instance
        """
        self.config = config or SpatialAwarePreprocessingConfig()
        
    def find_input_files(self):
        """Load raster pairs from the configuration."""
        print("Loading specified raster pairs...")
        
        paired_files = []
        for pair in self.config.raster_pairs:
            sat_path, bio_path, site_id = pair
            
            # Check if files exist
            if not os.path.exists(bio_path):
                print(f"Warning: Biomass file not found: {bio_path}")
                continue
                
            if not os.path.exists(sat_path):
                print(f"Warning: Satellite file not found: {sat_path}")
                continue
                
            # Add to paired files
            paired_files.append({
                'site_id': site_id,
                'biomass_file': bio_path,
                'satellite_file': sat_path
            })
        
        if not paired_files:
            print("Error: No valid raster pairs found")
            return None
            
        print(f"Found {len(paired_files)} valid raster pairs:")
        for pair in paired_files:
            print(f"  Site {pair['site_id']}: {os.path.basename(pair['satellite_file'])} + {os.path.basename(pair['biomass_file'])}")
        
        return paired_files

    def extract_chips(self, file_pairs):
        """Extract chips from biomass and satellite data rasters with robust NaN handling."""
        print("\n==== Extracting Chips ====")
        
        all_sat_chips = []
        all_bio_chips = []
        all_sources = []  # Track which site each chip comes from
        all_coords = []   # Store coordinates for each chip
        
        site_counts = []
        site_names = []
        site_coords = []  # Central coordinates of sites
        
        # Statistics for NaN handling
        nan_statistics = {
            'total_potential_chips': 0,
            'discarded_bio_nan': 0,
            'discarded_sat_excessive_nan': 0,
            'imputed_sat_minor_nan': 0,
            'clean_chips': 0
        }
        
        for i, file_pair in enumerate(file_pairs):
            site_id = file_pair['site_id']
            bio_file = file_pair['biomass_file']
            sat_file = file_pair['satellite_file']
            
            print(f"\nProcessing site {site_id}:")
            print(f"  Biomass: {bio_file}")
            print(f"  Satellite: {sat_file}")
            
            # Open raster files
            with rasterio.open(bio_file) as bio_src, rasterio.open(sat_file) as sat_src:
                # Read data
                bio_data = bio_src.read(1)  # Biomass data (single band)
                sat_data = sat_src.read()   # Satellite data (multiple bands)
                
                # Get site central coordinates
                site_center_x = bio_src.bounds.left + (bio_src.bounds.right - bio_src.bounds.left) / 2
                site_center_y = bio_src.bounds.bottom + (bio_src.bounds.top - bio_src.bounds.bottom) / 2
                site_coords.append([site_center_x, site_center_y])
                
                # Print shapes
                print(f"  Biomass shape: {bio_data.shape}")
                print(f"  Satellite shape: {sat_data.shape}")
                
                # Calculate chip extraction parameters
                height, width = bio_data.shape
                chip_size = self.config.chip_size
                stride = int(chip_size * (1 - self.config.overlap))
                
                # Calculate number of potential chips
                n_y = (height - chip_size) // stride + 1
                n_x = (width - chip_size) // stride + 1
                total_potential = n_y * n_x
                nan_statistics['total_potential_chips'] += total_potential
                print(f"  Potential chips: {total_potential} ({n_y}×{n_x})")
                
                # Extract chips
                site_chips = []
                site_bio = []
                site_coords_list = []
                
                site_stats = {
                    'discarded_bio_nan': 0,
                    'discarded_sat_excessive_nan': 0,
                    'imputed_sat_minor_nan': 0,
                    'clean_chips': 0
                }
                
                for y in range(0, height - chip_size + 1, stride):
                    for x in range(0, width - chip_size + 1, stride):
                        # Extract biomass chip
                        bio_chip = bio_data[y:y+chip_size, x:x+chip_size].copy()
                        
                        # Check if biomass chip has enough valid data
                        valid_mask_bio = ~np.isnan(bio_chip)
                        if bio_src.nodata is not None:
                            valid_mask_bio = valid_mask_bio & (bio_chip != bio_src.nodata)
                        
                        valid_fraction_bio = np.sum(valid_mask_bio) / (chip_size * chip_size)
                        
                        # If biomass chip doesn't have enough valid data, skip it
                        if valid_fraction_bio < self.config.min_valid_pixels:
                            site_stats['discarded_bio_nan'] += 1
                            continue
                        
                        # Calculate mean biomass from valid pixels
                        mean_biomass = np.mean(bio_chip[valid_mask_bio])
                        
                        # Skip if biomass mean is invalid (e.g., all zeros leading to NaN after log transform)
                        if np.isnan(mean_biomass) or mean_biomass <= 0:
                            site_stats['discarded_bio_nan'] += 1
                            continue
                        
                        # Transform biomass if configured
                        if self.config.use_log_transform:
                            mean_biomass = np.log(mean_biomass + 1)
                        
                        # Extract satellite chip
                        sat_chip = sat_data[:, y:y+chip_size, x:x+chip_size].copy()
                        
                        # Check for NaN values in satellite data
                        nan_mask_sat = np.isnan(sat_chip)
                        nan_fraction_sat = np.sum(nan_mask_sat) / sat_chip.size
                        
                        # Handle satellite NaN values based on fraction
                        if nan_fraction_sat > 0:
                            if nan_fraction_sat > self.config.max_sat_nan_fraction:  # If too many NaNs, discard the chip
                                site_stats['discarded_sat_excessive_nan'] += 1
                                continue
                            else:
                                # Impute NaNs in satellite data - band by band, pixel by pixel median
                                for band_idx in range(sat_chip.shape[0]):
                                    band = sat_chip[band_idx]
                                    if np.any(np.isnan(band)):
                                        # Get median of valid values in this band for this chip
                                        band_valid_values = band[~np.isnan(band)]
                                        if len(band_valid_values) > 0:
                                            band_median = np.median(band_valid_values)
                                            band[np.isnan(band)] = band_median
                                        else:
                                            # If entire band is NaN, use 0 (last resort)
                                            band[np.isnan(band)] = 0
                                
                                site_stats['imputed_sat_minor_nan'] += 1
                        else:
                            site_stats['clean_chips'] += 1
                        
                        # Get pixel coordinates in the original raster
                        center_y, center_x = y + chip_size // 2, x + chip_size // 2
                        
                        # Convert to geo-coordinates if available
                        if bio_src.transform:
                            geo_x, geo_y = bio_src.xy(center_y, center_x)
                            chip_coord = (geo_x, geo_y)
                        else:
                            chip_coord = (center_x, center_y)
                        
                        # Final check for any remaining NaNs
                        if np.any(np.isnan(sat_chip)):
                            print(f"WARNING: NaNs still present after imputation in chip at ({x},{y})")
                            # Replace any remaining NaNs with zeros as last resort
                            sat_chip = np.nan_to_num(sat_chip, nan=0.0)
                        
                        # Add to list
                        site_chips.append(sat_chip)
                        site_bio.append(mean_biomass)
                        site_coords_list.append(chip_coord)
                
                # Add site data to global lists
                n_chips = len(site_chips)
                if n_chips > 0:
                    all_sat_chips.extend(site_chips)
                    all_bio_chips.extend(site_bio)
                    all_sources.extend([i] * n_chips)
                    all_coords.extend(site_coords_list)
                    
                    site_counts.append(n_chips)
                    site_names.append(site_id)
                    
                    # Update global statistics
                    for key in site_stats:
                        nan_statistics[key] += site_stats[key]
                    
                    print(f"  Extracted {n_chips} valid chips:")
                    print(f"    - Clean chips: {site_stats['clean_chips']}")
                    print(f"    - Chips with minor NaN imputation: {site_stats['imputed_sat_minor_nan']}")
                    print(f"    - Discarded (excessive satellite NaN): {site_stats['discarded_sat_excessive_nan']}")
                    print(f"    - Discarded (invalid biomass): {site_stats['discarded_bio_nan']}")
                else:
                    print(f"  No valid chips extracted")
                    site_counts.append(0)
                    site_names.append(site_id)
        
        # Convert to numpy arrays
        X = np.array(all_sat_chips)
        y = np.array(all_bio_chips)
        sources = np.array(all_sources)
        
        # Print summary
        print(f"\nTotal extracted: {len(all_sat_chips)} chips from {len(site_counts)} sites")
        if nan_statistics['total_potential_chips'] > 0:
            print(f"NaN handling summary:")
            print(f"  - Total potential chips: {nan_statistics['total_potential_chips']}")
            print(f"  - Clean chips (no NaN): {nan_statistics['clean_chips']} ({100*nan_statistics['clean_chips']/nan_statistics['total_potential_chips']:.1f}%)")
            print(f"  - Chips with minor NaN imputation: {nan_statistics['imputed_sat_minor_nan']} ({100*nan_statistics['imputed_sat_minor_nan']/nan_statistics['total_potential_chips']:.1f}%)")
            print(f"  - Discarded (excessive satellite NaN): {nan_statistics['discarded_sat_excessive_nan']} ({100*nan_statistics['discarded_sat_excessive_nan']/nan_statistics['total_potential_chips']:.1f}%)")
            print(f"  - Discarded (invalid biomass): {nan_statistics['discarded_bio_nan']} ({100*nan_statistics['discarded_bio_nan']/nan_statistics['total_potential_chips']:.1f}%)")
            total_kept = nan_statistics['clean_chips'] + nan_statistics['imputed_sat_minor_nan']
            print(f"  - Total kept: {total_kept} ({100*total_kept/nan_statistics['total_potential_chips']:.1f}%)")
        
        for i, (count, name) in enumerate(zip(site_counts, site_names)):
            print(f"  Site {name}: {count} chips")
        
        site_info = {
            'counts': site_counts,
            'names': site_names,
            'coords': site_coords,
        }
        
        return X, y, sources, all_coords, site_info

    def analyze_spatial_autocorrelation(self, X, y, coordinates, sources, site_info):
        """Analyze spatial autocorrelation in the data."""
        print("\n==== Analyzing Spatial Autocorrelation ====")
        
        site_autocorr = {}
        site_ranges = {}
        
        # Group by site
        unique_sites = np.unique(sources)
        
        for site in unique_sites:
            site_name = site_info['names'][site]
            mask = (sources == site)
            
            # Skip sites with too few samples
            if np.sum(mask) < 10:
                print(f"Skipping site {site_name} (too few samples: {np.sum(mask)})")
                continue
            
            site_y = y[mask]
            site_coords = [coordinates[i] for i in range(len(coordinates)) if mask[i]]
            
            # Convert coordinates to numpy array
            coord_array = np.array(site_coords)
            
            # Calculate distances between all pairs of points
            distances = squareform(pdist(coord_array))
            
            # Create array of absolute biomass differences
            biomass_diffs = np.abs(site_y.reshape(-1, 1) - site_y.reshape(1, -1))
            
            # Flatten the arrays (excluding self-comparisons)
            mask = ~np.eye(distances.shape[0], dtype=bool)
            distances_flat = distances[mask]
            biomass_diffs_flat = biomass_diffs[mask]
            
            # Calculate correlation
            correlation, pvalue = spearmanr(distances_flat, biomass_diffs_flat)
            
            # Store results
            site_autocorr[site] = {
                'correlation': correlation,
                'pvalue': pvalue,
                'n_samples': np.sum(mask)
            }
            
            # Estimate range of spatial autocorrelation
            # (simplified approach using distance bins)
            max_dist = np.max(distances_flat)
            bins = np.linspace(0, max_dist, 10)
            bin_indices = np.digitize(distances_flat, bins) - 1
            
            bin_corrs = []
            for i in range(len(bins) - 1):
                bin_mask = (bin_indices == i)
                if np.sum(bin_mask) > 10:
                    bin_corr, _ = spearmanr(
                        distances_flat[bin_mask], 
                        biomass_diffs_flat[bin_mask]
                    )
                    bin_corrs.append((bins[i] + bins[i+1]) / 2)
            
            # Find distance where correlation is below threshold
            # (simplified estimate of spatial range)
            autocorr_range = max_dist / 2  # Default to half the maximum distance
            
            # Store estimated range
            site_ranges[site] = autocorr_range
            
            print(f"Site {site_name}:")
            print(f"  Spatial autocorrelation: {correlation:.4f} (p={pvalue:.4f})")
            print(f"  Estimated autocorrelation range: {autocorr_range:.2f} units")
        
        return site_autocorr, site_ranges

    def create_site_based_split(self, X, y, coordinates, sources, site_info):
        """Create a site-based train/val/test split that respects spatial boundaries."""
        print("\n==== Creating Spatially-Aware Data Split ====")
        
        # Get unique sites and their counts
        unique_sites = np.unique(sources)
        sites_with_counts = [(site, np.sum(sources == site), site_info['names'][i]) 
                            for i, site in enumerate(unique_sites)]
        
        # Sort sites by size for strategic selection
        sites_with_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Print site information
        print("Site distribution:")
        for site_id, count, name in sites_with_counts:
            print(f"  {name} (ID: {site_id}): {count} samples ({count/len(y)*100:.1f}%)")
        
        # Strategic site selection:
        # - Select medium-sized site for testing (Site_2 or Site_3)
        # - Use all other sites for train/val split
        
        # For testing, select either the second or third largest site
        # depending on which has a more suitable size
        if len(sites_with_counts) >= 3:
            if sites_with_counts[1][1] > 100:  # If second largest site has >100 samples
                test_sites = [sites_with_counts[1][0]]  # Use second largest site for testing
            else:
                test_sites = [sites_with_counts[2][0]]  # Use third largest site for testing
        else:
            # Fallback if fewer than 3 sites
            test_sites = [sites_with_counts[-1][0]]  # Use smallest site
        
        # Use all other sites for train/validation
        train_val_sites = [s[0] for s in sites_with_counts if s[0] not in test_sites]
        
        # Create masks for testing
        test_mask = np.zeros_like(sources, dtype=bool)
        for site in test_sites:
            test_mask |= (sources == site)
        
        # Create training/validation mask
        train_val_mask = np.zeros_like(sources, dtype=bool)
        for site in train_val_sites:
            train_val_mask |= (sources == site)
        
        # Randomly split training data into train and validation
        # (this ensures validation is representative)
        train_val_indices = np.where(train_val_mask)[0]
        np.random.shuffle(train_val_indices)
        
        # Ensure validation set has at least min_val_samples (default: 60)
        val_size = max(self.config.min_val_samples, int(self.config.val_ratio * len(train_val_indices)))
        val_size = min(val_size, len(train_val_indices) - 10)  # Ensure enough for training
        
        val_indices = train_val_indices[:val_size]
        train_indices = train_val_indices[val_size:]
        
        # Create final masks
        train_mask = np.zeros_like(sources, dtype=bool)
        train_mask[train_indices] = True
        
        val_mask = np.zeros_like(sources, dtype=bool)
        val_mask[val_indices] = True
        
        # Get final split counts
        train_count = np.sum(train_mask)
        val_count = np.sum(val_mask)
        test_count = np.sum(test_mask)
        
        # Print split information
        print("\nFinal data split:")
        print(f"  Training: {train_count} samples ({train_count/len(y)*100:.1f}%)")
        print(f"  Validation: {val_count} samples ({val_count/len(y)*100:.1f}%)")
        print(f"  Testing: {test_count} samples ({test_count/len(y)*100:.1f}%)")
        
        # Print site distribution in each split
        train_site_counts = [(np.sum((sources == site) & train_mask), site_info['names'][i])
                             for i, site in enumerate(unique_sites)]
        val_site_counts = [(np.sum((sources == site) & val_mask), site_info['names'][i])
                           for i, site in enumerate(unique_sites)]
        test_site_counts = [(np.sum((sources == site) & test_mask), site_info['names'][i])
                            for i, site in enumerate(unique_sites)]
        
        print("\nTraining set site distribution:")
        for count, name in sorted(train_site_counts, reverse=True):
            if count > 0:
                print(f"  {name}: {count} samples ({count/train_count*100:.1f}%)")
        
        print("\nValidation set site distribution:")
        for count, name in sorted(val_site_counts, reverse=True):
            if count > 0:
                print(f"  {name}: {count} samples ({count/val_count*100:.1f}%)")
        
        print("\nTest set site distribution:")
        for count, name in sorted(test_site_counts, reverse=True):
            if count > 0:
                print(f"  {name}: {count} samples ({count/test_count*100:.1f}%)")
        
        # Calculate distance between training and testing samples to verify spatial separation
        train_coords = np.array([coordinates[i] for i, m in enumerate(train_mask) if m])
        test_coords = np.array([coordinates[i] for i, m in enumerate(test_mask) if m])
        
        if len(train_coords) > 0 and len(test_coords) > 0:
            # Calculate minimum distance between any training and test point
            min_distances = []
            
            # Use a subset for efficiency if datasets are very large
            max_points = 1000
            train_subset = train_coords[:min(max_points, len(train_coords))]
            test_subset = test_coords[:min(max_points, len(test_coords))]
            
            for test_point in test_subset:
                distances = np.sqrt(np.sum((train_subset - test_point)**2, axis=1))
                min_distances.append(np.min(distances))
            
            avg_min_distance = np.mean(min_distances)
            print(f"\nSpatial separation: Average minimum distance between train and test samples: {avg_min_distance:.2f} units")
        
        # Create dictionary with split information
        split_info = {
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'train_sites': [site_info['names'][i] for i, site in enumerate(unique_sites) if np.sum((sources == site) & train_mask) > 0],
            'val_sites': [site_info['names'][i] for i, site in enumerate(unique_sites) if np.sum((sources == site) & val_mask) > 0],
            'test_sites': [site_info['names'][i] for i, site in enumerate(unique_sites) if np.sum((sources == site) & test_mask) > 0]
        }
        
        return split_info

    def save_processed_data(self, X, y, sources, coordinates, split_info, site_info, site_autocorr):
        """Save processed data for training."""
        print("\n==== Saving Processed Data ====")
        
        # Create timestamp for this preprocessing run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create paths
        X_path = os.path.join(self.config.processed_dir, f"X_{timestamp}.npy")
        y_path = os.path.join(self.config.processed_dir, f"y_{timestamp}.npy")
        sources_path = os.path.join(self.config.processed_dir, f"sources_{timestamp}.npy")
        coord_path = os.path.join(self.config.processed_dir, f"coordinates_{timestamp}.pkl")
        split_path = os.path.join(self.config.processed_dir, f"split_{timestamp}.npz")
        config_path = os.path.join(self.config.processed_dir, f"preprocessing_config_{timestamp}.json")
        latest_path = os.path.join(self.config.processed_dir, "latest.txt")
        
        # Save numpy arrays
        np.save(X_path, X)
        np.save(y_path, y)
        np.save(sources_path, sources)
        
        # Save coordinates
        with open(coord_path, 'wb') as f:
            pickle.dump(coordinates, f)
        
        # Save split masks
        np.savez(split_path, 
                 train_mask=split_info['train_mask'],
                 val_mask=split_info['val_mask'], 
                 test_mask=split_info['test_mask'])
        
        # Save preprocessing config and site information
        config_dict = {
            'timestamp': timestamp,
            'chip_size': self.config.chip_size,
            'overlap': self.config.overlap,
            'use_log_transform': self.config.use_log_transform,
            'site_counts': site_info['counts'],
            'site_names': site_info['names'],
            'train_sites': split_info['train_sites'],
            'val_sites': split_info['val_sites'],
            'test_sites': split_info['test_sites'],
            'spatial_autocorr': {str(site): {'correlation': float(info['correlation']), 
                                             'pvalue': float(info['pvalue'])} 
                                for site, info in site_autocorr.items()}
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Update latest timestamp reference
        with open(latest_path, 'w') as f:
            f.write(timestamp)
        
        print(f"Data saved with timestamp: {timestamp}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"sources shape: {sources.shape}")
        print(f"coordinates: {len(coordinates)} points")
        print(f"Split masks saved with train: {np.sum(split_info['train_mask'])}, " 
              f"val: {np.sum(split_info['val_mask'])}, test: {np.sum(split_info['test_mask'])} samples")

    def run_preprocessing(self):
        """Main function for preprocessing biomass data."""
        print("=" * 80)
        print("Revised Spatial-Aware Biomass Preprocessing")
        print("=" * 80)
        
        # Validate configuration
        if not self.config.validate_paths():
            print("Error: Some input files are missing. Please check your configuration.")
            return False
        
        # Find input files
        file_pairs = self.find_input_files()
        if not file_pairs:
            print("Error: No valid raster pairs found")
            return False
        
        # Extract chips from rasters
        X, y, sources, coordinates, site_info = self.extract_chips(file_pairs)
        
        if len(X) == 0:
            print("Error: No chips extracted")
            return False
        
        # Analyze spatial autocorrelation
        site_autocorr, site_ranges = self.analyze_spatial_autocorrelation(
            X, y, coordinates, sources, site_info
        )
        
        # Create data split
        split_info = self.create_site_based_split(
            X, y, coordinates, sources, site_info
        )
        
        # Save processed data
        self.save_processed_data(
            X, y, sources, coordinates, split_info, 
            site_info, site_autocorr
        )
        
        print("\n" + "=" * 80)
        print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return True


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run spatial-aware biomass preprocessing')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # TODO: Implement YAML config loading
        config = SpatialAwarePreprocessingConfig()
    else:
        config = SpatialAwarePreprocessingConfig()
    
    # Run preprocessing
    preprocessor = SpatialAwarePreprocessor(config)
    success = preprocessor.run_preprocessing()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()