#!/usr/bin/env python
"""
Script to run the complete biomass prediction pipeline.

Author: najahpokkiri
Date: 2025-05-28
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.spatial_preprocessing import SpatialAwarePreprocessor
from src.preprocessing.config import SpatialAwarePreprocessingConfig
from src.models.hybrid_cv import HybridSpatialCV
from src.models.config import HybridCVConfig
from src.utils.data_utils import load_yaml_config
from src.utils.visualisation import plot_data_distribution


def run_preprocessing(config_dict=None):
    """Run preprocessing step."""
    print("=" * 60)
    print("STEP 1: PREPROCESSING")
    print("=" * 60)
    
    # Create preprocessing config
    if config_dict and 'preprocessing' in config_dict:
        preprocess_config = SpatialAwarePreprocessingConfig.from_dict(config_dict['preprocessing'])
    else:
        preprocess_config = SpatialAwarePreprocessingConfig()
    
    # Validate paths
    if not preprocess_config.validate_paths():
        print("Error: Some input files are missing.")
        print("Please update the raster_pairs in your configuration.")
        return False
    
    # Run preprocessing
    preprocessor = SpatialAwarePreprocessor(preprocess_config)
    success = preprocessor.run_preprocessing()
    
    if success:
        print("✅ Preprocessing completed successfully!")
        return True
    else:
        print("❌ Preprocessing failed!")
        return False


def run_training(config_dict=None):
    """Run training step."""
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING")
    print("=" * 60)
    
    # Create training config
    if config_dict and 'training' in config_dict:
        train_config = HybridCVConfig.from_dict(config_dict['training'])
    else:
        train_config = HybridCVConfig()
    
    # Run cross-validation
    cv_trainer = HybridSpatialCV(train_config)
    fold_models, fold_results, fold_metrics, fold_histories = cv_trainer.run_cross_validation()
    
    print("✅ Training completed successfully!")
    return fold_models, fold_results, fold_metrics, fold_histories


def create_data_visualization(preprocess_config):
    """Create data distribution visualizations."""
    print("\n" + "=" * 60)
    print("STEP 3: DATA VISUALIZATION")
    print("=" * 60)
    
    try:
        from src.utils.data_utils import load_preprocessed_data
        
        # Load processed data
        data = load_preprocessed_data(preprocess_config)
        
        # Create visualization
        output_path = os.path.join(preprocess_config.output_dir, "data_distribution.png")
        plot_data_distribution(
            data['X'], data['y'], data['sources'], 
            data['coordinates'], output_path
        )
        
        print("✅ Data visualization completed!")
        return True
        
    except Exception as e:
        print(f"⚠️ Data visualization failed: {e}")
        return False


def main():
    """Main function to run complete pipeline."""
    parser = argparse.ArgumentParser(description='Run complete biomass prediction pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--skip-preprocessing', action='store_true', 
                        help='Skip preprocessing step (use existing processed data)')
    parser.add_argument('--preprocessing-only', action='store_true',
                        help='Run only preprocessing step')
    parser.add_argument('--training-only', action='store_true',
                        help='Run only training step')
    
    args = parser.parse_args()
    
    print("🌳 BIOMASS PREDICTION PIPELINE")
    print("=" * 60)
    
    # Load configuration
    config_dict = None
    if args.config:
        try:
            config_dict = load_yaml_config(args.config)
            print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration...")
    
    try:
        # Step 1: Preprocessing
        if not args.skip_preprocessing and not args.training_only:
            success = run_preprocessing(config_dict)
            if not success:
                return 1
            
            # Create data visualization
            if config_dict and 'preprocessing' in config_dict:
                preprocess_config = SpatialAwarePreprocessingConfig.from_dict(config_dict['preprocessing'])
            else:
                preprocess_config = SpatialAwarePreprocessingConfig()
            create_data_visualization(preprocess_config)
        
        # Exit if preprocessing only
        if args.preprocessing_only:
            print("\n🎉 Preprocessing pipeline completed!")
            return 0
        
        # Step 2: Training
        if not args.preprocessing_only:
            fold_models, fold_results, fold_metrics, fold_histories = run_training(config_dict)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        if not args.preprocessing_only:
            # Print final metrics summary
            import numpy as np
            # rmse_values = [m['rmse'] for m in fold_metrics]
            # r2_values = [m['r2'] for m in fold_metrics]
            # mae_values = [m['mae'] for m in fold_metrics]
            # spearman_values = [m['spearman'] for m in fold_metrics]

            # Choose which scale to report (log_scale or original_scale)
            scale_to_report = 'original_scale'  # or 'log_scale' depending on your preference

            rmse_values = [m[scale_to_report]['rmse'] for m in fold_metrics]
            r2_values = [m[scale_to_report]['r2'] for m in fold_metrics]
            mae_values = [m[scale_to_report]['mae'] for m in fold_metrics]
            spearman_values = [m[scale_to_report]['spearman'] for m in fold_metrics]

            print("\nFinal Cross-Validation Results (in", "original scale):" if scale_to_report == 'original_scale' else "log scale):")
            
            print("\nFinal Cross-Validation Results:")
            print(f"RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}")
            print(f"R²: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
            print(f"MAE: {np.mean(mae_values):.4f} ± {np.std(mae_values):.4f}")
            print(f"Spearman: {np.mean(spearman_values):.4f} ± {np.std(spearman_values):.4f}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())