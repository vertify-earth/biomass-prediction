#!/usr/bin/env python
"""
Script to run hybrid site-spatial cross-validation training.

Author: najahpokkiri
Date: 2025-05-28
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.hybrid_cv import HybridSpatialCV
from src.models.config import HybridCVConfig
from src.utils.data_utils import load_yaml_config


def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description='Run hybrid site-spatial cross-validation training')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory containing preprocessed data')
    parser.add_argument('--output-dir', type=str, default='results/cv_results',
                        help='Output directory for results')
    parser.add_argument('--n-folds', type=int, help='Number of CV folds')
    parser.add_argument('--model-type', type=str, help='Model type (cnn_coordinate)')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        try:
            config_dict = load_yaml_config(args.config)
            config = HybridCVConfig.from_dict(config_dict)
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration...")
            config = HybridCVConfig()
    else:
        config = HybridCVConfig()
    
    # Override configuration with command line arguments
    if args.data_dir:
        config.preprocessed_dir = args.data_dir
    if args.output_dir:
        config.cv_dir = args.output_dir
    if args.n_folds:
        config.n_folds = args.n_folds
    if args.model_type:
        config.model_type = args.model_type
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    
    # Print configuration
    print("Training Configuration:")
    print(f"  Data directory: {config.preprocessed_dir}")
    print(f"  Output directory: {config.cv_dir}")
    print(f"  Number of folds: {config.n_folds}")
    print(f"  Model type: {config.model_type}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.base_learning_rate}")
    print(f"  Loss function: {config.loss_function}")
    print(f"  Hard negative mining: {config.use_hard_negative_mining}")
    print(f"  Test-time augmentation: {config.use_test_time_augmentation}")
    
    # Run cross-validation
    cv_trainer = HybridSpatialCV(config)
    
    try:
        fold_models, fold_results, fold_metrics, fold_histories = cv_trainer.run_cross_validation()
        
        print("\n✅ Cross-validation completed successfully!")
        print(f"Results saved to: {config.cv_dir}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())