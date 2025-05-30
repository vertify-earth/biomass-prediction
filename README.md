# Deep Learning for Above Ground Biomass Estimation


<!-- # Above Ground Biomass Prediction using Deep Learning Models -->
<!-- # Spatial-Aware Biomass Prediction with Deep Learning and Hybrid Cross-Validation -->

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This repository provides a comprehensive machine learning pipeline for estimating above-ground biomass (AGB) using satellite remote sensing data. It emphasizes spatial-awareness in data preprocessing and model training, employing a hybrid site-spatial cross-validation strategy for robust model evaluation and deep learning models (CNNs) for prediction.

## Overview

The pipeline is designed to handle multi-source satellite imagery and corresponding biomass data from various study sites. Key aspects include:

-   **End-to-End Workflow**: From raw data ingestion to model training, evaluation, and results visualization.
-   **Spatial Autocorrelation Handling**: Techniques to mitigate the effects of spatial autocorrelation in biomass data, ensuring more reliable model performance assessment.
-   **Deep Learning Models**: Utilizes Convolutional Neural Networks (CNNs), specifically designed to incorporate spatial coordinate information as input channels, enhancing spatial awareness.
-   **Multi-Site Data Processing**: Capable of processing and integrating data from multiple geographically distinct study sites.
-   **Robust Evaluation**: Implements a hybrid site-spatial cross-validation method. This ensures that (a) data from all sites are adequately represented in training and testing, and (b) spatial separation is maintained between folds to avoid overly optimistic performance estimates.
-   **Advanced Training Techniques**: Incorporates features like configurable loss functions, data augmentation, hard negative mining, and test-time augmentation (TTA).

## Performance Metrics

Latest model evaluation results demonstrate strong predictive performance across multiple sites and diverse forest conditions:

### Cross-Validation Summary (5-fold)

**Log Scale (Training Scale):**
- **RMSE**: 0.1339 ± 0.0252
- **R²**: 0.8938 ± 0.0334
- **MAE**: 0.1012 ± 0.0158
- **Spearman Correlation**: 0.9393 ± 0.0141

**Original Scale (Biomass in Mg/ha):**
- **RMSE**: 25.5 ± 2.5 Mg/ha
- **R²**: 0.8739 ± 0.0228
- **MAE**: 19.1 ± 1.8 Mg/ha
- **Mean True Biomass**: 194.7 Mg/ha
- **Biomass Range**: 39.5 - 373.6 Mg/ha

**Performance Assessment:**
- **Relative RMSE**: 13.1% of mean biomass




### Data Augmentation Implementation
- Geometric augmentation (flips, rotations)
- Spectral augmentation (band jittering)
- Augmentation probability: 0.7
- Test-Time Augmentation: 4 augmented versions per prediction

## Training Data

The model was trained on data from four distinct forest sites in India and Thailand, covering a wide range of biomass conditions. This diverse dataset helps ensure the model's robustness across different forest types and biomass densities.

| Site | Area (km²) | Biomass Range (Mg/ha) | Mean ± Std Dev (Mg/ha) |
|------|------------|----------------------|------------------------|
| Yellapur | 312 | 47 to 322 | 215 ± 53 |
| Betul | 105 | 7 to 128 | 93 ± 27 |
| Achanakmar | 117 | 74 to 229 | 169 ± 28 |
| Khaoyai | 47 | 179 to 436 to 339 | 275 ± 47 |
| Uppangala | 21 | 244 to 436 | 337 ± 63 |



The training data is sourced from the study: Rodda, S.R., Fararoda, R., Gopalakrishnan, R. et al. LiDAR-based reference aboveground biomass maps for tropical forests of South Asia and Central Africa. Sci Data 11, 334 (2024). [https://www.nature.com/articles/s41597-024-03162-x](https://doi.org/10.1038/s41597-024-03162-x)

## Features

-   **Configurable Preprocessing**:
    -   Flexible input of raster pairs (satellite imagery and biomass maps).
    -   Chip extraction from rasters with configurable size and overlap.
    -   Optional log transformation for biomass values.
    -   Quality filtering of chips based on a minimum percentage of valid pixels.
    -   Robust NaN (Not a Number) handling for satellite data, including imputation for minor NaNs and discarding chips with excessive NaNs.
-   **Feature Engineering**:
    -   Automatic calculation of derived spectral indices (e.g., NDVI, EVI, SAVI, GNDVI, NDWI) if `add_derived_features` is enabled.
    -   Standardization of input features.
-   **Advanced CNN Architecture**:
    -   `CNNCoordinateModel`: A CNN that integrates normalized spatial coordinates (x, y) as additional input channels.
    -   Uses Instance Normalization for convolutional layers and Layer Normalization for fully connected layers.
-   **Hybrid Site-Spatial Cross-Validation**:
    -   Ensures each site's data is included in the cross-validation process.
    -   Applies spatial buffering and clustering techniques to create spatially distinct folds within sites.
    -   Configurable number of folds (`n_folds`) and spatial buffer distance.
-   **Flexible Training Options**:
    -   Support for multiple loss functions: MSE, Huber loss (robust to outliers), and a custom Spatial Loss (experimental, penalizes spatial autocorrelation in residuals).
    -   Data Augmentation: Geometric (flips, rotations) and spectral (band jittering) augmentations with configurable probability.
    -   Hard Negative Mining: Option to focus training on more difficult samples after a certain number of epochs.
    -   Test-Time Augmentation (TTA): Improves prediction robustness by averaging predictions over multiple augmented versions of test samples.
    -   Learning Rate Scheduling: Cosine Annealing Learning Rate scheduler.
    -   Early Stopping: Prevents overfitting by stopping training if validation loss doesn't improve.
-   **Comprehensive Output & Logging**:
    -   Saves processed data, trained model weights, and detailed prediction results.
    -   Generates visualizations for data distributions, training history, CV results (scatter plots, residual analysis), and site-specific performance.
    -   Logs configuration parameters for reproducibility.

## Repository Structure

```
.
├── configs/                # Configuration files for pipeline, preprocessing, training
│   ├── pipeline_config.yaml
│   ├── preprocessing_config.yaml
│   └── training_config.yaml
├── data/                   # Data directory (see data/README.md for structure)
│   ├── raw/                # (User-provided) Raw input satellite and biomass rasters
│   └── processed/          # Processed data (chips) ready for training
├── results/                # Output directory for results and visualizations
│   ├── cv_results/         # Cross-validation outputs (models, metrics, plots)
│   ├── hybrid_results/     # General training results (if not using specific cv_dir)
│   ├── preprocessing/      # Outputs from the preprocessing step (e.g., data distribution plots)
│   └── visualizations/     # General visualization outputs
├── scripts/                # Python scripts to run parts of or the full pipeline
│   ├── run_full_pipeline.py
│   ├── run_preprocessing.py
│   └── run_training.py
├── src/                    # Source code for the biomass prediction pipeline
│   ├── __init__.py
│   ├── models/             # Model architectures, training logic (HybridSpatialCV), loss functions
│   ├── preprocessing/      # Data preprocessing (SpatialAwarePreprocessor), chipping logic
│   └── utils/              # Utility functions (data loading, YAML config, visualization)
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   ├── test_models.py
│   └── test_preprocessing.py
├── .gitignore
├── environment.yml         # Conda environment definition
├── LICENSE
├── README.md               # This file
├── requirements.txt        # Pip requirements file
└── setup.py                # Package setup script
```

## Installation

### Requirements

-   Python 3.9+ (as per `environment.yml`)
-   Conda (recommended for environment management)
-   CUDA-compatible GPU (highly recommended for deep learning model training)
-   Sufficient RAM (e.g., >=16GB, dataset dependent) and disk space.

### Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vertify-earth/biomass-dl-model-training-1.git
    cd biomass-dl-model-training-1
    ```

2.  **Create and activate Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate biomass-spatial-cv
    ```
    Alternatively, if you prefer pip and have an existing Python environment:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the package (optional, for making scripts accessible):**
    ```bash
    pip install .
    ```
    Or for development:
    ```bash
    pip install -e .
    ```

## Configuration

The pipeline behavior is controlled by YAML configuration files located in the `configs/` directory.

1.  **`configs/preprocessing_config.yaml`**:
    *   Defines parameters for the data preprocessing step.
    *   **Crucially, update `raster_pairs`** with the correct paths to your raw satellite imagery (`satellite_path`) and biomass map (`biomass_path`) files, along with a `site_name` for each pair.
    *   Other key parameters: `chip_size`, `overlap`, `use_log_transform`, `min_valid_pixels`, `max_sat_nan_fraction`, `test_ratio`, `val_ratio`.

2.  **`configs/training_config.yaml`**:
    *   Defines parameters for the model training and cross-validation process.
    *   Key parameters: `preprocessed_dir` (path to processed data from the preprocessing step), `results_dir`, `cv_dir`, `n_folds`, `spatial_buffer`, `model_type` (e.g., "cnn_coordinate"), `batch_size`, `num_epochs`, `base_learning_rate`, `loss_function`, and settings for augmentation, TTA, and hard negative mining.

3.  **`configs/pipeline_config.yaml`**:
    *   A combined configuration file that includes both `preprocessing` and `training` sections. This is used by the `run_full_pipeline.py` script.
    *   Ensure paths and parameters are consistent here if running the full pipeline.

**Example `raster_pairs` entry in `preprocessing_config.yaml` or `pipeline_config.yaml`:**
```yaml
preprocessing:
  raster_pairs:
    - satellite_path: "/path/to/your/site1_satellite_data.tif"
      biomass_path: "/path/to/your/site1_biomass_data.tif"
      site_name: "SiteName1"
    - satellite_path: "/path/to/your/site2_satellite_data.tif"
      biomass_path: "/path/to/your/site2_biomass_data.tif"
      site_name: "SiteName2"
  # ... other preprocessing parameters
```

## Data Preparation

1.  **Input Data**:
    *   The pipeline expects raw input data in raster format (e.g., GeoTIFF).
    *   For each study site, you need:
        *   A multi-band satellite imagery file (e.g., stacked Sentinel-1, Sentinel-2, Landsat-8, PALSAR, DEM bands).
        *   A corresponding single-band biomass raster map (e.g., AGB values).
    *   Ensure these files are co-registered and have the same spatial resolution and extent for accurate chip extraction.
2.  **Configuration**:
    *   Update the `raster_pairs` section in `configs/preprocessing_config.yaml` (or `configs/pipeline_config.yaml` under the `preprocessing` key) to point to your data files.
3.  **Output**:
    *   The preprocessing step will generate chips (small image patches) and save them in NumPy format (`.npy`) along with metadata in the directory specified by `processed_dir` (default: `data/processed/`).
    *   A `latest.txt` file in `processed_dir` will point to the timestamp of the most recent preprocessing run, which is used by default during training.
    *   Refer to `data/README.md` for more details on the expected structure within the `data/` directory.

## Usage: Running the Pipeline

The pipeline can be run in modular steps or as a complete workflow using scripts in the `scripts/` directory.

### 1. Preprocessing Only

This step takes your raw raster data, extracts chips, performs transformations, and splits the data.

*Using `run_preprocessing.py` (recommended for focused preprocessing):*
```bash
python scripts/run_preprocessing.py --config configs/preprocessing_config.yaml
```
*You can override output directory:*
```bash
python scripts/run_preprocessing.py --config configs/preprocessing_config.yaml --output-dir data/my_processed_data
```

*Alternatively, using `run_full_pipeline.py`:*
```bash
python scripts/run_full_pipeline.py --config configs/pipeline_config.yaml --preprocessing-only
```

### 2. Training Only

This step trains the model using preprocessed data. Ensure preprocessing has been completed and the `preprocessed_dir` in `training_config.yaml` (or `pipeline_config.yaml`) points to the correct location.

*Using `run_training.py` (recommended for focused training):*
```bash
python scripts/run_training.py --config configs/training_config.yaml
```
*You can specify data and output directories:*
```bash
python scripts/run_training.py --config configs/training_config.yaml --data-dir data/processed --output-dir results/my_cv_run
```
*Override other parameters like number of folds, batch size, epochs:*
```bash
python scripts/run_training.py --config configs/training_config.yaml --n-folds 3 --batch-size 32 --epochs 50
```

*Alternatively, using `run_full_pipeline.py` (ensure preprocessed data exists):*
```bash
python scripts/run_full_pipeline.py --config configs/pipeline_config.yaml --training-only --skip-preprocessing
```

### 3. Full Pipeline (Preprocessing and Training)

This runs both preprocessing and training sequentially.

```bash
python scripts/run_full_pipeline.py --config configs/pipeline_config.yaml
```
*You can skip preprocessing if data is already processed:*
```bash
python scripts/run_full_pipeline.py --config configs/pipeline_config.yaml --skip-preprocessing
```

## Core Modules

-   **`src/preprocessing`**:
    -   `SpatialAwarePreprocessorConfig`: Dataclass for preprocessing configuration.
    -   `SpatialAwarePreprocessor`: Handles loading raw raster data, aligning them, extracting chips, applying transformations (e.g., log transform to biomass), robust NaN handling, and splitting data into training, validation, and test sets with spatial awareness.
-   **`src/models`**:
    -   `HybridCVConfig`: Dataclass for training configuration.
    -   `CNNCoordinateModel`: Defines the CNN architecture that incorporates spatial coordinates as input channels. Other model architectures can be added here.
    -   `create_model`: Factory function to instantiate models.
    -   `HuberLoss`, `SpatialLoss`: Custom loss functions.
    -   `create_loss_function`: Factory function for loss functions.
    -   `HybridSpatialCV`: Manages the hybrid site-spatial cross-validation training and evaluation loop. It handles data splitting per fold, feature engineering, model training, and evaluation.
-   **`src/utils`**:
    -   `data_utils.py`: Contains `load_preprocessed_data` to load data generated by the preprocessing step, and `load_yaml_config` for loading configuration files.
    -   `visualization.py`: Includes functions like `visualize_cv_results` to plot metrics, scatter plots, and training histories, and `plot_data_distribution` for initial data exploration.

## Model Architecture

The primary model implemented is `CNNCoordinateModel` (defined in `src/models/cnn_models.py`).
-   It's a Convolutional Neural Network designed for 2D image chip inputs.
-   **Spatial Awareness**: A key feature is the concatenation of two additional channels to the input satellite imagery chips. These channels represent the normalized X and Y coordinates of each pixel within the chip. This allows the model to learn location-specific patterns.
-   **Normalization**: Uses `InstanceNorm2d` after convolutional layers and `LayerNorm` after fully connected layers, which can be beneficial for stabilizing training.
-   The architecture consists of several convolutional blocks (Conv2D, Norm, ReLU, MaxPool, Dropout) followed by fully connected layers for regression.

## Cross-Validation Strategy

The project employs a **Hybrid Site-Spatial Cross-Validation** strategy, implemented in `src/models/hybrid_cv.py`. This approach is designed to provide a more realistic estimate of model performance on unseen data by:

1.  **Site Representation**: Ensuring that data from all available study sites are represented across the training and testing folds. This helps the model generalize better to new, unseen sites.
2.  **Spatial Separation**: Within each site (or across sites if applicable), the splitting mechanism attempts to maintain spatial separation between training, validation, and test sets. This is achieved through:
    *   Spatial clustering (KMeans) of samples within a site to group spatially contiguous data points.
    *   Assigning entire spatial clusters to test/train/validation sets.
    *   Applying a `spatial_buffer` to ensure a minimum distance between training samples and test samples, reducing data leakage due to spatial autocorrelation.
3.  **Configuration**: The number of folds (`n_folds`) and the `spatial_buffer` distance are configurable in `training_config.yaml`.

This strategy is more robust than simple random splitting, especially when dealing with spatially autocorrelated environmental data like biomass.

## Output

-   **Preprocessed Data**: Saved in the directory specified by `processed_dir` in the preprocessing configuration (default: `data/processed/`). Includes:
    -   `X_<timestamp>.npy`: Array of satellite data chips.
    -   `y_<timestamp>.npy`: Array of corresponding biomass values.
    -   `sources_<timestamp>.npy`: Array indicating the source site for each chip.
    -   `coordinates_<timestamp>.pkl`: List of coordinates for each chip.
    -   `split_<timestamp>.npz`: Train/validation/test masks.
    -   `preprocessing_config_<timestamp>.json`: Metadata about the preprocessing run.
    -   `latest.txt`: Contains the timestamp of the most recent run.
-   **Training & CV Results**: Saved in the directory specified by `cv_dir` in the training configuration (default: `results/cv_results/`). A new subdirectory is created for each run, timestamped. Includes:
    -   `fold_<i>_model.pt`: Saved PyTorch model state dictionary for each fold.
    -   `fold_<i>_results.csv`: Predictions, true values, and residuals for the test set of each fold.
    -   `cv_summary.json`: Aggregated metrics (mean/std of RMSE, R², MAE, Spearman) across all folds.
    -   `config.json`: The training configuration used for the run.
    -   Visualization plots:
        -   `cv_predictions_scatter.png`: Scatter plot of predicted vs. true biomass.
        -   `training_history.png`: Training/validation loss curves, learning rate schedule.
        -   `residual_analysis.png`: Plots of residuals vs. predicted, residual histogram, Q-Q plot.
        -   `site_performance.png`: Bar plots of metrics per site (if multiple sites).

## Testing

Unit tests are located in the `tests/` directory. They can be run using `pytest`:

```bash
conda activate biomass-spatial-cv # or your environment name
pytest
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation

```
@misc{vertify2025biomass,
  author = {vertify.earth},
  title = {Biomass Prediction Training Pipeline},
  year = {2025},
  publisher = {GitHub},
  note = {Developed for GIZ Forest Forward initiative},
  howpublished = {\url{https://github.com/vertify/biomass-prediction-training}}
}
```

## Contact

For questions, feedback, or collaboration opportunities, please reach out via:
- GitHub: [vertify](https://github.com/vertify)
- Email: info@vertify.earth
- Website: [vertify.earth](https://vertify.earth)