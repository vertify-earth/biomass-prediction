# Deep Learning for Above Ground Biomass Estimation: Patch-wise Prediction


<!-- # Above Ground Biomass Prediction using Deep Learning Models -->
<!-- # Spatial-Aware Biomass Prediction with Deep Learning and Hybrid Cross-Validation -->

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This repository provides a comprehensive machine learning pipeline for estimating above-ground biomass (AGB) using satellite remote sensing data. It emphasizes spatial-awareness in data preprocessing and model training, employing a hybrid site-spatial cross-validation strategy for robust model evaluation and deep learning models (CNNs) for prediction.

## What Is This Model For

This model is trainined to predict above-ground biomass (AGB) in tropical and subtropical forests using multi-source satellite imagery. Specifically:

- **Prediction Unit**: Estimates biomass at 24×24 pixel patches (approximately 960×960m at 40m resolution)
- **Output**: Biomass density in Mg/ha (megagrams per hectare)
- **Input Data**: Processes multi-sensor data including Sentinel-1, Sentinel-2, Landsat-8, PALSAR, and DEM
- **Application Scope**: Best suited for tropical and subtropical forest ecosystems in South/Southeast Asia
- **Biomass Range**: Validated for forests with biomass between ~40-460 Mg/ha
  
## Overview

The pipeline is designed to handle multi-source satellite imagery and corresponding biomass data from various study sites. Key aspects include:

-   **End-to-End Workflow**: From raw data ingestion to model training, evaluation, and results visualisation.
-   **Spatial Autocorrelation Handling**: Techniques to mitigate the effects of spatial autocorrelation in biomass data, ensuring more reliable model performance assessment.
-   **Deep Learning Models**: Utilises Convolutional Neural Networks (CNNs), specifically designed to incorporate spatial coordinate information as input channels, enhancing spatial awareness.
-   **Multi-Site Data Processing**: Capable of processing and integrating data from multiple geographically distinct study sites.
-   **Robust Evaluation**: Implements a hybrid site-spatial cross-validation method. This ensures that (a) data from all sites are adequately represented in training and testing, and (b) spatial separation is maintained between folds to avoid overly optimistic performance estimates.
-   **Advanced Training Techniques**: Incorporates features like configurable loss functions, data augmentation, hard negative mining, and test-time augmentation (TTA).

## Performance Metrics

### Model Architecture

The final prediction model is an ensemble that combines 5 spatially cross-validated CNN models. This ensemble approach:
- **Maintains spatial validation integrity** - each component model was validated on spatially separated data
- **Provides robust predictions** - averages out individual model uncertainties  
- **Generalisable across training regions** - performance estimates are realistic for similar forest ecosystems


### Overall Performance (5-Fold Spatial Cross-Validation)

- **RMSE**: 25.5 ± 1.9 Mg/ha
- **R²**: 0.873 ± 0.025  
- **MAE**: 18.9 ± 1.8 Mg/ha
- **Spearman Correlation**: 0.937 ± 0.017
- **Mean Biomass**: 191.2 Mg/ha
- **Biomass Range**: 5.2 - 460.7 Mg/ha
- **Relative RMSE**: 13.3% of mean biomass

### Site-Specific Performance 

Performance varies by forest site, reflecting differences in forest structure and biomass density:

| Site Name | Mean Biomass (Mg/ha) | RMSE (Mg/ha) | R² | MAE (Mg/ha) | Samples |
|-----------|---------------------|---------------|----|--------------|---------| 
| Yellapur | 214.8 | 19.0 | 0.887 | 14.7 | 421 |
| Betul | 93.8 | 7.9 | 0.922 | 6.3 | 132 |
| Achanakmar | 165.2 | 11.2 | 0.906 | 8.4 | 156 |
| Khaoyai | 276.1 | 19.6 | 0.880 | 15.8 | 63 |
| Uppangala | 328.5 | 68.4 | 0.088 | 48.0 | 12 |

*Uppangala shows lower performance due to very limited training samples (n=12).*

### Performance Assessment

With an R² of 0.873 and relative RMSE of 13.3%, the model demonstrates **very good performance** for biomass estimation across diverse tropical forest conditions. The spatial cross-validation methodology ensures these metrics represent realistic expectations for similar forest ecosystems in the training regions.


## Training Data

The model was trained on data from four distinct forest sites in India and Thailand, covering a wide range of biomass conditions. This diverse dataset helps ensure the model's robustness across different forest types and biomass densities.




| Site       | Location                    | Area (km²) | Biomass Range (Mg/ha) | Mean ± Std Dev (Mg/ha) | Forest / Terrain Type                  |
| ---------- | --------------------------- | ---------- | --------------------- | ---------------------- | -------------------------------------- |
| Yellapur   | Karnataka, India            | 312        | 47 to 322             | 215 ± 53               | Tropical semi-evergreen forest         |
| Betul      | Madhya Pradesh, India       | 105        | 7 to 128              | 93 ± 27                | Dry deciduous forest                   |
| Achanakmar | Chhattisgarh, India         | 117        | 74 to 229             | 169 ± 28               | Moist deciduous forest, hilly terrain  |
| Khaoyai    | Nakhon Ratchasima, Thailand | 47         | 179 to 436            | 275 ± 47               | Tropical evergreen forest, mountainous |
| Uppangala  | Karnataka, India            | 21         | 244 to 436            | 337 ± 63               | Tropical wet evergreen forest          |





The AGB ground data  is sourced from the study: Rodda, S.R., Fararoda, R., Gopalakrishnan, R. et al. LiDAR-based reference aboveground biomass maps for tropical forests of South Asia and Central Africa. Sci Data 11, 334 (2024). [https://www.nature.com/articles/s41597-024-03162-x](https://doi.org/10.1038/s41597-024-03162-x)

### Satellite Data Used
The model integrates data from multiple satellite sensors:
- **Sentinel-1**: C-band SAR (VV, VH polarizations)
- **Sentinel-2**: Multispectral 10-20m bands
- **Landsat-8**: Optical bands
- **PALSAR**: L-band SAR
- **Digital Elevation Model**: Topographic information

More on this here : `docs/satellite-data.md`

The input stack was generated in Google Earth Engine using this script: `scripts/satellite_data_preparation.js`

### Data Augmentation Implementation
- Geometric augmentation (flips, rotations)
- Spectral augmentation (band jittering)
- Augmentation probability: 0.7
- Test-Time Augmentation: 4 augmented versions per prediction
  
## Features

-   **Configurable Preprocessing**:
    -   Flexible input of raster pairs (satellite imagery and biomass maps).
    -   Chip extraction from rasters with configurable size and overlap.
    -   Optional log transformation for biomass values.
    -   Quality filtering of chips based on a minimum percentage of valid pixels.
    -   Robust NaN (Not a Number) handling for satellite data, including imputation for minor NaNs and discarding chips with excessive NaNs.
-   **Feature Engineering**:
    -   Automatic calculation of derived spectral indices (e.g., NDVI, EVI, SAVI, GNDVI, NDWI) if `add_derived_features` is enabled.
    -   Standardisation of input features.
-   **Advanced CNN Architecture**:
    -   `CNNCoordinateModel`: A CNN that integrates normalised spatial coordinates (x, y) as additional input channels.
    -   Uses Instance Normalisation for convolutional layers and Layer Normalisation for fully connected layers.
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
    -   Generates visualisations for data distributions, training history, CV results (scatter plots, residual analysis), and site-specific performance.
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
│   └── visualisations/     # General visualization outputs
├── scripts/                # Python scripts to run parts of or the full pipeline
│   ├── run_full_pipeline.py
│   ├── run_preprocessing.py
│   └── run_training.py
|   └── satellite_data_preparation.js
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
    -   `visualisation.py`: Includes functions like `visualise_cv_results` to plot metrics, scatter plots, and training histories, and `plot_data_distribution` for initial data exploration.

## Model Architecture

The primary model implemented is `CNNCoordinateModel` (defined in `src/models/cnn_models.py`).
-   It's a Convolutional Neural Network designed for 2D image chip inputs.
-   **Spatial Awareness**: A key feature is the concatenation of two additional channels to the input satellite imagery chips. These channels represent the normalized X and Y coordinates of each pixel within the chip. This allows the model to learn location-specific patterns.
-   **Normalisation**: Uses `InstanceNorm2d` after convolutional layers and `LayerNorm` after fully connected layers, which can be beneficial for stabilizing training.
-   The architecture consists of several convolutional blocks (Conv2D, Norm, ReLU, MaxPool, Dropout) followed by fully connected layers for regression.

## Cross-Validation Strategy

The project employs a **Hybrid Site-Spatial Cross-Validation** strategy, implemented in `src/models/hybrid_cv.py`. This approach is designed to provide a more realistic estimate of model performance on unseen data by:

1.  **Site Representation**: Ensuring that data from all available study sites are represented across the training and testing folds. This helps the model generalise better to new, unseen sites.
2.  **Spatial Separation**: Within each site (or across sites if applicable), the splitting mechanism attempts to maintain spatial separation between training, validation, and test sets. This is achieved through:
    *   Spatial clustering (KMeans) of samples within a site to group spatially contiguous data points.
    *   Assigning entire spatial clusters to test/train/validation sets.
    *   Applying a `spatial_buffer` to ensure a minimum distance between training samples and test samples, reducing data leakage due to spatial autocorrelation.
3.  **Configuration**: The number of folds (`n_folds`) and the `spatial_buffer` distance are configurable in `training_config.yaml`.
4. **Final Model and Ensemble Configuration**:
   - `train_final_model`: Train final model on entire dataset (default: true)
   - `create_ensemble`: Create ensemble from fold models (default: false)  
   - `final_model_epochs`: Number of epochs for final model training
   - `ensemble_method`: Method for combining predictions ("average" or "weighted")
   - `save_fold_models`: Whether to keep individual fold models

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
    -   Visualisation plots:
        -   `cv_predictions_scatter.png`: Scatter plot of predicted vs. true biomass.
        -   `training_history.png`: Training/validation loss curves, learning rate schedule.
        -   `residual_analysis.png`: Plots of residuals vs. predicted, residual histogram, Q-Q plot.
        -   `site_performance.png`: Bar plots of metrics per site (if multiple sites).

## Model Loading and Inference

### Loading the Final Model for Production

```python

import torch
from src.models.cnn_models import create_model

# Load final model for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model('cnn_coordinate', input_channels=10, height=24, width=24, device=device)
model.load_state_dict(torch.load('results/cv_results/ensemble_model.pt', map_location=device))
model.eval()

# Make predictions on new data
with torch.no_grad():
    predictions = model(new_satellite_data)
    # Convert from log scale to original biomass (Mg/ha)
    biomass_predictions = torch.exp(predictions) - 1

```


## Testing

Unit tests are located in the `tests/` directory. They can be run using `pytest`:

```bash
conda activate biomass-spatial-cv # or your environment name
pytest
```
## Limitations

While the model demonstrates strong performance across diverse tropical forest conditions, several limitations should be considered:

### 1. SAR Saturation Effects
- **High Biomass Saturation**: Synthetic Aperture Radar (SAR) signals, particularly C-band from Sentinel-1, experience saturation effects above approximately **400 Mg/ha**
- **Reduced Sensitivity**: This saturation leads to decreased model sensitivity and accuracy for very high biomass forests (>400 Mg/ha)
- **Impact on Dense Forests**: Particularly affects tropical wet evergreen forests and old-growth forest areas

### 2. Spatial Resolution Constraints
- **40m Resolution Limitation**: Current 40m pixel resolution may not capture fine-scale forest heterogeneity
- **Mixed Pixel Effects**: Pixels may contain multiple forest types or non-forest areas, affecting biomass estimates
- **Edge Effects**: Forest-non-forest boundaries may introduce uncertainty in biomass predictions

### 3. Training Data Limitations
- **Limited High Biomass Samples**: Fewer training samples available for forests >300 Mg/ha (e.g., Uppangala site with only 12 samples)
- **Geographic Coverage**: Training data concentrated in South/Southeast Asia, limiting generalizability to other tropical regions
- **Temporal Constraints**: Training data represents specific time periods, potentially missing seasonal variations

### 4. Sensor-Specific Constraints
- **C-band Limitations**: Current reliance on C-band SAR (Sentinel-1) which has limited penetration in dense canopies
- **Cloud Coverage**: Optical sensor data (Sentinel-2, Landsat-8) affected by persistent cloud cover in tropical regions
- **Temporal Compositing**: Multi-temporal compositing may mask important phenological signals

### 5. Model Architecture Limitations
- **Patch-based Approach**: 24×24 pixel patches may not capture landscape-scale biomass patterns
- **Fixed Input Channels**: Current architecture requires specific sensor combinations, limiting flexibility

## How to Improve

The following enhancements could significantly improve model performance and applicability:

### 1. Advanced SAR Integration
- **L-band SAR Data**: Integrate L-band SAR data for better penetration in dense forests
  - **ESA Biomass Mission**: Incorporate data from the upcoming ESA Biomass satellite (expected 2024-2025) specifically designed for forest biomass monitoring
  - **NISAR Mission**: Utilize NASA-ISRO NISAR L-band data when available for the corresponding 3 durations of the study period
  - **ALOS PALSAR-2**: Expand use of existing L-band data with improved temporal coverage

### 2. High-Resolution Ground Truth Data
- **Local LiDAR Integration**: 
  - Incorporate high-resolution airborne LiDAR data for local calibration and validation
  - Use LiDAR-derived metrics (canopy height, vertical structure) as additional input features
  - Develop site-specific calibration using local LiDAR campaigns
- **Field Plot Integration**: Add ground-measured forest inventory plots for enhanced validation

### 3. Enhanced Spatial Resolution
- **Higher Resolution Sensors**: Integrate 10m resolution Sentinel-2 bands and Planet imagery
- **Super-resolution Techniques**: Apply deep learning-based super-resolution methods to enhance spatial detail
- **Multi-scale Fusion**: Combine predictions from multiple spatial scales (10m, 20m, 40m)

### 4. Improved Model Architecture
- **Attention Mechanisms**: Implement spatial attention modules to focus on relevant image regions
- **Multi-scale CNNs**: Design architectures that process multiple spatial scales simultaneously
- **Transformer Models**: Explore vision transformers for better long-range spatial dependencies
- **Uncertainty Quantification**: Add probabilistic outputs to provide prediction confidence intervals

### 5. Extended Training Data
- **Global Training Data**: Expand training to include tropical forests from Africa, Central/South America
- **Synthetic Data Generation**: Use physics-based models to generate synthetic training samples
- **Transfer Learning**: Develop domain adaptation techniques for new geographic regions
- **Temporal Augmentation**: Include multi-year training data to capture temporal variations

### 6. Advanced Fusion Techniques
- **Physics-Informed Models**: Incorporate forest growth models and allometric relationships
- **Multi-sensor Fusion**: Develop sophisticated fusion algorithms for combining radar, optical, and LiDAR data
- **Temporal Modeling**: Add recurrent neural networks to model temporal biomass changes

### 7. Operational Enhancements
- **Real-time Processing**: Optimize for near real-time biomass monitoring
- **Cloud Platform Integration**: Deploy on cloud platforms (Google Earth Engine, AWS) for scalable processing
- **API Development**: Create APIs for easy integration with forest monitoring systems
- **Mobile Applications**: Develop field validation tools for ground truthing

### 8. Validation and Robustness
- **Cross-biome Validation**: Test model performance across different forest biomes
- **Seasonal Analysis**: Evaluate model stability across different seasons
- **Disturbance Detection**: Enhance capability to detect and account for forest disturbances
- **Error Propagation**: Implement comprehensive uncertainty analysis and error propagation methods

### Priority Recommendations
1. **Immediate (6-12 months)**: Integrate local LiDAR data for calibration, expand PALSAR-2 temporal coverage
2. **Medium-term (1-2 years)**: Incorporate ESA Biomass and NISAR L-band data when available
3. **Long-term (2-3 years)**: Develop global training datasets and advanced fusion architectures



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
