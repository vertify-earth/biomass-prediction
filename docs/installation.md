# Installation Guide

## System Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- At least 16GB RAM for large datasets
- 10GB+ free disk space

## Installation Methods

### Method 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/najahpokkiri/biomass-prediction-spatial-cv.git
cd biomass-prediction-spatial-cv

# Create conda environment
conda env create -f environment.yml
conda activate biomass-spatial-cv

# Install in development mode
pip install -e .