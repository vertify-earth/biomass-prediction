# Spatial-Aware Biomass Prediction with Hybrid Cross-Validation

A comprehensive machine learning pipeline for biomass estimation using satellite remote sensing data with spatial autocorrelation awareness and hybrid site-spatial cross-validation.

## Overview

This repository implements a spatial-aware biomass prediction system that:
- Processes multi-source satellite data (Sentinel-1, Sentinel-2, Landsat-8, PALSAR, DEM)
- Handles spatial autocorrelation in biomass data
- Uses hybrid site-spatial cross-validation for robust model evaluation
- Implements CNN models with coordinate channels for spatial awareness
- Includes advanced training techniques (hard negative mining, test-time augmentation)

## Features

- **Spatial-Aware Preprocessing**: Extracts chips while respecting spatial boundaries
- **Multi-Site Data Handling**: Processes data from multiple study sites
- **Advanced CNN Architecture**: Coordinate-aware CNNs with Instance/Layer Normalization
- **Robust Cross-Validation**: Hybrid approach ensuring both site representation and spatial separation
- **Production-Ready**: Comprehensive logging, error handling, and reproducible results

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 16GB RAM for large datasets

### Setup

1. Clone the repository:
```bash
git clone https://github.com/najahpokkiri/biomass-prediction-spatial-cv.git
cd biomass-prediction-spatial-cv