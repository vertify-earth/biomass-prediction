#!/usr/bin/env python
"""Setup script for biomass-prediction-spatial-cv package."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="biomass-prediction-spatial-cv",
    version="1.0.0",
    author="najahpokkiri",
    author_email="your.email@example.com",  # Update with your email
    description="Spatial-aware biomass prediction with hybrid cross-validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/najahpokkiri/biomass-prediction-spatial-cv",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.6.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "biomass-preprocess=scripts.run_preprocessing:main",
            "biomass-train=scripts.run_training:main",
            "biomass-pipeline=scripts.run_full_pipeline:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)