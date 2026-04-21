# Methane Plume Detection from Satellite Data

## Overview

This project explores the detection and future segmentation of methane plumes in multispectral satellite imagery (e.g. Sentinel-2).

The work is currently in progress and focuses on building a reliable machine learning pipeline and validating whether meaningful plume-related signals can be extracted from the data.

---

## Current Approach

* Extraction of selected spectral bands and Sánchez residual features
* Construction of a patch-based dataset around known emission locations
* Training a CNN model for binary classification (plume / no plume)
* Train/validation split performed at the scene level to avoid data leakage
* Initial experiments with Transformer-based architectures, including Vision Transformer (ViT), to evaluate their effectiveness on multispectral satellite data
* Comparative analysis of CNN and ViT models in terms of performance and generalization
* Planned exploration of fully self-attention-based models
---

## Status

The project is in an experimental stage.
Current results indicate that the model is able to learn useful patterns from the data, but further work is required.

---

## Tech Stack

* Python
* PyTorch
* NumPy, Pandas
* Rasterio

---

## Pipeline

### 1. Build dataset
To create patch-based dataset:

python -m src.features.build_patch_classification

This generates .npy files with patches and labels.

### 2. Train model
To train CNN classifier:

python -m src.models.train_patch_classifier

### 3. Outputs
Trained models and logs are saved in /outputs

## Notes

This is an ongoing research project. The current implementation is intended for experimentation and validation rather than final deployment.
