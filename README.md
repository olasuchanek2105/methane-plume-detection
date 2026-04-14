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

## Notes

This is an ongoing research project. The current implementation is intended for experimentation and validation rather than final deployment.
