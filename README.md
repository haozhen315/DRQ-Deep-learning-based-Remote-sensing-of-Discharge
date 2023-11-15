# Remote sensing of river discharge from medium-resolution satellite imagery based on deep learning

## Overview
The DRQ (Deep learning based Remote sensing of Discharge) method represents an advancement in remote sensing of river discharge (RSQ). Traditional RSQ methods, such as Mass Conserved Flow Law Inversion (McFLI), often struggle with accuracy, especially in estimating the width of narrow rivers from medium-resolution images. DRQ addresses these limitations by using reflectance data instead of width measurements, making it particularly effective for narrow rivers (<40m). The method employs an advanced Transformer architecture to map river discharge directly from time-series reflectance imagery, providing more density and accuracy in inversed discharge compared to McFLI-based methods.

### Key Features
- Utilizes reflectance data for discharge estimation in narrow rivers.
- Advanced Transformer architecture for enhanced accuracy.
- Resilient to cloud contamination and does not require precise river width.

## Contents
- `main.py`: Executes the DRQ algorithm.
- `model.py`: The deep learning model for DRQ.
- `utils.py`: Supporting utility functions.

## Usage
1. **Landsat Files Preparation**: Ensure files include "blue", "green", "red", "nir", "swir1", "swir2", and "qa" bands.
2. **File Organization**: Group Landsat files for the same river reach in one folder.

## Notes
- **Image Acceptance**: The algorithm processes images with clouds and NaN values without checks. This is to allow testing of a wider range of images. Given the incomplete detection of the FMask algorithm, the model during training has possibly encountered a wide variety of images, leading to an expected higher tolerance to noise than the nominated threshold of a maximum of 10% cloud cover ratio.
- **Environmental Performance**: DRQ works in all conditions but is less effective in drier ones. See our paper for details (under review).

## Environment Requirements
- GPU
- `python==3.9.12`
- `torch==2.0.1+cu118`
- `rasterio==1.3.6`
- `numpy==1.23.5`

## Pretrained Model [https://zenodo.org/records/10130554]

## Citation and Contact
For citing our work, refer to the research paper. For inquiries or collaboration, contact [haozhenuk@gmail.com].
