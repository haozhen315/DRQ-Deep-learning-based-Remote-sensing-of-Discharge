# Remote sensing of river discharge from medium-resolution satellite imagery based on deep learning

## Overview
The DRQ (Deep learning based Remote sensing of Discharge) method represents an advancement in remote sensing of river discharge (RSQ). Traditional RSQ methods, such as Mass Conserved Flow Law Inversion (McFLI), often struggle with accuracy, especially in estimating the width of narrow rivers from medium-resolution images. DRQ addresses these limitations by using reflectance data instead of width measurements, making it particularly effective for narrow rivers (<60m). The method employs an advanced Transformer architecture to map river discharge directly from time-series reflectance imagery, providing more density and accuracy in inversed discharge compared to McFLI-based methods.

### Key Features
- Utilizes reflectance data for discharge estimation in narrow rivers.
- Advanced Transformer architecture for enhanced accuracy.
- Median correlation score of 0.6 validated on 494 gauges with median river width 67m.

## Contents
- `main.py`: Executes the DRQ algorithm.
- `model.py`: The deep learning model for DRQ.
- `utils.py`: Supporting utility functions.

## Usage
1. **Landsat Files Preparation**: Ensure files include "blue", "green", "red", "nir", "swir1", "swir2", and "qa" bands. (blue, green, red, nir, swir1, swir2, qa = rasterio.open(tif_file).read()) The image should be at least 64 pixels wide, and ensure the river is centered in the image.
2. **File Organization**: Group Landsat files for the same river reach in one folder.

## Notes
- **Image Acceptance**: The algorithm processes images with clouds and NaN values without checks. Given the incomplete detection of the FMask algorithm, the model during training has possibly encountered a wide variety of images, leading to an expected higher tolerance to noise than the nominated threshold of a maximum of 1% cloud cover ratio.
- **When to expect success with RSQ**: First, narrow rivers still present difficulties for RSQ, though the term "narrowness" should be defined based on the image resolution at hand. For narrow rivers, when their width variation is significant, RSQ can be successful with both BAM/geoBAM and the DL model, though the DL model is expected to have higher accuracy. For narrow rivers with small width variation, if the river surface reflectance appears obviously different during high flow periods due to the carrying of suspended sediments, this difference is describable by the DL model and RSQ success can be expected, provided that the image resolution is sufficient to monitor this difference. For narrow rivers with small width variation where the discharge cannot be reflected by surface reflectance changes, these cases present significant challenges for RSQ. See our paper for details (under review).

<img width="468" alt="image" src="https://github.com/haozhen315/DRQ-Deep-learning-based-Remote-sensing-of-Discharge/assets/46937286/3a5b8c88-4e57-4f4c-8859-baf695b9a2fa">

## Environment Requirements
- GPU
- `python==3.9.12`
- `torch==2.0.1+cu118`
- `rasterio==1.3.6`
- `numpy==1.23.5`

## Pretrained Model
Pretrained weights can be downloaded from [here](https://zenodo.org/records/11139143).

## Contact
For inquiries, contact [haozhenuk@gmail.com].
