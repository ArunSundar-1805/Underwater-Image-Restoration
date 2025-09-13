# Underwater Image Restoration

This repository implements a traditional image processing pipeline for restoring and enhancing underwater images, focusing on color correction, contrast improvement, and image quality assessment.

## Features

- **Color Correction:** Grayscale world white balance and histogram auto-contrast adjustment.
- **Restoration Algorithm:** CCCBLSHL (Contrast Correction and Color Balance with Shadow Highlight Lifting).
- **Image Quality Metrics:** UCIQE, UIQM, BRISQUE, and NIQE for quantitative evaluation.
- **Modular Design:** Each step of the pipeline is implemented in a separate Python module.
- **Batch Processing:** Restore and evaluate multiple images in a directory.

## File Structure

- `main.py` – Main script for batch processing images. Computes metrics and invokes restoration pipeline.
- `CCCBLSHL.py` – Implements the core restoration algorithm, including color and contrast correction.
- `Grayscale_remapping.py` – Adjusts white balance and performs histogram auto-contrast.
- `estimate_airlight.py` – Custom color quantization and airlight estimation utilities.
- `wls_optimization.py` – Weighted Least Squares optimization for enhancing image details.
- `tempCodeRunnerFile.py` – Temporary file (can be ignored or deleted).

> **Note:** There may be more files in the repository. [See the full file list here.](https://github.com/ArunSundar-1805/Underwater-Image-Restoration/search)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ArunSundar-1805/Underwater-Image-Restoration.git
   cd Underwater-Image-Restoration
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Typical dependencies include `numpy`, `opencv-python`, `scipy`, `scikit-image`, `scikit-learn`, and `pyiqa`.

## Usage

1. Place input images in the directory: `image/11_Images`.
2. Run the main script:
   ```bash
   python main.py
   ```
   - Output images and metrics are saved to the `Results` directory.
   - Supported formats include BMP, JPG, JPEG, PNG, TIF, TIFF, WEBP.

## How It Works

- The pipeline loads each image, applies white balance and contrast correction (`Grayscale_remapping.py`).
- The CCCBLSHL algorithm (`CCCBLSHL.py`) further restores color and contrast.
- Image quality metrics are calculated using custom and library methods.
- Results and metrics are saved for review.

## References

- UCIQE: Underwater Color Image Quality Evaluation
- UIQM: Underwater Image Quality Measure
- BRISQUE, NIQE: No-reference image quality metrics
