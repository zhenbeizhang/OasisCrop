# OasisCrop Code

This repository contains the public code for **OasisCrop v1.0**, a 1 m visible net annual-crop surface dataset for arid oasis agriculture in Xinjiang, China.

The repository is organized as a reproducible code release rather than a copy of the local experiment workspace. It excludes restricted source imagery, released data products, annotation rasters, trained checkpoints, local paths, temporary outputs, and historical experiment folders. Data products and annotation samples are deposited separately in Zenodo or another long-term repository.

## Contents

- `oasiscrop/models/`: OasisCropNet, a boundary-guided multiscale encoder-decoder for binary visible net annual-crop segmentation.
- `oasiscrop/data/`: GeoTIFF image/mask dataset loader and augmentations.
- `oasiscrop/training/`: training entry point with main cropland, auxiliary cropland, and boundary-supervision losses.
- `oasiscrop/inference/`: sliding-window inference and 1 ha fraction aggregation.
- `oasiscrop/validation/`: binary raster validation and stratum-level summaries.
- `configs/`: production configuration used for the v1 release.
- `scripts/`: minimal PowerShell examples for training, inference, and validation.
- `docs/`: notes for model reporting, data-code mapping, and release checks.

## Installation

Create an environment with PyTorch, GDAL, OpenCV, NumPy, pandas, tifffile, tqdm, torchvision, and PyYAML. A Conda environment file is provided in `environment.yml`.

```powershell
conda env create -f environment.yml
conda activate oasiscrop
pip install -e .
```

## Expected annotation dataset layout

Training uses explicit image and mask directories. The released annotation sample record can be organized as:

```text
annotation_samples/
  train/
    images/*.tif
    masks/*.tif
  test/
    images/*.tif
    masks/*.tif
```

The file names of paired images and masks must match. Masks are treated as binary rasters: values greater than 0 are interpreted as visible net annual-crop pixels in the training loader. Released products should follow the class coding documented in the data dictionary.

## Main commands

```powershell
python -m oasiscrop.training.train `
  --train-image-dir "path\to\annotation_samples\train\images" `
  --train-mask-dir "path\to\annotation_samples\train\masks" `
  --test-image-dir "path\to\annotation_samples\test\images" `
  --test-mask-dir "path\to\annotation_samples\test\masks" `
  --save-dir outputs\checkpoints

python -m oasiscrop.inference.predict_tiles `
  --input-root "path\to\source_tiles" `
  --output-root outputs\predicted_masks `
  --weights "path\to\model_best.pth"

python -m oasiscrop.inference.aggregate_1ha_fraction `
  --input-root outputs\predicted_masks `
  --output-root outputs\fraction_1ha

python -m oasiscrop.validation.validate_binary_rasters `
  --prediction-root outputs\predicted_masks `
  --reference-root "path\to\reference_masks" `
  --output-csv outputs\validation\tile_metrics.csv
```

## Reproducibility notes

The public code can reproduce the documented model training, inference, post-processing, aggregation, and validation procedures when the annotation samples, source image tiles, trained checkpoint, and product files are available through the associated data repositories. Restricted third-party source imagery is not redistributed in this repository.

## Repository release

This folder is prepared for the OasisCrop v1.0 GitHub repository. After uploading, create a `v1.0.0` release and archive the release with Zenodo or another software DOI provider. Link the software DOI, data DOI, and manuscript bidirectionally once the repository records are available.
