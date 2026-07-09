# OasisCropNet Model Card

## Intended use

OasisCropNet is used for binary segmentation of visible net annual or short-cycle cropland surfaces from metre-level oasis-agriculture imagery. It is not designed to produce cadastral parcel boundaries, crop types, orchard areas, sowing intensity, or legally defined cultivated-land statistics.

## Architecture summary

OasisCropNet is a boundary-guided multiscale encoder-decoder. The released implementation contains:

- a ResNet-50 feature extractor returning 1/4- to 1/32-scale features;
- a top-down feature-pyramid decoder for spatial detail recovery;
- an atrous spatial pyramid pooling context branch;
- an explicit boundary auxiliary head trained from mask-derived boundary targets;
- a boundary-guided fusion block that combines spatial, contextual, and boundary features;
- an auxiliary cropland head and a final binary cropland head.

## Production settings

- Input tile size: 512 x 512 pixels.
- Input bands: first three image bands by default.
- Output: one-channel cropland logits.
- Training losses: main binary cross-entropy plus Dice loss, auxiliary cropland binary cross-entropy, and boundary binary cross-entropy.
- Inference threshold: 0.5.
- Post-processing: connected components smaller than 30 square metres are removed unless separately retained by manual review in the production log.

## Reporting caution

OasisCropNet is the production segmentation model for the OasisCrop data workflow. It should not be interpreted as a cadastral parcel detector, crop-type classifier, or planting-intensity model.
