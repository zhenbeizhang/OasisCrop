# Data and Code Mapping

| Manuscript component | Code location | Data dependency |
| --- | --- | --- |
| Annotation sample training | `oasiscrop/training/train.py` | public annotation sample tiles and masks |
| Full-product inference | `oasiscrop/inference/predict_tiles.py` | source image tiles and trained checkpoint |
| Small-component post-processing | `oasiscrop/inference/predict_tiles.py` | predicted binary masks |
| 1 ha fraction product | `oasiscrop/inference/aggregate_1ha_fraction.py` | 1 m binary product |
| Held-out tile validation | `oasiscrop/validation/validate_binary_rasters.py` | predicted masks and reference masks |
| Stratified validation summaries | `oasiscrop/validation/summarize_strata.py` | tile-level metrics and stratum table |

The associated data repository should include the public annotation dataset, main product rasters, metadata tables, validation tables, checksums, and README files. Restricted source imagery should be represented by a source-index table and valid-area masks where redistribution is not permitted.
