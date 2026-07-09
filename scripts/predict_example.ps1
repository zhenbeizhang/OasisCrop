python -m oasiscrop.inference.predict_tiles `
  --input-root "path\to\source_tiles" `
  --output-root "outputs\predicted_masks" `
  --weights "path\to\model_best.pth" `
  --threshold 0.5 `
  --min-area-m2 30
