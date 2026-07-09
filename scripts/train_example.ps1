python -m oasiscrop.training.train `
  --train-image-dir "path\to\annotation_samples\train\images" `
  --train-mask-dir "path\to\annotation_samples\train\masks" `
  --test-image-dir "path\to\annotation_samples\test\images" `
  --test-mask-dir "path\to\annotation_samples\test\masks" `
  --save-dir "outputs\checkpoints" `
  --device "cuda:0" `
  --epochs 500 `
  --batch-size 4 `
  --threshold 0.5
