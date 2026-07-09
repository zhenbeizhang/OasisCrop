from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal


def read_binary(path: Path, positive_value: int | None = None) -> np.ndarray:
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    dataset = None
    if positive_value is None:
        binary = arr > 0
    else:
        binary = arr == positive_value
    if nodata is not None:
        binary = np.where(arr == nodata, False, binary)
    return binary


def confusion(pred: np.ndarray, ref: np.ndarray) -> dict[str, int]:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    return {
        "tp": int(np.logical_and(pred, ref).sum()),
        "fp": int(np.logical_and(pred, ~ref).sum()),
        "fn": int(np.logical_and(~pred, ref).sum()),
        "tn": int(np.logical_and(~pred, ~ref).sum()),
    }


def metrics(row: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = row["tp"], row["fp"], row["fn"], row["tn"]
    eps = 1.0e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + fp + fn + tn + eps)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "oa": oa}


def run(args):
    pred_root = Path(args.prediction_root)
    ref_root = Path(args.reference_root)
    rows = []
    for pred_path in sorted(pred_root.rglob("*.tif")):
        rel = pred_path.relative_to(pred_root)
        ref_path = ref_root / rel
        if not ref_path.exists() and pred_path.name.endswith("_mask.tif"):
            ref_path = ref_root / rel.with_name(pred_path.name.replace("_mask.tif", ".tif"))
        if not ref_path.exists():
            rows.append({"file": str(rel), "status": "missing_reference"})
            continue
        counts = confusion(read_binary(pred_path, args.prediction_value), read_binary(ref_path, args.reference_value))
        rows.append({"file": str(rel), "status": "ok", **counts, **metrics(counts)})

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    ok = [r for r in rows if r.get("status") == "ok"]
    if ok:
        total = {key: sum(r[key] for r in ok) for key in ["tp", "fp", "fn", "tn"]}
        print({**total, **metrics(total)})


def parse_args():
    parser = argparse.ArgumentParser(description="Validate binary OasisCrop rasters against reference masks.")
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--prediction-value", type=int, default=None)
    parser.add_argument("--reference-value", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
