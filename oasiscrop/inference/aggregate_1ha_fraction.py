from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from osgeo import gdal
from tqdm import tqdm


def block_mean(array: np.ndarray, factor: int) -> np.ndarray:
    height, width = array.shape
    out_h = int(np.ceil(height / factor))
    out_w = int(np.ceil(width / factor))
    padded = np.full((out_h * factor, out_w * factor), np.nan, dtype=np.float32)
    padded[:height, :width] = array
    blocks = padded.reshape(out_h, factor, out_w, factor)
    return np.nanmean(blocks, axis=(1, 3)).astype(np.float32)


def aggregate_file(input_path: Path, output_path: Path, factor: int, cropland_value: int) -> None:
    dataset = gdal.Open(str(input_path))
    if dataset is None:
        raise RuntimeError(f"Cannot open raster: {input_path}")
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    valid = np.ones_like(arr, dtype=bool) if nodata is None else arr != nodata
    binary = np.where(valid, arr == cropland_value, np.nan).astype(np.float32)
    fraction = block_mean(binary, factor) * 100.0

    geotransform = list(dataset.GetGeoTransform())
    geotransform[1] *= factor
    geotransform[5] *= factor
    projection = dataset.GetProjection()
    dataset = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(str(output_path), fraction.shape[1], fraction.shape[0], 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"])
    out.SetGeoTransform(tuple(geotransform))
    out.SetProjection(projection)
    out.GetRasterBand(1).WriteArray(fraction)
    out.GetRasterBand(1).SetNoDataValue(-9999)
    out.FlushCache()
    out = None


def run(args):
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    for path in tqdm(sorted(input_root.rglob("*.tif")), desc="Aggregating"):
        rel = path.relative_to(input_root)
        aggregate_file(path, output_root / rel, args.factor, args.cropland_value)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate binary 1 m OasisCrop masks to 1 ha cropland fraction rasters.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--factor", type=int, default=100, help="Aggregation factor; 100 converts 1 m pixels to 100 m cells.")
    parser.add_argument("--cropland-value", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
