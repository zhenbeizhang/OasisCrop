from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from osgeo import gdal
from tqdm import tqdm

from oasiscrop.data.transforms import Compose, Normalize, ToTensor
from oasiscrop.models import OasisCropNet


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint


def load_model(weights: Path, device: torch.device) -> OasisCropNet:
    model = OasisCropNet(num_class=1)
    model.load_state_dict(load_checkpoint(weights, device), strict=True)
    model.to(device).eval()
    return model


def gaussian_weight(size: int) -> np.ndarray:
    grid = np.linspace(-1, 1, size)
    x, y = np.meshgrid(grid, grid)
    return np.exp(-(x**2 + y**2)).astype(np.float32)


def read_image(path: Path, band_combination: list[int]) -> tuple[np.ndarray, tuple, str]:
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise RuntimeError(f"Cannot open input GeoTIFF: {path}")
    bands = []
    for band_index in band_combination:
        if band_index + 1 > dataset.RasterCount:
            raise ValueError(f"Band index {band_index} exceeds band count in {path}")
        arr = dataset.GetRasterBand(band_index + 1).ReadAsArray().astype(np.float32)
        bands.append(np.nan_to_num(arr, nan=0.0, posinf=1.0e4, neginf=-1.0e4))
    image = np.stack(bands, axis=2)
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    dataset = None
    return image, geotransform, projection


def remove_small_components(mask: np.ndarray, min_area_pixels: int) -> np.ndarray:
    if min_area_pixels <= 1:
        return mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area_pixels:
            cleaned[labels == label_id] = 1
    return cleaned


def min_area_to_pixels(min_area_m2: float, geotransform: tuple, fallback_pixel_area_m2: float = 1.0) -> int:
    pixel_area = abs(float(geotransform[1]) * float(geotransform[5])) if geotransform else fallback_pixel_area_m2
    if pixel_area <= 0:
        pixel_area = fallback_pixel_area_m2
    return max(1, int(round(min_area_m2 / pixel_area)))


def predict_array(model, image: np.ndarray, device: torch.device, window_size: int, stride: int, batch_size: int) -> np.ndarray:
    height, width, channels = image.shape
    score_sum = np.zeros((height, width), np.float32)
    weight_sum = np.zeros((height, width), np.float32)
    transform = Compose([ToTensor(), Normalize()])
    weight = gaussian_weight(window_size)
    patches: list[torch.Tensor] = []
    coords: list[tuple[int, int, int, int]] = []

    def flush():
        if not patches:
            return
        batch = torch.stack(patches).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch)).cpu().numpy()
        for prob, (y1, y2, x1, x2) in zip(probs, coords):
            pred = prob[0, : y2 - y1, : x2 - x1]
            patch_weight = weight[: y2 - y1, : x2 - x1]
            score_sum[y1:y2, x1:x2] += pred * patch_weight
            weight_sum[y1:y2, x1:x2] += patch_weight
        patches.clear()
        coords.clear()

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y2 = min(y + window_size, height)
            x2 = min(x + window_size, width)
            patch = image[y:y2, x:x2]
            if patch.shape[:2] != (window_size, window_size):
                padded = np.zeros((window_size, window_size, channels), np.float32)
                padded[: patch.shape[0], : patch.shape[1]] = patch
                patch = padded
            patches.append(transform(patch))
            coords.append((y, y2, x, x2))
            if len(patches) == batch_size:
                flush()
    flush()
    return score_sum / np.maximum(weight_sum, 1.0e-6)


def write_mask(path: Path, mask: np.ndarray, geotransform: tuple, projection: str, nodata: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), mask.shape[1], mask.shape[0], 1, gdal.GDT_Byte, options=["COMPRESS=LZW", "TILED=YES"])
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(mask.astype(np.uint8))
    band.SetNoDataValue(nodata)
    dataset.FlushCache()
    dataset = None


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = load_model(Path(args.weights), device)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    files = sorted(input_root.rglob("*.tif"))
    for input_path in tqdm(files, desc="Predicting"):
        rel = input_path.relative_to(input_root)
        output_path = output_root / rel.with_name(rel.stem + "_mask.tif")
        if output_path.exists() and not args.overwrite:
            continue
        image, geotransform, projection = read_image(input_path, args.band_combination)
        scores = predict_array(model, image, device, args.window_size, args.stride, args.batch_size)
        mask = (scores > args.threshold).astype(np.uint8)
        if args.min_area_m2 > 0:
            mask = remove_small_components(mask, min_area_to_pixels(args.min_area_m2, geotransform))
        mask = mask * args.cropland_value
        write_mask(output_path, mask, geotransform, projection, nodata=args.nodata)


def parse_args():
    parser = argparse.ArgumentParser(description="Sliding-window inference for OasisCropNet.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area-m2", type=float, default=30.0)
    parser.add_argument("--band-combination", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--cropland-value", type=int, default=1)
    parser.add_argument("--nodata", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
