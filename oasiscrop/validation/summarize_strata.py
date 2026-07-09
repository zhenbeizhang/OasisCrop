from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    eps = 1.0e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + fp + fn + tn + eps)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "oa": oa}


def run(args):
    metric_df = pd.read_csv(args.metric_csv)
    strata_df = pd.read_csv(args.strata_csv)
    merged = metric_df.merge(strata_df, on=args.join_column, how="left")
    rows = []
    for name, group in merged.groupby(args.stratum_column, dropna=False):
        counts = {key: int(group[key].sum()) for key in ["tp", "fp", "fn", "tn"]}
        rows.append({args.stratum_column: name, "n": len(group), **counts, **metrics(**counts)})
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize OasisCrop validation metrics by a regional or confusion stratum.")
    parser.add_argument("--metric-csv", required=True)
    parser.add_argument("--strata-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--join-column", default="file")
    parser.add_argument("--stratum-column", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
