#!/usr/bin/env python3
"""Aggregate all benchmark results into a single CSV for analysis."""
import json
import os
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

benchmarks = ["kmeans", "knn", "mlp", "ga", "fuzzy"]

rows = []
for bench in benchmarks:
    fpath = os.path.join(RESULTS, f"{bench}.json")
    if not os.path.exists(fpath):
        print(f"  SKIP {bench} (no results)")
        continue
    with open(fpath) as f:
        data = json.load(f)
    for result in data.get("results", []):
        rows.append({
            "benchmark": bench,
            "language": result["command"],
            "mean_s": result["mean"],
            "stddev_s": result["stddev"],
            "median_s": result["median"],
            "min_s": result["min"],
            "max_s": result["max"],
        })

out = os.path.join(RESULTS, "all_results.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["benchmark", "language", "mean_s", "stddev_s", "median_s", "min_s", "max_s"])
    w.writeheader()
    w.writerows(rows)
print(f"Aggregated {len(rows)} results -> {out}")
